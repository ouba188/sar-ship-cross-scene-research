"""Image-only intervention selection for the selected OSS2 M1/M2/M3 model."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import os
import sys
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
OLD = WORK/'m1_guided_m2_20261106'
spec = importlib.util.spec_from_file_location('selected_guidance', OLD/'study.py')
s = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s
spec.loader.exec_module(s)
torch, np, pd, task = s.torch, s.np, s.pd, s.task
from sar_m123.current import interventions as intervention
from sar_m123.current.training import adaptation_loss, class_weights
from sar_m123.pool_replacement import grid_occupancy, regional_mean
from geometry import build_masks, pool_multiscale

_tensor = task.tensor
def tensor(value, ids):
    if isinstance(value, torch.Tensor):
        return value.index_select(0, torch.as_tensor(ids, device=value.device, dtype=torch.long))
    return _tensor(value, ids)
task.tensor = tensor


def write(name, value):
    task.write(ROOT/name, value)


class Teacher:
    def __init__(self):
        self.net = s.prev.Network('class')
        self.net.restore(torch.load(s.CHECKPOINT, map_location='cpu', weights_only=True))
        self.net.eval().requires_grad_(False)
        self.reader = s.RegionalGuidance('class').to(task.DEVICE)
        self.reader.load_state_dict(torch.load(OLD/'class_s17/final.pt', map_location=task.DEVICE, weights_only=True))
        self.reader.eval().requires_grad_(False)
        self.seen = {}
        self.handles = []
        self.handles.append(self.net.base.head.register_forward_pre_hook(self.head_hook))
        block = self.net.base.backbone.model.layer3[5]
        self.handles.append(block.conv1.register_forward_pre_hook(self.identity_hook))
        for p in range(2):
            self.handles.append(self.net.m1[p].reader.register_forward_hook(self.response_hook(p)))
            self.handles.append(self.net.base.m2[p].register_forward_hook(self.pool_hook(p)))

    def head_hook(self, module, args):
        self.seen['features'] = args[0]

    def identity_hook(self, module, args):
        self.seen['identity'] = args[0]

    def response_hook(self, p):
        def hook(module, args, output):
            self.seen[('response', p)] = output
        return hook

    def pool_hook(self, p):
        def hook(module, args, output):
            self.seen[('x', p)] = args[0]
            self.seen[('mask', p)] = torch.minimum(args[1], grid_occupancy(args[2], args[0].shape[-2:]))
            self.seen[('low', p)] = output['pooled']
        return hook

    @torch.no_grad()
    def forward(self, a, c, ids, details=False):
        self.net(a, c, ids)
        valid = tensor(a['valid'], ids)
        available = tensor(a['available'], ids)
        batch = {key: torch.stack([self.seen[(key, p)] for p in range(2)], 1)
                 for key in ('x', 'mask', 'response')}
        batch.update(features=self.seen['features'], available=available,
                     valid=grid_occupancy(valid.flatten(0, 1), (20, 20)).reshape(len(ids), 2, 1, 20, 20))
        score, trace = self.reader(batch, details=True)
        if not details:
            return score
        values = dict(identity=self.seen['identity'].reshape(len(ids), 2, 1024, 10, 10),
                      low=torch.stack([self.seen[('low', p)] for p in range(2)], 1),
                      guidance=torch.stack([t['delta'] for t in trace], 1),
                      delta=tensor(c['delta'], ids), valid=valid,
                      slots=tensor(a['slots'], ids), available=available, score=score)
        return score, values

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class RawTeacher:
    def __init__(self, teacher, rows, groups):
        self.teacher, self.rows, self.groups = teacher, rows, groups
        self.anchor = s.prev.e.previous.anchor()
        self.a, self.c = s.prev.arrays()
        self.rid = groups['references']
        self.refdesc = tensor(self.a['desc'], self.rid)
        self.refmass = tensor(self.a['mass'], self.rid)
        self.refavailable = tensor(self.a['available'], self.rid).bool()
        self.reflabels = torch.tensor([rows[i]['label'] for i in self.rid], device=task.DEVICE)

    @torch.no_grad()
    def forward(self, idx, items, geometry):
        b = task.data.batch(items, task.DEVICE)
        n = len(items)
        bb = self.anchor.base.backbone.model
        f1 = self.anchor.base.prefix(b['images'], stop=1)
        z = f1.flatten(0, 1)
        for layer in list(bb.layer2.children())[:3]:
            z = layer(z)
        identity = z
        block = bb.layer2[3]
        x = block.relu(block.bn1(block.conv1(z)))
        x = block.relu(block.bn2(block.conv2(x)))
        f2 = block.relu(block.bn3(block.conv3(x))+identity)
        read = torch.tensor(np.stack([g['read'] for g in geometry]), device=task.DEVICE).repeat(n, 1, 1, 1)
        desc = pool_multiscale(f1.flatten(0,1), f2, read, 'coarse_fixed').reshape(n, 2, 20, 768)
        mass = torch.tensor(np.stack([g['mass'] for g in geometry]), device=task.DEVICE)[None].expand(n,-1,-1)
        writer = torch.tensor(np.stack([g['write'] for g in geometry]), device=task.DEVICE)[None].expand(n,-1,-1,-1,-1)
        pair = task.legal(self.rows, [idx]*n, self.rid)[...,None] & b['available'][:,None].bool() & self.refavailable[None]
        delta = self.anchor.matcher(desc, self.refdesc, mass, self.refmass, writer, pair, self.reflabels, arm='aggregate')[0]
        a = dict(valid=b['valid'], available=b['available'], slots=b['slot_masks'])
        c = dict(early_x=x.reshape(n,2,128,20,20),
                 early_identity=identity.reshape(n,2,512,20,20),
                 local_slots=b['local_slot_masks'], delta=delta)
        return self.teacher.forward(a, c, np.arange(n))


@torch.no_grad()
def prepare():
    rows, groups = s.prev.records()
    teacher = Teacher()
    a, c = s.prev.arrays()
    if not (ROOT/'cache/complete.json').exists():
        dest = ROOT/'cache'
        dest.mkdir(parents=True, exist_ok=True)
        gids = np.r_[groups['source_adapt'], groups['validation'], groups['evaluation']]
        lookup = {int(g):i for i,g in enumerate(gids)}
        local = {key:[lookup[int(g)] for g in groups[key]] for key in ('source_adapt','validation','evaluation')}
        stores = {}
        for first in range(0, len(gids), 8):
            ids = gids[first:first+8]
            _, values = teacher.forward(a,c,ids,True)
            for key,value in values.items():
                arr=value.cpu().numpy()
                if key not in stores:
                    stores[key]=np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=np.float32,shape=(len(gids),)+arr.shape[1:])
                stores[key][first:first+len(ids)]=arr
            if first%160==0:
                print('CACHE',first+len(ids),len(gids),flush=True)
        errors={}
        for split in ('validation','evaluation'):
            old=np.load(OLD/f'class_s17/{split}.npz')
            actual=stores['score'][local[split]]
            assert np.array_equal(old['ids'],groups[split])
            errors[split]=float(abs(actual-old['score']).max())
            assert errors[split]<1e-4 and np.array_equal(actual.argmax(1),old['score'].argmax(1)),errors
        for store in stores.values():
            store.flush()
        write('groups.json',local)
        write('global_indices.json',gids.tolist())
        torch.save(teacher.net.base.state_dict(),dest/'base.pt')
        torch.save(teacher.reader.head.state_dict(),dest/'head.pt')
        write('cache/complete.json',dict(n=len(gids),replay_errors=errors,teacher='current class_s17, fixed aggregate M3'))
        del stores
    if not (ROOT/'selection.json').exists():
        raw = RawTeacher(teacher, rows, groups)
        selected=[]
        original=np.load(ROOT/'cache/score.npy',mmap_mode='r')
        max_error=0.
        tick=time.perf_counter()
        for number,idx in enumerate(groups['source_adapt']):
            idx=int(idx);row=rows[idx]
            task.data.assert_source([row])
            items,meta=intervention.variants(row)
            views=task.data.prepare(row)
            geometry=[build_masks(v['geometry'] if v is not None and v['valid_target'] else None,
                                   v['footprint'] if v is not None and v['valid_target'] else np.zeros((160,160)),
                                   'coarse_fixed') for v in views]
            scores=np.concatenate([raw.forward(idx,items[first:first+8],geometry).cpu().numpy()
                                   for first in range(0,len(items),8)])
            error=float(abs(scores[0]-original[number]).max())
            max_error=max(max_error,error)
            assert error<1e-4,(idx,error)
            choice=intervention.select_teacher(scores,meta,row['label'])
            y=row['label'];clean=torch.tensor(scores[0]);altered=torch.tensor(scores[choice['chosen_index']])
            def margin(score):
                other=score.clone();other[y]=-torch.inf
                return score[y]-other.max()
            useful=choice['useful'] and bool(altered.log_softmax(0)[y]>clean.log_softmax(0)[y] and margin(altered)>margin(clean))
            selected.append(dict(id=row['id'],label=y,useful=useful,original=clean.tolist(),altered=altered.tolist(),
                                 choice=choice,selected_intervention=meta[choice['chosen_index']],
                                 metadata=meta,scores=scores.tolist(),original_replay_error=error))
            if number%20==0 or number+1==len(groups['source_adapt']):
                write('selection_progress.json',dict(done=number+1,total=len(groups['source_adapt']),
                      useful=sum(z['useful'] for z in selected),seconds=time.perf_counter()-tick,max_original_error=max_error))
                print('SELECT',number+1,len(groups['source_adapt']),sum(z['useful'] for z in selected),flush=True)
        write('selection.json',selected)
        del raw
    teacher.close()
    del teacher,a,c
    gc.collect();torch.cuda.empty_cache()


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=task.base_model()
        self.base.m3=torch.nn.ModuleList()
        self.base.load_state_dict(torch.load(ROOT/'cache/base.pt',map_location=task.DEVICE,weights_only=True),strict=True)
        self.base.head.load_state_dict(torch.load(ROOT/'cache/head.pt',map_location=task.DEVICE,weights_only=True))
        self.base.eval().requires_grad_(False)
        for name,p in self.base.named_parameters():
            p.requires_grad_(name in task.current.LIVE)

    def forward(self,b):
        valid,slots,av=b['valid'],b['slots'],b['available']
        block=self.base.backbone.model.layer3[5]
        identity=b['identity'].flatten(0,1)
        x=block.relu(block.bn1(block.conv1(identity))).reshape(len(av),2,256,10,10)
        out=[]
        for p in range(2):
            reg,en=self.base.region_integrals(slots[:,p],slots.new_zeros(len(av),20,3,10,10))
            out.append(self.base.internal_m2[p](x[:,p],reg,valid[:,p],en)[0])
        x=torch.stack(out,1).flatten(0,1)
        x=block.relu(block.bn2(block.conv2(x)))
        x=block.relu(block.bn3(block.conv3(x))+identity).reshape(len(av),2,1024,10,10)
        updated=x+b['delta']
        deep=self.base.backbone.model.layer4(updated.flatten(0,1))
        glob=regional_mean(deep,valid.flatten(0,1),valid.flatten(0,1))[0][:,0].reshape(len(av),2,-1)
        pooled=[]
        for p in range(2):
            reg,en=self.base.region_integrals(slots[:,p],slots.new_zeros(len(av),20,3,10,10))
            extra=self.base.deep_pool[p](updated[:,p],reg,valid[:,p],en)
            low=b['low'][:,p]+self.base.deep_output[p](extra['pooled'])
            low=low*av[:,p,None,None]+b['guidance'][:,p]
            pooled.append(low.flatten(1))
        features=torch.cat([glob[:,0]*av[:,0,None],pooled[0],glob[:,1]*av[:,1,None],pooled[1],av],1)
        return self.base.head(features)


def cache():
    out={}
    for file in (ROOT/'cache').glob('*.npy'):
        a=np.load(file,mmap_mode='r')
        value=torch.empty(a.shape,dtype=torch.float32,device=task.DEVICE)
        for first in range(0,len(a),16):
            value[first:first+16]=torch.from_numpy(np.array(a[first:first+16],copy=True)).to(task.DEVICE)
        out[file.stem]=value
    return out


@torch.no_grad()
def predict(net,arrays,ids):
    return np.concatenate([net({key:tensor(value,ids[first:first+16]) for key,value in arrays.items()}).cpu().numpy()
                           for first in range(0,len(ids),16)])


def checks():
    arrays=cache();net=Student()
    probe=np.arange(len(arrays['score']))
    z=predict(net,arrays,probe)
    old=arrays['score'].cpu().numpy()
    error=float(abs(z-old).max())
    assert error<1e-4 and np.array_equal(z.argmax(1),old.argmax(1)),error
    params={n:p for n,p in net.base.named_parameters() if p.requires_grad}
    assert set(params)==set(task.current.LIVE),list(params)
    write('student_checks.json',dict(replay_error=error,n=len(probe),trainable=list(params),
                                    parameters=sum(p.numel() for p in params.values())))
    print('STUDENT_CHECK',error,list(params),flush=True)


def train():
    assert (ROOT/'student_checks.json').exists(), 'Run exact suffix checks first.'
    arrays=cache()
    rows,_=s.prev.records()
    gids=np.array(json.loads((ROOT/'global_indices.json').read_text(encoding='utf-8')))
    groups=json.loads((ROOT/'groups.json').read_text(encoding='utf-8'))
    ids=np.array(groups['source_adapt'])
    records=[rows[gids[i]] for i in ids]
    selections=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))
    assert [r['id'] for r in records]==[r['id'] for r in selections]
    labels=np.array([r['label'] for r in records])
    useful=np.array([r['useful'] for r in selections],bool)
    quota={int(k):int(useful[labels==k].sum()) for k in np.unique(labels)}
    weights=class_weights(records,'adapt')
    coefficient=.3683066666126251
    write('selection_summary.json',dict(n=len(records),selected=int(useful.sum()),by_class=quota,
                                      initial_teacher='class_s17',seeds=[17,29,43]))
    assert useful.any(), 'No eligible source interventions; no selection contrast is available.'
    for seed in (17,29,43):
        ordinary=np.zeros(len(ids),bool)
        selector=np.random.default_rng(seed+10000)
        for label,n in quota.items():
            ordinary[selector.choice(np.flatnonzero(labels==label),n,replace=False)]=True
        for arm,mask in (('ordinary',ordinary),('intervention',useful)):
            dest=ROOT/f'{arm}_s{seed}'
            if (dest/'complete.json').exists():
                continue
            dest.mkdir(parents=True,exist_ok=True)
            net=Student()
            params={n:p for n,p in net.base.named_parameters() if p.requires_grad}
            optimizer=torch.optim.AdamW(params.values(),lr=3e-5,weight_decay=.01)
            rng=np.random.default_rng(seed)
            curve=[];steps=0;tick=time.perf_counter()
            for epoch in range(1,7):
                values=[]
                for local in np.array_split(rng.permutation(len(ids)),range(16,len(ids),16)):
                    batch={k:tensor(v,ids[local]) for k,v in arrays.items()}
                    optimizer.zero_grad(set_to_none=True)
                    score=net(batch)
                    teacher=batch['score']
                    if steps==0:
                        assert float((score.detach()-teacher).abs().max())<1e-4
                    loss,terms=adaptation_loss(score,torch.tensor(labels[local],device=task.DEVICE),teacher,
                        torch.tensor(weights[local],device=task.DEVICE),
                        torch.tensor(mask[local],device=task.DEVICE),coefficient)
                    assert bool(torch.isfinite(loss))
                    loss.backward();optimizer.step();steps+=1
                    values.append({k:float(v.detach()) for k,v in terms.items()})
                row=dict(epoch=epoch,steps=steps,**{k:float(np.mean([v[k] for v in values])) for k in values[0]})
                curve.append(row)
                print('TRAIN',arm,seed,row,flush=True)
            assert steps==96
            state={n:p.detach().cpu() for n,p in params.items()}
            torch.save(state,dest/'adaptation.pt')
            metrics={}
            for split in ('validation','evaluation'):
                splitids=np.array(groups[split]);y=np.array([rows[gids[i]]['label'] for i in splitids])
                z=predict(net,arrays,splitids)
                np.savez_compressed(dest/f'{split}.npz',ids=gids[splitids],labels=y,score=z)
                metrics[split]=task.metrics(y,z)
            write(f'{arm}_s{seed}/complete.json',dict(arm=arm,seed=seed,steps=steps,curve=curve,metrics=metrics,
                  selected=int(mask.sum()),by_class=quota,selected_local_indices=np.flatnonzero(mask).tolist(),
                  seconds=time.perf_counter()-tick,trainable=list(params),coefficient=coefficient))
            del net,optimizer,params,state
            gc.collect();torch.cuda.empty_cache()


def metrics(y,z):
    result=task.metrics(y,z)
    p=torch.tensor(z,dtype=torch.float64).softmax(-1).numpy()
    result['brier']=float(((p-np.eye(z.shape[1])[y])**2).sum(1).mean())
    return result


def analyze():
    rows,globalgroups=s.prev.records()
    groups=json.loads((ROOT/'groups.json').read_text(encoding='utf-8'))
    gids=np.array(json.loads((ROOT/'global_indices.json').read_text(encoding='utf-8')))
    original=np.load(ROOT/'cache/score.npy',mmap_mode='r')
    summary=[];pairs=[];strata=[];events=[];entries={}
    for split in ('validation','evaluation'):
        ids=gids[groups[split]];y=np.array([rows[i]['label'] for i in ids])
        base=original[groups[split]]
        summary.append(dict(split=split,seed=17,arm='before_adaptation',**metrics(y,base)))
        for seed in (17,29,43):
            scores={'before_adaptation':base}
            for arm in ('ordinary','intervention'):
                z=np.load(ROOT/f'{arm}_s{seed}/{split}.npz')
                assert np.array_equal(z['ids'],ids) and np.array_equal(z['labels'],y)
                scores[arm]=z['score']
                summary.append(dict(split=split,seed=seed,arm=arm,**metrics(y,z['score'])))
            entries[(split,seed)]=scores
            for control in ('ordinary','before_adaptation'):
                mm=metrics(y,scores['intervention']);bb=metrics(y,scores[control])
                good=scores['intervention'].argmax(1)==y;old=scores[control].argmax(1)==y
                pairs.append(dict(split=split,seed=seed,control=control,rescues=int((good&~old).sum()),
                                  harms=int((~good&old).sum()),**{'delta_'+k:mm[k]-bb[k] for k in ('oa','ba','nll','brier')}))
                if control!='ordinary':continue
                for key in ('label','orbit'):
                    values=np.array([rows[i]['label'] if key=='label' else task.orbit(rows[i]) for i in ids])
                    for val in np.unique(values):
                        mask=values==val
                        strata.append(dict(split=split,seed=seed,group=key,value=str(val),n=int(mask.sum()),
                            delta_oa=100*float((good[mask].astype(float)-old[mask]).mean()),
                            rescues=int((good&~old&mask).sum()),harms=int((~good&old&mask).sum())))
                for pos in np.flatnonzero(good!=old):
                    row=rows[ids[pos]]
                    events.append(dict(split=split,seed=seed,id=row['id'],mmsi=row['mmsi'],orbit=task.orbit(row),
                                       label=int(y[pos]),effect='rescue' if good[pos] else 'harm'))
    frame=pd.DataFrame(summary);pair=pd.DataFrame(pairs)
    frame.to_csv(ROOT/'summary.csv',index=False);pair.to_csv(ROOT/'paired.csv',index=False)
    pd.DataFrame(strata).to_csv(ROOT/'strata.csv',index=False)
    pd.DataFrame(events,columns=['split','seed','id','mmsi','orbit','label','effect']).to_csv(ROOT/'events.csv',index=False)
    averages=frame.groupby(['split','arm'])[['oa','ba','nll','brier']].agg(['mean','std'])
    averages.to_csv(ROOT/'averages.csv')
    pair.groupby(['split','control'])[['delta_oa','delta_ba','delta_nll','delta_brier','rescues','harms']].mean().to_csv(ROOT/'paired_means.csv')
    spec=importlib.util.spec_from_file_location('intervention_gallery',task.ROOT/'review_task.py')
    gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery);gallery.ROOT=ROOT
    a,_=s.prev.arrays();ids=globalgroups['evaluation'];y=np.array([rows[i]['label'] for i in ids])
    cases=[];statuses=[];used=set()
    for group in ('rescue','harm','stable_correct','stable_wrong'):
        choices=[]
        for seed in (17,29,43):
            scores=entries[('evaluation',seed)];z=scores['intervention'];b=scores['ordinary']
            good=z.argmax(1)==y;old=b.argmax(1)==y
            mask={'rescue':good&~old,'harm':~good&old,'stable_correct':good&old,'stable_wrong':~good&~old}[group]
            delta=abs(gallery.margins(z,y)-gallery.margins(b,y))
            choices.extend((float(delta[p]),seed,int(p)) for p in np.flatnonzero(mask))
        choices.sort(reverse=True)
        pick=next((v for v in choices if int(ids[v[2]]) not in used),None)
        statuses.append(dict(group=group,events=len(choices),selected=pick))
        if pick is None:continue
        _,seed,pos=pick;idx=int(ids[pos]);used.add(idx)
        scores=dict(entries[('evaluation',seed)]);scores['baseline']=scores['ordinary']
        case=gallery.draw_case(rows,a,globalgroups['references'],idx,f'{group}_s{seed}',scores,y,pos,focus='intervention')
        case.update(seed=seed,split='evaluation');cases.append(case)
    write('cases.json',dict(cases=cases,groups=statuses,selection='Largest absolute paired margin difference per group, unique observation; not prevalence'))
    import html
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>新版干预训练对照</title>',
          '<style>body{font-family:Arial,sans-serif;max-width:1300px;margin:24px auto;padding:16px}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>',
          '<h1>当前完整模型：干预引导与同配额普通选样</h1>',
          '<p>OSS2四分类；固定当前教师seed17；三种子仅改变后续适配顺序和普通组抽样。未改模块、未按目标结果选参数。</p>',
          '<p>所有241条源原图参与训练；只在选中样本加原教师判别分差下界。NLL/Brier负差更好。参考图是固定合法库中的案例，不是注意力最大值。</p>',
          averages.to_html(),pair.to_html(index=False)]
    for case in cases:
        body.extend(['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2><img src="'+case['image']+'">',
                     '<pre>'+html.escape(json.dumps(case['evidence'],indent=2))+'</pre>'])
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(averages.to_string(),flush=True);print(pair.to_string(index=False),flush=True)


def diagnose():
    arrays=cache();rows,_=s.prev.records()
    gids=np.array(json.loads((ROOT/'global_indices.json').read_text(encoding='utf-8')))
    groups=json.loads((ROOT/'groups.json').read_text(encoding='utf-8'))
    ids=np.array(groups['source_adapt']);y=np.array([rows[gids[i]]['label'] for i in ids])
    selections=json.loads((ROOT/'selection.json').read_text(encoding='utf-8'))
    useful=np.array([r['useful'] for r in selections])
    evidence=[];checks=[];composition=[];flips=[]
    original=arrays['score'].cpu().numpy()
    for name,mask in (('all',np.ones(len(ids),bool)),('selected',useful),('not_selected',~useful)):
        if not mask.any():continue
        rr=[rows[gids[i]] for i in ids[mask]]
        composition.append(dict(group=name,n=len(rr),vessels=len({r['mmsi'] for r in rr}),
                               passes=len({task.orbit(r) for r in rr}),
                               counts={int(k):int((y[mask]==k).sum()) for k in np.unique(y)},
                               teacher_metrics=metrics(y[mask],original[ids[mask]])))
    evidence.append(dict(arm='before_adaptation',seed=17,**metrics(y,original[ids])))
    for seed in (17,29,43):
        for arm in ('ordinary','intervention'):
            net=Student();params=dict(net.base.named_parameters())
            state=torch.load(ROOT/f'{arm}_s{seed}/adaptation.pt',map_location=task.DEVICE,weights_only=True)
            assert set(state)==set(task.current.LIVE)
            with torch.no_grad():
                for k,v in state.items():params[k].copy_(v)
            z=predict(net,arrays,ids)
            evidence.append(dict(arm=arm,seed=seed,**metrics(y,z)))
            probe=np.array(groups['evaluation'])[:16]
            actual=predict(net,arrays,probe)
            old=np.load(ROOT/f'{arm}_s{seed}/evaluation.npz')['score'][:16]
            error=float(abs(actual-old).max());assert error<1e-4
            checks.append(dict(arm=arm,seed=seed,reload_error=error))
            for split in ('validation','evaluation'):
                ii=np.array(groups[split]);yy=np.array([rows[gids[i]]['label'] for i in ii])
                zz=np.load(ROOT/f'{arm}_s{seed}/{split}.npz')['score'];base=original[ii]
                good=zz.argmax(1)==yy;before=base.argmax(1)==yy
                for label in np.unique(yy):
                    m=yy==label
                    flips.append(dict(arm=arm,seed=seed,split=split,label=int(label),n=int(m.sum()),
                                      rescues=int((good&~before&m).sum()),harms=int((~good&before&m).sum())))
            del net,params,state;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(evidence).to_csv(ROOT/'source_fit_diagnostics.csv',index=False)
    pd.DataFrame(flips).to_csv(ROOT/'adaptation_vs_initial_classes.csv',index=False)
    write('diagnostics.json',dict(source_composition=composition,reload_checks=checks,
                                 interpretation='Source metrics diagnose optimization versus transfer; not causal attribution to a specific image factor.'))
    print(pd.DataFrame(evidence).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','checks','train','analyze','diagnose'))
    args=parser.parse_args();task.current.setup(task.DEVICE)
    if args.action=='prepare':prepare()
    elif args.action=='checks':checks()
    elif args.action=='train':train()
    elif args.action=='analyze':analyze()
    else:diagnose()


if __name__=='__main__':main()
