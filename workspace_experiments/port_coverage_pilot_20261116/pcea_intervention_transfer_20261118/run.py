"""Isolated PCEA candidate. Default: bounded source-only integration check."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parents[1]/'paper_draft_20261108'
spec = importlib.util.spec_from_file_location('side_intervention_routed', PAPER/'pcea_routed_correction_20261118/run.py')
recipe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = recipe
spec.loader.exec_module(recipe)
m, torch, np = recipe.m, recipe.torch, recipe.np
from objective import CONDITIONS, REGIONS, make_interventions, select_targets, auxiliary, margin, MATCHED_MODES, matched_auxiliary

CHECKPOINT = PAPER/'pcea_routed_correction_20261118/fits/routed_s17/best.pt'


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def model(variant='legacy'):
    if variant == 'unified':
        from unified_model import Model
        net = Model()
    elif variant == 'legacy':
        net = m.Model('generic')
        net.load_state_dict(torch.load(CHECKPOINT, map_location='cuda', weights_only=True), strict=True)
    else:
        raise ValueError(variant)
    net.accelerate()
    return net


def runtime(variant='legacy'):
    # Do not call the inherited runtime(), which writes main-thread reports.
    if variant == 'unified':
        from unified_model import Inputs
        inputs = Inputs()
    elif variant == 'legacy':
        inputs = m.Inputs()
    else:
        raise ValueError(variant)
    split = json.loads((PAPER/'pcea_routed_correction_20261118/split_local.json').read_text(encoding='utf-8'))
    fit, inner = [np.asarray(split[k], np.int64) for k in ('fit','inner')]
    inputs.weights = m.old.t.e.weights(inputs.rows, fit)
    return inputs, fit, inner


def optimizer_groups(net, variant='legacy'):
    if variant == 'unified':
        from unified_model import groups_for
        return groups_for(net)
    return recipe.previous.groups_for(net)


def native_masks(inputs, ids, code):
    masks = np.zeros((len(ids), 2, 3, 160, 160), bool)
    raw, task = m.old.t.raw, m.old.t.e.task
    for j, i in enumerate(ids):
        for p, view in enumerate(task.data.prepare(inputs.rows[i])):
            if view is None or not view['valid_target']:
                continue
            soft = raw.tokens.soft_regions(view['geometry'])
            used = np.zeros((160,160), bool)
            for r in range(3):
                mask = (soft[r] >= .5) & view['footprint'].astype(bool) & ~used
                used |= mask
                masks[j,p,r] = m.old.t.ft.transform(mask, code)
    return masks


def forward_data(net, data, inputs, ids, reference):
    return net(data, [inputs.rows[i] for i in ids], reference, inputs.reference_rows)


@torch.no_grad()
def measure(teacher, inputs, ids, code=0, include_off=False):
    teacher.eval()
    teacher.exchange.disabled = False
    data = inputs.data(ids, 'small', code)
    masks = native_masks(inputs, ids, code)
    arrays = [data[k].cpu().numpy() for k in ('images','valid')]
    interventions, controls, valid, diagnostics = make_interventions(
        *arrays, masks, data['available'].cpu().numpy())
    reference = teacher.reference(inputs.data(inputs.groups['references'], 'small'))
    clean, _ = forward_data(teacher,data,inputs,ids,reference)
    active = teacher.exchange.trace['active'].any(1).detach().clone()
    clean_off = None
    if include_off:
        teacher.exchange.disabled = True
        clean_off, _ = forward_data(teacher,data,inputs,ids,reference)
        teacher.exchange.disabled = False
    iz, cz = [], []
    for candidate, control in zip(interventions,controls):
        for arr, scores in ((candidate,iz),(control,cz)):
            changed = dict(data, images=torch.as_tensor(arr, device='cuda'))
            z,_ = forward_data(teacher,changed,inputs,ids,reference)
            scores.append(z)
    y = torch.as_tensor(inputs.y[ids],device='cuda')
    targets = select_targets(clean, torch.stack(iz), torch.stack(cz), y,
                             torch.as_tensor(valid,device='cuda'))
    untouched_error = 0.
    for c,(p,r,_) in enumerate(CONDITIONS):
        change = interventions[c]-arrays[0]
        allowed = np.zeros_like(change,dtype=bool)
        allowed[:,p,0] = masks[:,p,r]
        untouched_error = max(untouched_error,float(abs(change[~allowed]).max()))
    assert untouched_error == 0.
    return targets, dict(images=arrays[0], masks=masks, interventions=interventions,
                         controls=controls, intervention_scores=torch.stack(iz).cpu().numpy(),
                         control_scores=torch.stack(cz).cpu().numpy(), diagnostics=diagnostics,
                         untouched_error=untouched_error, active=active.cpu().numpy(),
                         clean_off=None if clean_off is None else clean_off.cpu().numpy())


def backward(net, inputs, ids, code, targets, mode='relations', audit_routing=False):
    net.train()
    net.exchange.disabled = True
    off_loss, off = m.full.loss_for(net,inputs,ids,code)
    detached = off.detach()
    (.5*off_loss).backward()
    del off_loss, off
    net.exchange.disabled = False
    on_loss, on = m.full.loss_for(net,inputs,ids,code)
    y = torch.as_tensor(inputs.y[ids],device='cuda')
    weights = torch.as_tensor(inputs.weights[ids],device='cuda')
    active = net.exchange.trace['active'].any(1)
    if mode == 'routed':
        extra, stats = recipe.b.correction(on,detached,targets['clean'],y,weights,active)
    elif mode in MATCHED_MODES:
        extra, stats = matched_auxiliary(on,detached,y,weights,targets,active,mode)
    else:
        extra, stats = auxiliary(on,detached,y,weights,targets,active,mode)
    params = [(n,p) for n,p in net.named_parameters()
              if p.requires_grad and n.startswith(('ccrel.','exchange.'))]
    grads = torch.autograd.grad(extra,[p for _,p in params],retain_graph=True,allow_unused=True)
    (.5*on_loss).backward()
    before = {n:p.grad.detach().clone() for n,p in net.named_parameters()
              if audit_routing and p.grad is not None and not n.startswith(('ccrel.','exchange.'))}
    norms = {'reader':0.,'exchange':0.}
    for (name,p),g in zip(params,grads):
        if g is not None:
            if p.grad is None:
                p.grad = g.detach().clone()
            else:
                p.grad.add_(g.detach())
            key = 'reader' if name.startswith('ccrel.') else 'exchange'
            norms[key] += float(g.detach().square().sum())
    stats.update(extra=float(extra.detach()), auxiliary_norm={k:v**.5 for k,v in norms.items()})
    if audit_routing:
        delta = max((float((p.grad-before[n]).abs().max()) for n,p in net.named_parameters() if n in before), default=0.)
        assert delta == 0., delta
        stats['routing_non_target_delta'] = delta
    return on.detach(), detached, stats


def save_cases(inputs, ids, targets, details, after, off):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    selected=targets['selected'].cpu().numpy()
    base=targets['clean'].cpu().numpy()
    reference_ids=inputs.groups['references']
    reference_pixels=inputs.data(reference_ids,'small')['images'].cpu().numpy()
    for j, idx in enumerate(ids):
        c=int(selected[j]); p,r,band=CONDITIONS[c]
        row=inputs.rows[idx]
        record=dict(index=int(idx),id=row['id'],port=row['port'],label=int(row['label']),
            condition=[('VV','VH')[p],REGIONS[r],band],selected=bool(targets['use'][j]),
            teacher_before=base[j].tolist(),teacher_intervention=details['intervention_scores'][c,j].tolist(),
            teacher_level_control=details['control_scores'][c,j].tolist(),
            student_on=after[j].tolist(),student_off=off[j].tolist(),
            raw_paths={pol:row.get(pol,{}).get('image') if row.get(pol) else None for pol in ('vv','vh')})
        records.append(record)
        if j >= 4:
            continue
        rival=int(np.argmax(np.where(np.arange(base.shape[1]) == row['label'],-np.inf,base[j])))
        same=int(np.flatnonzero(inputs.y[reference_ids] == row['label'])[0])
        other=int(np.flatnonzero(inputs.y[reference_ids] == rival)[0])
        record['reference_examples']=[int(reference_ids[same]),int(reference_ids[other])]
        fig,axes=plt.subplots(2,5,figsize=(13,5))
        for polar in range(2):
            for col,(name,img) in enumerate((('Original',details['images'][j,polar,0]),
                ('Intervention',details['interventions'][c,j,polar,0]),
                ('Difference',details['interventions'][c,j,polar,0]-details['images'][j,polar,0]),
                ('Same-class ref',reference_pixels[same,polar,0]),
                ('Rival-class ref',reference_pixels[other,polar,0]))):
                if col == 2:
                    v=max(float(abs(img).max()),1e-6)
                    axes[polar,col].imshow(img,cmap='RdBu_r',vmin=-v,vmax=v,interpolation='nearest')
                else:
                    axes[polar,col].imshow(img,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[polar,col].set_title(('VV','VH')[polar]+' '+name)
                axes[polar,col].axis('off')
        fig.suptitle(f"Source {row['id']} | {record['condition']} | selected={record['selected']}")
        fig.tight_layout()
        fig.savefig(ROOT/f'case_{j:02d}.png',dpi=120)
        plt.close(fig)
    write('cases_local.json',records)


def verify():
    tick=time.perf_counter()
    inputs,fit,_=runtime()
    # One source-fit ship per class, selected without examining predictions.
    refs=set(inputs.groups['references'].tolist())
    ids=np.array([next(int(i) for i in fit if inputs.y[i] == c and i not in refs
                       and bool(np.asarray(inputs.arrays['available'][i]).all())) for c in range(8)])
    net=model()
    targets,details=measure(net,inputs,ids)
    before=targets['clean'].cpu().numpy()
    y=torch.as_tensor(inputs.y[ids],device='cuda')
    measured=dict(selected=int(targets['use'].sum()), total=len(ids),
                  condition_names=[list(c) for c in CONDITIONS],
                  good=targets['good'].cpu().tolist(), source_ids=ids.tolist())
    write('teacher_measurements_local.json',measured)
    np.savez_compressed(ROOT/'source_probe_local.npz',ids=ids,label=inputs.y[ids],
        clean=before,interventions=details['intervention_scores'],controls=details['control_scores'],
        selected=targets['selected'].cpu().numpy(),use=targets['use'].cpu().numpy())
    # Fixed target tensors remain detached while the original-image student trains.
    groups=recipe.previous.groups_for(net)
    net.zero_grad(set_to_none=True)
    z,off,stats=backward(net,inputs,ids,0,targets)
    total={g['name']:sum(float(p.grad.square().sum()) for p in g['params'] if p.grad is not None)**.5 for g in groups}
    assert all(np.isfinite(v) and v > 0 for v in total.values()), total
    assert all(np.isfinite(v) and v > 0 for v in stats['auxiliary_norm'].values()), stats
    replay=float(np.max(abs(before-z.cpu().numpy())))
    assert replay < 1e-4, replay
    assert all(not v.requires_grad for v in targets.values())
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    save_cases(inputs,ids,targets,details,z.cpu().numpy(),off.cpu().numpy())
    write('verification.json',dict(passed=True,scope='source-only forward/backward, zero optimizer steps',
        checkpoint=str(CHECKPOINT),source_samples=len(ids),selected=int(targets['use'].sum()),
        predictions_changed=0,initial_forward_replay_error=replay,gradients=total,
        untouched_pixel_error=details['untouched_error'],
        maximum_mean_control_error=max(r['mean_control_error'] for c in details['diagnostics'] for r in c),
        loss=stats,seconds=time.perf_counter()-tick,formal_model_modified=False,
        target_evaluation_accessed=False,max_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))
    print(json.dumps(json.loads((ROOT/'verification.json').read_text(encoding='utf-8')),ensure_ascii=False),flush=True)


def train(mode,seed,steps):
    inputs,fit,inner=runtime()
    teacher=model().requires_grad_(False).eval()
    net=model()
    groups=recipe.previous.groups_for(net)
    opt=torch.optim.AdamW(groups,weight_decay=.01)
    dest=ROOT/f'{mode}_s{seed}'
    if dest.exists():
        raise FileExistsError(dest)
    dest.mkdir()
    order=np.random.default_rng(seed+20261118).permutation(fit)[:steps*8]
    if len(order) != steps*8:
        raise ValueError('Requested budget exceeds the fixed source order')
    np.save(dest/'ids_local.npy',order)
    aug=np.random.default_rng(seed+20261119)
    z=m.full.predict(net,inputs,inner,'small')
    best=m.full.metrics(inputs.y[inner],z)
    torch.save(net.state_dict(),dest/'best.pt')
    np.savez_compressed(dest/'inner_best.npz',ids=inner,label=inputs.y[inner],score=z)
    (dest/'config.json').write_text(json.dumps(dict(mode=mode,seed=seed,steps=steps,batch=8,
        start=str(CHECKPOINT),rates=recipe.previous.RATES,teacher='fixed same starting checkpoint',
        selection='source-inner BA/OA/-NLL including step0; previously exposed development set',
        references=16,target_evaluation=False)),encoding='utf-8')
    for step in range(steps):
        ids=order[step*8:(step+1)*8]
        code=int(aug.integers(8))
        targets,_=measure(teacher,inputs,ids,code)
        opt.zero_grad(set_to_none=True)
        _,_,stats=backward(net,inputs,ids,code,targets,mode)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],5.)
        opt.step()
        if (step+1)%30 == 0 or step+1 == steps:
            z=m.full.predict(net,inputs,inner,'small')
            metrics=m.full.metrics(inputs.y[inner],z)
            if recipe.previous.key(metrics) > recipe.previous.key(best):
                best=metrics
                torch.save(net.state_dict(),dest/'best.pt')
                np.savez_compressed(dest/'inner_best.npz',ids=inner,label=inputs.y[inner],score=z)
            (dest/f'step_{step+1}.json').write_text(json.dumps(dict(metrics=metrics,loss=stats)),encoding='utf-8')
            print(mode,seed,step+1,metrics['oa'],metrics['ba'],flush=True)
    torch.save(net.state_dict(),dest/'last.pt')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('verify','train'),default='verify')
    parser.add_argument('--mode',choices=('routed','margin','relations'),default='relations')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--steps',type=int,default=120)
    args=parser.parse_args()
    verify() if args.phase == 'verify' else train(args.mode,args.seed,args.steps)
