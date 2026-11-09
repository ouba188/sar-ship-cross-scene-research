"""Source-pass early stopping of the fixed current intervention adaptation."""
from pathlib import Path
import argparse
import gc
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'intervention_validation_20261109'
spec=importlib.util.spec_from_file_location('prior_intervention_validation',OLD/'study.py')
p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
torch,np,pd,task=p.torch,p.np,p.pd,p.task


def write(name,value):
    task.write(ROOT/name,value)


def context():
    rows,globalgroups=p.s.prev.records()
    gids=np.array(json.loads((OLD/'global_indices.json').read_text(encoding='utf-8')))
    groups=json.loads((OLD/'groups.json').read_text(encoding='utf-8'))
    adapt=np.array(groups['source_adapt'])
    labels=np.array([rows[i]['label'] for i in gids])
    source=[rows[gids[i]] for i in adapt]
    passes=np.array([task.orbit(r) for r in source])
    eligible=[v for v in np.unique(passes) if len(set(labels[adapt[passes==v]]))==4]
    holdpass=sorted(eligible,key=lambda v:(-int((passes==v).sum()),v))[0]
    hold=adapt[passes==holdpass]
    ships={str(rows[gids[i]]['mmsi']) for i in hold}
    train=np.array([i for i in adapt if i not in set(hold) and str(rows[gids[i]]['mmsi']) not in ships])
    assert len(hold)==105 and len(train)==130
    assert not {task.orbit(rows[gids[i]]) for i in train}&{holdpass}
    assert not {str(rows[gids[i]]['mmsi']) for i in train}&ships
    assert set(labels[train])==set(labels[hold])==set(range(4))
    for i in train:
        assert rows[gids[i]]['split']=='source_adapt'
    return rows,globalgroups,gids,groups,labels,train,hold,holdpass


def state(net):
    return {n:v.detach().cpu().clone() for n,v in net.base.named_parameters() if v.requires_grad}


def restore(net,weights):
    params=dict(net.base.named_parameters())
    assert set(weights)==set(task.current.LIVE)
    with torch.no_grad():
        for key,value in weights.items():params[key].copy_(value)


def fit():
    assert (OLD/'student_checks.json').exists()
    rows,_,gids,groups,y,train,hold,holdpass=context()
    arrays=p.cache()
    selections=json.loads((OLD/'selection.json').read_text(encoding='utf-8'))
    lookup={r['id']:r for r in selections}
    useful=np.array([lookup[rows[gids[i]]['id']]['useful'] for i in train],bool)
    quota={int(c):int(useful[y[train]==c].sum()) for c in range(4)}
    weights=p.class_weights([rows[gids[i]] for i in train],'adapt')
    base=p.metrics(y[hold],p.tensor(arrays['score'],hold).cpu().numpy())
    write('split_receipt.json',dict(teacher='current class_s17; source data previously seen',
          holdpass=holdpass,n_hold=len(hold),n_train=len(train),excluded_same_vessel=6,
          train_global_indices=gids[train].tolist(),hold_global_indices=gids[hold].tolist(),
          train_classes={c:int((y[train]==c).sum()) for c in range(4)},
          hold_classes={c:int((y[hold]==c).sum()) for c in range(4)},quota=quota,source_initial=base))
    choices=[]
    for seed in (17,29,43):
        ordinary=np.zeros(len(train),bool);rng=np.random.default_rng(seed+10000)
        for c,n in quota.items():ordinary[rng.choice(np.flatnonzero(y[train]==c),n,replace=False)]=True
        assert all(int(ordinary[y[train]==c].sum())==quota[c] for c in range(4))
        for arm,mask in (('ordinary',ordinary),('intervention',useful)):
            candidates=[]
            for lr in (3e-5,1e-5):
                name=f'{arm}_s{seed}_lr{lr:.0e}'
                dest=ROOT/name;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'complete.json').exists():
                    rec=json.loads((dest/'complete.json').read_text(encoding='utf-8'))
                    candidates.append(rec['selected']);continue
                net=p.Student();initial=state(net)
                score=p.predict(net,arrays,hold)
                error=float(abs(score-p.tensor(arrays['score'],hold).cpu().numpy()).max())
                assert error<1e-4
                params=[v for v in net.parameters() if v.requires_grad]
                optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=.01)
                rng=np.random.default_rng(seed);tick=time.perf_counter();steps=0
                best=dict(arm=arm,seed=seed,lr=lr,epoch=0,steps=0,path=name+'/best.pt',source=base)
                torch.save(initial,dest/'best.pt')
                curve=[dict(epoch=0,steps=0,source=base,relative_update=0.)];orders=[]
                for epoch in range(1,7):
                    order=rng.permutation(len(train));orders.append(order.tolist());terms_all=[]
                    for local in np.array_split(order,range(16,len(train),16)):
                        batch={k:p.tensor(v,train[local]) for k,v in arrays.items()}
                        optimizer.zero_grad(set_to_none=True)
                        z=net(batch)
                        loss,terms=p.adaptation_loss(z,torch.tensor(y[train[local]],device=task.DEVICE),batch['score'],
                            torch.tensor(weights[local],device=task.DEVICE),
                            torch.tensor(mask[local],device=task.DEVICE),.3683066666126251)
                        assert bool(torch.isfinite(loss))
                        loss.backward()
                        if steps==0:
                            gradients={n:float(v.grad.norm()) for n,v in net.base.named_parameters() if v.requires_grad}
                            assert all(np.isfinite(v) and v>0 for v in gradients.values())
                        optimizer.step();steps+=1
                        terms_all.append({k:float(v.detach()) for k,v in terms.items()})
                    score=p.predict(net,arrays,hold);m=p.metrics(y[hold],score)
                    now=state(net)
                    norm=(sum(float((now[k]-initial[k]).square().sum()) for k in now)/
                          sum(float(v.square().sum()) for v in initial.values()))**.5
                    curve.append(dict(epoch=epoch,steps=steps,source=m,relative_update=norm,
                        losses={k:float(np.mean([t[k] for t in terms_all])) for k in terms_all[0]}))
                    if m['balanced_nll']<best['source']['balanced_nll']-1e-6:
                        best=dict(arm=arm,seed=seed,lr=lr,epoch=epoch,steps=steps,path=name+'/best.pt',source=m)
                        torch.save(now,dest/'best.pt')
                    print('SOURCE',name,epoch,round(m['balanced_nll'],6),'best',best['epoch'],flush=True)
                assert steps==54
                torch.save(now,dest/'last.pt')
                rec=dict(selected=best,curve=curve,orders=orders,quota=quota,selected_count=int(mask.sum()),
                         gradients_first=gradients,seconds=time.perf_counter()-tick,teacher_error=error)
                write(name+'/complete.json',rec);candidates.append(best)
                del net,optimizer,params,initial,now;gc.collect();torch.cuda.empty_cache()
            selected=min(candidates,key=lambda c:(c['source']['balanced_nll'],c['steps'],c['lr']))
            choices.append(selected)
            write('selection_progress.json',choices)
    assert len(choices)==6
    write('locked_selection.json',dict(choices=choices,criterion='source held-pass balanced NLL; includes no update',
                                      target_used_for_selection=False,candidates_per_arm_seed=14))


def evaluate():
    lock=json.loads((ROOT/'locked_selection.json').read_text(encoding='utf-8'))
    rows,_,gids,groups,y,train,hold,_=context();arrays=p.cache()
    for choice in lock['choices']:
        arm,seed=choice['arm'],choice['seed'];dest=ROOT/f'{arm}_s{seed}';dest.mkdir(exist_ok=True)
        if (dest/'evaluation_complete.json').exists():continue
        net=p.Student()
        for stage,filename in (('selected','best.pt'),('last','last.pt')):
            path=ROOT/Path(choice['path']).parent/filename
            restore(net,torch.load(path,map_location=task.DEVICE,weights_only=True))
            for split,ids in (('source_train',train),('source_hold',hold),('validation',np.array(groups['validation'])),
                              ('evaluation',np.array(groups['evaluation']))):
                z=p.predict(net,arrays,ids)
                np.savez_compressed(dest/f'{stage}_{split}.npz',ids=gids[ids],labels=y[ids],score=z)
        write(f'{arm}_s{seed}/evaluation_complete.json',dict(selection=choice,complete=True))
        del net;gc.collect();torch.cuda.empty_cache()


def analyze():
    rows,gg,gids,groups,yall,train,hold,_=context()
    base=np.load(OLD/'cache/score.npy',mmap_mode='r')
    scores_all={};summary=[];paired=[];strata=[];events=[]
    curves=[]
    for file in sorted(ROOT.glob('*_lr*/complete.json')):
        rec=json.loads(file.read_text(encoding='utf-8'));sel=rec['selected']
        for point in rec['curve']:
            curves.append(dict(arm=sel['arm'],seed=sel['seed'],lr=sel['lr'],epoch=point['epoch'],
                               steps=point['steps'],relative_update=point['relative_update'],**point['source']))
    pd.DataFrame(curves).to_csv(ROOT/'source_curves.csv',index=False)
    for split,local in (('source_train',train),('source_hold',hold),('validation',np.array(groups['validation'])),
                       ('evaluation',np.array(groups['evaluation']))):
        y=yall[local];ids=gids[local];original=base[local]
        summary.append(dict(split=split,arm='initial',seed=17,**p.metrics(y,original)))
        for seed in (17,29,43):
            scores={'initial':original}
            for arm in ('ordinary','intervention'):
                for stage in ('selected','last'):
                    data=np.load(ROOT/f'{arm}_s{seed}/{stage}_{split}.npz')
                    assert np.array_equal(data['ids'],ids) and np.array_equal(data['labels'],y)
                    key=arm if stage=='selected' else arm+'_last'
                    scores[key]=data['score']
                    summary.append(dict(split=split,arm=key,seed=seed,**p.metrics(y,data['score'])))
            scores_all[(split,seed)]=scores
            for a,b in (('ordinary','initial'),('intervention','initial'),('intervention','ordinary'),
                        ('ordinary','ordinary_last'),('intervention','intervention_last')):
                am=p.metrics(y,scores[a]);bm=p.metrics(y,scores[b])
                ac=scores[a].argmax(1)==y;bc=scores[b].argmax(1)==y
                paired.append(dict(split=split,seed=seed,candidate=a,control=b,rescues=int((ac&~bc).sum()),
                                   harms=int((~ac&bc).sum()),**{'delta_'+k:am[k]-bm[k] for k in ('oa','ba','nll','brier')}))
                if split not in ('validation','evaluation') or a!='intervention' or b!='ordinary':continue
                for cls in np.unique(y):
                    m=y==cls
                    strata.append(dict(split=split,seed=seed,label=int(cls),n=int(m.sum()),
                                       rescues=int((ac&~bc&m).sum()),harms=int((~ac&bc&m).sum())))
                for j in np.flatnonzero(ac!=bc):
                    row=rows[ids[j]]
                    events.append(dict(split=split,seed=seed,id=row['id'],mmsi=row['mmsi'],pass_group=task.orbit(row),
                                       effect='rescue' if ac[j] else 'harm'))
    frame=pd.DataFrame(summary);pairs=pd.DataFrame(paired)
    frame.to_csv(ROOT/'summary.csv',index=False);pairs.to_csv(ROOT/'paired.csv',index=False)
    pd.DataFrame(strata).to_csv(ROOT/'strata.csv',index=False)
    pd.DataFrame(events).to_csv(ROOT/'events.csv',index=False)
    averages=frame.groupby(['split','arm'])[['oa','ba','nll','brier']].agg(['mean','std'])
    averages.to_csv(ROOT/'averages.csv')
    pairs.groupby(['split','candidate','control'])[['delta_oa','delta_ba','delta_nll','delta_brier','rescues','harms']].mean().to_csv(ROOT/'paired_means.csv')
    spec=importlib.util.spec_from_file_location('earlystop_gallery',task.ROOT/'review_task.py')
    gallery=importlib.util.module_from_spec(spec);spec.loader.exec_module(gallery);gallery.ROOT=ROOT
    arrays,_=p.s.prev.arrays();ids=gg['evaluation'];y=np.array([rows[i]['label'] for i in ids])
    cases=[];status=[];used=set()
    for group in ('rescue','harm','stable_correct','stable_wrong'):
        candidates=[]
        for seed in (17,29,43):
            scores=scores_all[('evaluation',seed)];a=scores['intervention'];b=scores['ordinary']
            good=a.argmax(1)==y;old=b.argmax(1)==y
            mask={'rescue':good&~old,'harm':~good&old,'stable_correct':good&old,'stable_wrong':~good&~old}[group]
            delta=abs(gallery.margins(a,y)-gallery.margins(b,y))
            if float(delta.max())<1e-8:
                initial_margin=gallery.margins(scores['initial'],y)
                delta=1/(1+abs(initial_margin)) if group=='stable_correct' else abs(initial_margin)
            candidates.extend((float(delta[j]),seed,int(j)) for j in np.flatnonzero(mask))
        candidates.sort(reverse=True);pick=next((c for c in candidates if int(ids[c[2]]) not in used),None)
        status.append(dict(group=group,events=len(candidates),selected=pick))
        if pick is None:continue
        _,seed,j=pick;idx=int(ids[j]);used.add(idx)
        scores=dict(scores_all[('evaluation',seed)]);scores['baseline']=scores['ordinary']
        case=gallery.draw_case(rows,arrays,gg['references'],idx,f'{group}_s{seed}',scores,y,j,focus='intervention')
        case['seed']=seed;cases.append(case)
    # When early stopping chooses no update, visualize what forced continuation would change.
    for group in ('earlystop_protection','forgone_training_rescue'):
        candidates=[]
        for seed in (17,29,43):
            scores=scores_all[('evaluation',seed)];z=scores['intervention'];b=scores['intervention_last']
            good=z.argmax(1)==y;old=b.argmax(1)==y
            mask=good&~old if group=='earlystop_protection' else ~good&old
            delta=abs(gallery.margins(z,y)-gallery.margins(b,y))
            candidates.extend((float(delta[j]),seed,int(j)) for j in np.flatnonzero(mask))
        candidates.sort(reverse=True);pick=next((c for c in candidates if int(ids[c[2]]) not in used),None)
        status.append(dict(group=group,events=len(candidates),selected=pick,control='same-LR forced final epoch'))
        if pick is None:continue
        _,seed,j=pick;idx=int(ids[j]);used.add(idx)
        scores=dict(scores_all[('evaluation',seed)]);scores['baseline']=scores['intervention_last']
        case=gallery.draw_case(rows,arrays,gg['references'],idx,f'{group}_s{seed}',scores,y,j,focus='intervention')
        case.update(seed=seed,comparison='source-selected versus forced final epoch, NOT intervention versus ordinary')
        cases.append(case)
    write('cases.json',dict(cases=cases,groups=status,selection='Largest paired margin change; not prevalence'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    curveframe=pd.DataFrame(curves)
    for ax,lr in zip(axes,(3e-5,1e-5)):
        for arm,color in (('ordinary','#3474a6'),('intervention','#bd4e4e')):
            for seed in (17,29,43):
                df=curveframe[(curveframe.lr==lr)&(curveframe.arm==arm)&(curveframe.seed==seed)]
                ax.plot(df.epoch,df.balanced_nll,color=color,alpha=.65,marker='.',label=arm if seed==17 else None)
        ax.axhline(curveframe.iloc[0].balanced_nll,color='black',linestyle='--',label='Before adaptation')
        ax.set(title=f'Learning rate {lr:.0e}',xlabel='Epoch',ylabel='Source held-pass balanced NLL')
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'source_earlystop_curves.png',dpi=160);plt.close(fig)
    import html
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>源端早停验证</title>',
          '<style>body{font-family:Arial;max-width:1350px;margin:24px auto;padding:16px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style>',
          '<h1>适配阶段源过境留出：早停与小步更新</h1><p>源留出105条、训练130条；教师历史见过源数据。只用源balanced NLL选学习率和轮次；目标只在锁定后评价。</p>',
          '<p>当前两组均选第0轮，数值相同来自同一权重，不是重复训练的稳定提升。earlystop_protection比较保留模型与强行训练末轮，不能称为干预选样的额外收益。</p>',
          '<img src="source_earlystop_curves.png">',averages.to_html(),pairs[pairs.split=='evaluation'].to_html(index=False)]
    for case in cases:
        body.extend(['<h2>'+html.escape(case['group']+' '+case['id'])+'</h2><img src="'+case['image']+'">',
                     '<pre>'+html.escape(json.dumps(case['evidence'],indent=2))+'</pre>'])
    (ROOT/'index.html').write_text('\n'.join(body+['</html>']),encoding='utf-8')
    print(averages.to_string(),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('fit','evaluate','analyze'))
    args=parser.parse_args();task.current.setup(task.DEVICE)
    {'fit':fit,'evaluate':evaluate,'analyze':analyze}[args.action]()


if __name__=='__main__':main()
