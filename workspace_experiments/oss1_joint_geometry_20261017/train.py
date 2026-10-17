"""Source-only episodic local reader; frozen public full-classifier integration."""
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from common import HERE,FULL,MANIFEST,folds,public
from local_reader import ARMS,LocalReader,support_features

ARMS=ARMS+('body_only',)
SEEDS=(0,1,2)
STEPS=200
ALPHAS=(.75,4.)


def distinct(frame):
    return frame.sort_values('sample_id').drop_duplicates('mmsi')


def make_episodes(data,kind):
    rng=np.random.default_rng(20261017)
    frame=data['source_train' if kind=='fit' else 'source_val'].copy()
    frame['local']=data['train' if kind=='fit' else 'val']
    episodes=[]
    for acquisition,group in frame.groupby('acquisition',sort=True):
        group=distinct(group)
        if len(group)<8:
            continue
        order=group.iloc[rng.permutation(len(group))]
        q=order.iloc[:min(32,max(4,len(order)//2))]
        other=frame[~frame.acquisition.eq(acquisition)&~frame.mmsi.isin(set(q.mmsi))]
        candidates=[(a,distinct(f)) for a,f in other.groupby('acquisition',sort=True) if distinct(f).shape[0]>=16]
        rng.shuffle(candidates)
        for origin in ('same','cross'):
            possibilities=[(acquisition,distinct(group[~group.mmsi.isin(set(q.mmsi))]))] if origin=='same' else candidates
            for support_acq,pool in possibilities:
                if len(pool)<16:
                    continue
                s=public.balanced_support(pool.local.to_numpy(int),data['labels'],rng)
                if kind=='fit':
                    excluded=set(q.mmsi)|set(frame[frame.local.isin(s)].mmsi)
                    fit_frame=frame[~frame.acquisition.isin([acquisition,support_acq])&~frame.mmsi.isin(excluded)]
                    prior_rows=fit_frame.local.to_numpy(int)
                else:
                    prior_rows=data['train']
                if len(prior_rows)<128 or len(np.unique(data['labels'][prior_rows]))<8:
                    continue
                qr=q.local.to_numpy(int)
                assert not set(prior_rows)&set(np.r_[qr,s])
                assert not set(q.mmsi)&set(frame[frame.local.isin(s)].mmsi)
                episodes.append(dict(q=qr,s=s,prior_rows=prior_rows,origin=origin,acquisition=acquisition,
                                     support_acquisition=support_acq))
                break
    return episodes


def make_task(data,base,z,episode,prior_cache):
    q,s=episode['q'],episode['s'];fit=episode['prior_rows'];labels=data['labels']
    key=tuple(sorted(fit))
    if key not in prior_cache:
        bh=public.fit_head(base[:,:-1],labels,fit)
        eh=public.fit_head(z[:,:-1],labels,fit)
        relation=public.class_map(z[:,:-1],labels,fit)
        prior_cache[key]=(base@bh,z@eh,relation)
    bp,ep,r=prior_cache[key]
    ss=.75*(z[s]@z[s].T)+.25*(r[s]@r[s].T)
    qs=.75*(z[q]@z[s].T)+.25*(r[q]@r[s].T)
    bscore=bp[q]+base[q]@base[s].T@public.coefficients(base[s]@base[s].T,bp[s],labels[s],8.)
    bp_q=softmax(bscore,axis=1)
    old=.5*(bp_q+softmax(ep[q]+qs@public.coefficients(ss,ep[s],labels[s],8.),axis=1))
    sy=labels[s]
    root=np.sqrt(len(s)/(len(np.unique(sy))*np.bincount(sy,minlength=8)[sy]))
    task=dict(s=s,q=q,ids=np.r_[s,q],ns=len(s),origin=episode['origin'],
              acquisition=episode['acquisition'])
    for key,value in dict(ss=ss,qs=qs,prior_s=ep[s],prior_q=ep[q],base=bp_q,old=old,
                          desired=2*np.eye(8)[sy]-1,root=root).items():
        task[key]=torch.tensor(value,dtype=torch.float64)
    return task


def predict(raw,valid,task,alpha):
    ns=task['ns'];h=support_features(raw,valid,ns).double()
    ss=task['ss']+alpha*(h[:ns]@h[:ns].T)
    qs=task['qs']+alpha*(h[ns:]@h[:ns].T)
    root=task['root']
    coefficient=root[:,None]*torch.linalg.solve(root[:,None]*ss*root[None,:]+8*torch.eye(ns,dtype=ss.dtype),
                   root[:,None]*(task['desired']-task['prior_s']))
    probability=.5*(task['base']+(task['prior_q']+qs@coefficient).softmax(-1))
    active=valid[ns:]&(valid[:ns].sum()>=2)
    return torch.where(active[:,None],probability,task['old']),h


def risk(probability,truth):
    target=torch.nn.functional.one_hot(truth,8).double()
    per=(probability-target).square().sum(1)
    return torch.stack([per[truth==c].mean() for c in truth.unique()]).mean()


def forward(model,features,task):
    ids=task['ids']
    return model(features['body'][ids],features['tokens'][ids],features['positions'][ids])


@torch.no_grad()
def select(model,features,tasks,labels):
    risks={a:[] for a in (0.,)+ALPHAS}
    for task in tasks:
        raw,_=forward(model,features,task)
        for alpha in risks:
            p,_=predict(raw,features['valid'][task['ids']],task,alpha)
            risks[alpha].append(float(risk(p,labels[task['q']])))
    means={a:float(np.mean(v)) for a,v in risks.items()}
    alpha=min(means,key=lambda a:(means[a],a))
    return means[alpha],alpha,means


def fit_reader(features,fit_tasks,val_tasks,labels,seed,folder):
    torch.manual_seed(20261017+seed)
    model=LocalReader()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(20261017+seed)
    history=[];best=None;selected=None
    for step in range(STEPS+1):
        if step:
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for i in rng.choice(len(fit_tasks),size=min(2,len(fit_tasks)),replace=False):
                task=fit_tasks[i]
                raw,_=forward(model,features,task)
                p,_=predict(raw,features['valid'][task['ids']],task,4.)
                losses.append(risk(p,labels[task['q']]))
            loss=torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
        if step in (0,50,100,200):
            value,alpha,means=select(model,features,val_tasks,labels)
            entry=dict(step=step,selection_brier=value,alpha=alpha,alpha_risks=means,
                       training_loss=None if step==0 else float(loss.detach()))
            history.append(entry)
            if best is None or value<best['selection_brier']-1e-9:
                best=entry;selected=copy.deepcopy(model.state_dict())
    terminal=copy.deepcopy(model.state_dict())
    torch.save(dict(selected=selected,terminal=terminal,best=best),folder/'reader.pt')
    (folder/'training.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    model.load_state_dict(selected)
    return model,best,terminal


def scores(y,p):
    result=public.metrics(y,p.argmax(1))
    result['brier']=float(((p-np.eye(8)[y])**2).sum(1).mean())
    result['nll']=float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean())
    return result


def run_fold(fold,cache,episode,prefix,manifest):
    folder=HERE/fold;folder.mkdir(exist_ok=True)
    data=public.load_fold('oss1',fold,cache,episode,prefix,manifest,
         np.load(FULL/'oss1_c4.npy'),np.load(FULL/'oss1_zero.npy'))
    source=np.r_[data['train'],data['val']]
    g=public.pca(data['raw']['G50'],source)
    d=public.pca(data['raw']['D18'],source)
    c=public.pca(data['c4'],source)
    base=public.design_block(c,np.arange(len(c)))
    z=public.design_block((g,d),np.arange(len(g)))
    fit_episodes=make_episodes(data,'fit');val_episodes=make_episodes(data,'selection')
    if not fit_episodes or not val_episodes:
        raise RuntimeError(f'No usable episodes: {fold}, {len(fit_episodes)}, {len(val_episodes)}')
    prior_cache={}
    fit_tasks=[make_task(data,base,z,e,prior_cache) for e in fit_episodes]
    val_tasks=[make_task(data,base,z,e,prior_cache) for e in val_episodes]
    # The public baseline fits class reference on source TRAIN, heads on train+val.
    reference=public.class_map(z[:,:-1],data['labels'],data['train'])
    bp=base@public.fit_head(c,data['labels'],source)
    ep=z@public.fit_head((g,d),data['labels'],source)
    final_key=tuple(sorted(source))
    prior_cache[final_key]=(bp,ep,reference)
    target_tasks=[make_task(data,base,z,dict(q=data['query'],s=s,prior_rows=source,
                  origin=origin,acquisition=str(data['query_frame'].acquisition.iloc[0])),prior_cache)
                  for (origin,draw),s in data['support'].items()]
    for t,(_,draw) in zip(target_tasks,data['support']):
        t['draw']=int(draw)
    replay=[]
    old_metrics=pd.read_csv(FULL/'oss1'/f'{fold}_metrics.csv')
    for task in target_tasks:
        old=task['old'].numpy()
        m=scores(data['labels'][task['q']],old)
        previous=old_metrics[(old_metrics.arm=='c4_class_fixed')&(old_metrics.origin==task['origin'])&(old_metrics.draw==task['draw'])].iloc[0]
        error=max(abs(m['oa']-previous.chip_oa),abs(m['ba']-previous.chip_ba))
        if error>1e-7:
            raise AssertionError(f'Baseline replay differs: {fold} {error}')
        replay.append(error)
    with np.load(folder/'local_features.npz') as f:
        np.testing.assert_array_equal(f['sample_ids'][data['rows']],manifest.sample_id.to_numpy(str)[data['rows']])
        common={'body':torch.tensor(f['body'][data['rows']]),'valid':torch.tensor(f['valid'][data['rows']])}
        source_arrays={a:(torch.tensor(f[a+'_tokens'][data['rows']]),torch.tensor(f[a+'_positions'][data['rows']]))
                       for a in ARMS if a!='body_only'}
        body_tokens=common['body'][:,:64,None].transpose(1,2).expand(-1,12,-1).clone()
        body_positions=torch.zeros(len(body_tokens),12,16)
        body_positions[:,:,3]=1
        source_arrays['body_only']=(body_tokens,body_positions)
    source_table=[]
    for kind,episodes,tasks in [('fit',fit_episodes,fit_tasks),('selection',val_episodes,val_tasks)]:
        for i,(e,t) in enumerate(zip(episodes,tasks)):
            source_table.append(dict(kind=kind,task=i,origin=e['origin'],acquisition=e['acquisition'],
                support_acquisition=e['support_acquisition'],query=len(e['q']),support=len(e['s']),
                prior=len(e['prior_rows']),valid_query=int(common['valid'][e['q']].sum()),
                valid_support=int(common['valid'][e['s']].sum())))
    pd.DataFrame(source_table).to_csv(folder/'source_tasks.csv',index=False)
    serial=[dict(kind=kind,q=data['rows'][e['q']].tolist(),s=data['rows'][e['s']].tolist(),prior=data['rows'][e['prior_rows']].tolist())
            for kind,episodes in [('fit',fit_episodes),('selection',val_episodes)] for e in episodes]
    (folder/'source_task_rows.json').write_text(json.dumps(serial),encoding='utf-8')
    print('EPISODES',fold,len(fit_tasks),len(val_tasks),flush=True)
    labels=torch.tensor(data['labels'],dtype=torch.long)
    records=[];predictions=[];fits=[]
    for arm in ARMS:
        features={**common,'tokens':source_arrays[arm][0],'positions':source_arrays[arm][1]}
        for seed in SEEDS:
            fit_folder=folder/f'{arm}_{seed}';fit_folder.mkdir(exist_ok=True)
            start=time.perf_counter()
            model,best,terminal=fit_reader(features,fit_tasks,val_tasks,labels,seed,fit_folder)
            selected=copy.deepcopy(model.state_dict())
            fits.append(dict(fold=fold,arm=arm,seed=seed,seconds=time.perf_counter()-start,
                             step=best['step'],alpha=best['alpha'],source_brier=best['selection_brier']))
            with torch.no_grad():
                for policy,state,alpha in [('source_selected',selected,best['alpha']),('terminal_fixed4',terminal,4.)]:
                    model.load_state_dict(state)
                    for task in target_tasks:
                        raw,attention=forward(model,features,task)
                        p,h=predict(raw,features['valid'][task['ids']],task,alpha)
                        p=p.numpy();y=data['labels'][task['q']];old=task['old'].numpy()
                        result=scores(y,p);before=scores(y,old)
                        result.update(fold=fold,arm=arm,seed=seed,policy=policy,alpha=alpha,
                            origin=task['origin'],draw=task['draw'],query=len(y),
                            valid_query=int(features['valid'][task['q']].sum()),
                            valid_support=int(features['valid'][task['s']].sum()),
                            baseline_oa=before['oa'],baseline_ba=before['ba'],
                            delta_oa=result['oa']-before['oa'],delta_ba=result['ba']-before['ba'],
                            delta_brier=result['brier']-before['brier'],
                            rescues=int(((old.argmax(1)!=y)&(p.argmax(1)==y)).sum()),
                            harms=int(((old.argmax(1)==y)&(p.argmax(1)!=y)).sum()),
                            mean_descriptor_energy=float(h.square().sum(1).mean()),
                            attention_entropy=float((-(attention*attention.clamp_min(1e-9).log()).sum(1)).mean()))
                        records.append(result)
                        for j,row in enumerate(data['rows'][task['q']]):
                            predictions.append(dict(fold=fold,arm=arm,seed=seed,policy=policy,
                                origin=task['origin'],draw=task['draw'],global_row=int(row),
                                truth=int(y[j]),baseline=int(old[j].argmax()),prediction=int(p[j].argmax()),
                                baseline_true_probability=float(old[j,y[j]]),true_probability=float(p[j,y[j]]),
                                **{f'p{k}':float(p[j,k]) for k in range(8)}))
            pd.DataFrame(records).to_csv(folder/'metrics.csv',index=False)
            pd.DataFrame(fits).to_csv(folder/'fits.csv',index=False)
            print('FIT',fold,arm,seed,'alpha',best['alpha'],'step',best['step'],
                  'seconds',round(fits[-1]['seconds'],1),flush=True)
    pd.DataFrame(predictions).to_csv(folder/'predictions.csv',index=False)
    (folder/'DONE.json').write_text(json.dumps(dict(baseline_max_error=max(replay),fits=len(fits),
                 fit_episodes=len(fit_tasks),selection_episodes=len(val_tasks)),indent=2),encoding='utf-8')


def main():
    torch.set_num_threads(2)
    manifest=pd.read_csv(MANIFEST)
    protocol=dict(date='2026-10-17',dataset='OpenSARShip1.0',folds=[f[0] for f in folds()],
        arms=ARMS,seeds=SEEDS,steps=STEPS,selection_steps=[0,50,100,200],
        alpha_candidates=[0.]+list(ALPHAS),training_alpha=4.,selection='source-only macro Brier',
        training='source-held acquisition episodes; priors exclude query/support acquisitions and identities',
        baseline='existing C4+source-only SAR-R18+fixed class relation, .5 full probability fusion',
        target_support=16,support_draws=5,source_pca='existing all-legal-source unsupervised PCA, held target excluded',
        backbone_retrained=False,ais_heading_used=False,neighborhood_tokens=12,
        target_tuning=False,historical_development_data=True,
        implementation_revision='conv1 center j+.25; consistent complete-mask coverage; initial pass archived',
        body_control='same reader width, body mean replicated as context, constant positions; no outer pixels',
        secondary_policy='terminal_fixed4 is predeclared diagnostic, not target-selected replacement')
    (HERE/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for args in folds():
        run_fold(*args,manifest)


if __name__=='__main__':
    main()
