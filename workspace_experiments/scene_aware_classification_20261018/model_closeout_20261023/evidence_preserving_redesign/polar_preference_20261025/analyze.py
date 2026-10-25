"""Inspect learned preference, fixed-head dependence and native-image cases."""
import os
from pathlib import Path
import importlib.util
import sys
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
spec=importlib.util.spec_from_file_location('train',HERE/'train.py')
run=importlib.util.module_from_spec(spec);sys.modules['train']=run;spec.loader.exec_module(run)
import json
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

if (HERE/'conflict_complete.json').exists():
    run.ARMS=run.ARMS+('conflict_spatial','conflict_freq')

stats=run.load('preference_stats',HERE.parent/'typed_m3_20261025/analyze.py');stats.HERE=HERE
stats.PAIRS=[(a,'hybrid_equal') for a in run.ARMS]+[
    ('ce','common'),('aux','ce'),('aux_freq','aux'),('aux_freq','aux_wide'),
    ('aux_freq','globalpref'),('aux_freq','spatial'),('aux_freq','frequency'),
    ('aux_freq','old_full'),('aux_freq','prior_late'),('aux_freq','hybrid_source')]+[
    (a,a+'__no_preference') for a in ('ce','aux','aux_freq','globalpref')]+[
    ('aux_freq','aux_freq__fixed_strength'),('aux_freq','aux_freq__no_frequency')]
if 'conflict_freq' in run.ARMS:
    stats.PAIRS += [('conflict_spatial','aux'),('conflict_freq','aux_freq'),
        ('conflict_freq','conflict_spatial'),('conflict_freq','globalpref'),('conflict_freq','spatial'),
        ('conflict_freq','old_full'),('conflict_freq','prior_late'),('conflict_freq','hybrid_source'),
        ('conflict_freq','conflict_freq__no_preference'),('conflict_freq','conflict_freq__no_frequency')]


def agreement(pred,target):
    # Indifference contributes half, not a silently preferred modality.
    return np.where(abs(pred)<1e-7,.5,(pred>0)==target).astype(float)


def preference(data,scores):
    rows,details=[],[]
    for seed in run.SEEDS:
        raw,z=run.inputs(data,seed)
        y=data.table.iloc[:data.n].class_id.to_numpy(int)
        sv=(z['base']+z['delta_vv']).numpy();sh=(z['base']+z['delta_vh']).numpy()
        gap=logsumexp(sh,axis=1)-sh[np.arange(data.n),y]-logsumexp(sv,axis=1)+sv[np.arange(data.n),y]
        av=z['available'].numpy().astype(bool);cv=sv.argmax(1)==y;ch=sh.argmax(1)==y
        a,b=run.folds(data)
        constant=np.zeros(data.n)
        for tr,va in ((a,b),(b,a),(data.va,data.te)):
            take=av[tr].all(1)&(cv[tr]!=ch[tr])
            value=np.mean(cv[tr][take])
            constant[va]=np.sign(value-.5)
        for arm in run.ARMS:
            for role,ids in (('source_cross_selection',data.va),('source_refit',data.va),('evaluation',data.te)):
                is_oof=role=='source_cross_selection'
                path=HERE/'results'/f'{"source_cv_" if is_oof else ""}{arm}_{seed}.npz'
                with np.load(path) as f:
                    np.testing.assert_array_equal(f['up_index'],data.up[ids] if is_oof else data.up[:data.n])
                    g=f['weight'].copy();r=f['preference'].copy();t=f['strength'].copy()
                    if not is_oof:g,r,t=g[ids],r[ids],t[ids]
                paired=av[ids].all(1);v,h=cv[ids],ch[ids]
                baseline=constant[ids]
                if role=='source_refit':
                    source_conflict=av[data.va].all(1)&(cv[data.va]!=ch[data.va])
                    baseline=np.full(len(ids),np.sign(cv[data.va][source_conflict].mean()-.5))
                for name,mask in (('paired',paired),('only_VV_correct',paired&v&~h),
                    ('only_VH_correct',paired&h&~v),('conflict',paired&(v!=h)),
                    ('both_correct',paired&v&h),('both_wrong',paired&~v&~h),('VH_missing',~av[ids,1])):
                    if not mask.any():continue
                    gg=g[mask];rr=r[mask];ggap=gap[ids][mask]
                    corr=float(np.corrcoef(rr.astype(float),ggap.astype(float))[0,1]) if np.ptp(rr)>1e-7 and np.ptp(ggap)>1e-7 else None
                    rows.append(dict(seed=seed,arm=arm,role=role,group=name,n=int(mask.sum()),
                        VV_mean=float(gg[:,0].mean()),VH_mean=float(gg[:,1].mean()),strength_mean=float(t[mask].mean()),
                        preference_abs_mean=float(abs(rr).mean()),weight_gap_abs_mean=float(abs(gg[:,0]-gg[:,1]).mean()),
                        CE_direction_agreement=float(agreement(rr,ggap>0).mean()),CE_gap_correlation=corr,
                        correct_endpoint_agreement=float(agreement(rr,v[mask]).mean()) if name in ('conflict','only_VV_correct','only_VH_correct') else None,
                        source_majority_endpoint_agreement=float(agreement(baseline[mask],v[mask]).mean()) if name=='conflict' else None,
                        CE_vs_top1_discord=int(((ggap>0)!=v[mask]).sum()) if name=='conflict' else None))
                frame=data.table.iloc[ids][['obs_id','mmsi','pass_group','class_id','class_name']].copy()
                frame['seed']=seed;frame['arm']=arm;frame['role']=role
                frame['VV_correct']=v;frame['VH_correct']=h;frame['paired']=paired
                frame['CE_gap']=gap[ids];frame['r']=r;frame['t']=t;frame['gV']=g[:,0];frame['gH']=g[:,1]
                details.append(frame)
    pd.DataFrame(rows).to_csv(HERE/'preference_groups.csv',index=False)
    pd.concat(details).to_csv(HERE/'preference_samples.csv.gz',index=False,compression='gzip')


def teacher_mass(data):
    rows=[];a,b=run.folds(data)
    y=data.table.iloc[:data.n].class_id.to_numpy(int)
    for seed in run.SEEDS:
        _,z=run.inputs(data,seed)
        for fold,ids in (('A',a),('B',b),('all',data.va)):
            _,q,confidence=run.targets(z,y,ids)
            count=np.bincount(y[ids],minlength=8)
            wt=.5/len(ids)+.5/(8*count[y[ids]])
            mass=wt*confidence.numpy();mass/=mass.sum()
            paired=z['available'][ids].numpy().all(1)
            v=(z['base']+z['delta_vv'])[ids].numpy().argmax(1)==y[ids]
            h=(z['base']+z['delta_vh'])[ids].numpy().argmax(1)==y[ids]
            for group,take in (('both_correct',paired&v&h),('conflict',paired&(v!=h)),
                               ('both_wrong',paired&~v&~h),('VH_missing',~paired)):
                rows.append(dict(seed=seed,fold=fold,group=group,n=int(take.sum()),
                    auxiliary_weight_mass=float(mass[take].sum()),
                    initial_preference_gradient=float((mass[take]*(.5-q.numpy()[take])).sum())))
    pd.DataFrame(rows).to_csv(HERE/'teacher_mass.csv',index=False)


def population(data,scores):
    rows,flips=[] ,[]
    ids=data.te;f=data.table.iloc[ids];y=f.class_id.to_numpy(int)
    for seed in run.SEEDS:
        _,z=run.inputs(data,seed);available=z['available'][ids].numpy().astype(bool)
        cv=(z['base']+z['delta_vv'])[ids].numpy().argmax(1)==y
        ch=(z['base']+z['delta_vh'])[ids].numpy().argmax(1)==y
        for arm,comp in stats.PAIRS:
            a,b=scores[seed,arm][ids],scores[seed,comp][ids]
            for name,take in (('all',np.ones(len(ids),bool)),('paired',available.all(1)),
                ('VH_missing',~available[:,1]),('conflict',available.all(1)&(cv!=ch)),
                ('both_wrong',available.all(1)&~cv&~ch),('both_correct',available.all(1)&cv&ch)):
                if take.any():rows.append(dict(seed=seed,arm=arm,comparator=comp,group=name)|stats.effect(a[take],b[take],y[take]))
            for k in np.flatnonzero((a.argmax(1)==y)!=(b.argmax(1)==y)):
                row=f.iloc[k]
                flips.append(dict(seed=seed,arm=arm,comparator=comp,obs_id=int(row.obs_id),mmsi=row.mmsi,
                    pass_group=row.pass_group,port=row.port,class_id=int(y[k]),class_name=row.class_name,
                    case='rescue' if a[k].argmax()==y[k] else 'harm',VH_present=bool(available[k,1]),
                    before=int(b[k].argmax()),after=int(a[k].argmax())))
    pd.DataFrame(rows).to_csv(HERE/'population_pairs.csv',index=False)
    pd.DataFrame(flips).to_csv(HERE/'flips.csv',index=False)


def cases(data,scores):
    ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int);out=[];used=set()
    comparisons=[('aux_freq','ce',('rescue','harm','CC','WW')),
        ('aux_freq','aux_freq__no_preference',('rescue','harm')),
        ('aux_freq','aux_freq__no_frequency',('rescue','harm')),
        ('aux','ce',('rescue','harm'))]
    if 'conflict_freq' in run.ARMS:
        comparisons += [('conflict_freq','aux_freq',('rescue','harm')),
            ('conflict_freq','conflict_freq__no_preference',('rescue','harm'))]
    for arm,comp,names in comparisons:
        for name in names:
            options=[]
            for seed in run.SEEDS:
                a=scores[seed,arm][ids].argmax(1)==y;b=scores[seed,comp][ids].argmax(1)==y
                take={'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[name]
                for j in ids[take]:
                    obs=int(data.table.iloc[j].obs_id)
                    if obs not in used:options.append((seed,obs,int(j)))
            if options:
                options.sort();seed,obs,j=options[0] if name in ('rescue','harm') else options[len(options)//2]
                used.add(obs);out.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm=arm,comparator=comp))
    for obs,seed in ((22572,17),(2080,43),(5585,29)):
        if obs in used:continue
        j=int(np.flatnonzero(data.table.obs_id.eq(obs)&data.table.role.eq('evaluation'))[0]);yy=int(data.table.iloc[j].class_id)
        a=scores[seed,'aux_freq'][j].argmax()==yy;b=scores[seed,'frequency'][j].argmax()==yy
        name='CC' if a and b else 'WW' if not a and not b else 'rescue' if a else 'harm'
        out.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm='aux_freq',comparator='frequency',historical=True))
    return out


def main():
    assert (HERE/'complete.json').exists();torch.set_num_threads(2)
    data=run.Data();scores={};weights={};replay=[]
    # Keep the inherited statistics writer's explicit evaluation filter.
    frame=pd.read_csv(HERE/'metrics.csv');frame['role']='evaluation'
    if 'conflict_freq' in run.ARMS:
        frame=pd.concat([frame,pd.read_csv(HERE/'conflict_metrics.csv')]).drop_duplicates(['seed','arm'],keep='last')
    frame.to_csv(HERE/'metrics.csv',index=False)
    for seed in run.SEEDS:
        raw,z=run.inputs(data,seed)
        for arm in run.ARMS:
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as f:
                np.testing.assert_array_equal(f['up_index'],data.up[:data.n]);scores[seed,arm]=f['score'].copy();weights[seed,arm]=f['weight'].copy()
            model,x,_=run.restore(seed,arm,raw)
            with torch.no_grad():s=run.prediction(model,x,z,np.arange(data.n))[0].numpy()
            err=float(abs(s-scores[seed,arm]).max());assert err<1e-6
            replay.append(dict(seed=seed,arm=arm,error=err))
            with np.load(HERE/'results'/f'diagnostics_{arm}_{seed}.npz') as f:
                for mode in f.files:
                    if mode!='up_index':scores[seed,arm+'__'+mode]=f[mode].copy()
        for arm in run.COMPARATORS:
            with np.load(run.PREVIOUS/'results'/f'{arm}_{seed}.npz') as f:scores[seed,arm]=f['score'].copy()
    receipt=stats.statistics(data,scores);preference(data,scores);population(data,scores);teacher_mass(data)
    native=run.load('preference_native',HERE.parent/'null_centered_readout_20261025/analyze.py');native.HERE=HERE
    geo=run.load('preference_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in cases(data,scores):
        path,prov,true,rival=native.panel(case,data,scores,geo,up,valid,snapshots,refs)
        seed,j=case['seed'],case['index'];_,z=run.inputs(data,seed)
        records.append(case|dict(figure=str(path),provenance=prov,true=true,rival=rival,
            base_margin=float(z['base'][j,true]-z['base'][j,rival]),
            delta_VV=float(z['delta_vv'][j,true]-z['delta_vv'][j,rival]),
            delta_VH=float(z['delta_vh'][j,true]-z['delta_vh'][j,rival]),
            weights={a:weights[seed,a][j] for a in run.ARMS},
            comparisons={a:dict(prediction=int(s[j].argmax()),margin=float(s[j,true]-s[j,rival]))
                         for (ss,a),s in scores.items() if ss==seed}))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),frozen_replay=replay,
        limitation='Repeated development and historically selected source. Fixed-head neutralization is functional dependence, not physical intervention.')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'paired.csv').groupby(['arm','comparator'])[['OA_pp','BA_pp','rescued','harmed']].mean().to_string(),flush=True)
    print(pd.read_csv(HERE/'preference_groups.csv').query("group=='conflict'").groupby(['role','arm'])[['correct_endpoint_agreement','source_majority_endpoint_agreement','weight_gap_abs_mean']].mean().to_string(),flush=True)


if __name__=='__main__':
    main()
