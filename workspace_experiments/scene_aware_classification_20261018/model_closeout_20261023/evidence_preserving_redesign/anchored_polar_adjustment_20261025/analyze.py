"""Paired outcomes, anchor displacement, fixed-head diagnostics and native chips."""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'.mplcache')
import train as run
import json
import numpy as np
import pandas as pd
import torch

stats = run.load('anchor_stats', HERE.parent/'typed_m3_20261025/analyze.py')
stats.HERE = HERE
stats.PAIRS = [(a,'hybrid_source') for a in run.ARMS] + [
    ('frequency','spatial'), ('frequency','spatial_wide'), ('spatial','constant'),
    ('frequency','frequency__no_frequency'),
    *[(a,a+'__source_mean') for a in run.ARMS],
    ('spatial__source_mean','hybrid_source'), ('spatial','old_full'), ('spatial','prior_late')]


def select_cases(data, scores):
    result, used = [], set()
    ids = data.te; y = data.table.iloc[ids].class_id.to_numpy(int)
    requests = [('spatial','hybrid_source',('rescue','harm','CC','WW')),
                ('frequency','hybrid_source',('rescue','harm')),
                ('frequency','frequency__no_frequency',('rescue','harm')),
                ('spatial','spatial__source_mean',('rescue','harm'))]
    for arm, comp, names in requests:
        for name in names:
            options = []
            for seed in run.SEEDS:
                a = scores[seed,arm][ids].argmax(1)==y
                b = scores[seed,comp][ids].argmax(1)==y
                take = {'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[name]
                for j in ids[take]:
                    obs = int(data.table.iloc[j].obs_id)
                    if obs not in used: options.append((seed,obs,int(j)))
            if options:
                options.sort()
                seed,obs,j = options[0] if name in ('rescue','harm') else options[len(options)//2]
                used.add(obs)
                result.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm=arm,comparator=comp))
    return result


def main():
    assert (HERE/'complete.json').exists()
    torch.set_num_threads(2)
    data = run.Data(); scores, states, replay = {}, {}, []
    details, groups, population, flips = [], [], [], []
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    for seed in run.SEEDS:
        raw,z = run.inputs(data,seed)
        av = z['available'].bool().all(1).numpy()
        for arm in run.ARMS:
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as f:
                np.testing.assert_array_equal(f['up_index'],data.up[:data.n])
                scores[seed,arm] = f['score'].copy()
                states[seed,arm] = {k:f[k].copy() for k in ('weight','alpha','logit')}
            model,x,ck = run.restore(seed,arm,raw)
            with torch.no_grad():
                s = run.prediction(model,x,z,np.arange(data.n))[0].numpy()
            error = float(abs(s-scores[seed,arm]).max()); assert error < 1e-6
            replay.append(dict(seed=seed,arm=arm,error=error))
            alpha = states[seed,arm]['alpha']; anchor = model.alpha
            with np.load(HERE/'results'/f'diagnostics_{arm}_{seed}.npz') as f:
                for key in f.files:
                    if key != 'up_index': scores[seed,arm+'__'+key] = f[key].copy()
            for role,ids in (('source_refit',data.va),('evaluation',data.te)):
                aa = alpha[ids][av[ids]]; rr = states[seed,arm]['logit'][ids][av[ids]]
                groups.append(dict(seed=seed,arm=arm,role=role,n=len(aa),anchor=anchor,
                    alpha_mean=float(aa.mean()),alpha_std=float(aa.std()),
                    mean_abs_shift=float(abs(aa-anchor).mean()),min=float(aa.min()),max=float(aa.max()),
                    zero_fraction=float((aa==0).mean()),one_fraction=float((aa==1).mean()),
                    outward_clipped_fraction=float(((anchor+run.RADIUS*np.tanh(rr)<0)|(anchor+run.RADIUS*np.tanh(rr)>1)).mean()),
                    mean_preference_logit=float(rr.mean())))
            frame = data.table.iloc[data.te][['obs_id','class_id','class_name','mmsi','pass_group','port']].copy()
            frame['seed']=seed;frame['arm']=arm;frame['anchor']=anchor
            frame['alpha']=alpha[data.te];frame['paired']=av[data.te]
            frame['VV_delta_true']=z['delta_vv'].numpy()[data.te,y[data.te]]
            frame['VH_delta_true']=z['delta_vh'].numpy()[data.te,y[data.te]]
            details.append(frame)
        for arm in ('base','old_full','prior_late','hybrid_equal','hybrid_source'):
            with np.load(HERE/'results'/f'{arm}_{seed}.npz') as f:
                scores[seed,arm]=f['score'].copy()
        ids=data.te; yy=y[ids]
        for arm,comp in stats.PAIRS:
            a,b=scores[seed,arm][ids],scores[seed,comp][ids]
            for name,take in (('all',np.ones(len(ids),bool)),('paired',av[ids]),('missing',~av[ids])):
                population.append(dict(seed=seed,arm=arm,comparator=comp,group=name,**stats.effect(a[take],b[take],yy[take])))
            for k in np.flatnonzero((a.argmax(1)==yy)!=(b.argmax(1)==yy)):
                row=data.table.iloc[ids[k]]
                flips.append(dict(seed=seed,arm=arm,comparator=comp,obs_id=int(row.obs_id),
                    mmsi=row.mmsi,pass_group=row.pass_group,port=row.port,class_id=int(yy[k]),class_name=row.class_name,
                    case='rescue' if a[k].argmax()==yy[k] else 'harm',before=int(b[k].argmax()),after=int(a[k].argmax())))
        for arm in run.ARMS:
            assert np.max(abs(scores[seed,arm][ids[~av[ids]]]-scores[seed,'hybrid_source'][ids[~av[ids]]])) < 3e-6
    pd.DataFrame(groups).to_csv(HERE/'weight_groups.csv',index=False)
    pd.concat(details).to_csv(HERE/'weight_samples.csv',index=False)
    pd.DataFrame(population).to_csv(HERE/'population_pairs.csv',index=False)
    pd.DataFrame(flips).to_csv(HERE/'flips.csv',index=False)
    receipt=stats.statistics(data,scores)
    native=run.load('anchor_native',HERE.parent/'null_centered_readout_20261025/analyze.py');native.HERE=HERE
    geo=run.load('anchor_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in select_cases(data,scores):
        path,provenance,true,rival=native.panel(case,data,scores,geo,up,valid,snapshots,refs)
        seed,j=case['seed'],case['index'];_,z=run.inputs(data,seed)
        records.append(case|dict(figure=str(path),provenance=provenance,true=true,rival=rival,
            base_margin=float(z['base'][j,true]-z['base'][j,rival]),
            delta_VV=float(z['delta_vv'][j,true]-z['delta_vv'][j,rival]),
            delta_VH=float(z['delta_vh'][j,true]-z['delta_vh'][j,rival]),
            weights={a:states[seed,a]['weight'][j] for a in run.ARMS},
            comparisons={a:dict(prediction=int(s[j].argmax()),margin=float(s[j,true]-s[j,rival]))
                         for (ss,a),s in scores.items() if ss==seed}))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),frozen_replay=replay,missing_unchanged=True,
        limitation='Reused development, historical source; source-mean replay is diagnostic, not a separately selected new method.')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'paired.csv').groupby(['arm','comparator'])[['OA_pp','BA_pp','rescued','harmed']].mean().to_string(),flush=True)
    print(pd.read_csv(HERE/'weight_groups.csv').query("role=='evaluation'").to_string(index=False),flush=True)


if __name__=='__main__':
    main()
