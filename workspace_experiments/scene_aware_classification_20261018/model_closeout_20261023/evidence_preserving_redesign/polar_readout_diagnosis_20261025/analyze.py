"""Paired polar readout diagnosis and native query/reference review."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import train as run
import json
import numpy as np
import pandas as pd
import torch

shared=run.load('polar_analysis_shared',HERE.parent/'anchored_m3_20261025/analyze.py')
shared.HERE=HERE
stats=shared.previous
stats.HERE=HERE
stats.PAIRS=[(a,'old_full') for a in run.ARMS]+[
    ('late_average','joint'),('late_average','joint_wide'),
    ('late_average','base'),('late_average','strong'),('vh','vv')]
effect,margin=stats.effect,stats.margin


def populations(data,scores):
    rows=[];kt=[];carry=[]
    for seed in run.SEEDS:
        z=np.load(HERE/'results'/f'inputs_{seed}.npz')
        cross=np.load(HERE/'results'/f'cross_scores_{seed}.npz')
        for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
            y=data.table.iloc[ids].class_id.to_numpy(int)
            for arm,comp in stats.PAIRS:
                for group,mask in (('all',np.ones(len(ids),bool)),
                    ('VV_VH_present',z['query_valid'][ids].all(1)),
                    ('VH_missing',~z['query_valid'][ids,1])):
                    a,b=scores[seed,arm][ids][mask],scores[seed,comp][ids][mask]
                    rows.append(dict(seed=seed,role=role,group=group,arm=arm,comparator=comp)|effect(a,b,y[mask]))
            for arm in ('old_full',)+run.ARMS:
                plain=cross[arm+'__plain'][ids]
                for mode in ('K_only','T_only','both'):
                    value=cross[arm+'__'+mode][ids]
                    kt.append(dict(seed=seed,role=role,arm=arm,mode=mode)|effect(value,plain,y))
        ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
        old=scores[seed,'old_full'][ids].argmax(1)==y
        strong=scores[seed,'strong'][ids].argmax(1)==y
        for arm in run.ARMS:
            ok=scores[seed,arm][ids].argmax(1)==y
            carry.append(dict(seed=seed,arm=arm,old_metric_rescues=int((old&~strong).sum()),
                preserved=int((ok&old&~strong).sum()),old_metric_harms=int((~old&strong).sum()),
                repaired=int((ok&~old&strong).sum())))
    pd.DataFrame(rows).to_csv(HERE/'population_pairs.csv',index=False)
    pd.DataFrame(kt).to_csv(HERE/'fixed_head_KT.csv',index=False)
    pd.DataFrame(carry).to_csv(HERE/'old_metric_carryover.csv',index=False)


@torch.inference_mode()
def polar_interventions(data):
    summary=[];cache={}
    for seed in run.SEEDS:
        reader=run.prior.Reader('pair_joint',seed)
        z=np.load(HERE/'results'/f'inputs_{seed}.npz')
        base,ix=z['base'],z['ix'];scaler=(z['mu'],z['scale'],z['active'])
        heads={a:run.restore(seed,a) for a in run.LEARNED}
        variants={'plain':z['plain'].copy(),'both':z['local'].copy()}
        for p,name in enumerate(('VV','VH')):
            v=z['plain'].copy();v[:,:,p]=z['local'][:,:,p];variants[name]=v
        for mode,v in variants.items():
            x,_=run.normalize(run.raw_features(v,base),data.tr,scaler)
            pred={a:base+m(x).cpu().numpy() for a,m in heads.items()}
            pred['late_average']=(pred['vv']+pred['vh'])*.5
            pred['old_full']=(torch.tensor(base,device='cuda')+reader.delta(
                torch.tensor(v,device='cuda'),torch.tensor(ix,device='cuda'))).cpu().numpy()
            for arm,s in pred.items():cache[seed,arm,mode]=s
        for arm in ('old_full',)+run.ARMS:
            ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
            s0=cache[seed,arm,'plain'][ids]
            rival=np.where(np.arange(8)[None]==y[:,None],-np.inf,s0).argmax(1)
            fm=lambda s:s[np.arange(len(ids)),y]-s[np.arange(len(ids)),rival]
            m0=fm(s0)
            for mode in ('VV','VH','both'):
                s=cache[seed,arm,mode][ids]
                summary.append(dict(seed=seed,arm=arm,mode=mode,
                    fixed_rival_margin_change=float((fm(s)-m0).mean()))|effect(s,s0,y))
            interaction=fm(cache[seed,arm,'both'][ids])-fm(cache[seed,arm,'VV'][ids])-fm(cache[seed,arm,'VH'][ids])+m0
            summary.append(dict(seed=seed,arm=arm,mode='interaction',
                fixed_rival_margin_change=float(interaction.mean()),max_abs_interaction=float(abs(interaction).max())))
        # The unused polar local evidence cannot affect an independent head.
        for arm,other in (('vv','VH'),('vh','VV')):
            np.testing.assert_allclose(cache[seed,arm,'plain'],cache[seed,arm,other],atol=1e-6)
        del reader,heads;torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(HERE/'polar_interventions.csv',index=False)
    return cache


def choose(data,scores):
    out=[];used=set();ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
    for case in ('rescue','harm','CC','WW'):
        candidates=[]
        for seed in run.SEEDS:
            a=scores[seed,'late_average'][ids].argmax(1)==y
            b=scores[seed,'old_full'][ids].argmax(1)==y
            mask={'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[case]
            for j in ids[mask]:
                obs=int(data.table.iloc[j].obs_id)
                if obs not in used:candidates.append((seed,obs,int(j)))
        candidates.sort()
        if candidates:
            seed,obs,j=candidates[0] if case in ('rescue','harm') else candidates[len(candidates)//2]
            out.append(dict(seed=seed,obs_id=obs,index=j,case=case,arm='late_average'));used.add(obs)
    for obs,seed in ((28839,29),(7875,29),(2080,43),(29687,43)):
        where=[int(j) for j in ids if int(data.table.iloc[j].obs_id)==obs]
        if where and obs not in used:
            j=where[0];y0=int(data.table.iloc[j].class_id)
            a=scores[seed,'late_average'][j].argmax()==y0;b=scores[seed,'old_full'][j].argmax()==y0
            case='CC' if a and b else 'WW' if not a and not b else 'rescue' if a else 'harm'
            out.append(dict(seed=seed,obs_id=obs,index=j,case=case,arm='late_average',historical=True));used.add(obs)
    return out


def main():
    assert (HERE/'complete.json').exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();scores={}
    for seed in run.SEEDS:
        for arm in ('old_full','base','strong')+run.ARMS:
            z=np.load(HERE/'results'/f'{arm}_{seed}.npz')
            np.testing.assert_array_equal(z['up_index'],data.up[:data.n])
            scores[seed,arm]=z['score'].copy()
    receipt=stats.statistics(data,scores);populations(data,scores)
    cache=polar_interventions(data)
    geo=run.load('polar_native_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in choose(data,scores):
        seed,j=case['seed'],case['index'];before=scores[seed,'old_full'][j];after=scores[seed,'late_average'][j]
        path,prov,true,rival=shared.panel(case,data,before,after,geo,up,valid,snapshots,refs)
        traces=[]
        cross=np.load(HERE/'results'/f'cross_scores_{seed}.npz')
        for arm in ('old_full',)+run.ARMS:
            for mode in ('plain','K_only','T_only','both'):
                s=cross[arm+'__'+mode][j]
                traces.append(dict(arm=arm,mode=mode,prediction=int(s.argmax()),fixed_margin=float(s[true]-s[rival])))
            for mode in ('VV','VH'):
                s=cache[seed,arm,mode][j]
                traces.append(dict(arm=arm,mode=mode,prediction=int(s.argmax()),fixed_margin=float(s[true]-s[rival])))
        z=np.load(HERE/'results'/f'inputs_{seed}.npz')
        records.append(case|dict(true=true,rival=rival,figure=str(path),provenance=prov,
            comparisons={a:dict(prediction=int(scores[seed,a][j].argmax()),fixed_margin=float(scores[seed,a][j,true]-scores[seed,a][j,rival])) for a in ('old_full','base','strong')+run.ARMS},
            local_true=z['local'][j,true],local_rival=z['local'][j,rival],
            plain_true=z['plain'][j,true],plain_rival=z['plain'][j,rival],traces=traces))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),note='Development diagnosis; no physical-causal attribution; source-selected independent heads.')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'paired.csv').to_string(index=False),flush=True)


if __name__=='__main__':main()
