"""Class/pass/availability outcomes and native-image case review."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplcache')
import train as run
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stats=run.load('null_stats',HERE.parent/'typed_m3_20261025/analyze.py')
stats.HERE=HERE
stats.PAIRS=[(a,'old_full') for a in run.ARMS]+[(a,b) for a,b in (
    ('center_late','prior_late'),('center_late','mask_late'),('mask_late','prior_late'),
    ('center_late','center_joint'),('center_joint','prior_wide'),('center_late','base'),
    ('center_vh','prior_vh'),('center_vv','prior_vv'))]

def population(data,scores):
    rows=[];kt=[];flips=[]
    for seed in run.SEEDS:
        z=np.load(run.PREVIOUS/'results'/f'inputs_{seed}.npz')
        cross=np.load(HERE/'results'/f'cross_scores_{seed}.npz')
        for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
            y=data.table.iloc[ids].class_id.to_numpy(int)
            for arm,comp in stats.PAIRS:
                a,b=scores[seed,arm][ids],scores[seed,comp][ids]
                for group,take in (('all',np.ones(len(ids),bool)),('paired',z['query_valid'][ids].all(1)),
                    ('VH_missing',~z['query_valid'][ids,1])):
                    rows.append(dict(seed=seed,role=role,arm=arm,comparator=comp,group=group)|stats.effect(a[take],b[take],y[take]))
                if role=='evaluation' and arm in ('center_late','mask_late'):
                    changes=(a.argmax(1)==y)!=(b.argmax(1)==y)
                    for k in np.flatnonzero(changes):
                        j=ids[k];r=data.table.iloc[j]
                        flips.append(dict(seed=seed,arm=arm,comparator=comp,obs_id=int(r.obs_id),
                            mmsi=r.mmsi,port=r.port,pass_group=r.pass_group,class_id=int(y[k]),class_name=r.class_name,
                            case='rescue' if a[k].argmax()==y[k] else 'harm',VH_present=bool(z['query_valid'][j,1]),
                            before=int(b[k].argmax()),after=int(a[k].argmax())))
            for arm in run.ARMS:
                plain=cross[arm+'__plain'][ids]
                for mode in ('K_only','T_only','both'):
                    kt.append(dict(seed=seed,role=role,arm=arm,mode=mode)|stats.effect(cross[arm+'__'+mode][ids],plain,y))
    pd.DataFrame(rows).to_csv(HERE/'population_pairs.csv',index=False)
    pd.DataFrame(kt).to_csv(HERE/'fixed_head_KT.csv',index=False)
    pd.DataFrame(flips).to_csv(HERE/'flips.csv',index=False)

def cases(data,scores):
    result=[];used=set();ids=data.te;y=data.table.iloc[ids].class_id.to_numpy(int)
    for name in ('rescue','harm','CC','WW'):
        choices=[]
        for seed in run.SEEDS:
            a=scores[seed,'center_late'][ids].argmax(1)==y;b=scores[seed,'prior_late'][ids].argmax(1)==y
            select={'rescue':a&~b,'harm':~a&b,'CC':a&b,'WW':~a&~b}[name]
            for j in ids[select]:
                obs=int(data.table.iloc[j].obs_id)
                if obs not in used:choices.append((seed,obs,int(j)))
        choices.sort()
        if choices:
            seed,obs,j=choices[0] if name in ('rescue','harm') else choices[len(choices)//2]
            result.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm='center_late',comparator='prior_late'));used.add(obs)
    for obs,seed in ((911,17),(4718,17),(2080,43),(29687,43)):
        ids0=[int(j) for j in ids if int(data.table.iloc[j].obs_id)==obs]
        if ids0 and obs not in used:
            j=ids0[0];yy=int(data.table.iloc[j].class_id)
            a=scores[seed,'center_late'][j].argmax()==yy;b=scores[seed,'prior_late'][j].argmax()==yy
            name='CC' if a and b else 'WW' if not a and not b else 'rescue' if a else 'harm'
            result.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm='center_late',comparator='prior_late',historical=True));used.add(obs)
    # Inspect at least one actual missing-VH correctness change, if present.
    f=pd.read_csv(HERE/'flips.csv')
    f=f[f.arm.eq('center_late')&f.comparator.eq('prior_late')&~f.VH_present]
    for row in f.sort_values(['seed','obs_id']).itertuples():
        if row.obs_id not in used:
            j=int(np.flatnonzero(data.table.obs_id.eq(row.obs_id)&data.table.role.eq('evaluation'))[0])
            result.append(dict(seed=int(row.seed),obs_id=int(row.obs_id),index=j,case=row.case,arm='center_late',comparator='prior_late',missing_VH=True));break
    for name in ('rescue','harm'):
        choices=[]
        for seed in run.SEEDS:
            a=scores[seed,'center_vh'][ids].argmax(1)==y;b=scores[seed,'prior_vh'][ids].argmax(1)==y
            select=a&~b if name=='rescue' else ~a&b
            for j in ids[select]:
                obs=int(data.table.iloc[j].obs_id)
                if obs not in used:choices.append((seed,obs,int(j)))
        if choices:
            seed,obs,j=sorted(choices)[0];used.add(obs)
            result.append(dict(seed=seed,obs_id=obs,index=j,case=name,arm='center_vh',comparator='prior_vh'))
    return result

def panel(case,data,scores,geo,up,valid,snapshots,refs):
    j=case['index'];i=int(data.up[j]);seed=case['seed'];true=int(up.iloc[i].class_id)
    before=scores[seed,case['comparator']][j];after=scores[seed,case['arm']][j]
    s=after if case['case']=='harm' else before
    rival=int(np.where(np.arange(8)==true,-np.inf,s).argmax())
    views=[(i,'QUERY')]+[(int(refs[i,c,k]),name+str(k+1)) for c,name in ((true,'TRUE '),(rival,'RIVAL ')) for k in range(2)]
    fig,axs=plt.subplots(2,5,figsize=(16,7));prov=[]
    for p,pol in enumerate(('vv','vh')):
        for col,(ri,name) in enumerate(views):
            row=up.iloc[ri];obs=int(row.obs_id);ax=axs[p,col]
            entry=dict(up_index=ri,obs_id=obs,polarization=pol,path=str(row['image_'+pol]))
            if valid[ri,p] and obs in snapshots and pol in snapshots[obs]['images']:
                pix,fp,g,place=geo.native(up,snapshots,ri,p)
                ax.imshow(pix,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                for region,color in zip(geo.soft_regions(g)*fp,('#20a780','#d77329','#3287d0')):
                    if region.min()<.5<region.max():ax.contour(region,levels=[.5],colors=[color],linewidths=.55)
                entry.update(status='rendered',geometry=g,placement=place)
            else:ax.text(.5,.5,'MISSING VIEW',ha='center',transform=ax.transAxes);entry['status']='missing'
            ax.set_title(f'{pol.upper()} {name} {obs}\n{row.class_name}',fontsize=8);ax.set(xticks=[],yticks=[]);prov.append(entry)
    fig.suptitle(f"{case['arm']} vs {case['comparator']} | {case['case']} | seed{seed} | obs{case['obs_id']}\n"
        f"true={true},rival={rival}; pred {before.argmax()} -> {after.argmax()}; fixed margin {before[true]-before[rival]:+.5f} -> {after[true]-after[rival]:+.5f}",fontsize=11)
    fig.text(.02,.02,'Native160 crop/pad; no ship resize. Green body / orange ends / blue sides. Actual frozen references.',fontsize=9)
    fig.subplots_adjust(top=.84,bottom=.09,wspace=.12,hspace=.18)
    path=HERE/'analysis'/f"{case['case']}_{case['obs_id']}_{seed}.png";fig.savefig(path,dpi=120);plt.close(fig)
    return path,prov,true,rival

def main():
    assert (HERE/'complete.json').exists()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=run.Data();scores={}
    for seed in run.SEEDS:
        for arm in run.ARMS+run.COMPARATORS:
            z=np.load(HERE/'results'/f'{arm}_{seed}.npz')
            np.testing.assert_array_equal(z['up_index'],data.up[:data.n]);scores[seed,arm]=z['score']
    receipt=stats.statistics(data,scores);population(data,scores)
    geo=run.load('null_native_geo',run.BASE.parent/'eight_class_evidence_20261023/obb_dynamic_readout/common.py')
    up=pd.read_csv(run.BASE/'regional_utility_pilot/upstream/table.csv')
    valid=np.load(run.BASE/'regional_utility_pilot/upstream/summaries.npz')['valid']
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8'))}
    refs=np.load(run.LOCAL/'scores_layer1.npz')['refs'];records=[]
    for case in cases(data,scores):
        seed,j=case['seed'],case['index'];path,prov,true,rival=panel(case,data,scores,geo,up,valid,snapshots,refs)
        cross=np.load(HERE/'results'/f'cross_scores_{seed}.npz');traces=[]
        for arm in run.ARMS:
            for mode in ('plain','K_only','T_only','both'):
                s=cross[arm+'__'+mode][j];traces.append(dict(arm=arm,mode=mode,prediction=int(s.argmax()),fixed_margin=float(s[true]-s[rival])))
        records.append(case|dict(true=true,rival=rival,figure=str(path),provenance=prov,traces=traces,
            comparisons={a:dict(prediction=int(scores[seed,a][j].argmax()),fixed_margin=float(scores[seed,a][j,true]-scores[seed,a][j,rival])) for a in run.ARMS+run.COMPARATORS}))
        print('CASE',case,flush=True)
    run.write_json(HERE/'analysis/cases.json',records)
    receipt.update(cases=len(records),interpretation='Frozen input/metric, matched readout refits; not an independent confirmation or a new frequency mechanism.')
    run.write_json(HERE/'analysis_receipt.json',receipt)
    print(pd.read_csv(HERE/'paired.csv').to_string(index=False),flush=True)

if __name__=='__main__':main()
