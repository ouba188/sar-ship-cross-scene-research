"""Native-case review for direct fixed-relation evidence, not learned deltas."""
import os
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE/'fixed_readout'
spec=importlib.util.spec_from_file_location('fixed_distribution_analysis',HERE/'analyze.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
run=analysis.run
exp=run.exp


def main():
    dest=OUT/'analysis'
    dest.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(dest/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo,slots
    table,x,mass,bank,idx,tr,va,te=exp.source_data()
    receipts=pd.read_csv(OUT/'receipts.csv')
    active=receipts.query("output=='rbf' and step>0").seed.to_list()
    seed=int(active[0]) if active else 17
    pred=np.load(OUT/f'predictions_rbf_{seed}.npz')
    a=np.load(OUT/'weights.npz')[f'rbf_{seed}_a']
    divisor=pd.read_csv(OUT/'scales.csv').query("output=='rbf'").divisor.to_numpy()
    reader=run.Reader(x,mass,bank,'excess',device='cpu',dtype=torch.float64)
    y=table.class_id.to_numpy(int)
    old,new=pred['full'],pred['original']
    before=old[te].argmax(-1)==y[idx[te]]
    after=new[te].argmax(-1)==y[idx[te]]
    cases,margins,groups,refs=[],[],[],[]
    for name,mask in [('rescued',~before&after),('harmed',before&~after),('CC',before&after),('WW',~before&~after)]:
        candidates=te[mask]
        if not len(candidates):
            continue
        loc=int(candidates[np.argmin(table.iloc[idx[candidates]].obs_id.to_numpy())])
        i=int(idx[loc]); c=int(y[i])
        scores=new[loc] if name=='harmed' else old[loc]
        rival=int(np.argmax(np.where(np.arange(8)==c,-np.inf,scores)))
        raw,values,d,err=analysis.trace(reader,i,np.eye(32),np.eye(32))
        parts=a*(raw[c]-raw[rival])/divisor
        replay=abs(float(parts.sum()-(new[loc,c]-new[loc,rival]-old[loc,c]+old[loc,rival])))
        assert replay<4e-5,replay
        active_case=bool(np.any(parts))
        g=int(np.argmax(abs(parts) if active_case else abs((raw[c]-raw[rival])/divisor)))
        chosen=[]
        for cl in (c,rival):
            v=values[g,cl].sum(-1)/divisor[g]
            contributions=a[g]*v
            j=int(np.argmax(np.where(d['rv'][g,cl],abs(contributions) if active_case else abs(v),-np.inf)))
            chosen.append(int(bank[cl,j]))
            for r in np.flatnonzero(d['rv'][g,cl]):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,r]].obs_id),group=exp.edge.GROUPS[g],identity_evidence=float(v[r]),contribution=float(contributions[r])))
        cases.append(dict(i=i,loc=loc,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=rival,group=g,refs=chosen,seed=seed,active=active_case,replay_error=replay,allocation_error=err))
        groups.extend(dict(obs_id=int(table.iloc[i].obs_id),group=label,increment=float(parts[j])) for j,label in enumerate(exp.edge.GROUPS))
        for arm in ('full','linear','rbf'):
            value=old[loc] if arm=='full' else np.load(OUT/f'predictions_{arm}_{seed}.npz')['original'][loc]
            margins.append(dict(obs_id=int(table.iloc[i].obs_id),outcome=name,seed=seed,arm=arm,prediction=int(value.argmax()),margin=float(value[c]-value[rival])))
    wanted={int(table.iloc[i].obs_id) for ca in cases for i in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens=np.load(exp.ROOT/'tokens.npz'); provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pixels,fp,g,place=geo.native(table,snapshots,i,p)
                ax=axes[p,col]
                ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][i,p]:
                    masks=slots(g,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region=masks[lo:hi].sum(0)
                        if region.min()<.5<region.max():
                            ax.contour(region,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(tokens['valid'][i,p])}",xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=place))
        text='; '.join(v['arm']+f": pred{v['prediction']}, margin{v['margin']:+.3f}" for v in margins if v['obs_id']==ca['obs_id'])
        fig.suptitle(f"Fixed output-RKHS evidence | {ca['outcome']} {ca['obs_id']}; n={ca['n']}; seed{seed}\nGroup {exp.edge.GROUPS[ca['group']]}; active={ca['active']}")
        fig.text(.02,.02,text+'\nNative160 crop/pad; no resize. Symmetric reference allocation, not causal removal.',fontsize=8)
        fig.subplots_adjust(top=.83,bottom=.13,wspace=.25,hspace=.5)
        fig.savefig(dest/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    for name,rows in [('margins',margins),('groups',groups),('references',refs)]:
        pd.DataFrame(rows).to_csv(dest/f'{name}.csv',index=False)
    (dest/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (dest/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
