"""Replay the deployed single-bank episodic operator and inspect native cases."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('episodic_analysis_impl',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp


def main():
    out=HERE/'analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo,slots
    table,x,mass,bank,idx,tr,va,te=exp.source_data()
    arm='excess'
    receipts=pd.read_csv(HERE/'receipts.csv')
    active=receipts[receipts.method.eq('episodic')&receipts.arm.eq(arm)&receipts.selected_step.gt(0)].seed.to_list()
    seed=int(active[0]) if active else 17
    data=np.load(HERE/f'predictions_{arm}_{seed}.npz')
    weights=np.load(HERE/'weights.npz')
    mb,mn,a=[weights[f'episodic_{arm}_{seed}_{k}'] for k in ('Mb','Mn','a')]
    divisor=pd.read_csv(exp.HERE/'scales.csv').query("arm=='excess'").sort_values('feature').divisor.to_numpy()
    labels=table.class_id.to_numpy(int)
    old=data['full']; new=data['episodic_original']
    before=old[te].argmax(-1)==labels[idx[te]]
    after=new[te].argmax(-1)==labels[idx[te]]
    cases,margins,groups,refs,edges=[],[],[],[],[]
    for name,mask in [('rescued',~before&after),('harmed',before&~after),('CC',before&after),('WW',~before&~after)]:
        candidates=te[mask]
        if not len(candidates):
            continue
        loc=int(candidates[np.argmin(table.iloc[idx[candidates]].obs_id.to_numpy())])
        i=int(idx[loc]); c=int(labels[i])
        score=new[loc] if name=='harmed' else old[loc]
        rival=int(np.argmax(np.where(np.arange(8)==c,-np.inf,score)))
        identity=exp.trace_query(i,arm,np.eye(32),np.eye(32),x,mass,bank)
        learned=exp.trace_query(i,arm,mb,mn,x,mass,bank)
        parts=a*((learned['raw_scores'][c]-identity['raw_scores'][c])-(learned['raw_scores'][rival]-identity['raw_scores'][rival]))/divisor
        observed=new[loc,c]-new[loc,rival]-old[loc,c]+old[loc,rival]
        cached=data['episodic_original_parts'][loc,c]-data['episodic_original_parts'][loc,rival]
        err=max(abs(float(parts.sum()-observed)),float(np.max(abs(parts-cached))))
        assert err<3e-5,err
        group=int(np.argmax(abs(parts)))
        chosen=[]
        for cl in (c,rival):
            valid=learned['reference_valid'][group,0,cl]
            change=learned['reference_scores'][group,0,cl]-identity['reference_scores'][group,0,cl]
            increments=a[group]*change/divisor[group]/valid.sum()
            j=int(np.argmax(np.where(valid,abs(increments),-np.inf)))
            chosen.append(int(bank[cl,j]))
            for k in np.flatnonzero(valid):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,k]].obs_id),group=exp.edge.GROUPS[group],increment=float(increments[k])))
            ref=cl*8+j
            for state,trace in [('identity',identity),('learned',learned)]:
                A=trace['A'][group,0,ref]
                qn,sn=trace['qn'][group,0],trace['sn'][group,ref]
                prediction=A@sn
                T=2*np.sum(qn*prediction,-1)-np.sum(prediction*prediction,-1)
                ms=trace['ms'][group,ref]
                V=np.sum(sn*sn)/(ms-1)*(np.sum(A*A,-1)-A.sum(-1)**2/ms)
                for k in np.flatnonzero(trace['legal_edges'][0,group]):
                    edges.append(dict(obs_id=int(table.iloc[i].obs_id),ref_obs=int(table.iloc[bank[cl,j]].obs_id),state=state,group=exp.edge.GROUPS[group],edge=int(k),transfer=float(T[k]),null=float(V[k]),excess=float(T[k]+V[k])))
        cases.append(dict(i=i,loc=loc,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=rival,group=group,refs=chosen,seed=seed,replay_error=err))
        groups.extend(dict(obs_id=int(table.iloc[i].obs_id),group=label,increment=float(parts[g])) for g,label in enumerate(exp.edge.GROUPS))
        for label in ('full',*run.ARMS):
            value=old[loc] if label=='full' else np.load(HERE/f'predictions_{label}_{seed}.npz')['episodic_original'][loc]
            margins.append(dict(obs_id=int(table.iloc[i].obs_id),outcome=name,seed=seed,arm=label,prediction=int(value.argmax()),margin=float(value[c]-value[rival])))
    wanted={int(table.iloc[j].obs_id) for ca in cases for j in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens=np.load(exp.ROOT/'tokens.npz')
    provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pixels,fp,geometry,placement=geo.native(table,snapshots,int(i),p)
                ax=axes[p,col]
                ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][i,p]:
                    masks=slots(geometry,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region=masks[lo:hi].sum(0)
                        if region.min()<.5<region.max():
                            ax.contour(region,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(tokens['valid'][i,p])}",xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=placement))
        vals=[v for v in margins if v['obs_id']==ca['obs_id']]
        fig.suptitle(f"Multi-reference training, single original-bank deployment | {ca['outcome']} {ca['obs_id']}; n={ca['n']}; seed{seed}\nActual refs: strongest operator contribution in {exp.edge.GROUPS[ca['group']]}")
        fig.text(.02,.015,'; '.join(v['arm']+f": pred{v['prediction']}, margin{v['margin']:+.3f}" for v in vals)+'\nNative160 crop/pad, no resize. RF35. These are computational contributions, not physical attribution.',fontsize=8)
        fig.subplots_adjust(top=.83,bottom=.13,wspace=.25,hspace=.5)
        fig.savefig(out/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130)
        plt.close(fig)
    for name,rows in [('margins',margins),('groups',groups),('references',refs),('edges',edges)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))


if __name__=='__main__':
    exp.torch.set_num_threads(2)
    main()
