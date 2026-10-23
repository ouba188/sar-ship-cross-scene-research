"""Exact pooled-reference contribution allocation and native VV/VH review."""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('class_operator_analysis',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp


def trace(reader,i,mb,mn):
    with torch.no_grad():
        z,raw,valid,d=reader.scores(np.array([i]),torch.tensor(mb,dtype=torch.float64),torch.tensor(mn,dtype=torch.float64),trace=True)
    d={key:value.cpu().numpy() for key,value in d.items()}
    values=np.zeros((4,8,8,8))
    for g in range(4):
        if reader.counts[g,i]<2:
            continue
        qb,qn=d['qb'][g,0],d['qn'][g,0]
        for c in range(8):
            H=qb@d['operator'][g,c]
            for r in range(8):
                if not d['rv'][g,c,r]:
                    continue
                sb,sn=d['sb'][g,c,r],d['sn'][g,c,r]
                m=d['ms'][g,c,r]
                A=(qb@d['R'][g,c]@sb.T)*d['weight'][g,c,r]
                h=A@sn
                null=np.sum(sn*sn)/(m-1)*(np.sum(A*A,-1)-np.sum(A,-1)**2/m)
                # Split the total quadratic energy symmetrically across contributors.
                values[g,c,r]=(2*np.sum(qn*h,-1)-np.sum(h*H,-1)+null)/d['mq'][g,0]
    error=float(np.max(abs(values.sum((-1,-2)).T-raw[0].numpy())))
    assert error<1e-10,error
    return raw[0].numpy(),values,d,error


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
    receipt=pd.read_csv(HERE/'receipts.csv')
    active=receipt[receipt.arm.eq(arm)&receipt.selected_step.gt(0)].seed.to_list()
    seed=int(active[0]) if active else 17
    data=np.load(HERE/f'predictions_{arm}_{seed}.npz')
    weights=np.load(HERE/'weights.npz')
    mb,mn,a=[weights[f'{arm}_{seed}_{k}'] for k in ('Mb','Mn','a')]
    divisor=pd.read_csv(HERE/'scales.csv').query("arm=='excess'").sort_values('feature').divisor.to_numpy()
    reader=run.Reader(x,mass,bank,arm,device='cpu',dtype=torch.float64)
    labels=table.class_id.to_numpy(int)
    old,new=data['full'],data['original']
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
        raw0,v0,d0,e0=trace(reader,i,np.eye(32),np.eye(32))
        raw1,v1,d1,e1=trace(reader,i,mb,mn)
        parts=a*((raw1[c]-raw0[c])-(raw1[rival]-raw0[rival]))/divisor
        observed=new[loc,c]-new[loc,rival]-old[loc,c]+old[loc,rival]
        err=max(abs(float(parts.sum()-observed)),float(np.max(abs(parts-(data['parts'][loc,c]-data['parts'][loc,rival])))))
        assert err<3e-5,err
        g=int(np.argmax(abs(parts)))
        chosen=[]
        for cl in (c,rival):
            valid=d1['rv'][g,cl]
            increments=a[g]*(v1[g,cl]-v0[g,cl]).sum(-1)/divisor[g]
            j=int(np.argmax(np.where(valid,abs(increments),-np.inf)))
            chosen.append(int(bank[cl,j]))
            for k in np.flatnonzero(valid):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,k]].obs_id),group=exp.edge.GROUPS[g],increment=float(increments[k])))
            for k in np.flatnonzero(reader.numpy_legal[i,g]):
                edges.append(dict(obs_id=int(table.iloc[i].obs_id),ref_obs=int(table.iloc[bank[cl,j]].obs_id),group=exp.edge.GROUPS[g],edge=int(k),identity=float(v0[g,cl,j,k]),learned=float(v1[g,cl,j,k]),increment=float(a[g]*(v1[g,cl,j,k]-v0[g,cl,j,k])/divisor[g])))
        cases.append(dict(i=i,loc=loc,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=rival,group=g,refs=chosen,seed=seed,replay_error=err,allocation_error=max(e0,e1)))
        groups.extend(dict(obs_id=int(table.iloc[i].obs_id),group=label,increment=float(parts[g])) for g,label in enumerate(exp.edge.GROUPS))
        for label in ('full',*run.ARMS):
            value=old[loc] if label=='full' else np.load(HERE/f'predictions_{label}_{seed}.npz')['original'][loc]
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
        fig.suptitle(f"Class-pooled conditional operator | {ca['outcome']} {ca['obs_id']}; n={ca['n']}; seed{seed}\nRefs from exact symmetric energy allocation in {exp.edge.GROUPS[ca['group']]}")
        fig.text(.02,.015,'; '.join(v['arm']+f": pred{v['prediction']}, margin{v['margin']:+.3f}" for v in vals)+'\nNative160 crop/pad, no resize; RF35. Contributions allocate cross-reference interactions, not individual causal effects.',fontsize=7.5)
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
