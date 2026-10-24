"""Explain the positive independent output-RKHS candidate by body/context channel."""
import os
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_output_analysis',HERE/'run.py')
run=importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
exp=run.exp


def trace(reader,i,mb,mn):
    with torch.no_grad():
        _,raw,_,detail=reader.scores(np.array([i]),torch.tensor(mb,dtype=torch.float64),torch.tensor(mn,dtype=torch.float64),trace=True)
    d={k:v.cpu().numpy() for k,v in detail.items()}
    v=np.zeros((4,8,8,8,2))
    for g in range(4):
        if reader.counts[g,i]<2:
            continue
        for c in range(8):
            for p,(ak,qk,sk) in enumerate((('A','Bqs','Bss'),('An','Lqs','Lss'))):
                A=d[ak][g,0,c]
                L=d[qk][g,0,c]
                AL=A@d[sk][g,c]
                for r in range(8):
                    if not d['rv'][g,c,r]:
                        continue
                    s=slice(8*r,8*(r+1))
                    v[g,c,r,:,p]=.5*(2*np.sum(A[:,s]*L[:,s],-1)-np.sum(A[:,s]*AL[:,s],-1))/d['mq'][g,0]
    error=float(abs(v.sum((-1,-2,-3)).T-raw[0].numpy()).max())
    assert error<1e-10,error
    return raw[0].numpy(),v,d,error


def main():
    out=HERE/'independent_analysis'
    out.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(out/'.mplcache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from extract_tokens import geo,slots
    table,x,mass,bank,idx,tr,va,te=exp.source_data()
    seed=17
    data=np.load(HERE/f'predictions_independent_{seed}.npz')
    w=np.load(HERE/'weights.npz')
    mb,mn,a=[w[f'independent_{seed}_{k}'] for k in ('Mb','Mn','a')]
    div=pd.read_csv(HERE/'scales.csv').query("arm=='independent'").sort_values('feature').divisor.to_numpy()
    reader=run.Reader(x,mass,bank,'independent',device='cpu',dtype=torch.float64)
    labels=table.class_id.to_numpy(int)
    old,new=data['full'],data['original']
    before=old[te].argmax(-1)==labels[idx[te]]
    after=new[te].argmax(-1)==labels[idx[te]]
    cases,margins,channels,refs=[],[],[],[]
    for name,mask in [('rescued',~before&after),('harmed',before&~after),('CC',before&after),('WW',~before&~after)]:
        options=te[mask]
        if not len(options):
            continue
        loc=int(options[np.argmin(table.iloc[idx[options]].obs_id.to_numpy())])
        i=int(idx[loc]); c=int(labels[i])
        pred=new[loc] if name=='harmed' else old[loc]
        rival=int(np.argmax(np.where(np.arange(8)==c,-np.inf,pred)))
        raw0,v0,d0,e0=trace(reader,i,np.eye(32),np.eye(32))
        raw1,v1,d1,e1=trace(reader,i,mb,mn)
        contributions=(v1-v0).sum((2,3))
        channel=a[:,None]*(contributions[:,c]-contributions[:,rival])/div[:,None]
        error=abs(float(channel.sum()-(new[loc,c]-new[loc,rival]-old[loc,c]+old[loc,rival])))
        assert error<4e-5,error
        g=int(np.argmax(abs(channel.sum(-1))))
        chosen=[]
        for cl in (c,rival):
            val=a[g]*(v1[g,cl]-v0[g,cl]).sum((-1,-2))/div[g]
            j=int(np.argmax(np.where(d1['rv'][g,cl],abs(val),-np.inf)))
            chosen.append(int(bank[cl,j]))
            for r in np.flatnonzero(d1['rv'][g,cl]):
                refs.append(dict(obs_id=int(table.iloc[i].obs_id),class_id=cl,ref_obs=int(table.iloc[bank[cl,r]].obs_id),group=exp.edge.GROUPS[g],increment=float(val[r])))
        cases.append(dict(i=i,loc=loc,obs_id=int(table.iloc[i].obs_id),outcome=name,n=int(mask.sum()),true=c,rival=rival,refs=chosen,group=g,seed=seed,replay_error=error,allocation_error=max(e0,e1)))
        for j,group in enumerate(exp.edge.GROUPS):
            channels.append(dict(obs_id=int(table.iloc[i].obs_id),group=group,body=float(channel[j,0]),neighbor=float(channel[j,1]),total=float(channel[j].sum())))
        candidates={'full':old[loc],'independent_rbf':new[loc],
            'independent_linear':np.load(HERE.parent/f'kernel_condition_operator/predictions_independent_{seed}.npz')['original'][loc],
            'excess_rbf':np.load(HERE/f'predictions_excess_{seed}.npz')['original'][loc]}
        for arm,value in candidates.items():
            margins.append(dict(obs_id=int(table.iloc[i].obs_id),outcome=name,seed=seed,arm=arm,prediction=int(value.argmax()),margin=float(value[c]-value[rival])))
    wanted={int(table.iloc[j].obs_id) for ca in cases for j in [ca['i'],*ca['refs']]}
    snapshots={int(s['obs_id']):s for s in json.loads((geo.OLD/'local_condition_module/view_snapshot.json').read_text(encoding='utf-8')) if int(s['obs_id']) in wanted}
    tokens=np.load(exp.ROOT/'tokens.npz'); provenance=[]
    for ca in cases:
        fig,axes=plt.subplots(2,3,figsize=(12,8),squeeze=False)
        for col,(i,label) in enumerate(zip([ca['i'],*ca['refs']],('QUERY','TRUE REF','RIVAL REF'))):
            for p,pol in enumerate(('vv','vh')):
                pixels,fp,g,place=geo.native(table,snapshots,i,p)
                ax=axes[p,col]; ax.imshow(pixels,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                if tokens['valid'][i,p]:
                    masks=slots(g,fp)
                    for (lo,hi),color in zip(((0,8),(8,16),(16,20)),('#24a878','#2c84c6','#d39222')):
                        region=masks[lo:hi].sum(0)
                        if region.min()<.5<region.max():
                            ax.contour(region,levels=[.5],colors=[color],linewidths=.7)
                row=table.iloc[i]
                ax.set(title=f"{pol.upper()} {label} {int(row.obs_id)}\n{row.class_name}\n{row.port}; valid={int(tokens['valid'][i,p])}",xlim=(-.5,159.5),ylim=(159.5,-.5),xticks=[0,80,159],yticks=[0,80,159])
                provenance.append(dict(query_obs=ca['obs_id'],obs_id=int(row.obs_id),pol=pol,path=row['image_'+pol],placement=place))
        fig.suptitle(f"Independent output-RKHS | {ca['outcome']} {ca['obs_id']}; n={ca['n']}; seed{seed}\nActual contributions in {exp.edge.GROUPS[ca['group']]}")
        text='; '.join(v['arm']+f": pred{v['prediction']}, margin{v['margin']:+.3f}" for v in margins if v['obs_id']==ca['obs_id'])
        fig.text(.02,.02,text+'\nNative160 crop/pad, no resize. Separate body/context reconstruction; no conditional pairing claim.',fontsize=7.5)
        fig.subplots_adjust(top=.83,bottom=.13,wspace=.25,hspace=.5)
        fig.savefig(out/f"case_{ca['outcome']}_{ca['obs_id']}.png",dpi=130); plt.close(fig)
    for name,rows in [('margins',margins),('channels',channels),('references',refs)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    (out/'image_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(pd.DataFrame(margins).to_string(index=False))
    print(pd.DataFrame(channels).to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
