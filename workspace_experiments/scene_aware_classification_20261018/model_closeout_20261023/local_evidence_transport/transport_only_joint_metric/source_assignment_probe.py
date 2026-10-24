"""Frozen source-only decomposition of real-match mass and its allocation."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('transport_train',HERE/'train.py')
train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)

@torch.no_grad()
def main():
    torch.set_num_threads(2);rows=[];receipts=[]
    for seed in train.SEEDS:
        rd=train.Reader('transport_only',seed);model=train.Metric('transport_only').cuda()
        ck=torch.load(HERE/'models'/f'transport_only_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
        indices=np.flatnonzero(rd.table.role.isin(['utility_train','selection']))
        outputs={k:[] for k in ('old','mass_only','allocation_only','both')};stats=[];h=model.project(rd.d)
        for start in range(0,len(indices),32):
            idx=torch.as_tensor(indices[start:start+32],device='cuda');ref=rd.refs[idx]
            q,s=rd.x[idx],rd.x[ref];hq,hs=h[idx],h[ref];a,b=rd.mass[idx],rd.mass[ref]
            groups={k:[] for k in outputs}
            for ri,r in enumerate(train.old.REGIONS):
                qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
                hi=hq[:,:,r][:,None,None,:,:,None,:];hj=hs[...,r,:][...,None,:,:]
                k0=(qi*sj).sum(-1);k1=train.previous.similarity(qi,sj,model.weights(hi,hj))
                qa=a[:,None,None,:,r];sa=b[...,r];keep=(qa>0)[...,None]&(sa>0)[...,None,:]
                t0=rd.fixed[ri][idx]*keep;t1=train.old.sinkhorn(k1,qa,sa)[0]*keep
                m0=t0.sum((-1,-2),keepdim=True);m1=t1.sum((-1,-2),keepdim=True)
                p0=t0/m0.clamp_min(1e-20);p1=t1/m1.clamp_min(1e-20)
                for name,t in [('old',t0),('mass_only',m1*p0),('allocation_only',m0*p1),('both',t1)]:
                    groups[name].append(train.old.pair_score(t,k0,qa,sa))
                valid=keep.any((-1,-2))
                for p in range(2):
                    v=valid[...,p]
                    stats.append(dict(region=ri,pol=p,n=int(v.sum()),mass_absolute_sum=float((m1-m0).abs()[...,p,0,0][v].sum()),allocation_L1_sum=float((p1-p0).abs().sum((-1,-2))[...,p][v].sum())))
            for name,g in groups.items():
                local=torch.stack(g,-2).mean(2)
                score=rd.anchor[idx]+rd.delta(local,idx)-rd.old_delta[idx]
                outputs[name].append(score.cpu().numpy())
        outputs={k:np.concatenate(v) for k,v in outputs.items()}
        cached=np.load(HERE/'results'/f'transport_only_{seed}.npz')['score'][indices]
        err0=float(abs(outputs['old']-rd.strong[indices]).max());err1=float(abs(outputs['both']-cached).max())
        assert err0<3e-5 and err1<3e-5,(err0,err1)
        for name,score in outputs.items():
            for role in ('utility_train','selection'):
                take=np.flatnonzero(rd.table.iloc[indices].role.eq(role));y=rd.table.iloc[indices[take]].class_id.to_numpy(int)
                rows.append(dict(seed=seed,mode=name,role=role,**train.old.metrics(y,score[take])))
        st=pd.DataFrame(stats).groupby(['region','pol']).sum().reset_index()
        st['mean_abs_mass_change']=st.mass_absolute_sum/st.n;st['mean_allocation_L1']=st.allocation_L1_sum/st.n
        st['seed']=seed;st.to_csv(HERE/f'source_assignment_change_{seed}.csv',index=False)
        sel=rd.table.iloc[indices].role.eq('selection').to_numpy()
        receipts.append(dict(seed=seed,n=len(indices),selected_step=ck['receipt']['selected_step'],old_replay=err0,new_replay=err1,
            selection_mass_vs_both_prediction_disagreements=int((outputs['mass_only'][sel].argmax(1)!=outputs['both'][sel].argmax(1)).sum()),
            selection_allocation_vs_old_prediction_disagreements=int((outputs['allocation_only'][sel].argmax(1)!=outputs['old'][sel].argmax(1)).sum())))
        del rd,model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(HERE/'source_assignment_metrics.csv',index=False)
    (HERE/'source_assignment_receipt.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).groupby(['role','mode'])[['OA','BA','NLL']].mean().to_string())

if __name__=='__main__':main()
