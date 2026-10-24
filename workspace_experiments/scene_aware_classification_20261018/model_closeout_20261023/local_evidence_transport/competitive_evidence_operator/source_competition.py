"""Does the selected pair gate lose its useful margin in other class contests?"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('source_pair_run',HERE/'run.py')
run=importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


@torch.no_grad()
def main():
    z=np.load(HERE/'inputs.npz'); idx=z['idx']; locs=np.r_[z['tr'],z['va']]
    e=torch.tensor(z['e'][locs]/z['scale'][None,:,None,None],device='cuda')
    q=torch.tensor(z['q'][locs],device='cuda'); mask=torch.tensor(z['mask'][locs],device='cuda')
    cv=torch.tensor(z['cv'],device='cuda'); table=pd.read_csv(HERE/'table.csv').iloc[idx[locs]].reset_index(drop=True)
    rows=[]
    for seed in (17,29,43):
        m=run.Pool('pair',seed).cuda(); m.load_state_dict(torch.load(HERE/f'model_pair_{seed}.pt',weights_only=True))
        full=np.load(HERE/f'predictions_pair_{seed}.npz')['full'][locs]
        y=table.class_id.to_numpy(int)
        rival=np.where(np.arange(8)[None]==y[:,None],-np.inf,full).argmax(-1)
        ii,jj=m.pairs.T
        for first in range(0,len(locs),64):
            part=slice(first,first+64); ee=e[part]; mm=mask[part]
            _,_,w,edges=m(ee,q[part],mm,cv,True)
            uniform=mm.float()/mm.sum(-1,keepdim=True).clamp_min(1)
            old=((ee[...,ii]-ee[...,jj])*uniform[...,None]).sum(2)*(cv[:,ii]&cv[:,jj])[None]
            delta=(edges-old)*m.a[None,:,None]/cv.sum(-1)[None,:,None].clamp_min(1)
            mat=ee.new_zeros((len(ee),6,8,8)); mat[:,:,ii,jj]=delta; mat[:,:,jj,ii]=-delta
            node=mat.sum(-1)
            rr=torch.arange(len(ee),device='cuda'); yc=torch.tensor(y[part],device='cuda'); rc=torch.tensor(rival[part],device='cuda')
            total=(node[rr,:,yc]-node[rr,:,rc]).sum(-1)
            direct=2*mat[rr,:,yc,rc].sum(-1)
            other=total-direct
            for b in range(len(ee)):
                k=first+b
                rows.append(dict(seed=seed,role=table.iloc[k].role,obs_id=int(table.iloc[k].obs_id),true=int(y[k]),rival=int(rival[k]),base_correct=bool(full[k].argmax()==y[k]),direct=float(direct[b]),other=float(other[b]),total=float(total[b])))
    frame=pd.DataFrame(rows); frame.to_csv(HERE/'source_contests.csv',index=False)
    summary=[]
    for (seed,role),f in frame.groupby(['seed','role']):
        summary.append(dict(seed=seed,role=role,n=len(f),direct_mean=f.direct.mean(),other_mean=f.other.mean(),total_mean=f.total.mean(),
            direct_positive=int((f.direct>1e-6).sum()),direct_positive_other_overturn=int(((f.direct>1e-6)&(f.total<0)).sum()),
            direct_negative_other_rescue=int(((f.direct<-1e-6)&(f.total>0)).sum()),both_negative=int(((f.direct<-1e-6)&(f.other<0)).sum())))
    out=pd.DataFrame(summary); out.to_csv(HERE/'source_contest_summary.csv',index=False); print(out.to_string(index=False))


if __name__=='__main__':
    torch.set_num_threads(2); main()
