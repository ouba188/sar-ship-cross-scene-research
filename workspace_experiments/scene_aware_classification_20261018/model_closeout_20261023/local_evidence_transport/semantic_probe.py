"""Source-only diagnostic: does the correspondence key carry class evidence?

Same tokens, gallery and selected references. Only change the descriptor used
to compare them: PCA variance coordinates versus frozen local class contrasts.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from match_readout import unit,operations,pair_score,ARMS
from train_readout import metrics
HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'
OUT=HERE/'semantic_projection';OUT.mkdir(exist_ok=True)


def main():
    torch.set_num_threads(2)
    table=pd.read_csv(UP/'table.csv')
    fit=np.load(UP/'indices.npz')['fit_train']
    z=np.load(HERE/'tokens.npz');f=z['layer1'];mass=z['mass']
    w=np.load(UP/'models/reader_layer1.npz')['W'].reshape(8,2,256).transpose(1,0,2)
    response=np.einsum('nptd,pcd->nptc',f,w)
    response-=response.mean(-1,keepdims=True)
    mean=np.stack([response[fit,p][mass[fit,p]>0].mean(0) for p in range(2)])
    x=unit(response-mean[None,:,None,:]).astype(np.float32)
    x[mass<=0]=0
    np.savez_compressed(OUT/'projection.npz',W=w,mean=mean,scope=np.array('frozen reader fitted only original fit_train; mean also fit_train'))
    original=np.load(HERE/'scores_layer1.npz');refs=original['refs']
    rng=np.random.default_rng(17)
    perm=np.stack([rng.permutation(8) for _ in range(len(x)*2)]).reshape(len(x),2,8)
    xm=x.copy();mm=mass.copy()
    for i in range(len(x)):
        for p in range(2):
            xm[i,p,8:16]=x[i,p,8+perm[i,p]]
            mm[i,p,8:16]=mass[i,p,8+perm[i,p]]
    scores={arm:np.zeros((len(x),8,2,3,2),np.float32) for arm in ARMS}
    with torch.inference_mode():
        for first in range(0,len(x),32):
            idx=np.arange(first,min(first+32,len(x)));r=refs[idx]
            q,s,a,b=[torch.tensor(v,device='cuda') for v in (x[idx],x[r],mass[idx],mass[r])]
            values,t=operations(q,s,a,b)
            mq,ms,ma,mb=[torch.tensor(v,device='cuda') for v in (xm[idx],xm[r],mm[idx],mm[r])]
            side=torch.einsum('bpid,bckpjd->bckpij',mq[:,:,8:16],ms[...,8:16,:])
            mis=values['anchored'].clone()
            mis[...,1,:]=pair_score(t,side,ma[:,None,None,:,8:16],mb[...,8:16])
            values['mispaired']=mis
            for arm,value in values.items():scores[arm][idx]=value.mean(2).cpu().numpy()
    np.savez_compressed(OUT/'scores_layer1.npz',**scores,refs=refs)
    # Inspect source training role only; this is a mechanism diagnostic, not a
    # new selection rule or an additional target-label budget.
    take=np.flatnonzero(table.role.eq('utility_train'));y=table.iloc[take].class_id.to_numpy()
    rows=[]
    for key,data in [('PCA32',original),('class_contrast8',scores)]:
        for arm in ['pooled','independent','anchored']:
            v=data[arm][take]
            for region,ri in [('body',0),('side',1),('end',2)]:
                score=v[:,:,:,ri,0].mean(-1)
                true=score[np.arange(len(y)),y]
                wrong=np.where(np.arange(8)[None,:]==y[:,None],-np.inf,score).max(1)
                m=v[:,:,:,ri,1].mean(-1)
                rows.append(dict(descriptor=key,arm=arm,region=region,**metrics(y,score),mean_true_minus_best_wrong=float((true-wrong).mean()),true_matched_mass=float(m[np.arange(len(y)),y].mean()),mean_wrong_mass=float(((m.sum(1)-m[np.arange(len(y)),y])/7).mean())))
    pd.DataFrame(rows).to_csv(OUT/'source_key_diagnostic.csv',index=False)
    print(pd.DataFrame(rows)[['descriptor','arm','region','OA','BA','mean_true_minus_best_wrong','true_matched_mass','mean_wrong_mass']].to_string(index=False),flush=True)


if __name__=='__main__':main()
