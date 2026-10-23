"""Separate pair correspondence from mere marginal token weighting."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from match_readout import unit,sinkhorn,pair_score,END,IS_END
from train_readout import Head,inputs
HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'


def main():
    torch.set_num_threads(2)
    table=pd.read_csv(UP/'table.csv')
    z=np.load(HERE/'tokens.npz');mass=z['mass']
    layer='layer1';f=z[layer]
    projection=np.load(HERE/f'projection_{layer}.npz')
    x=unit((f-projection['mean'])@projection['components'].T).astype(np.float32)
    x[mass<=0]=0
    original=np.load(HERE/f'scores_{layer}.npz')
    refs=original['refs']; substitute=np.zeros_like(original['anchored'])
    stats=[]
    with torch.inference_mode():
        for first in range(0,len(x),32):
            idx=np.arange(first,min(first+32,len(x)));r=refs[idx]
            q,s,a,b=[torch.tensor(v,device='cuda') for v in (x[idx],x[r],mass[idx],mass[r])]
            sim=torch.einsum('bpid,bckpjd->bckpij',q[:,:,:8],s[...,:8,:])
            t,whole,aa,bb=sinkhorn(sim,a[:,None,None,:,:8],b[...,:8])
            total=t.sum((-1,-2))
            factor=t.sum(-1)[..., :,None]*t.sum(-2)[...,None,:]/total[...,None,None].clamp_min(1e-20)
            relative=(t-factor).abs().sum((-1,-2))/total.clamp_min(1e-20)
            norm=t/total[...,None,None].clamp_min(1e-20)
            entropy=-(norm*norm.clamp_min(1e-20).log()).sum((-1,-2))/np.log(64)
            side=torch.einsum('bpid,bckpjd->bckpij',q[:,:,8:16],s[...,8:16,:])
            end=torch.einsum('bpid,bckpjd->bckpij',q[:,:,END],s[...,END,:])
            ea=a[:,None,None,:,END]*torch.tensor(IS_END,device='cuda')
            eb=b[...,END]*torch.tensor(IS_END,device='cuda')
            packed=torch.stack([pair_score(factor,sim,a[:,None,None,:,:8],b[...,:8]),pair_score(factor,side,a[:,None,None,:,8:16],b[...,8:16]),pair_score(factor,end,ea,eb)],-2)
            substitute[idx]=packed.mean(2).cpu().numpy()
            rel=relative.cpu().numpy(); ent=entropy.cpu().numpy();tot=total.cpu().numpy()
            for j,i in enumerate(idx):
                for p in range(2):
                    if not z['valid'][i,p]:continue
                    stats.append(dict(local_index=i,obs_id=int(table.iloc[i].obs_id),role=table.iloc[i].role,pol=p,relative_nonfactorization=float(rel[j,:,:,p].mean()),normalized_entropy=float(ent[j,:,:,p].mean()),matched_mass=float(tot[j,:,:,p].mean())))
    np.savez_compressed(HERE/'scores_layer1_factorized.npz',factorized=substitute,refs=refs)
    pd.DataFrame(stats).to_csv(HERE/'analysis/transport_structure.csv',index=False)
    print(pd.DataFrame(stats).groupby('role')[['relative_nonfactorization','normalized_entropy','matched_mass']].mean().to_string(),flush=True)
    # Same trained head, same normalizer, only replacing the local pairing.
    tr=np.flatnonzero(table.role.eq('utility_train'))
    te=np.flatnonzero(table.role.eq('evaluation'))
    rows=[]; case_rows=[]
    for seed in (17,29,43):
        base=np.load(UP/f'strategies_seed{seed}.npz')['logits'][:,2].astype(np.float32)
        ck=torch.load(HERE/'models'/f'layer1_anchored_{seed}.pt',map_location='cpu',weights_only=False)
        head=Head();head.load_state_dict(ck['state']);head.eval()
        zz=base-base.mean(-1,keepdims=True)
        rival=np.stack([np.delete(zz,c,axis=1).max(1) for c in range(8)],1)
        def predict(local):
            raw=np.concatenate([local.reshape(len(x),8,12),zz[...,None],rival[...,None],(zz-rival)[...,None]],-1)
            xx=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
            with torch.no_grad():return base+head(torch.tensor(xx)).numpy()
        actual=predict(original['anchored']); changed=predict(substitute)
        cached=np.load(HERE/'results'/f'layer1_anchored_{seed}.npz')['score']
        assert np.max(abs(actual-cached))<1e-5
        for role in ['utility_train','selection','evaluation']:
            take=np.flatnonzero(table.role.eq(role)); y=table.iloc[take].class_id.to_numpy()
            a=actual[take].argmax(1)==y;b=changed[take].argmax(1)==y;d=a.astype(float)-b
            rows.append(dict(seed=seed,role=role,n=len(take),actual_better=int((a&~b).sum()),factorized_better=int((~a&b).sum()),delta_OA=100*d.mean(),delta_BA=100*np.mean([d[y==c].mean() for c in np.unique(y)]),mean_abs_logit_change=float(abs(actual[take]-changed[take]).mean())))
        for case in pd.read_csv(HERE/'analysis/case_selection.csv').to_dict('records'):
            i=case['local_index'];y=int(table.iloc[i].class_id)
            c=int(np.argmax(np.where(np.arange(8)==y,-np.inf,actual[i])))
            case_rows.append(dict(seed=seed,obs_id=int(table.iloc[i].obs_id),rival=c,actual_margin=float(actual[i,y]-actual[i,c]),factorized_margin=float(changed[i,y]-changed[i,c])))
    pd.DataFrame(rows).to_csv(HERE/'analysis/fixed_head_factorization.csv',index=False)
    pd.DataFrame(case_rows).to_csv(HERE/'analysis/case_factorization.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
