"""Apply frozen local operators and heads to broader historical target coverage."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from match_readout import unit,operations,pair_score,END,IS_END,ARMS
from train_readout import Head
from pipeline import metrics

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'
EXT=HERE/'external_confirmation'


def project(z,pr):
    f=z['layer1']
    x=unit((f-pr['mean'])@pr['components'].T).astype(np.float32)
    x[z['mass']<=0]=0
    return x


def body(x,m):
    v=(x[:,:,:8]*m[:,:,:8,None]).sum(2)/np.maximum(m[:,:,:8].sum(2)[...,None],1e-8)
    return unit(v).reshape(len(x),-1)


def main():
    tick=time.perf_counter();torch.set_num_threads(2)
    assert (HERE/'EXTERNAL_FREEZE_ZH.md').exists()
    table=pd.read_csv(EXT/'table.csv',dtype={'mmsi':str})
    source=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    for col in ('mmsi','pass_group'):
        assert not set(source[col])&set(table[col]),col
    z=np.load(EXT/'tokens.npz');src=np.load(HERE/'tokens.npz')
    np.testing.assert_array_equal(z['instance_ids'].astype(str),table.instance_id.to_numpy())
    pr=np.load(HERE/'projection_layer1.npz');bank=pr['bank']
    x=project(z,pr);xs=project(src,pr)
    m=z['mass'];ms=src['mass']
    cs=np.einsum('nd,ckd->nck',body(x,m),body(xs,ms)[bank])
    pick=np.argsort(-cs,axis=-1)[...,:2]
    refs=np.take_along_axis(np.broadcast_to(bank,(len(x),8,8)),pick,-1)
    out={k:np.zeros((len(x),8,2,3,2),np.float32) for k in (*ARMS,'factorized')}
    rng=np.random.default_rng(17)
    source_perm=np.stack([rng.permutation(8) for _ in range(len(xs)*2)]).reshape(len(xs),2,8)
    target_perm=np.stack([rng.permutation(8) for _ in range(len(x)*2)]).reshape(len(x),2,8)
    xmis=x.copy();xsmis=xs.copy();mmis=m.copy();msmis=ms.copy()
    for f,ff,a,aa,perm in [(x,xmis,m,mmis,target_perm),(xs,xsmis,ms,msmis,source_perm)]:
        for i in range(len(f)):
            for p in range(2):
                ff[i,p,8:16]=f[i,p,8+perm[i,p]]
                aa[i,p,8:16]=a[i,p,8+perm[i,p]]
    with torch.inference_mode():
        for first in range(0,len(x),32):
            idx=np.arange(first,min(first+32,len(x)));r=refs[idx]
            q,s,a,b=[torch.tensor(v,device='cuda') for v in (x[idx],xs[r],m[idx],ms[r])]
            values,t=operations(q,s,a,b)
            mq,ss,ma,mb=[torch.tensor(v,device='cuda') for v in (xmis[idx],xsmis[r],mmis[idx],msmis[r])]
            side=torch.einsum('bpid,bckpjd->bckpij',mq[:,:,8:16],ss[...,8:16,:])
            mis=values['anchored'].clone()
            mis[...,1,:]=pair_score(t,side,ma[:,None,None,:,8:16],mb[...,8:16])
            values['mispaired']=mis
            total=t.sum((-1,-2))
            fac=t.sum(-1)[..., :,None]*t.sum(-2)[...,None,:]/total[...,None,None].clamp_min(1e-20)
            sim=torch.einsum('bpid,bckpjd->bckpij',q[:,:,:8],s[...,:8,:])
            side=torch.einsum('bpid,bckpjd->bckpij',q[:,:,8:16],s[...,8:16,:])
            end=torch.einsum('bpid,bckpjd->bckpij',q[:,:,END],s[...,END,:])
            ea=a[:,None,None,:,END]*torch.tensor(IS_END,device='cuda')
            eb=b[...,END]*torch.tensor(IS_END,device='cuda')
            values['factorized']=torch.stack([pair_score(fac,sim,a[:,None,None,:,:8],b[...,:8]),pair_score(fac,side,a[:,None,None,:,8:16],b[...,8:16]),pair_score(fac,end,ea,eb)],-2)
            for arm,v in values.items():out[arm][idx]=v.mean(2).cpu().numpy()
    np.savez_compressed(EXT/'local_scores.npz',**out,refs=refs)
    results=EXT/'results';results.mkdir(exist_ok=True)
    rows=[];pairs=[];cls=[];coverage=[]
    y=table.class_id.to_numpy()
    for seed in (17,29,43):
        old=np.load(EXT/f'strategies_seed{seed}.npz')['logits']
        base=old[:,2].astype(np.float32);zscore=base-base.mean(-1,keepdims=True)
        competitor=np.stack([np.delete(zscore,c,axis=1).max(1) for c in range(8)],1)
        allscores={name:old[:,k] for k,name in enumerate(['base','body','full'])}
        for arm in (*ARMS,'factorized','body_match','score_only'):
            if arm=='score_only':
                others=np.stack([np.sort(np.delete(zscore,c,axis=1),axis=1) for c in range(8)],1)
                raw=np.concatenate([zscore[...,None],others,zscore[...,None]-others],-1)
                name='none_score_only'
            else:
                local=out['anchored' if arm=='body_match' else arm].copy()
                if arm=='body_match':local[:,:,:,1:]=0
                raw=np.concatenate([local.reshape(len(x),8,12),zscore[...,None],competitor[...,None],(zscore-competitor)[...,None]],-1)
                name='layer1_'+arm
            ck=torch.load(HERE/'models'/f'{name}_{seed}.pt',map_location='cpu',weights_only=False)
            inputs=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
            head=Head();head.load_state_dict(ck['state']);head.eval()
            with torch.no_grad():
                delta=np.concatenate([head(torch.tensor(inputs[i:i+256])).numpy() for i in range(0,len(x),256)])
            pred=base+delta
            np.savez_compressed(results/f'{arm}_{seed}.npz',score=pred,delta=delta)
            allscores[arm]=pred
        for arm,pred in allscores.items():
            rows.append(dict(seed=seed,arm=arm,**metrics(y,pred)))
            for scope,mask in [('original_fold0',table.original_test_fold.eq(0).to_numpy()),('additional_folds',table.original_test_fold.ne(0).to_numpy())]:
                coverage.append(dict(seed=seed,arm=arm,scope=scope,**metrics(y[mask],pred[mask])))
        a=allscores['anchored'].argmax(1)==y
        for arm,pred in allscores.items():
            b=pred.argmax(1)==y;d=a.astype(float)-b
            pairs.append(dict(seed=seed,comparison=arm,n=len(y),rescue=int((a&~b).sum()),harm=int((~a&b).sum()),delta_OA=100*d.mean(),delta_BA=100*np.mean([d[y==c].mean() for c in np.unique(y)])))
            for c in np.unique(y):
                take=y==c
                cls.append(dict(seed=seed,comparison=arm,class_id=int(c),n=int(take.sum()),rescue=int((a[take]&~b[take]).sum()),harm=int((~a[take]&b[take]).sum()),delta_recall=100*d[take].mean()))
    pd.DataFrame(rows).to_csv(EXT/'metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(EXT/'paired.csv',index=False)
    pd.DataFrame(cls).to_csv(EXT/'classes.csv',index=False)
    pd.DataFrame(coverage).to_csv(EXT/'coverage_metrics.csv',index=False)
    summary=pd.DataFrame(rows).groupby('arm')[['OA','BA','NLL','Brier']].agg(['mean','std'])
    summary.to_csv(EXT/'summary.csv')
    print(summary.to_string(),flush=True)
    (EXT/'local_complete.json').write_text(json.dumps(dict(n=len(y),seconds=time.perf_counter()-tick,new_fits=0,source_bank_unchanged=True,checkpoint_selection='frozen source selection',scope='historical target coverage; not independent new collection'),indent=2),encoding='utf-8')


if __name__=='__main__':main()
