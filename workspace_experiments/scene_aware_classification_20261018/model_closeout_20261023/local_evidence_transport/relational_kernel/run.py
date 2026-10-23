"""Matched pre-aggregation regional products and role-sensitive relations."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
UP=ROOT.parent/'regional_utility_pilot/upstream'
EXT=ROOT/'external_confirmation'
sys.path.insert(0,str(ROOT))
from match_readout import unit,sinkhorn,pair_score,END,IS_END
from train_readout import Head,inputs,joint_loss,exact_j,metrics
ARMS=('anchored','ordered','role_contrast','mispaired')
SEEDS=(17,29,43)


def project(data,pr):
    x=unit((data['layer1']-pr['mean'])@pr['components'].T).astype(np.float32)
    x[data['mass']<=0]=0
    return x


def shuffled(x,m,permutation):
    xx=x.copy();mm=m.copy()
    for i in range(len(x)):
        for p in range(2):
            xx[i,p,8:16]=x[i,p,8+permutation[i,p]]
            mm[i,p,8:16]=m[i,p,8+permutation[i,p]]
    return xx,mm


def dot(a,b):
    return torch.einsum('...id,...jd->...ij',a,b)


def compute(q,s,qm,sm,misq,miss,misqm,missm):
    q=q[:,None,None];qm=qm[:,None,None]
    misq=misq[:,None,None];misqm=misqm[:,None,None]
    b,bb=q[...,:8,:],s[...,:8,:]
    bm,bbm=qm[...,:8],sm[...,:8]
    body=dot(b,bb)
    t,_,_,_=sinkhorn(body,bm,bbm)
    body_score=pair_score(t,body,bm,bbm)
    outputs={arm:[body_score] for arm in ARMS}
    body_ok=(bm.sum(-1)>0)&(bbm.sum(-1)>0)
    availability=[body_ok]
    flag=torch.as_tensor(IS_END,device=q.device)
    for region in ('side','end'):
        ix=np.arange(8,16) if region=='side' else END
        n,nn=q[...,ix,:],s[...,ix,:]
        a,aa=qm[...,ix],sm[...,ix]
        if region=='end':a=a*flag;aa=aa*flag
        similarity=dot(n,nn)
        ordered=body*similarity
        contrast=ordered-dot(b,nn)*dot(n,bb)
        outputs['anchored'].append(pair_score(t,similarity,a,aa))
        outputs['ordered'].append(pair_score(t,ordered,a,aa))
        outputs['role_contrast'].append(pair_score(t,contrast,a,aa))
        if region=='side':
            mn,mnn=misq[...,ix,:],miss[...,ix,:]
            ma,maa=misqm[...,ix],missm[...,ix]
            rel=body*dot(mn,mnn)-dot(b,mnn)*dot(mn,bb)
            outputs['mispaired'].append(pair_score(t,rel,ma,maa))
        else:outputs['mispaired'].append(outputs['role_contrast'][-1])
        availability.append(body_ok&(a.sum(-1)>0)&(aa.sum(-1)>0))
    ok=torch.stack(availability,-1)
    count=ok.sum(2)
    # Structural absence only: keep each available reference's rejection mass.
    result={arm:(torch.stack(v,-2)*ok[...,None]).sum(2)/count.clamp_min(1)[...,None]
            for arm,v in outputs.items()}
    return result,count


def extract(external=False):
    source=np.load(ROOT/'tokens.npz');pr=np.load(ROOT/'projection_layer1.npz')
    other=np.load(EXT/'tokens.npz') if external else source
    xs=project(source,pr);x=project(other,pr)
    ms=source['mass'];m=other['mass']
    path=EXT/'local_scores.npz' if external else ROOT/'scores_layer1.npz'
    refs=np.load(path)['refs']
    rng=np.random.default_rng(17)
    ps=np.stack([rng.permutation(8) for _ in range(len(xs)*2)]).reshape(len(xs),2,8)
    pq=np.stack([rng.permutation(8) for _ in range(len(x)*2)]).reshape(len(x),2,8) if external else ps
    sx,sm=shuffled(xs,ms,ps);qx,qm=shuffled(x,m,pq)
    scores={arm:np.zeros((len(x),8,2,3,2),np.float32) for arm in ARMS}
    count=np.zeros((len(x),8,2,3),np.int16)
    with torch.inference_mode():
        for first in range(0,len(x),32):
            idx=np.arange(first,min(first+32,len(x)));r=refs[idx]
            args=[torch.tensor(v,device='cuda') for v in
                  (x[idx],xs[r],m[idx],ms[r],qx[idx],sx[r],qm[idx],sm[r])]
            out,n=compute(*args)
            for arm,v in out.items():scores[arm][idx]=v.cpu().numpy()
            count[idx]=n.cpu().numpy()
    name='external' if external else 'source'
    np.savez_compressed(HERE/f'{name}_local.npz',**scores,count=count,refs=refs)
    if not external:
        expected=np.load(ROOT/'availability_pool/scores_valid_layer1.npz')['anchored']
        err=float(np.max(abs(expected-scores['anchored'])))
        assert err<5e-5,err
        table=pd.read_csv(UP/'table.csv');tr=np.flatnonzero(table.role.eq('utility_train'))
        rows=[]
        for p in range(2):
            for region,ix,flag in [('side',np.arange(8,16),np.ones(8,bool)),('end',END,IS_END)]:
                cosine=(x[tr,p,:8]*x[tr,p][:,ix]).sum(-1)
                valid=(m[tr,p,:8]>0)&(m[tr,p][:,ix]>0)&flag
                v=cosine[valid]
                rows.append(dict(pol=p,region=region,n=len(v),mean_cosine=float(v.mean()),
                                 fraction_above_08=float((v>.8).mean()),
                                 mean_wedge_self_energy=float((1-v*v).mean())))
        pd.DataFrame(rows).to_csv(HERE/'source_region_dependence.csv',index=False)
        y=table.iloc[tr].class_id.to_numpy();rows=[]
        for arm in ARMS:
            for ri,region in enumerate(('body','side','end')):
                v=scores[arm][tr,:,:,ri,0].mean(-1)
                true=v[np.arange(len(y)),y]
                wrong=np.where(np.arange(8)==y[:,None],-np.inf,v).max(1)
                rows.append(dict(arm=arm,region=region,**metrics(y,v),margin=float((true-wrong).mean())))
        pd.DataFrame(rows).to_csv(HERE/'source_diagnostic.csv',index=False)
        print('source anchor replay max',err,flush=True)
    return scores


def raw_inputs(local,base):
    z=base-base.mean(-1,keepdims=True)
    other=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
    return np.concatenate([local.reshape(len(local),8,12),z[...,None],other[...,None],(z-other)[...,None]],-1)


def train():
    scores=np.load(HERE/'source_local.npz')
    table=pd.read_csv(UP/'table.csv');y=table.class_id.to_numpy()
    tr=np.flatnonzero(table.role.eq('utility_train'));va=np.flatnonzero(table.role.eq('selection'))
    te=np.flatnonzero(table.role.eq('evaluation'))
    (HERE/'models').mkdir(exist_ok=True);(HERE/'results').mkdir(exist_ok=True)
    rows=[];curves=[]
    for seed in SEEDS:
        base=np.load(UP/f'strategies_seed{seed}.npz')['logits'][:,2].astype(np.float32)
        for arm in ARMS:
            x,mu,sd,active=inputs(scores[arm],base,tr)
            torch.manual_seed(seed);head=Head();opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
            best=exact_j(y[va],base[va].argmax(1));step_selected=0
            state={k:v.detach().clone() for k,v in head.state_dict().items()}
            tx=torch.tensor(x[tr]);tb=torch.tensor(base[tr]);ty=torch.tensor(y[tr]);vx=torch.tensor(x[va])
            for step in range(1,201):
                opt.zero_grad(set_to_none=True);loss=joint_loss(tb+head(tx),ty)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
                if step%5==0:
                    with torch.no_grad():v=base[va]+head(vx).numpy()
                    value=exact_j(y[va],v.argmax(1))
                    curves.append(dict(seed=seed,arm=arm,step=step,loss=float(loss.detach()),**metrics(y[va],v)))
                    if value>best:
                        best=value;step_selected=step;state={k:v.detach().clone() for k,v in head.state_dict().items()}
            head.load_state_dict(state)
            torch.save(dict(state=state,mu=mu,scale=sd,active=active,selected_step=step_selected),HERE/'models'/f'{arm}_{seed}.pt')
            with torch.no_grad():score=base+head(torch.tensor(x)).numpy()
            np.savez_compressed(HERE/'results'/f'source_{arm}_{seed}.npz',score=score)
            for role,ids in [('utility_train',tr),('selection',va),('evaluation',te)]:
                rows.append(dict(seed=seed,arm=arm,step=step_selected,role=role,**metrics(y[ids],score[ids])))
            print(seed,arm,step_selected,rows[-1],flush=True)
    pd.DataFrame(rows).to_csv(HERE/'source_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)


def evaluate():
    scores=np.load(HERE/'external_local.npz');table=pd.read_csv(EXT/'table.csv');y=table.class_id.to_numpy()
    rows=[];pairs=[]
    for seed in SEEDS:
        base=np.load(EXT/f'strategies_seed{seed}.npz')['logits'][:,2].astype(np.float32)
        allscores={'full':base}
        for arm in ARMS:
            ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
            raw=raw_inputs(scores[arm],base)
            x=np.where(ck['active'],(raw-ck['mu'])/ck['scale'],0).astype(np.float32)
            head=Head();head.load_state_dict(ck['state'])
            with torch.no_grad():score=base+head(torch.tensor(x)).numpy()
            np.savez_compressed(HERE/'results'/f'external_{arm}_{seed}.npz',score=score)
            allscores[arm]=score
        for arm,score in allscores.items():
            for scope,mask in [('all',np.ones(len(y),bool)),('fold0',table.original_test_fold.eq(0).to_numpy()),('extra',table.original_test_fold.ne(0).to_numpy())]:
                rows.append(dict(seed=seed,arm=arm,scope=scope,**metrics(y[mask],score[mask])))
            a=allscores['role_contrast'].argmax(1)==y;b=score.argmax(1)==y;delta=a.astype(float)-b
            pairs.append(dict(seed=seed,comparison=arm,rescue=int((a&~b).sum()),harm=int((~a&b).sum()),
                              delta_OA=100*delta.mean(),delta_BA=100*np.mean([delta[y==c].mean() for c in np.unique(y)])))
    pd.DataFrame(rows).to_csv(HERE/'external_metrics.csv',index=False)
    pd.DataFrame(pairs).to_csv(HERE/'external_paired.csv',index=False)
    print(pd.DataFrame(rows).query("scope=='all'").groupby('arm')[['OA','BA']].mean().to_string(),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['source','train','external']);args=ap.parse_args()
    tick=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    # Distinguishes role sensitivity from another ordinary product kernel.
    e=torch.eye(2);b=e[0];n=e[1]
    kernel=lambda a,c,d,f:torch.dot(a,d)*torch.dot(c,f)-torch.dot(a,f)*torch.dot(c,d)
    assert kernel(b,n,b,n)==1 and kernel(b,n,n,b)==-1 and kernel(b,b,b,n)==0
    if args.stage=='source':extract()
    elif args.stage=='train':train()
    else:extract(True);evaluate()
    (HERE/f'{args.stage}_complete.json').write_text(json.dumps(dict(stage=args.stage,seconds=time.perf_counter()-tick,scope='historical development; no target fitting'),indent=2),encoding='utf-8')
