"""Matched local-evidence readouts, including body-anchored context transport.

Transport is an established primitive, not claimed as the contribution. The
experimental question is whether keeping body/context correspondence helps.
"""
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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'
ARMS=('pooled','fixed','free','independent','joint','anchored','mispaired')
REGIONS=(np.arange(8),np.arange(8,16),np.arange(16,20))
REV=np.r_[np.arange(8)[::-1],np.arange(8,16)[::-1],np.arange(16,20)[::-1]]
END=np.array([16,17,16,17,18,19,18,19])
IS_END=np.array([1,1,0,0,0,0,1,1],bool)


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)


def sinkhorn(sim, qa, sa, steps=60, eps=.1):
    """Dustbin-augmented balanced OT; each real side has total unit mass.

    The null has unit mass, allowing every real token to remain unmatched.
    Real match mass is not renormalized after discarding the null.
    """
    qa=qa/qa.sum(-1,keepdim=True).clamp_min(1e-20)
    sa=sa/sa.sum(-1,keepdim=True).clamp_min(1e-20)
    a=torch.cat([qa,2-qa.sum(-1,keepdim=True)],-1)
    b=torch.cat([sa,2-sa.sum(-1,keepdim=True)],-1)
    loga=a.clamp_min(1e-20).log()
    logb=b.clamp_min(1e-20).log()
    logk=torch.nn.functional.pad(sim/eps,(0,1,0,1))
    u=torch.zeros_like(a); v=torch.zeros_like(b)
    for _ in range(steps):
        u=loga-torch.logsumexp(logk+v[...,None,:],-1)
        v=logb-torch.logsumexp(logk+u[...,:,None],-2)
    t=(logk+u[..., :,None]+v[...,None,:]).exp()
    return t[...,:-1,:-1],t,a,b


def pair_score(t,sim,qa,sa):
    keep=(qa>0)[...,:,None]&(sa>0)[...,None,:]
    t=t*keep
    return torch.stack([(t*sim).sum((-1,-2)),t.sum((-1,-2))],-1)


def make_bank(x,mass,table,fit):
    # One image per MMSI: repeated observations cannot dominate reference choice.
    eligible=fit[(mass[fit,:,:8].sum((1,2))>0)]
    unique=table.iloc[eligible].drop_duplicates('mmsi').index.to_numpy()
    body=(x[:,:,:8]*mass[:,:,:8,None]).sum(2)/np.maximum(mass[:,:,:8].sum(2)[...,None],1e-8)
    desc=unit(body).reshape(len(x),-1)
    bank=[]
    for c in range(8):
        idx=unique[table.iloc[unique].class_id.to_numpy()==c]
        assert len(idx)>=8,(c,len(idx))
        km=KMeans(n_clusters=8,random_state=17,n_init=10).fit(desc[idx])
        pick=[]
        for ctr in km.cluster_centers_:
            order=np.argsort(((desc[idx]-ctr)**2).sum(1))
            pick.append(next(int(idx[j]) for j in order if int(idx[j]) not in pick))
        bank.append(pick)
    return np.array(bank),desc


def operations(q,s,qm,sm):
    # q:B,P,T,D; s:B,C,K,P,T,D; masks use identical corresponding axes.
    q=q[:,None,None].expand(*s.shape[:-2],q.shape[-2],q.shape[-1])
    qm=qm[:,None,None].expand(*sm.shape[:-1],qm.shape[-1])
    sims=[torch.einsum('...id,...jd->...ij',q[...,r,:],s[...,r,:]) for r in REGIONS]
    qmass=[qm[...,r] for r in REGIONS]; smass=[sm[...,r] for r in REGIONS]
    out={arm:[] for arm in ARMS}
    transports=[]
    for r,(sim,a,b) in enumerate(zip(sims,qmass,smass)):
        aa=a/a.sum(-1,keepdim=True).clamp_min(1e-20)
        bb=b/b.sum(-1,keepdim=True).clamp_min(1e-20)
        has=(a.sum(-1)>0)&(b.sum(-1)>0)
        outer=aa[..., :,None]*bb[...,None,:]
        out['pooled'].append(pair_score(outer,sim,a,b))
        # Whole-axis reversal is a coordinate ambiguity, not a different class.
        diag=sim.diagonal(dim1=-2,dim2=-1)
        reverse=sim.flip(-1).diagonal(dim1=-2,dim2=-1)
        w=(aa*bb).sqrt(); wr=(aa*bb.flip(-1)).sqrt()
        d=(diag*w).sum(-1)/w.sum(-1).clamp_min(1e-20)
        dr=(reverse*wr).sum(-1)/wr.sum(-1).clamp_min(1e-20)
        # Use one coherent flip chosen from the body for all region scores.
        if r==0:
            choose_reverse=dr>d
        fixed=torch.where(choose_reverse,dr,d)
        out['fixed'].append(torch.stack([fixed*has,has.float()],-1))
        masked=sim.masked_fill(b[...,None,:]<=0,-2.)
        f1=(masked.max(-1).values*aa).sum(-1)
        masked=sim.masked_fill(a[..., :,None]<=0,-2.)
        f2=(masked.max(-2).values*bb).sum(-1)
        out['free'].append(torch.stack([.5*(f1+f2)*has,has.float()],-1))
        t,_,_,_=sinkhorn(sim,a,b)
        transports.append(t)
        out['independent'].append(pair_score(t,sim,a,b))
    body_t=transports[0]
    end_q=q[...,END,:]; end_s=s[...,END,:]
    ea=qm[...,END]*torch.as_tensor(IS_END,device=q.device)
    eb=sm[...,END]*torch.as_tensor(IS_END,device=q.device)
    end_sim=torch.einsum('...id,...jd->...ij',end_q,end_s)
    # Side tokens attach to the matching longitudinal body slot and lateral half.
    side_sim=sims[1]
    joint_sim=(sims[0]+side_sim)*.5
    joint_t,_,_,_=sinkhorn(joint_sim,qmass[0],smass[0])
    # Per-observation permutations supplied externally retain each region's bag
    # of responses but remove its body association in the mispaired control.
    for arm,t in [('anchored',body_t),('joint',joint_t),('mispaired',body_t)]:
        out[arm]=[pair_score(t,sims[0],qmass[0],smass[0]),
                  pair_score(t,side_sim,qmass[1],smass[1]),
                  pair_score(t,end_sim,ea,eb)]
    return {k:torch.stack(v,-2) for k,v in out.items()},body_t


def main():
    torch.set_num_threads(2)
    tick=time.perf_counter()
    table=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    fit=np.load(UP/'indices.npz')['fit_train']
    src=np.load(HERE/'tokens.npz')
    mass=src['mass'].copy()
    rng=np.random.default_rng(17)
    perm=np.stack([rng.permutation(8) for _ in range(len(table)*2)]).reshape(len(table),2,8)
    checks=[]
    # Detect transposed transport axes or rejection normalization errors.
    sim=torch.tensor([[[1.,-1.],[-1.,1.]],[[-1.,-1.],[-1.,-1.]]])
    a=torch.ones(2,2)
    t,whole,aa,bb=sinkhorn(sim,a,a,steps=300)
    err=max(float(abs(whole.sum(-1)-aa).max()),float(abs(whole.sum(-2)-bb).max()))
    assert err<.002 and t[0].trace()>.9 and t[1].sum()<.02,(err,t)
    checks.append(dict(test='transport_known_match_and_reject',marginal_error=err,matched=t.sum((-1,-2)).tolist()))
    for layer in ('stem','layer1'):
        f=src[layer].copy()
        training=f[fit].reshape(-1,f.shape[-1])
        mask=mass[fit].reshape(-1)>0
        training=training[mask]
        sample=np.random.default_rng(17).choice(len(training),min(20000,len(training)),replace=False)
        pca=PCA(n_components=32,svd_solver='full').fit(training[sample])
        x=unit(pca.transform(f.reshape(-1,f.shape[-1])).reshape(*f.shape[:-1],32)).astype(np.float32)
        x[mass<=0]=0
        bank,globalbody=make_bank(x,mass,table,fit)
        np.savez_compressed(HERE/f'projection_{layer}.npz',mean=pca.mean_,components=pca.components_,bank=bank)
        rows=[]
        for c in range(8):
            for j,i in enumerate(bank[c]):
                rows.append(dict(class_id=c,slot=j,local_index=int(i),**table.iloc[i][['obs_id','mmsi','pass_group','port']].to_dict()))
        pd.DataFrame(rows).to_csv(HERE/f'bank_{layer}.csv',index=False)
        # All methods see exactly the same two per-class references, selected
        # solely by body pooled similarity, never by evaluation labels.
        cosine=np.einsum('nd,ckd->nck',globalbody,globalbody[bank])
        selected=np.argsort(-cosine,axis=-1)[...,:2]
        refs=np.take_along_axis(np.broadcast_to(bank,(len(x),8,8)),selected,axis=-1)
        scores={k:np.zeros((len(x),8,2,3,2),np.float32) for k in ARMS}
        xmis=x.copy(); mmis=mass.copy()
        for i in range(len(x)):
            for p in range(2):
                xmis[i,p,8:16]=x[i,p,8+perm[i,p]]
                mmis[i,p,8:16]=mass[i,p,8+perm[i,p]]
        with torch.inference_mode():
            for start in range(0,len(x),32):
                take=slice(start,start+32); r=refs[take]
                args=[torch.tensor(v,device='cuda') for v in (x[take],x[r],mass[take],mass[r])]
                values,body_t=operations(*args)
                mq,ms,ma,mb=[torch.tensor(v,device='cuda') for v in (xmis[take],xmis[r],mmis[take],mmis[r])]
                # The deliberate mismatch affects only side association; body/end remain identical.
                mis=values['anchored'].clone()
                side=torch.einsum('bpid,bckpjd->bckpij',mq[:,:,8:16],ms[...,8:16,:])
                mis[...,1,:]=pair_score(body_t,side,ma[:,None,None,:,8:16],mb[...,8:16])
                values['mispaired']=mis
                for k,v in values.items():
                    scores[k][take]=v.mean(2).cpu().numpy()
                if start%512==0:
                    print(layer,start,'/',len(x),'seconds',round(time.perf_counter()-tick,1),flush=True)
        np.savez_compressed(HERE/f'scores_{layer}.npz',**scores,refs=refs)
        for k,v in scores.items():
            assert np.isfinite(v).all(),k
        print(layer,'complete',flush=True)
    (HERE/'matching_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    (HERE/'matching_receipt.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick,arms=ARMS,pca_fit='fit_train only',bank_per_class=8,selected_per_class=2,trainable_parameters=0),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
