"""Candidate-pair conditional local pooling, on frozen source-reference evidence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('competition_epi',ROOT/'conditional_relation_operator/reference_resampling/episodic/run.py')
epi=importlib.util.module_from_spec(spec); spec.loader.exec_module(epi)
exp=epi.exp
GROUPS=('VV_body','VV_side','VV_end','VH_body','VH_side','VH_end')
ARMS=('uniform','own','pair')
CHECKS=(0,20,50,100,200)


def organize(x,mass):
    q=np.zeros((len(x),6,8,32),np.float32)
    mask=np.zeros((len(x),6,8),bool)
    for g in range(6):
        p,r=divmod(g,3); lo,hi=((0,8),(8,16),(16,20))[r]
        q[:,g,:hi-lo]=x[:,p,lo:hi]
        mask[:,g,:hi-lo]=mass[:,p,lo:hi]>0
    return q,mask


@torch.no_grad()
def evidence(q,mask,bank,ids):
    out=[]
    s=q[bank]; sm=mask[bank]
    counts=sm.sum(-1)
    rv=counts>0
    lm=-counts.clamp_min(1).log()-rv.sum(1).clamp_min(1).log()[:,None,:]
    lm=(lm[...,None].expand_as(sm)).masked_fill(~sm,-1e4)
    cv=rv.any(1).T
    for first in range(0,len(ids),128):
        v=torch.einsum('bgtd,crgud->bgtcru',q[ids[first:first+128]],s)*np.sqrt(32)
        v=torch.logsumexp((v+lm.permute(2,0,1,3)[None,:,None]).flatten(-2),-1)
        v=torch.where(cv[None,:,None],v,0.)
        center=(v*cv[None,:,None]).sum(-1,keepdim=True)/cv.sum(-1)[None,:,None,None].clamp_min(1)
        out.append(torch.where(cv[None,:,None],v-center,0.).cpu())
    return torch.cat(out),cv.cpu()


class Pool(nn.Module):
    def __init__(self,arm,seed=17):
        super().__init__(); torch.manual_seed(seed)
        self.arm=arm
        self.a=nn.Parameter(torch.zeros(6))
        if arm!='uniform':
            self.net=nn.Sequential(nn.Linear(36,16),nn.Tanh(),nn.Linear(16,1))
        self.register_buffer('pairs',torch.combinations(torch.arange(8),r=2))

    def forward(self,e,q,mask,cv,trace=False):
        # B,G,T,C. Mask reference-empty classes as well as query padding.
        if self.arm=='pair':
            i,j=self.pairs.T
            a,b=e[...,i],e[...,j]
            features=torch.stack(((a+b)*.5,(a-b).abs(),a*b,(a-b).square()),-1)
            content=q[:,:,:,None].expand(*features.shape[:-1],32)
            gate=self.net(torch.cat((features,content),-1)).squeeze(-1)
            w=gate.masked_fill(~mask[...,None],-1e4).softmax(2)*mask[...,None]
            w=w/w.sum(2,keepdim=True).clamp_min(1e-8)
            edges=(w*(a-b)).sum(2)*(cv[:,i]&cv[:,j])[None]
            pooled=e.new_zeros((len(e),6,8))
            pooled.index_add_(2,i,edges); pooled.index_add_(2,j,-edges)
            pooled=pooled/cv.sum(-1)[None,:,None].clamp_min(1)
        else:
            if self.arm=='own':
                features=torch.stack((e,e.square(),e.pow(3),e.pow(4)),-1)
                content=q[:,:,:,None].expand(*features.shape[:-1],32)
                gate=self.net(torch.cat((features,content),-1)).squeeze(-1)
            else:
                gate=torch.zeros_like(e)
            w=gate.masked_fill(~mask[...,None],-1e4).softmax(2)*mask[...,None]
            w=w/w.sum(2,keepdim=True).clamp_min(1e-8)
            pooled=(w*e).sum(2)*cv[None]
            pooled=(pooled-pooled.sum(-1,keepdim=True)/cv.sum(-1)[None,:,None].clamp_min(1))*cv[None]
            edges=None
        parts=(pooled*self.a[None,:,None]).transpose(1,2)
        score=parts.sum(-1)
        return (score,parts,w,edges) if trace else score


def checks():
    torch.manual_seed(91)
    e=torch.randn(3,6,8,8,dtype=torch.float64)
    q=torch.randn(3,6,8,32,dtype=torch.float64)
    mask=torch.rand(3,6,8)>.2; mask[0,1]=False
    cv=torch.ones(6,8,dtype=torch.bool); cv[2,3]=False
    uniform=Pool('uniform').double(); uniform.a.data.fill_(1)
    ref=uniform(e,q,mask,cv)
    rows={}
    for arm in ('own','pair'):
        m=Pool(arm).double(); m.a.data.fill_(1)
        with torch.no_grad():
            for p in m.net.parameters(): p.zero_()
        err=float((m(e,q,mask,cv)-ref).abs().max())
        assert err<1e-10,err
        m=Pool(arm).double(); m.a.data.fill_(.3)
        pred,parts,w,edges=m(e,q,mask,cv,True)
        perm=torch.tensor([3,0,7,1,6,2,4,5]); tp=torch.randperm(8)
        ce=float((m(e[...,perm],q,mask,cv[:,perm])-pred[:,perm]).abs().max())
        te=float((m(e[:,:,tp],q[:,:,tp],mask[:,:,tp],cv)-pred).abs().max())
        assert max(ce,te)<1e-10 and parts[0,:,1].abs().max()==0
        loss=pred.square().sum(); loss.backward()
        p=m.net[0].weight; pos=np.unravel_index(int(p.grad.abs().argmax()),p.shape)
        actual=float(p.grad[pos]); before=float(p[pos]); eps=1e-5
        with torch.no_grad():
            p[pos]=before+eps; plus=float(m(e,q,mask,cv).square().sum())
            p[pos]=before-eps; minus=float(m(e,q,mask,cv).square().sum()); p[pos]=before
        ge=abs(actual-(plus-minus)/(2*eps)); assert ge<1e-7 and abs(actual)>1e-8
        # Pair edges need not be differences of independently pooled class potentials.
        cycle=None
        if edges is not None:
            pairs=[tuple(v) for v in m.pairs.tolist()]
            cycle=float((edges[...,pairs.index((0,1))]+edges[...,pairs.index((1,2))]-edges[...,pairs.index((0,2))]).abs().max())
            assert cycle>1e-5
        rows[arm]=dict(uniform_replay=err,class_permutation=ce,token_permutation=te,gradient_error=ge,nonadditive_cycle=cycle)
    (HERE/'checks.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('CHECKS',rows,flush=True)


def main():
    tick=time.perf_counter(); checks()
    table,x,mass,old,idx,tr,va,te=exp.source_data()
    table.to_csv(HERE/'table.csv',index=False)
    q,mask=organize(x,mass)
    q=torch.tensor(q,device='cuda'); mask=torch.tensor(mask,device='cuda')
    _,_,legal,_=exp.edge.edges(x,mass)
    banks,_=epi.diag.banks_from_source(table,mass,legal,old)
    schedule=epi.schedule(table,mass,old,legal)
    fixed={k:evidence(q,mask,b,idx) for k,b in banks.items()}
    e0,cv0=fixed['original']
    valid=mask[idx[tr]].cpu()[...,None]&cv0[None,:,None]
    scale=((e0[tr].square()*valid).sum((0,2,3))/valid.sum((0,2,3)).clamp_min(1)).sqrt().clamp_min(.05)
    np.savez_compressed(HERE/'inputs.npz',idx=idx,tr=tr,va=va,te=te,bank=old,scale=scale.numpy(),q=q[idx].cpu().numpy(),mask=mask[idx].cpu().numpy(),e=e0.numpy(),cv=cv0.numpy())
    fixed={k:(e/scale[None,:,None,None],v.to('cuda')) for k,(e,v) in fixed.items()}
    training=[]
    for step,b in enumerate(schedule):
        e,v=evidence(q,mask,b,idx[tr]); training.append((e/scale[None,:,None,None],v.to('cuda')))
        if step%50==0: print('CACHE',step,round(time.perf_counter()-tick,1),flush=True)
    qt=q[idx]; mt=mask[idx]; y=table.iloc[idx].class_id.to_numpy(int)
    counts=np.bincount(y[tr],minlength=8)
    weight=torch.tensor(.5/len(tr)+.5/(8*counts[y[tr]]),device='cuda',dtype=torch.float32)
    target=torch.tensor(y[tr],device='cuda')
    metrics=[]; paired=[]; receipts=[]; curves=[]; classes=[]
    @torch.no_grad()
    def infer(model,e,cv,loc,base):
        out=[]
        for first in range(0,len(loc),64):
            take=loc[first:first+64]
            out.append(model(e[take].to('cuda'),qt[take],mt[take],cv).cpu().numpy())
        return base[loc]+np.concatenate(out)
    for seed in (17,29,43):
        full=np.load(exp.HERE/f'predictions_{seed}.npz')['full'][idx]
        for arm in ARMS:
            start=time.perf_counter(); model=Pool(arm,seed).cuda()
            opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.0001)
            best=-np.inf; beststep=0; state=None
            for step in range(201):
                if step:
                    e,cv=training[step-1]; opt.zero_grad(set_to_none=True)
                    for first in range(0,len(tr),64):
                        take=tr[first:first+64]
                        score=torch.tensor(full[take],device='cuda')+model(e[first:first+64].to('cuda'),qt[take],mt[take],cv)
                        loss=(F.cross_entropy(score,target[first:first+64],reduction='none')*weight[first:first+64]).sum()
                        loss.backward()
                    opt.step()
                if step not in CHECKS: continue
                scores=[infer(model,e,cv,va,full) for e,cv in fixed.values()]
                j=np.mean([exp.dep.metrics(y[va],s)['J'] for s in scores])
                curves.append(dict(arm=arm,seed=seed,step=step,J=j))
                if j>best:
                    best=float(j); beststep=step; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                print('FIT',arm,seed,step,round(float(j),6),round(time.perf_counter()-start,1),flush=True)
            model.load_state_dict(state); torch.save(state,HERE/f'model_{arm}_{seed}.pt')
            saved=dict(idx=idx,y=y,full=full)
            for name,(e,cv) in fixed.items():
                score=infer(model,e,cv,np.arange(len(idx)),full); saved[name]=score
                for role,loc in [('utility_train',tr),('selection',va),('evaluation',te)]:
                    meta=dict(arm=arm,seed=seed,bank=name,role=role,step=beststep)
                    metrics.append(dict(**meta,**exp.dep.metrics(y[loc],score[loc])))
                    before=full[loc].argmax(-1)==y[loc]; after=score[loc].argmax(-1)==y[loc]
                    paired.append(dict(**meta,rescued=int((~before&after).sum()),harmed=int((before&~after).sum()),net=int(after.sum()-before.sum())))
                    for c in range(8):
                        keep=y[loc]==c
                        classes.append(dict(**meta,class_id=c,n=int(keep.sum()),correct=int(after[keep].sum()),full_correct=int(before[keep].sum())))
            np.savez_compressed(HERE/f'predictions_{arm}_{seed}.npz',**saved)
            receipts.append(dict(arm=arm,seed=seed,step=beststep,selection_J=best,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start))
            for name,rows in [('metrics',metrics),('paired',paired),('receipts',receipts),('curves',curves),('classes',classes)]: pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    pd.DataFrame(metrics).query("role=='evaluation'").groupby(['arm','bank'])[['OA','BA','J','NLL','Brier']].mean().to_csv(HERE/'summary.csv')
    (HERE/'execution.json').write_text(json.dumps(dict(fits=9,seconds=time.perf_counter()-tick,steps=200,selection='six source banks J',raw_review='pending'),indent=2),encoding='utf-8')


if __name__=='__main__':
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32=False
    if '--checks' in sys.argv: checks()
    else: main()
