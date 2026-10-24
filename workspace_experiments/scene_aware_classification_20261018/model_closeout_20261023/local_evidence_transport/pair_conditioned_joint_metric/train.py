"""Pair-conditioned normalized local similarities from joint regional evidence."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('joint_pool_train',ROOT/'spatial_frequency_joint_pooling/train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
old=previous.old
SEEDS=old.SEEDS
ARMS=('pair_joint','separable_joint','pair_marginal')

def similarity(q,s,w):
    numerator=(w*q*s).sum(-1)
    denominator=((w*q.square()).sum(-1)*(w*s.square()).sum(-1)).clamp_min(1e-20).sqrt()
    return numerator/denominator

class Metric(nn.Module):
    def __init__(self,arm):
        super().__init__();self.separable=arm=='separable_joint'
        self.project=nn.Linear(96,4,bias=False);self.output=nn.Linear(8,32,bias=False)
        nn.init.zeros_(self.output.weight)
    def member(self,h):
        return self.output(nn.functional.gelu(torch.cat([h,h.abs()],-1))).tanh()
    def weights(self,q,s):
        if self.separable:change=(self.member(q)+self.member(s))*.5
        else:
            phi=torch.cat([(q+s)*.5,(q-s).abs()*.5],-1)
            change=self.output(nn.functional.gelu(phi)).tanh()
        return 1+.25*change

class Reader(previous.Reader):
    def __init__(self,arm,seed):
        super().__init__('marginal' if arm=='pair_marginal' else 'joint',seed)
    def forward(self,idx,model,diagnostic=False):
        idx=torch.as_tensor(idx,device='cuda');ref=self.refs[idx]
        h=model.project(self.d)
        q,s=self.x[idx],self.x[ref];hq,hs=h[idx],h[ref]
        a,b=self.mass[idx],self.mass[ref];groups=[];details=[]
        for r in old.REGIONS:
            qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
            hi=hq[:,:,r][:,None,None,:,:,None,:];hj=hs[...,r,:][...,None,:,:]
            w=model.weights(hi,hj);sim=similarity(qi,sj,w)
            qa=a[:,None,None,:,r];sa=b[...,r]
            t,_,_,_=old.sinkhorn(sim,qa,sa)
            groups.append(old.pair_score(t,sim,qa,sa))
            if diagnostic:
                raw=(qi*sj).sum(-1);keep=(qa>0)[...,None]&(sa>0)[...,None,:]
                t0,_,_,_=old.sinkhorn(raw,qa,sa)
                details.append(torch.stack([(t-t0).abs().sum((-1,-2)),
                    ((sim-raw).abs()*keep).sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),
                    (t*(sim-raw)).sum((-1,-2))],-1))
        local=torch.stack(groups,-2).mean(2)
        score=self.anchor[idx]+self.delta(local,idx)-self.old_delta[idx]
        if diagnostic:return score,local,torch.stack(details,-2).mean(2)
        return score

def checks(reader,arm):
    torch.manual_seed(17);model=Metric(arm).cuda()
    idx=np.flatnonzero(reader.table.role.eq('utility_train'))[:3]
    score,local,_=reader.predict(idx,model,True)
    replay=float(abs(score-reader.strong[idx]).max());le=float(abs(local-reader.oldlocal[idx].cpu().numpy()).max())
    assert replay<3e-5 and le<2e-5,(replay,le)
    y=torch.tensor(reader.table.iloc[idx].class_id.to_numpy(),device='cuda')
    loss=nn.functional.cross_entropy(reader.forward(idx,model),y);loss.backward()
    p=model.output.weight;flat=int(p.grad.abs().argmax());auto=float(p.grad.flatten()[flat]);assert abs(auto)>1e-7
    eps=.01;values=[]
    with torch.no_grad():
        for sign in (1,-1):
            p.flatten()[flat]=sign*eps;values.append(float(nn.functional.cross_entropy(reader.forward(idx,model),y)))
        p.flatten()[flat]=0
    numeric=(values[0]-values[1])/(2*eps)
    assert abs(numeric-auto)<max(1e-4,abs(auto)*.08),(auto,numeric)
    with torch.no_grad():
        model.output.weight.normal_(0,.2)
        q=torch.randn(2,5,1,32,device='cuda');s=torch.randn(2,1,7,32,device='cuda')
        hq=torch.randn(2,5,1,4,device='cuda');hs=torch.randn(2,1,7,4,device='cuda')
        w=model.weights(hq,hs);k=similarity(q,s,w)
        reverse=similarity(s.transpose(1,2),q.transpose(1,2),model.weights(hs.transpose(1,2),hq.transpose(1,2))).transpose(1,2)
        symmetry=float(abs(k-reverse).max());scale_error=float(abs(k-similarity(q,s,w*2.3)).max())
        assert symmetry<1e-6 and scale_error<1e-6 and float(k.abs().max())<=1+1e-6
    return dict(initial_replay=replay,local_replay=le,gradient_auto=auto,gradient_numeric=numeric,
        symmetry=symmetry,uniform_scale_error=scale_error,parameters=sum(p.numel() for p in model.parameters()))

def fit(reader,arm,seed):
    tick=time.perf_counter();table=reader.table
    tr=np.flatnonzero(table.role.eq('utility_train'));va=np.flatnonzero(table.role.eq('selection'))
    y=table.class_id.to_numpy(int);yt=torch.tensor(y,device='cuda');counts=np.bincount(y[tr],minlength=8)
    wt=torch.tensor(.5/len(tr)+.5/(8*counts[y[tr]]),dtype=torch.float32,device='cuda')
    torch.manual_seed(seed);model=Metric(arm).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=old.exact_j(y[va],reader.strong[va].argmax(1));chosen=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve=[dict(seed=seed,arm=arm,step=0,train_loss=None,**old.metrics(y[va],reader.strong[va]))]
    for step in range(1,101):
        opt.zero_grad(set_to_none=True);score=reader.forward(tr,model)
        loss=(nn.functional.cross_entropy(score,yt[tr],reduction='none')*wt).sum()
        assert torch.isfinite(loss);loss.backward();opt.step()
        if step in old.CHECKPOINTS:
            s=reader.predict(va,model);m=old.metrics(y[va],s);j=old.exact_j(y[va],s.argmax(1))
            curve.append(dict(seed=seed,arm=arm,step=step,train_loss=float(loss.detach()),**m))
            if j>best:best=j;chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT',seed,arm,step,'source_J',round(m['J'],6),'seconds',round(time.perf_counter()-tick,1),flush=True)
    receipt=dict(seed=seed,arm=arm,selected_step=chosen,parameters=640,steps=100,seconds=time.perf_counter()-tick)
    torch.save(dict(state=state,scaler=reader.scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt

def main(check_only=False):
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    for name in ('models','results'):(HERE/name).mkdir(exist_ok=True)
    verification={}
    for arm in ARMS:
        reader=Reader(arm,17);verification[arm]=checks(reader,arm);del reader;torch.cuda.empty_cache()
    (HERE/'checks.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print('CHECKS',verification,flush=True)
    if check_only:return
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);receipts.append(fit(reader,arm,seed));del reader;torch.cuda.empty_cache()
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    results=[];inference=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);model=Metric(arm).cuda()
            ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
            torch.cuda.synchronize();start=time.perf_counter()
            score,local,diagnostic=reader.predict(np.arange(len(reader.ids)),model,True);torch.cuda.synchronize()
            inference.append(dict(seed=seed,arm=arm,n=len(reader.ids),seconds=time.perf_counter()-start))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,local=local,diagnostic=diagnostic,ids=reader.ids)
            for role in ('utility_train','selection','evaluation'):
                take=np.flatnonzero(reader.table.role.eq(role));y=reader.table.iloc[take].class_id.to_numpy(int)
                results.append(dict(seed=seed,arm=arm,role=role,**old.metrics(y,score[take])))
                if arm=='pair_joint':
                    baselines=[('strong',reader.strong),('full',reader.base),('frequency',np.load(ROOT/'frequency_conditioned_matching/results'/f'frequency_{seed}.npz')['score']),('independent_joint',np.load(ROOT/'spatial_frequency_joint_pooling/results'/f'joint_{seed}.npz')['score'])]
                    for name,s in baselines:results.append(dict(seed=seed,arm=name,role=role,**old.metrics(y,s[take])))
            del reader,model;torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(inference).to_csv(HERE/'inference_receipts.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick,fits=9,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checks=verification),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','J']].mean().to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();main(a.check_only)
