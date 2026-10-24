"""Matched joint-versus-marginal pooling with a frozen classification reader."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
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
spec=importlib.util.spec_from_file_location('frequency_train',ROOT/'frequency_conditioned_matching/train.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
SEEDS=old.SEEDS
ARMS=('joint','marginal')

class Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(96,4,bias=False),nn.GELU(),nn.Linear(4,32,bias=False))
        nn.init.zeros_(self.net[-1].weight)
    def forward(self,x,d,mass):
        residual=.25*self.net(d).tanh()/(32**.5)
        return nn.functional.normalize(x+residual,dim=-1)*mass.gt(0)[...,None]

class Reader(old.Reader):
    def __init__(self,arm,seed):
        super().__init__('content',seed)
        f=np.load(HERE/'features.npz');np.testing.assert_array_equal(f['ids'],self.ids)
        d=f[arm].copy();mass=self.mass.cpu().numpy()
        tr=self.table.role.eq('utility_train').to_numpy()
        values=d[tr][mass[tr]>0]
        mu=values.mean(0);sd=values.std(0);active=sd>1e-6;sd=np.where(active,sd,1.)
        d=np.where(active,(d-mu)/sd,0).astype(np.float32);d[mass<=0]=0
        self.d=torch.tensor(d,device='cuda')
        self.scaler=dict(mu=mu.tolist(),scale=sd.tolist(),active=active.tolist())
    def forward(self,idx,model,diagnostic=False):
        idx=torch.as_tensor(idx,device='cuda');ref=self.refs[idx]
        x=model(self.x,self.d,self.mass)
        q,s=x[idx],x[ref];a,b=self.mass[idx],self.mass[ref]
        groups=[]
        for r in old.REGIONS:
            sim=torch.einsum('bpid,bckpjd->bckpij',q[:,:,r],s[...,r,:])
            qa=a[:,None,None,:,r];sa=b[...,r]
            t,_,_,_=old.sinkhorn(sim,qa,sa)
            groups.append(old.pair_score(t,sim,qa,sa))
        local=torch.stack(groups,-2).mean(2)
        score=self.anchor[idx]+self.delta(local,idx)-self.old_delta[idx]
        if diagnostic:
            change=(x[idx]-self.x[idx]).norm(dim=-1)
            return score,local,change
        return score

def checks(reader):
    torch.manual_seed(17);model=Pool().cuda()
    tr=np.flatnonzero(reader.table.role.eq('utility_train'));idx=tr[:3]
    score,local,_=reader.predict(idx,model,True)
    replay=float(abs(score-reader.strong[idx]).max())
    local_error=float(abs(local-reader.oldlocal[idx].cpu().numpy()).max())
    assert replay<3e-5 and local_error<2e-5,(replay,local_error)
    y=torch.tensor(reader.table.iloc[idx].class_id.to_numpy(),device='cuda')
    loss=nn.functional.cross_entropy(reader.forward(idx,model),y);loss.backward()
    p=model.net[-1].weight;flat=int(p.grad.abs().argmax());auto=float(p.grad.flatten()[flat]);assert abs(auto)>1e-7
    vals=[];eps=.01
    with torch.no_grad():
        for sign in (1,-1):
            p.flatten()[flat]=sign*eps
            vals.append(float(nn.functional.cross_entropy(reader.forward(idx,model),y)))
        p.flatten()[flat]=0
    numeric=(vals[0]-vals[1])/(2*eps)
    assert abs(auto-numeric)<max(1e-4,abs(auto)*.08),(auto,numeric)
    return dict(replay=replay,local_replay=local_error,gradient_auto=auto,gradient_numeric=numeric,parameters=sum(p.numel() for p in model.parameters()))

def fit(reader,arm,seed):
    start=time.perf_counter();table=reader.table
    tr=np.flatnonzero(table.role.eq('utility_train'));va=np.flatnonzero(table.role.eq('selection'))
    y=table.class_id.to_numpy(int);yt=torch.tensor(y,device='cuda')
    counts=np.bincount(y[tr],minlength=8)
    wt=torch.tensor(.5/len(tr)+.5/(8*counts[y[tr]]),dtype=torch.float32,device='cuda')
    torch.manual_seed(seed);model=Pool().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=old.exact_j(y[va],reader.strong[va].argmax(1));chosen=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve=[dict(seed=seed,arm=arm,step=0,train_loss=None,**old.metrics(y[va],reader.strong[va]))]
    for step in range(1,101):
        opt.zero_grad(set_to_none=True)
        score=reader.forward(tr,model)
        loss=(nn.functional.cross_entropy(score,yt[tr],reduction='none')*wt).sum()
        assert torch.isfinite(loss);loss.backward();opt.step()
        if step in old.CHECKPOINTS:
            s=reader.predict(va,model);m=old.metrics(y[va],s);j=old.exact_j(y[va],s.argmax(1))
            curve.append(dict(seed=seed,arm=arm,step=step,train_loss=float(loss.detach()),**m))
            if j>best:
                best=j;chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT',seed,arm,step,'selection_J',round(m['J'],5),'seconds',round(time.perf_counter()-start,1),flush=True)
    receipt=dict(seed=seed,arm=arm,selected_step=chosen,parameters=512,steps=100,seconds=time.perf_counter()-start)
    torch.save(dict(state=state,scaler=reader.scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt

def main(check_only=False):
    start=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    for name in ('models','results'): (HERE/name).mkdir(exist_ok=True)
    verification={}
    for arm in ARMS:
        reader=Reader(arm,17);verification[arm]=checks(reader);del reader;torch.cuda.empty_cache()
    (HERE/'checks.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print('CHECKS',verification,flush=True)
    if check_only:return
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);receipts.append(fit(reader,arm,seed));del reader;torch.cuda.empty_cache()
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    results=[];inference=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);model=Pool().cuda()
            ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
            torch.cuda.synchronize();tick=time.perf_counter()
            score,local,change=reader.predict(np.arange(len(reader.ids)),model,True)
            torch.cuda.synchronize()
            inference.append(dict(seed=seed,arm=arm,observations=len(reader.ids),seconds=time.perf_counter()-tick))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,local=local,change=change,ids=reader.ids)
            for role in ('utility_train','selection','evaluation'):
                take=np.flatnonzero(reader.table.role.eq(role));y=reader.table.iloc[take].class_id.to_numpy(int)
                results.append(dict(seed=seed,arm=arm,role=role,**old.metrics(y,score[take])))
                if arm=='joint':
                    for name,s in [('strong',reader.strong),('full',reader.base),('frequency',np.load(ROOT/'frequency_conditioned_matching/results'/f'frequency_{seed}.npz')['score'])]:
                        results.append(dict(seed=seed,arm=name,role=role,**old.metrics(y,s[take])))
            del reader,model;torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(inference).to_csv(HERE/'inference_receipts.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,fits=6,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checks=verification),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','J']].mean().to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();main(a.check_only)
