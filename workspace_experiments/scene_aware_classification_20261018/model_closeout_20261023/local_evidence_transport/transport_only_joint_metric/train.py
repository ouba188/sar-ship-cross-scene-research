"""Train assignment-only versus score-only joint-conditioned local matching."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import importlib.util
import json
import time
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('previous_pair',ROOT/'pair_conditioned_joint_metric/train.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
old=previous.old
Metric=previous.Metric
SEEDS=previous.SEEDS
ARMS=('transport_only','metric_only')

class Reader(previous.Reader):
    def __init__(self,arm,seed):
        super().__init__(arm,seed);self.arm=arm
        self.fixed=[]
        # Only unchanged content is cached; no labels or learned weights enter T0.
        with torch.no_grad():
            for r in old.REGIONS:
                parts=[]
                for start in range(0,len(self.ids),64):
                    idx=torch.arange(start,min(start+64,len(self.ids)),device='cuda');ref=self.refs[idx]
                    q=self.x[idx][:,:,r][:,None,None,:,:,None,:];s=self.x[ref][...,r,:][...,None,:,:]
                    k=(q*s).sum(-1);a=self.mass[idx][:,None,None,:,r];b=self.mass[ref][...,r]
                    parts.append(old.sinkhorn(k,a,b)[0])
                self.fixed.append(torch.cat(parts))
    def forward(self,idx,model,diagnostic=False):
        idx=torch.as_tensor(idx,device='cuda');ref=self.refs[idx];h=model.project(self.d)
        q,s=self.x[idx],self.x[ref];hq,hs=h[idx],h[ref]
        a,b=self.mass[idx],self.mass[ref];groups=[];details=[]
        for ri,r in enumerate(old.REGIONS):
            qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
            hi=hq[:,:,r][:,None,None,:,:,None,:];hj=hs[...,r,:][...,None,:,:]
            k0=(qi*sj).sum(-1);k1=previous.similarity(qi,sj,model.weights(hi,hj))
            qa=a[:,None,None,:,r];sa=b[...,r];t0=self.fixed[ri][idx]
            if self.arm=='transport_only':
                t=old.sinkhorn(k1,qa,sa)[0];k=k0
            else:t=t0;k=k1
            groups.append(old.pair_score(t,k,qa,sa))
            if diagnostic:
                keep=(qa>0)[...,None]&(sa>0)[...,None,:]
                details.append(torch.stack([(t-t0).abs().sum((-1,-2)),((k-k0).abs()*keep).sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),(t*(k-k0)).sum((-1,-2))],-1))
        local=torch.stack(groups,-2).mean(2)
        score=self.anchor[idx]+self.delta(local,idx)-self.old_delta[idx]
        if diagnostic:return score,local,torch.stack(details,-2).mean(2)
        return score

def checks(reader,arm):
    torch.manual_seed(17);model=Metric(arm).cuda()
    tr=np.flatnonzero(reader.table.role.eq('utility_train'))
    score,local,_=reader.predict(tr,model,True)
    replay=float(abs(score-reader.strong[tr]).max());le=float(abs(local-reader.oldlocal[tr].cpu().numpy()).max())
    assert replay<3e-5 and le<2e-5,(replay,le)
    idx=tr[:3];y=torch.tensor(reader.table.iloc[idx].class_id.to_numpy(),device='cuda')
    loss=torch.nn.functional.cross_entropy(reader.forward(idx,model),y);loss.backward()
    p=model.output.weight;flat=int(p.grad.abs().argmax());auto=float(p.grad.flatten()[flat]);assert abs(auto)>1e-7
    vals=[];eps=.01
    with torch.no_grad():
        for sign in (1,-1):
            p.flatten()[flat]=sign*eps;vals.append(float(torch.nn.functional.cross_entropy(reader.forward(idx,model),y)))
        p.flatten()[flat]=0
    numeric=(vals[0]-vals[1])/(2*eps);assert abs(numeric-auto)<max(1e-4,abs(auto)*.08),(auto,numeric)
    return dict(source_count=len(tr),initial_score_replay=replay,initial_local_replay=le,gradient_auto=auto,gradient_numeric=numeric)

def main():
    tick=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.reset_peak_memory_stats()
    for name in ('models','results'):(HERE/name).mkdir(exist_ok=True)
    verification={}
    for arm in ARMS:
        reader=Reader(arm,17);verification[arm]=checks(reader,arm);del reader;torch.cuda.empty_cache()
    (HERE/'checks.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print('CHECKS',verification,flush=True)
    # Reuse the exact previous fit routine; its output directory is the only change.
    previous.HERE=HERE
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);receipts.append(previous.fit(reader,arm,seed));del reader;torch.cuda.empty_cache()
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    rows=[];times=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);model=Metric(arm).cuda();ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
            torch.cuda.synchronize();start=time.perf_counter();score,local,diagnostic=reader.predict(np.arange(len(reader.ids)),model,True);torch.cuda.synchronize()
            times.append(dict(seed=seed,arm=arm,n=len(reader.ids),seconds=time.perf_counter()-start))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,local=local,diagnostic=diagnostic,ids=reader.ids)
            data=[(arm,score)]
            if arm=='transport_only':
                data.extend([('strong',reader.strong),('full',reader.base),('both',np.load(ROOT/'pair_conditioned_joint_metric/results'/f'pair_joint_{seed}.npz')['score'])])
            for name,scores in data:
                for role in ('utility_train','selection','evaluation'):
                    idx=np.flatnonzero(reader.table.role.eq(role));y=reader.table.iloc[idx].class_id.to_numpy(int)
                    rows.append(dict(seed=seed,arm=name,role=role,**old.metrics(y,scores[idx])))
            del reader,model;torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(times).to_csv(HERE/'inference_receipts.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(fits=6,seconds=time.perf_counter()-tick,peak_cuda_bytes=torch.cuda.max_memory_allocated()),indent=2),encoding='utf-8')
    m=pd.DataFrame(rows);print(m[m.role.eq('evaluation')].groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)

if __name__=='__main__':main()
