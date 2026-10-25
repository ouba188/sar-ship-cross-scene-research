"""Source-only learning of typed query/reference evidence, on a frozen system."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
LOCAL = BASE/'local_evidence_transport'
NATIVE = HERE.parent/'native_validation'
SEEDS = (17, 29, 43)
ARMS = ('score_only', 'content_both', 'typed_fixed', 'typed_both')
CHECKPOINTS = (0, 20, 40, 60, 80, 100)
MICRO = 8

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

matching = load('typed_existing_transport', LOCAL/'match_readout.py')
ops = load('typed_evidence_ops', HERE.parent/'operators.py')

def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False), encoding='utf-8')

def metrics(y, score):
    pred = score.argmax(1)
    lp = score.astype(float)-logsumexp(score.astype(float), axis=1, keepdims=True)
    oa = float((pred == y).mean())
    ba = float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))
    return dict(OA=oa, BA=ba, J=(oa+ba)/2, NLL=float(-lp[np.arange(len(y)), y].mean()),
                Brier=float(((np.exp(lp)-np.eye(8)[y])**2).sum(1).mean()), n=len(y))

class Data:
    def __init__(self):
        self.table = pd.read_csv(NATIVE/'table.csv', dtype={'mmsi':str})
        self.n = int((self.table.role != 'strategy_fit').sum())
        assert self.n == 1741 and len(self.table) == 1805
        self.tr = np.flatnonzero(self.table.role.eq('utility_train'))
        self.va = np.flatnonzero(self.table.role.eq('selection'))
        self.te = np.flatnonzero(self.table.role.eq('evaluation'))
        self.up = self.table.up_index.to_numpy(int)
        with np.load(HERE/'features.npz') as z:
            np.testing.assert_array_equal(z['up_index'], self.up)
            desc, mass = z['desc'].astype(np.float32), z['mass'].astype(np.float32)
        assert desc.shape == (1805,18,8,91) and mass.shape == (1805,18,8)
        assert np.isfinite(desc).all() and (mass >= 0).all()
        mu, sd, active = [], [], []
        for t in range(18):
            x = desc[self.tr,t][mass[self.tr,t] > 0]
            m, s = x.mean(0), x.std(0)
            a = s > 1e-6
            mu.append(m); sd.append(np.where(a,s,1)); active.append(a)
        mu, sd, active = np.array(mu), np.array(sd), np.array(active)
        desc = np.where(active[None,:,None], (desc-mu[None,:,None])/sd[None,:,None], 0)
        desc[mass <= 0] = 0
        self.scaler = dict(mu=mu, scale=sd, active=active)
        self.x = torch.tensor(desc, device='cuda')
        self.mass = torch.tensor(mass, device='cuda')
        inverse = {int(v):i for i,v in enumerate(self.up)}
        oldrefs = np.load(LOCAL/'scores_layer1.npz')['refs'][self.up[:self.n]]
        self.refs = torch.tensor(np.array([inverse[int(v)] for v in oldrefs.ravel()]).reshape(oldrefs.shape), device='cuda')
        assert (self.refs >= self.n).all()
        compact = pd.read_csv(LOCAL/'paired_scale_readout/table.csv')
        lookup = {int(v):i for i,v in enumerate(compact.table_index)}
        ix = np.array([lookup[int(v)] for v in self.up[:self.n]])
        self.baselines = {s:np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{s}.npz')['score'][ix].astype(np.float32) for s in SEEDS}
        self.fixed = {}
        self.cover = ((self.mass[:self.n,None,None].sum(-1)>0) & (self.mass[self.refs].sum(-1)>0)).float().mean(2)
        for a,b in ((self.tr,self.va),(self.tr,self.te),(self.va,self.te),(np.arange(self.n),np.arange(self.n,1805))):
            for field in ('mmsi','pass_group'):
                assert not set(self.table.iloc[a][field]) & set(self.table.iloc[b][field]), field

    def evidence(self, ids, model, mode='both', diagnostic=False):
        ids = torch.as_tensor(ids,device='cuda'); ref = self.refs[ids]
        q = self.x[ids][:,None,None].expand(-1,8,2,-1,-1,-1)
        s = self.x[ref]
        if model.arm == 'content_both':
            q = q*model.dim_mask; s = s*model.dim_mask
        shape = q.shape[:-2]
        q, s = q.reshape(-1,8,91), s.reshape(-1,8,91)
        tid = torch.arange(18,device='cuda').expand(*shape).reshape(-1)
        if model.arm == 'content_both':
            # Zero-padded non-content coordinates must not dilute the distance.
            hq, hs = model.metric.project(q), model.metric.project(s)
            a,b = hq[:,:,None],hs[:,None]
            emb = model.metric.embedding(tid)[:,None,None].expand(-1,8,8,-1)
            phi = torch.cat(((a+b)*.5,(a-b).abs()*.5,emb),-1)
            w = (1+.25*model.metric.output(F.gelu(phi)).tanh())*model.dim_mask
            distance = (w*(q[:,:,None]-s[:,None]).square()).sum(-1)/w.sum(-1)
            k = 2*torch.exp(-.5*distance)-1
        else:
            k = model.metric(q,s,tid)
        qa = self.mass[ids][:,None,None].expand(*shape,8).reshape(-1,8)
        sa = self.mass[ref].reshape(-1,8)
        k0 = None
        if mode != 'both' or diagnostic:
            dim = 32 if model.arm == 'content_both' else 91
            d0 = (q[:,:,:dim,None].transpose(2,3)-s[:,None,:,:dim]).square().mean(-1)
            k0 = 2*torch.exp(-.5*d0)-1
        kt = k0 if mode in ('initial','score_only') else k
        ks = k0 if mode in ('initial','transport_only') else k
        t,_,_,_ = matching.sinkhorn(kt,qa,sa)
        local = matching.pair_score(t,ks,qa,sa).reshape(*shape,2).mean(2)
        if diagnostic:
            t0,_,_,_ = matching.sinkhorn(k0,qa,sa)
            keep=(qa>0)[:,:,None]&(sa>0)[:,None,:]
            d=torch.stack(((t-t0).abs().sum((-1,-2)),
                           ((k-k0).abs()*keep).sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),
                           (t*k).sum((-1,-2))-(t0*k0).sum((-1,-2))),-1)
            return local,d.reshape(*shape,3).mean(2)
        return local

class Model(nn.Module):
    def __init__(self,arm):
        super().__init__(); self.arm=arm
        self.metric=ops.TypedPairMetric(91,18)
        if arm in ('score_only','typed_fixed'): self.metric.requires_grad_(False)
        self.head=nn.Sequential(nn.Linear(129,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.register_buffer('mu',torch.zeros(129));self.register_buffer('scale',torch.ones(129))
        self.register_buffer('active',torch.ones(129,dtype=torch.bool))
        self.register_buffer('dim_mask',torch.cat((torch.ones(32),torch.zeros(59))))

    def raw(self,local,base,cover):
        v=local.flatten(2)
        others=(v.sum(1,keepdim=True)-v)/7
        mask=torch.eye(8,device=base.device,dtype=torch.bool)[None,:,:,None]
        rival=torch.logsumexp(v[:,None].expand(-1,8,-1,-1).masked_fill(mask,-torch.inf),2)-np.log(7.)
        z=base-base.mean(1,keepdim=True)
        competing=z[:,None].expand(-1,8,-1).masked_fill(torch.eye(8,device=z.device,dtype=torch.bool)[None],-torch.inf).max(-1).values
        return torch.cat((v,others,rival,cover,z[...,None],competing[...,None],(z-competing)[...,None]),-1)

    def read(self,local,base,cover):
        raw=self.raw(local,base,cover)
        delta=self.head(torch.where(self.active,(raw-self.mu)/self.scale,0.)).squeeze(-1)
        delta=delta-delta.mean(1,keepdim=True)
        if self.arm!='score_only':delta=delta*(cover.sum((1,2))>0)[:,None]
        return base+delta

    def forward(self,data,ids,baseline,mode='both'):
        ids=torch.as_tensor(ids,device='cuda')
        if self.arm=='score_only':
            local=torch.zeros(len(ids),8,18,2,device='cuda');cover=torch.zeros(len(ids),8,18,device='cuda')
        elif self.arm=='typed_fixed':local=data.fixed['typed'][ids];cover=data.cover[ids]
        else:local=data.evidence(ids,self,mode);cover=data.cover[ids]
        return self.read(local,baseline[ids],cover)

@torch.inference_mode()
def predict(model,data,ids,baseline,mode='both'):
    return np.concatenate([model(data,ids[i:i+MICRO],baseline,mode).cpu().numpy() for i in range(0,len(ids),MICRO)])

@torch.inference_mode()
def prepare_fixed(data):
    for key,arm in (('typed','typed_fixed'),('content','content_both')):
        model=Model(arm).cuda();path=HERE/f'fixed_{key}.npz'
        if path.is_file():
            with np.load(path) as z:
                np.testing.assert_array_equal(z['up_index'],data.up[:data.n]);value=z['local'].copy()
        else:
            rows=[];tick=time.perf_counter()
            for i in range(0,data.n,MICRO):
                rows.append(data.evidence(np.arange(i,min(i+MICRO,data.n)),model).cpu().numpy())
            value=np.concatenate(rows)
            np.savez_compressed(path,up_index=data.up[:data.n],local=value)
            print('FIXED',key,'seconds',round(time.perf_counter()-tick,2),flush=True)
        data.fixed[key]=torch.tensor(value,device='cuda')
    write_json(HERE/'source_matching.json',{
        k:dict(score_quantiles=np.quantile(v[data.tr,...,0].cpu(),[0,.1,.5,.9,1]),
               mass_quantiles=np.quantile(v[data.tr,...,1].cpu(),[0,.1,.5,.9,1])) for k,v in data.fixed.items()})

@torch.inference_mode()
def head_stats(model,data,baseline):
    ids=data.tr
    if model.arm=='score_only':local=torch.zeros(len(ids),8,18,2,device='cuda');cover=torch.zeros(len(ids),8,18,device='cuda')
    else:local=data.fixed['content' if model.arm=='content_both' else 'typed'][ids];cover=data.cover[ids]
    raw=model.raw(local,baseline[ids],cover)
    model.mu.copy_(raw.mean((0,1)));sd=raw.std((0,1),unbiased=False)
    model.active.copy_(sd>1e-6);model.scale.copy_(torch.where(model.active,sd,1.))

def checks(data):
    ids=data.tr[:2];base=torch.tensor(data.baselines[17],device='cuda')
    torch.manual_seed(17);model=Model('typed_both').cuda();head_stats(model,data,base)
    initial=model(data,ids,base);error=float((initial-base[ids]).abs().max());assert error==0
    with torch.no_grad():model.head[-1].weight.normal_(0,.02)
    score=model(data,ids,base);loss=F.cross_entropy(score,torch.tensor(data.table.iloc[ids].class_id.to_numpy(),device='cuda'));loss.backward()
    grad=float(model.metric.output.weight.grad.abs().max());assert grad>1e-9,grad
    q=data.x[ids,0,:3];s=data.x[ids,0,3:7];tid=torch.zeros(2,device='cuda',dtype=torch.long)
    with torch.no_grad():
        model.metric.output.weight.normal_(0,.05)
        symmetry=float((model.metric(q,s,tid)-model.metric(s,q,tid).transpose(-1,-2)).abs().max())
    assert symmetry<1e-6,symmetry
    # Verify both paths actually carry derivatives, separately from a trained benefit claim.
    paths={}
    for mode in ('score_only','transport_only','both'):
        model.zero_grad(set_to_none=True)
        out=model(data,ids,base,mode)
        F.cross_entropy(out,torch.tensor(data.table.iloc[ids].class_id.to_numpy(),device='cuda')).backward()
        paths[mode]=float(model.metric.output.weight.grad.abs().max())
        assert paths[mode]>1e-10,(mode,paths)
    return dict(zero_head_replay=error,metric_gradient=grad,pair_symmetry=symmetry,path_gradients=paths,
                source_n=len(data.tr),selection_n=len(data.va),development_n=len(data.te),reference_n=64)

def fit(data,arm,seed):
    tick=time.perf_counter();torch.manual_seed(seed);model=Model(arm).cuda()
    base=torch.tensor(data.baselines[seed],device='cuda');head_stats(model,data,base)
    y=data.table.class_id.to_numpy(int);yt=torch.tensor(y,device='cuda')
    counts=np.bincount(y[data.tr],minlength=8);weights=.5+.5*len(data.tr)/(8*counts)
    weight=torch.tensor(weights,dtype=torch.float32,device='cuda')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    rng=np.random.default_rng(seed);best=metrics(y[data.va],data.baselines[seed][data.va])['J'];chosen=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve=[dict(seed=seed,arm=arm,step=0,loss=None,**metrics(y[data.va],data.baselines[seed][data.va]))]
    for step in range(1,101):
        batch=rng.choice(data.tr,32,replace=False);opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,32,MICRO):
            ix=batch[start:start+MICRO];score=model(data,ix,base)
            loss=(F.cross_entropy(score,yt[ix],reduction='none')*weight[yt[ix]]).sum()/32
            assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())
        opt.step()
        if step in CHECKPOINTS:
            m=metrics(y[data.va],predict(model,data,data.va,base))
            curve.append(dict(seed=seed,arm=arm,step=step,loss=total,**m))
            if m['J']>best+1e-12:best=m['J'];chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT',arm,seed,step,'source_J',round(m['J'],6),'seconds',round(time.perf_counter()-tick,1),flush=True)
    receipt=dict(arm=arm,seed=seed,selected_step=chosen,steps=100,batch=32,microbatch=MICRO,
                 parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=time.perf_counter()-tick)
    torch.save(dict(state=state,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt

def main(check_only=False):
    tick=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for d in ('models','results'):(HERE/d).mkdir(exist_ok=True)
    data=Data();np.savez_compressed(HERE/'scaler.npz',**data.scaler)
    prepare_fixed(data);verification=checks(data);write_json(HERE/'checks.json',verification);print('CHECKS',verification,flush=True)
    if check_only:return
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            receipts.append(fit(data,arm,seed));pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    rows=[];timings=[]
    for seed in SEEDS:
        base=torch.tensor(data.baselines[seed],device='cuda')
        np.savez_compressed(HERE/'results'/f'old_full_{seed}.npz',score=data.baselines[seed],up_index=data.up[:data.n])
        for arm in ARMS:
            model=Model(arm).cuda();ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
            torch.cuda.synchronize();start=time.perf_counter();score=predict(model,data,np.arange(data.n),base);torch.cuda.synchronize()
            timings.append(dict(arm=arm,seed=seed,n=data.n,seconds=time.perf_counter()-start))
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,up_index=data.up[:data.n])
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(arm=arm,seed=seed,role=role,**metrics(data.table.iloc[ids].class_id.to_numpy(int),score[ids])))
                if arm==ARMS[0]:rows.append(dict(arm='old_full',seed=seed,role=role,**metrics(data.table.iloc[ids].class_id.to_numpy(int),data.baselines[seed][ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False);pd.DataFrame(timings).to_csv(HERE/'inference_receipts.csv',index=False)
    write_json(HERE/'complete.json',dict(fits=12,seconds=time.perf_counter()-tick,peak_cuda_bytes=torch.cuda.max_memory_allocated(),checks=verification))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args();main(args.check_only)
