"""Matched native-resolution M2 fits; old complete model stays frozen."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
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
OLD = BASE/'m2_operator_validation'
UP = BASE/'regional_utility_pilot/upstream'
LOCAL = BASE/'local_evidence_transport'
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
ops = load('native_redesign_ops', HERE.parent/'operators.py')
oldops = load('native_old_ops', OLD/'operators.py')
SEEDS = (17, 29, 43)
ARMS = ('mean', 'moment', 'grid_moment', 'axis_moment')
CHECKPOINTS = (0, 20, 40, 80)
SLOTS = (slice(0,8), slice(16,20), slice(8,16))
CENTERS = torch.tensor([(u,v) for u in (-.375,-.125,.125,.375) for v in (-.25,.25)]*2
                       +[(u,v) for u in (-.625,.625) for v in (-.25,.25)])

def metrics(y, score):
    pred = score.argmax(1)
    lp = score-logsumexp(score, axis=1, keepdims=True)
    oa = float((pred==y).mean())
    ba = float(np.mean([(pred[y==c]==c).mean() for c in np.unique(y)]))
    return dict(OA=oa, BA=ba, J=(oa+ba)/2, NLL=float(-lp[np.arange(len(y)),y].mean()),
                Brier=float(((np.exp(lp)-np.eye(8)[y])**2).sum(1).mean()), n=len(y))

class Data:
    def __init__(self):
        self.table = pd.read_csv(OLD/'table.csv', dtype={'mmsi':str})
        self.n = len(self.table)
        native = pd.read_csv(HERE/'table.csv')
        np.testing.assert_array_equal(native.obs_id.iloc[:self.n], self.table.obs_id)
        self.bands = np.load(HERE/'native_cache/native_bands.npy', mmap_mode='r')
        self.valid = np.load(HERE/'native_cache/valid_pixels.npy', mmap_mode='r')
        self.g = np.load(HERE/'native_cache/geometry.npy')[:self.n]
        self.maps = np.load(OLD/'x.npy', mmap_mode='r')
        self.flags = np.load(OLD/'metadata.npz')['region_valid'].astype(np.float32)
        self.tr = np.flatnonzero(self.table.role.eq('utility_train'))
        self.va = np.flatnonzero(self.table.role.eq('selection'))
        self.te = np.flatnonzero(self.table.role.eq('evaluation'))
        self.up = self.table.up_index.to_numpy(int)
        compact = pd.read_csv(LOCAL/'paired_scale_readout/table.csv')
        mapping = {int(i):j for j,i in enumerate(compact.table_index)}
        self.compact = np.array([mapping[int(i)] for i in self.up])
        self.baselines = {s:np.load(LOCAL/'pair_conditioned_joint_metric/results'/f'pair_joint_{s}.npz')['score'][self.compact].astype(np.float32) for s in SEEDS}
        self.yy, self.xx = torch.meshgrid(torch.arange(160,device='cuda'),torch.arange(160,device='cuda'),indexing='ij')
        self.content_grid=torch.stack((2*(self.xx/4)/39-1,2*(self.yy/4)/39-1),-1)[None]

    def batch(self, ids, rotation=0):
        ids = np.asarray(ids)
        n = len(ids)
        band = torch.tensor(np.array(self.bands[ids]), device='cuda').rot90(rotation,(-2,-1))
        valid = torch.tensor(np.array(self.valid[ids]), device='cuda',dtype=torch.float32).rot90(rotation,(-2,-1))
        coarse = torch.tensor(np.stack([self.maps[i,:,rotation,:7] for i in ids]),device='cuda',dtype=torch.float32)
        coarse = F.grid_sample(coarse.flatten(0,1),self.content_grid.expand(n*2,-1,-1,-1),
                               mode='bilinear',padding_mode='border',align_corners=True).reshape(n,2,7,160,160)
        x = torch.cat((coarse[:,:,None].expand(-1,-1,3,-1,-1,-1),band[:,:,:,None]),3).flatten(1,2)
        g = torch.tensor(self.g[ids],device='cuda',dtype=torch.float32)
        for _ in range(rotation):
            a,b = g[...,0].clone(),g[...,1].clone()
            g[...,0],g[...,1] = b,159-a
            g[...,4] -= torch.pi/2
        dx,dy = self.xx-g[...,0,None,None],self.yy-g[...,1,None,None]
        angle = g[...,4,None,None]
        u = dx*angle.cos()+dy*angle.sin()
        v = -dx*angle.sin()+dy*angle.cos()
        length,width = g[...,2,None,None],g[...,3,None,None]
        def rectangle(l,w):
            return (l/2+.5-u.abs()).clamp(0,1)*(w/2+.5-v.abs()).clamp(0,1)
        body = rectangle(length,width)
        regions = torch.stack((body,(rectangle(1.5*length,width)-body).clamp_min(0),
                               (rectangle(length,2*width)-body).clamp_min(0)),2)*valid[:,:,None]
        regions *= (g[...,2]>0)[:,:,None,None,None]
        occupied = regions.sum((0,1,2))>0
        yy,xx = torch.where(occupied)
        if len(yy):
            ys = slice(max(0,int(yy.min())-5),min(160,int(yy.max())+6))
            xs = slice(max(0,int(xx.min())-5),min(160,int(xx.max())+6))
        else:
            ys,xs = slice(0,5),slice(0,5)
        return dict(x=x[...,ys,xs],regions=regions[...,ys,xs],valid=valid[...,ys,xs],
                    u=(u/length.clamp_min(1))[...,ys,xs],v=(v/width.clamp_min(1))[...,ys,xs],
                    theta=g[...,4],flags=torch.tensor(self.flags[ids],device='cuda'),ids=ids)

def typed_regions(b):
    n,_,_,h,w = b['regions'].shape
    return b['regions'][:,:,None].expand(n,2,3,3,h,w).flatten(1,2)

@torch.inference_mode()
def fit_reader(data):
    started=time.perf_counter()
    sums=torch.zeros(6,8,device='cuda',dtype=torch.float64)
    sums2=sums.clone(); masses=torch.zeros(6,device='cuda',dtype=torch.float64)
    means=[]; all_valid=[]
    for first in range(0,len(data.tr),8):
        ids=data.tr[first:first+8]
        xx=[]; vv=[]
        for k in range(4):
            b=data.batch(ids,k); x=b['x'].double()
            w=typed_regions(b)[:,:,0].double()
            mass=w.sum((-2,-1)); safe=mass.clamp_min(1)
            vals=(x*w[:,:,None]).sum((-2,-1))/safe[:,:,None]
            xx.append(vals.cpu().numpy()); vv.append((mass>0).cpu().numpy())
            sums += (x*w[:,:,None]).sum((0,3,4))
            sums2 += (x.square()*w[:,:,None]).sum((0,3,4))
            masses += mass.sum(0)
        means.append(np.mean(xx,axis=0)); all_valid.append(np.all(vv,axis=0))
    mu=(sums/masses[:,None]).cpu().numpy()
    scale=(sums2/masses[:,None]-torch.tensor(mu,device='cuda').square()).clamp_min(1e-8).sqrt().cpu().numpy()
    m=np.concatenate(means); ok=np.concatenate(all_valid)
    labels=data.table.iloc[data.tr].class_id.to_numpy(int)
    W=[]; bias=[]
    for t in range(6):
        z=(m[ok[:,t],t]-mu[t])/scale[t]
        yy=labels[ok[:,t]]
        count=np.bincount(yy,minlength=8)
        weight=.5+.5*len(yy)/(8*np.maximum(count[yy],1))
        design=np.c_[z,np.ones(len(z))]
        target=np.eye(8)[yy]
        penalty=np.eye(9); penalty[-1,-1]=0
        coef=np.linalg.solve(design.T@(weight[:,None]*design)+penalty,design.T@(weight[:,None]*target))
        W.append(coef[:8].T); bias.append(coef[8])
    # Fixed source response scale, not a target-selected attention temperature.
    W=np.array(W); bias=np.array(bias)
    tau=np.maximum(np.sqrt((W-W.mean(1,keepdims=True))**2).mean((1,2)),.02)
    np.savez_compressed(HERE/'reader.npz',mu=mu,scale=scale,W=W,bias=bias,tau=tau)
    receipt=dict(source_n=len(data.tr),source_classes=np.bincount(labels,minlength=8).tolist(),
                 source_only=True,seconds=time.perf_counter()-started,tau=tau.tolist(),
                 construction='7 frozen source PCA coordinates + native DoG, separate 6 typed ridge readers',
                 fitting='pixel body standardization; C4 mean body descriptors; balanced-half ridge alpha1')
    (HERE/'reader_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('READER',receipt,flush=True)

class GridResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth=nn.Parameter(torch.randn(6,8,1,3,3)*.05)
        self.out=nn.Parameter(torch.zeros(6,8,8))
    def forward(self,x,regions,valid,theta,tid):
        # Independent typed depthwise convolutions with no spatial resizing.
        n,c,h,w=x.shape
        y=F.conv2d((x*valid).reshape(1,n*c,h,w),self.depth[tid].reshape(n*c,1,3,3),padding=1,groups=n*c).reshape(n,c,h,w)
        return x+torch.einsum('noc,nchw->nohw',self.out[tid],y)*valid

class Model(nn.Module):
    def __init__(self,arm,reader,raw_mu=None,raw_scale=None):
        super().__init__(); self.arm=arm
        for key in ('mu','scale','W','bias','tau'):
            self.register_buffer(key,torch.tensor(reader[key],dtype=torch.float32))
        self.head=nn.Sequential(nn.Linear(128,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.conv=ops.AxisPairConv(8) if arm=='axis_moment' else GridResidual() if arm=='grid_moment' else None
        self.register_buffer('raw_mu',torch.zeros(128) if raw_mu is None else torch.tensor(raw_mu,dtype=torch.float32))
        self.register_buffer('raw_scale',torch.ones(128) if raw_scale is None else torch.tensor(raw_scale,dtype=torch.float32))

    def describe(self,b):
        n,_,c,h,w=b['x'].shape
        x=((b['x']-self.mu[None,:,:,None,None])/self.scale[None,:,:,None,None]).flatten(0,1)
        r=typed_regions(b).flatten(0,1)
        tid=torch.arange(6,device=x.device).repeat(n)
        valid=b['valid'][:,:,None].expand(n,2,3,h,w).reshape(n*6,1,h,w)
        theta=b['theta'][:,:,None].expand(n,2,3).reshape(-1)
        if self.conv is not None:
            x=self.conv(x,r,valid,theta,tid)
        response=torch.einsum('ncd,ndhw->nchw',self.W[tid],x)+self.bias[tid,:,None,None]
        if self.arm=='mean':
            u=b['u'][:,:,None].expand(n,2,3,h,w).reshape(n*6,-1)
            v=b['v'][:,:,None].expand(n,2,3,h,w).reshape(n*6,-1)
            centers=CENTERS.to(x.device)
            score=-((u[:,None]-centers[None,:,0,None])/.35)**2-((v[:,None]-centers[None,:,1,None])/.5)**2
            desc=[]
            for role,slots in enumerate(SLOTS):
                assignment=score[:,slots].softmax(1)*r[:,role].flatten(1)[:,None]
                mass=assignment.sum(-1)
                token=torch.einsum('nsl,ncl->ncs',assignment,response.flatten(2))/mass[:,None].clamp_min(1e-20)
                desc.append(oldops.class_summary(token,mass[:,None],self.tau[tid])[:,0])
            desc=torch.stack(desc,1)
        else:
            # Exact regional merge of pre-averaging exponential moments.
            desc=oldops.class_summary(response.flatten(2),r.flatten(2),self.tau[tid])
        desc=desc.reshape(n,2,3,3,8,3)*b['flags'][:,:,None,:,None,None]
        return desc

    def raw(self,desc,baseline,flags):
        # B,P,Band,R,C,Stat -> B,C,54; every polarization remains explicit.
        a=desc.permute(0,4,1,2,3,5).flatten(2)
        center=a-a.mean(1,keepdim=True)
        s=baseline-baseline.mean(1,keepdim=True)
        other=baseline[:,None].expand(-1,8,-1).masked_fill(torch.eye(8,device=s.device,dtype=torch.bool)[None],-torch.inf).max(-1).values
        flag=flags[:,:,None].expand(-1,2,3,3).flatten(1)
        return torch.cat((a,center,flag[:,None].expand(-1,8,-1),s[:,:,None],(baseline-other)[:,:,None]),-1)

    def forward(self,b,baseline,return_desc=False):
        desc=self.describe(b)
        raw=self.raw(desc,baseline,b['flags'])
        delta=self.head((raw-self.raw_mu)/self.raw_scale).squeeze(-1)
        delta=delta-delta.mean(1,keepdim=True)
        delta=delta*(b['flags'].sum((1,2))>0)[:,None]
        score=baseline+delta
        return (score,desc) if return_desc else score

@torch.inference_mode()
def predict(model,data,ids,baseline,size=8,descriptors=False):
    scores=[]; dd=[]
    for first in range(0,len(ids),size):
        ix=ids[first:first+size]; base=torch.tensor(baseline[ix],device='cuda')
        ss=[]; ds=[]
        for k in range(4):
            s,d=model(data.batch(ix,k),base,True);ss.append(s);ds.append(d)
        scores.append(torch.stack(ss).mean(0).cpu().numpy())
        if descriptors:dd.append(torch.stack(ds).mean(0).cpu().numpy())
    score=np.concatenate(scores)
    return (score,np.concatenate(dd)) if descriptors else score

@torch.inference_mode()
def head_stats(model,data,baseline):
    rows=[]
    for first in range(0,len(data.tr),8):
        ids=data.tr[first:first+8]
        base=torch.tensor(baseline[ids],device='cuda')
        for k in range(4):
            b=data.batch(ids,k); d=model.describe(b)
            rows.append(model.raw(d,base,b['flags']).flatten(0,1).cpu().numpy())
    raw=np.concatenate(rows)
    return raw.mean(0),np.maximum(raw.std(0),.05)

def fit(data,reader,arm,seed):
    tick=time.perf_counter();torch.manual_seed(seed)
    baseline=data.baselines[seed]
    model=Model(arm,reader).cuda()
    mu,scale=head_stats(model,data,baseline)
    model.raw_mu.copy_(torch.tensor(mu,device='cuda'));model.raw_scale.copy_(torch.tensor(scale,device='cuda'))
    y=data.table.class_id.to_numpy(int)
    ytr=y[data.tr];count=np.bincount(ytr,minlength=8)
    weights=.5+.5*len(ytr)/(8*count[ytr])
    rng=np.random.default_rng(seed);draws=rng.integers(len(data.tr),size=(80,32));rotations=rng.integers(4,size=(80,8))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=metrics(y[data.va],baseline[data.va])['J'];chosen=0
    beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curves=[dict(seed=seed,arm=arm,step=0,loss=None,**metrics(y[data.va],baseline[data.va]))]
    gradlog=[]
    for step in range(1,81):
        opt.zero_grad(set_to_none=True);loss_total=0.
        for micro in range(8):
            take=draws[step-1,micro*4:(micro+1)*4];ids=data.tr[take]
            base=torch.tensor(baseline[ids],device='cuda')
            out=model(data.batch(ids,int(rotations[step-1,micro])),base)
            loss=(F.cross_entropy(out,torch.tensor(ytr[take],device='cuda'),reduction='none')*torch.tensor(weights[take],device='cuda')).sum()/32
            assert torch.isfinite(loss)
            loss.backward();loss_total+=float(loss.detach())
        if step<=3:
            gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
            gradlog.append(dict(step=step,gradients=gradients))
        opt.step()
        if step in CHECKPOINTS:
            score=predict(model,data,data.va,baseline)
            m=metrics(y[data.va],score)
            if m['J']>best+1e-12:
                best=m['J'];chosen=step
                beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            curves.append(dict(seed=seed,arm=arm,step=step,loss=loss_total,**m))
            print('FIT',arm,seed,step,'J',m['J'],'chosen',chosen,'seconds',round(time.perf_counter()-tick,1),flush=True)
    receipt=dict(arm=arm,seed=seed,selected_step=chosen,seconds=time.perf_counter()-tick,parameters=sum(p.numel() for p in model.parameters()),gradients=gradlog)
    torch.save(dict(state=beststate,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    pd.DataFrame(curves).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt

def checks(data,reader):
    heads=[]
    for arm in ARMS:
        torch.manual_seed(17)
        candidate=Model(arm,reader)
        heads.append(candidate.head[0].weight.detach().clone())
    head_error=max(float((h-heads[0]).abs().max()) for h in heads)
    assert head_error==0
    torch.manual_seed(7)
    model=Model('axis_moment',reader).cuda()
    ix=data.tr[:2];b=data.batch(ix);base=torch.tensor(data.baselines[17][ix],device='cuda')
    out=model(b,base)
    identity=float((out-base).abs().max().detach());assert identity==0
    with torch.no_grad():model.conv.output.normal_(0,.03)
    d=model.describe(b)
    changed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
    changed['x'][:,3:] *= 1.7
    d2=model.describe(changed)
    pol=float((d[:,0]-d2[:,0]).abs().max().detach());assert pol==0
    change=float((d[:,1]-d2[:,1]).abs().max().detach());assert change>0
    changed={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in b.items()}
    changed['x'][:,:3]*=1.7
    reverse=float((d[:,1]-model.describe(changed)[:,1]).abs().max().detach());assert reverse==0
    ramp=torch.arange(40,device='cuda',dtype=torch.float32)[None,None,None].expand(1,1,40,40)
    sampled=F.grid_sample(ramp,data.content_grid,align_corners=True,padding_mode='border')
    coordinate=float((sampled[0,0,::4,::4]-ramp[0,0]).abs().max());assert coordinate<1e-5
    d.square().mean().backward()
    cg=float(model.conv.output.grad.norm());assert cg>0 and np.isfinite(cg)
    return dict(matched_head_initialization=head_error,zero_delta_replay=identity,unchanged_VV_under_VH_change=pol,
                unchanged_VH_under_VV_change=reverse,content_center_replay=coordinate,
                VH_descriptor_change=change,conv_gradient=cg)

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for folder in ('models','results'):(HERE/folder).mkdir(exist_ok=True)
    data=Data()
    if not (HERE/'reader.npz').exists():fit_reader(data)
    reader=dict(np.load(HERE/'reader.npz'))
    check=checks(data,reader);(HERE/'checks.json').write_text(json.dumps(check,indent=2));print('CHECKS',check,flush=True)
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            path=HERE/'models'/f'{arm}_{seed}.pt'
            receipts.append(torch.load(path,map_location='cpu',weights_only=False)['receipt'] if path.exists() else fit(data,reader,arm,seed))
            (HERE/'training_receipts.json').write_text(json.dumps(receipts,indent=2))
    rows=[];y=data.table.class_id.to_numpy(int)
    for seed in SEEDS:
        for arm in ('old_full',*ARMS):
            if arm=='old_full':score=data.baselines[seed]
            else:
                ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False)
                model=Model(arm,reader).cuda();model.load_state_dict(ck['state'])
                score=data.baselines[seed].copy() if ck['receipt']['selected_step']==0 else predict(model,data,np.arange(data.n),data.baselines[seed])
                del model
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,up_index=data.up)
            for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
                rows.append(dict(seed=seed,arm=arm,role=role,**metrics(y[ids],score[ids])))
    pd.DataFrame(rows).to_csv(HERE/'metrics.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(fits=12,checks=check,stage='M2 only; old M3 frozen'),indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA','BA']].mean(),flush=True)

if __name__=='__main__':main()
