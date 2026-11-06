"""Explicit frozen M1 -> shallow M2 interface, paired source-only readout training."""
from pathlib import Path
import importlib.util
import sys
import os
import json
import gc
import time
import argparse

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
ROOT=Path(__file__).resolve().parent
CACHE_ROOT=ROOT/'cache'
PREVIOUS=ROOT.parent/'m1_coverage_20261101'
spec=importlib.util.spec_from_file_location('coverage_source',PREVIOUS/'experiment.py')
prev=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prev;spec.loader.exec_module(prev)
torch,np,pd,task=prev.torch,prev.np,prev.pd,prev.task
nn=torch.nn
F=prev.F
CHECKPOINT=PREVIOUS/'small_class_s17/final.pt'


def write(name,data):
    task.write(ROOT/name,data)


@torch.no_grad()
def prepare():
    dest=ROOT/'cache';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    rows,groups=prev.records();a,c=prev.arrays()
    net=prev.Network('class')
    net.restore(torch.load(CHECKPOINT,map_location='cpu',weights_only=True))
    net.eval();net.requires_grad_(False)
    stores={};seen={};handles=[]
    def pool_hook(p):
        def hook(module,args):
            seen[('x',p)]=args[0].detach()
            seen[('mask',p)]=torch.minimum(args[1],prev.old.grid_occupancy(args[2],args[0].shape[-2:]))
        return hook
    def response_hook(p):
        def hook(module,args,out):seen[('response',p)]=out.detach()
        return hook
    def head_hook(module,args):seen['features']=args[0].detach()
    for p in range(2):
        handles.append(net.base.m2[p].register_forward_pre_hook(pool_hook(p)))
        handles.append(net.m1[p].reader.register_forward_hook(response_hook(p)))
    handles.append(net.base.head.register_forward_pre_hook(head_hook))
    tick=time.perf_counter()
    # Preserve original manifest indexing; selection rows are cached but never fitted.
    for start in range(0,len(rows),8):
        ids=np.arange(start,min(start+8,len(rows)))
        score=net(a,c,ids)[0]
        values={k:torch.stack([seen[(k,p)] for p in range(2)],1) for k in ('x','mask','response')}
        values.update(features=seen['features'],score=score,available=task.tensor(a['available'],ids))
        values['valid']=prev.old.grid_occupancy(task.tensor(a['valid'],ids).flatten(0,1),(20,20)).reshape(len(ids),2,1,20,20)
        for key,value in values.items():
            arr=value.cpu().numpy()
            if key not in stores:
                stores[key]=np.lib.format.open_memmap(dest/(key+'.npy'),mode='w+',dtype=np.float32,shape=(len(rows),)+arr.shape[1:])
            stores[key][ids]=arr
        if start%160==0:print('CACHE',start+len(ids),len(rows),flush=True)
    for handle in handles:handle.remove()
    for s in stores.values():s.flush()
    errors={}
    for split in ('validation','evaluation'):
        expected=np.load(CHECKPOINT.parent/(split+'.npz'))
        errors[split]=float(abs(stores['score'][groups[split]]-expected['score']).max())
        assert errors[split]<1e-4,errors
    assert stores['features'].shape[1]==7170,stores['features'].shape
    torch.save(net.base.head.state_dict(),dest/'head.pt')
    write('groups.json',{k:v.tolist() for k,v in groups.items()})
    write('cache/complete.json',dict(n=len(rows),shapes={k:list(v.shape) for k,v in stores.items()},errors=errors,seconds=time.perf_counter()-tick,checkpoint=str(CHECKPOINT)))
    del net,a,c;gc.collect();torch.cuda.empty_cache()


class RegionalGuidance(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        self.value=nn.ModuleList(nn.Conv2d(512,32,1,bias=False) for _ in range(2))
        self.output=nn.ModuleList(nn.Linear(160,512,bias=False) for _ in range(2))
        self.scorer=nn.ModuleList(nn.Conv2d(512,4,1,bias=False) for _ in range(2))
        for layer in self.output:nn.init.zeros_(layer.weight)
        for layer in self.scorer:
            nn.init.normal_(layer.weight,std=.005)
            layer.requires_grad_(arm=='content')
        self.head=nn.Linear(7170,4)
        self.head.load_state_dict(torch.load(CACHE_ROOT/'head.pt',map_location='cpu',weights_only=True))

    def forward(self,b,mode=None,details=False):
        deltas=[];traces=[]
        for p in range(2):
            x=b['x'][:,p];mask=b['mask'][:,p].flatten(2)
            mass=mask.sum(-1,keepdim=True);uniform=mask/mass.clamp_min(1e-8)
            normalized=F.layer_norm(x.permute(0,2,3,1),(512,)).permute(0,3,1,2)
            v=self.value[p](normalized).flatten(2)
            response=b['response'][:,p].flatten(2)
            if self.arm=='class':
                relative=(4*response-response.sum(1,keepdim=True))/3
                valid=b['valid'][:,p].flatten(2)
                vw=valid/valid.sum(-1,keepdim=True).clamp_min(1e-8)
                centered=relative-(relative*vw).sum(-1,keepdim=True)
                score=2*torch.tanh(centered/((centered.square()*vw).sum(-1,keepdim=True)+.01).sqrt())
            elif self.arm=='content':score=2*torch.tanh(self.scorer[p](normalized).flatten(2))
            else:score=torch.zeros_like(response)
            logits=score[:,None]+mask.clamp_min(1e-12).log()[:,:,None]
            logits=logits.masked_fill(mask[:,:,None]<=0,-1e4)
            weights=logits.softmax(-1)*(mask[:,:,None]>0)
            weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-8)
            if mode=='uniform':weights=uniform[:,:,None].expand(-1,-1,4,-1)
            elif mode=='shuffle':
                # Reverse occupied positions only: same mass/histogram, different alignment.
                weights=weights.clone()
                for i in range(len(x)):
                    for r in range(3):
                        ix=torch.where(mask[i,r]>0)[0]
                        weights[i,r,:,ix]=weights[i,r,:,ix.flip(0)]
            mean_v=torch.einsum('bru,bdu->brd',uniform,v)
            token=torch.einsum('brku,bdu->brkd',weights,v)
            descriptors=torch.cat((mean_v,token.flatten(2)),-1)
            mean_x=torch.einsum('bru,bdu->brd',uniform,x.flatten(2))
            scale=mean_x.square().mean(-1,keepdim=True).sqrt().detach()
            delta=.1*scale*self.output[p](descriptors).tanh()*(mass>0)
            delta=delta*b['available'][:,p,None,None]
            if mode=='off':delta=delta*0
            deltas.append(delta)
            if details:traces.append(dict(weights=weights,uniform=uniform,delta=delta,mean=mean_x))
        # Original per-polarization layout: global2048 + 3 regional512, then availability2.
        parts=[]
        for p in range(2):
            z=b['features'][:,p*3584:(p+1)*3584]
            parts.extend((z[:,:2048],z[:,2048:]+deltas[p].flatten(1)))
        features=torch.cat(parts+[b['features'][:,-2:]],1)
        score=self.head(features)
        return (score,traces) if details else score


def load_cache():
    cache={}
    for path in CACHE_ROOT.glob('*.npy'):
        a=np.load(path,mmap_mode='r');out=torch.empty(a.shape,dtype=torch.float32,device=task.DEVICE)
        for first in range(0,len(a),16):out[first:first+16]=torch.from_numpy(np.array(a[first:first+16],copy=True)).to(task.DEVICE)
        cache[path.stem]=out
    print('RESIDENT_GIB',torch.cuda.memory_allocated()/1024**3,flush=True)
    return cache


def batch(cache,ids):
    ids=torch.as_tensor(ids,dtype=torch.long,device=task.DEVICE)
    return {k:v.index_select(0,ids) for k,v in cache.items() if k!='score'}


@torch.no_grad()
def predict(net,cache,ids,mode=None):
    return np.concatenate([net(batch(cache,ids[s:s+24]),mode).cpu().numpy() for s in range(0,len(ids),24)])


def orders(train,seed):
    rng=np.random.default_rng(seed);stream=[]
    while len(stream)<4320:stream.extend(rng.permutation(train).tolist())
    return np.array(stream[:4320]).reshape(180,24)


def checks(cache,groups):
    probe=groups['small'][:2];b=batch(cache,probe);record={}
    for arm in ('mean','content','class'):
        torch.manual_seed(17);net=RegionalGuidance(arm).to(task.DEVICE)
        z,tr=net(b,details=True)
        err=float((z-cache['score'][torch.tensor(probe,device=task.DEVICE)]).abs().max())
        assert err<1e-4,(arm,err);record[arm+'_zero_start']=err
        for p,t in enumerate(tr):
            nonempty=b['mask'][:,p].flatten(2).sum(-1)>0
            assert torch.allclose(t['weights'].sum(-1),nonempty[:,:,None].expand(-1,-1,4).float(),atol=1e-6)
            assert (t['weights']*(b['mask'][:,p].flatten(2)[:,:,None]==0)).abs().max()==0
        torch.nn.init.normal_(net.output[0].weight,std=.01)
        z,tr=net(b,details=True)
        manual=b['features'].clone()
        for p in range(2):manual[:,p*3584+2048:(p+1)*3584]+=tr[p]['delta'].flatten(1)
        assert torch.allclose(z,F.linear(manual,net.head.weight,net.head.bias),atol=1e-6)
        empty={k:v.clone() for k,v in b.items()};empty['mask'].zero_()
        _,et=net(empty,details=True)
        assert all(t['weights'].abs().max()==0 and t['delta'].abs().max()==0 for t in et)
        # Independent loop checks one weighted feature aggregate rather than its own einsum.
        p=0;x=b['x'][:,p];norm=F.layer_norm(x.permute(0,2,3,1),(512,)).permute(0,3,1,2)
        v=net.value[p](norm).flatten(2)
        direct=sum(tr[p]['weights'][0,0,0,u]*v[0,:,u] for u in range(400))
        fast=torch.einsum('u,du->d',tr[p]['weights'][0,0,0],v[0])
        assert torch.allclose(direct,fast,atol=1e-5)
    write('checks.json',record)


def train(arm,seed,cache,rows,groups,freeze_head=False):
    dest=ROOT/f'{arm}_s{seed}'
    if (dest/'receipt.json').exists():return
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);net=RegionalGuidance(arm).to(task.DEVICE)
    if freeze_head:net.head.requires_grad_(False)
    initial_head={n:p.detach().clone() for n,p in net.head.named_parameters()}
    params={n:p for n,p in net.named_parameters() if p.requires_grad}
    buckets={}
    for name,p in params.items():
        lr=1e-5 if name.startswith('head') else (1e-3 if name.startswith('output') else 3e-4)
        buckets.setdefault(lr,[]).append(p)
    opt=torch.optim.AdamW([dict(params=ps,lr=lr) for lr,ps in buckets.items()],weight_decay=.01)
    y=np.array([r['label'] for r in rows]);train_ids=groups['small']
    fr=pd.DataFrame([rows[i] for i in train_ids]);weight=np.zeros(len(rows),np.float32)
    weight[train_ids]=1/(fr.groupby('label').mmsi.transform('nunique').to_numpy()*fr.groupby('mmsi').mmsi.transform('size').to_numpy())
    baseline=cache['score'].cpu().numpy()
    margin=baseline[train_ids,y[train_ids]]-np.where(np.arange(4)[None]==y[train_ids,None],-np.inf,baseline[train_ids]).max(1)
    weight[train_ids]*=np.where(margin<1,2,1);weight[train_ids]/=weight[train_ids].mean()
    weight=torch.tensor(weight,device=task.DEVICE);steps=orders(train_ids,seed)
    task.write(dest/'orders.json',steps.tolist());curve=[];losses=[];grad=[];tick=time.perf_counter()
    for step,ids in enumerate(steps,1):
        opt.zero_grad(set_to_none=True);z=net(batch(cache,ids))
        loss=(F.cross_entropy(z,torch.tensor(y[ids],device=task.DEVICE),reduction='none')*weight[ids]).mean()
        assert torch.isfinite(loss);loss.backward()
        if step in (1,2):
            norms={n:float(p.grad.norm()) for n,p in params.items() if p.grad is not None}
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params.values())
            assert all(norms[f'output.{p}.weight']>0 for p in range(2))
            grad.append(dict(step=step,norms=norms))
        torch.nn.utils.clip_grad_norm_(list(params.values()),5);opt.step();losses.append(float(loss.detach()))
        if step%60==0:
            z=predict(net,cache,groups['validation'])
            rec=dict(step=step,loss=float(np.mean(losses)),**task.metrics(y[groups['validation']],z));curve.append(rec)
            pd.DataFrame(curve).to_csv(dest/'curve.csv',index=False);print('STEP',dest.name,rec,flush=True);losses=[]
    torch.save(net.state_dict(),dest/'final.pt');metrics={}
    head_change=max(float((p-initial_head[n]).abs().max()) for n,p in net.head.named_parameters())
    if freeze_head:
        assert head_change==0
        assert all(p.grad is None for p in net.head.parameters())
    off_errors={}
    for split in ('validation','evaluation'):
        ids=groups[split]
        for mode in (None,'off','uniform','shuffle') if arm=='class' else (None,'off'):
            z=predict(net,cache,ids,mode);name=split+('_'+mode if mode else '')
            np.savez_compressed(dest/(name+'.npz'),ids=ids,score=z)
            metrics[name]=task.metrics(y[ids],z)
            if mode=='off':
                off_errors[split]=float(abs(z-baseline[ids]).max())
                if freeze_head:assert off_errors[split]<1e-4,off_errors
    before=predict(net,cache,groups['validation'][:3])
    net.load_state_dict(torch.load(dest/'final.pt',map_location=task.DEVICE,weights_only=True))
    error=float(abs(before-predict(net,cache,groups['validation'][:3])).max());assert error==0
    task.write(dest/'receipt.json',dict(arm=arm,seed=seed,steps=180,n_train=719,examples=4320,
        upstream_seed=17,trainable=sum(p.numel() for p in params.values()),metrics=metrics,gradients=grad,
        seconds=time.perf_counter()-tick,reload_error=error,checkpoint_selection='fixed180',formal_changed=False))
    receipt=json.loads((dest/'receipt.json').read_text())
    receipt.update(head_frozen=freeze_head,head_max_change=head_change,off_errors=off_errors)
    task.write(dest/'receipt.json',receipt)
    print('DONE',dest.name,metrics,flush=True)


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--freeze-head',action='store_true');args=parser.parse_args()
    if args.freeze_head:
        assert not args.prepare,'Reuse the existing frozen cache.'
        ROOT=ROOT/'frozen_head';ROOT.mkdir(parents=True,exist_ok=True)
    task.current.setup(task.DEVICE)
    if args.prepare:prepare();return
    rows,groups=prev.records();cache=load_cache()
    if not args.freeze_head:checks(cache,groups)
    for seed in (17,29,43):
        for arm in ('mean','content','class'):train(arm,seed,cache,rows,groups,args.freeze_head)


if __name__=='__main__':main()
