"""Fixed OSS2 split; validation-only model selection; no test during training."""
from pathlib import Path
import argparse
import gc
import json
import os
import time

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cpinet import CPINet, GradNorm, losses
from data import ROOT, records, prepare

torch.set_num_threads(2)
torch.backends.cudnn.benchmark=True
torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision('high')
DEVICE='cuda'


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def metrics(y,z):
    pred=z.argmax(1)
    cm=np.zeros((4,4),int)
    np.add.at(cm,(y,pred),1)
    recall=cm.diagonal()/np.maximum(cm.sum(1),1)
    f1=2*cm.diagonal()/np.maximum(cm.sum(1)+cm.sum(0),1)
    prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    return dict(oa=float(100*(pred==y).mean()),ba=float(100*recall.mean()),
                macro_f1=float(100*f1.mean()),nll=float(-np.log(prob[np.arange(len(y)),y].clip(1e-12)).mean()),
                brier=float(((prob-np.eye(4)[y])**2).sum(1).mean()),confusion=cm.tolist())


class Inputs:
    def __init__(self):
        self.rows,self.groups=records()
        cache=ROOT/'native_cache'
        self.pixels=np.load(cache/'pixels.npy',mmap_mode='r')
        self.footprint=np.load(cache/'footprint.npy',mmap_mode='r')
        self.available=np.load(cache/'available.npy')
        self.y=np.array([r['label'] for r in self.rows])
        # Only legal training and validation pixels are moved to the GPU here.
        self.train_ids=np.r_[self.groups['small'],self.groups['validation']]
        self.lookup={int(i):j for j,i in enumerate(self.train_ids)}
        x=torch.tensor(np.array(self.pixels[self.train_ids]),device=DEVICE,dtype=torch.float32)/255
        mask=torch.tensor(np.array(self.footprint[self.train_ids]),device=DEVICE,dtype=torch.float32)
        count=mask.sum((-2,-1),keepdim=True).clamp_min(1)
        mean=(x*mask).sum((-2,-1),keepdim=True)/count
        std=(((x-mean).square()*mask).sum((-2,-1),keepdim=True)/count).sqrt().clamp_min(1e-5)
        self.x=(x-mean)/std*mask
        self.av=torch.tensor(self.available[self.train_ids],device=DEVICE)
        del x,mask

    def batch(self,ids,augment=False,generator=None):
        local=torch.tensor([self.lookup[int(i)] for i in ids],device=DEVICE)
        x=self.x[local]
        if augment:
            # Paired, scale-preserving dihedral transforms; no arbitrary rotation interpolation.
            k=int(torch.randint(4,(),generator=generator).item())
            x=torch.rot90(x,k,(-2,-1))
            if torch.rand((),generator=generator)<.5:x=x.flip(-1)
        return x,self.av[local],torch.tensor(self.y[ids],device=DEVICE)


@torch.no_grad()
def predict(net,inputs,ids):
    net.eval()
    result=[]
    for start in range(0,len(ids),32):
        x,av,y=inputs.batch(ids[start:start+32])
        result.append(net(x,av)[0].cpu().numpy())
    return np.concatenate(result)


def build(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net=CPINet().to(DEVICE)
    return net


def fit(inputs,seed,lr,epochs=300):
    dest=ROOT/f'cpinet_s{seed}_lr{lr:g}'
    dest.mkdir(exist_ok=True)
    if (dest/'complete.json').exists():
        return json.loads((dest/'complete.json').read_text(encoding='utf-8'))
    net=build(seed)
    gn=GradNorm().to(DEVICE)
    optimizer=torch.optim.SGD(net.parameters(),lr=lr,momentum=.9,weight_decay=5e-4)
    weight_optimizer=torch.optim.SGD(gn.parameters(),lr=lr,momentum=.9)
    rng=np.random.default_rng(seed+20261111)
    generator=torch.Generator().manual_seed(seed+20261112)
    train=inputs.groups['small'];val=inputs.groups['validation']
    by_class=[train[inputs.y[train]==k] for k in range(4)]
    steps=int(np.ceil(len(train)/32))
    start_epoch=0;best=None;curve=[];elapsed=0
    resume=dest/'resume.pt'
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        net.load_state_dict(saved['model']);gn.load_state_dict(saved['gradnorm'])
        optimizer.load_state_dict(saved['optimizer']);weight_optimizer.load_state_dict(saved['weight_optimizer'])
        start_epoch=saved['epoch'];best=saved['best'];curve=saved['curve'];elapsed=saved['elapsed']
        rng.bit_generator.state=saved['rng'];generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
    tick=time.perf_counter()
    for epoch in range(start_epoch,epochs):
        net.train();sum_loss=0;train_right=0;first_losses=[]
        scale=.1**sum(epoch>=p for p in (150,200,250))
        for opt in (optimizer,weight_optimizer):
            for group in opt.param_groups:group['lr']=lr*scale
        for step in range(steps):
            ids=np.concatenate([rng.choice(ids,8,replace=True) for ids in by_class])
            rng.shuffle(ids)
            x,av,y=inputs.batch(ids,True,generator)
            optimizer.zero_grad(set_to_none=True);weight_optimizer.zero_grad(set_to_none=True)
            z,scores,active=net(x,av)
            terms=losses(scores,active,y)
            if epoch==0:
                first_losses.append(terms.detach())
            else:
                loss_g=gn.objective(terms,net.classifier)
                gn.weights.grad=torch.autograd.grad(loss_g,gn.weights)[0]
                weight_optimizer.step();gn.normalize()
            total=(gn.weights.detach()*terms).sum()
            if not torch.isfinite(total):raise RuntimeError(f'Nonfinite loss epoch={epoch} step={step}')
            total.backward()
            optimizer.step()
            sum_loss+=float(total.detach());train_right+=int((z.argmax(1)==y).sum())
        if epoch==0:
            gn.initial.copy_(torch.stack(first_losses).mean(0).clamp_min(1e-8))
        row=dict(epoch=epoch+1,loss=sum_loss/steps,train_oa=100*train_right/(steps*32),
                 weights=gn.weights.detach().cpu().tolist(),seconds=elapsed+time.perf_counter()-tick)
        if (epoch+1)%10==0 or epoch==0:
            score=predict(net,inputs,val);m=metrics(inputs.y[val],score)
            row['validation']=m
            key=(m['ba'],m['oa'],-m['nll'])
            if best is None or key>tuple(best['key']):
                best=dict(epoch=epoch+1,key=list(key),metrics=m)
                torch.save(net.state_dict(),dest/'best.pt')
                np.savez_compressed(dest/'validation.npz',ids=val,score=score,y=inputs.y[val])
            print('VAL',seed,lr,epoch+1,round(row['train_oa'],2),round(m['oa'],3),round(m['ba'],3),round(row['seconds'],1),flush=True)
            write(ROOT/'status.json',dict(status='training',seed=seed,lr=lr,epoch=epoch+1,epochs=epochs,best=best,test_read=False))
        curve.append(row)
        if (epoch+1)%10==0:
            torch.save(dict(model=net.state_dict(),gradnorm=gn.state_dict(),optimizer=optimizer.state_dict(),
                weight_optimizer=weight_optimizer.state_dict(),epoch=epoch+1,best=best,curve=curve,
                elapsed=row['seconds'],rng=rng.bit_generator.state,generator=generator.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),resume)
            write(dest/'curve.json',curve)
    receipt=dict(seed=seed,lr=lr,epochs=epochs,steps=epochs*steps,exposures=epochs*steps*32,
        seconds=elapsed+time.perf_counter()-tick,best=best,parameters=sum(p.numel() for p in net.parameters()),
        trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),test_read=False,
        precision='float32, TF32 enabled',peak_tensor_gib=torch.cuda.max_memory_allocated()/1024**3)
    torch.save(net.state_dict(),dest/'final.pt')
    write(dest/'complete.json',receipt)
    del net,gn,optimizer,weight_optimizer
    gc.collect();torch.cuda.empty_cache()
    return receipt


def smoke():
    prepare();inputs=Inputs();net=build(17);net.train();gn=GradNorm().to(DEVICE)
    ids=inputs.groups['small'][:8];x,av,y=inputs.batch(ids)
    tick=time.perf_counter()
    z,scores,active=net(x,av);terms=losses(scores,active,y)
    objective=gn.objective(terms,net.classifier)
    gw=torch.autograd.grad(objective,gn.weights)[0]
    terms.sum().backward()
    grads={k:float(p.grad.norm()) for k,p in net.named_parameters() if p.grad is not None}
    assert all(np.isfinite(v) for v in grads.values())
    assert any('backbones.0' in k and v>0 for k,v in grads.items())
    assert any('backbones.1' in k and v>0 for k,v in grads.items())
    assert all(any(f'codes.{i}.' in k and v>0 for k,v in grads.items()) for i in range(4))
    net.eval()
    with torch.no_grad():
        av1=av.clone();av1[:,1]=0
        first=net(x,av1)[0];changed=x.clone();changed[:,1]=torch.randn_like(changed[:,1])*10
        second=net(changed,av1)[0]
        error=float((first-second).abs().max())
        assert error==0,error
    report=dict(parameters=sum(p.numel() for p in net.parameters()),scores=list(scores.shape),
        initial_losses=terms.detach().cpu().tolist(),gradnorm_gradient=gw.cpu().tolist(),missing_branch_error=error,
        finite_nonzero_gradients=sum(v>0 for v in grads.values()),seconds=time.perf_counter()-tick,
        channels=net.backbones[0].channels)
    write(ROOT/'smoke.json',report);print(report,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:return smoke()
    prepare();inputs=Inputs()
    scouts=[fit(inputs,17,lr) for lr in (.01,.003)]
    chosen=max(scouts,key=lambda r:tuple(r['best']['key']))
    selection=dict(lr=chosen['lr'],scouts=scouts,selection='validation BA, then OA, then NLL',test_read=False)
    write(ROOT/'selected.json',selection)
    for seed in (29,43):fit(inputs,seed,chosen['lr'])
    write(ROOT/'status.json',dict(status='training_complete',test_read=False,selected_lr=chosen['lr']))


if __name__=='__main__':main()
