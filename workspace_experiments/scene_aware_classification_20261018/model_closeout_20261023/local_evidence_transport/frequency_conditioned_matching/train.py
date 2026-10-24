"""Train bounded content/frequency-conditioned matching on frozen local readers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
UP=ROOT.parent/'regional_utility_pilot/upstream'
sys.path.insert(0,str(ROOT))
from match_readout import unit,sinkhorn,pair_score,REGIONS
from train_readout import Head,metrics,exact_j
SEEDS=(17,29,43)
ARMS=('content','frequency')
CHECKPOINTS=(0,10,20,40,60,80,100)
BATCH=644

class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(48,8),nn.GELU(),nn.Linear(8,32))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,q,s):
        return self.net(torch.cat([q+s,(q-s).abs()],-1)).tanh()

class Reader:
    def __init__(self,arm,seed,features_required=True):
        self.table=pd.read_csv(ROOT/'paired_scale_readout/table.csv',dtype={'mmsi':str})
        ids=self.table.table_index.to_numpy(int);self.ids=ids
        original=np.load(ROOT/'tokens.npz');pr=np.load(ROOT/'projection_layer1.npz')
        x=unit((original['layer1'][ids]-pr['mean'])@pr['components'].T).astype(np.float32)
        mass=original['mass'][ids];x[mass<=0]=0
        old=np.load(ROOT/'scores_layer1.npz')
        inv=np.full(len(old['refs']),-1);inv[ids]=np.arange(len(ids))
        refs=inv[old['refs'][ids]];assert (refs>=0).all()
        if arm=='frequency':
            f=np.load(HERE/'features.npz');np.testing.assert_array_equal(f['ids'],ids)
            e=f['e_ship'].copy()
            # Retain orientation strength but remove absolute image-angle labels outside the OBB.
            exterior=f['e_image'][:,:,8:].reshape(len(ids),2,12,2,3,4)
            e[:,:,8:]=np.sort(exterior,axis=-1).reshape(len(ids),2,12,24)
        else:e=x[...,:24].copy()
        tr=np.flatnonzero(self.table.role.eq('utility_train'))
        values=e[tr][mass[tr]>0]
        mu=values.mean(0);sd=values.std(0);active=sd>1e-6;sd=np.where(active,sd,1.)
        e=np.where(active,(e-mu)/sd,0).astype(np.float32);e[mass<=0]=0
        self.scaler=dict(mu=mu.tolist(),scale=sd.tolist(),active=active.tolist())
        self.x=torch.tensor(x,device='cuda');self.e=torch.tensor(e,device='cuda')
        self.mass=torch.tensor(mass,device='cuda');self.refs=torch.tensor(refs,device='cuda')
        self.oldlocal=torch.tensor(old['independent'][ids],device='cuda')
        self.base=np.load(UP/f'strategies_seed{seed}.npz')['logits'][ids,2].astype(np.float32)
        self.strong=np.load(ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][ids].astype(np.float32)
        z=self.base-self.base.mean(-1,keepdims=True)
        rival=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
        self.context=torch.tensor(np.stack([z,rival,z-rival],-1),device='cuda')
        self.anchor=torch.tensor(self.strong,device='cuda')
        ck=torch.load(ROOT/'models'/f'layer1_independent_{seed}.pt',map_location='cpu',weights_only=False)
        self.head=Head().cuda().eval().requires_grad_(False);self.head.load_state_dict(ck['state'])
        self.mu=torch.tensor(ck['mu'],device='cuda');self.sd=torch.tensor(ck['scale'],device='cuda')
        self.active=torch.tensor(ck['active'],device='cuda')
        with torch.no_grad():self.old_delta=self.delta(self.oldlocal,torch.arange(len(ids),device='cuda'))
        error=float(abs(torch.tensor(self.base,device='cuda')+self.old_delta-self.anchor).max())
        assert error<3e-5,('oldhead replay',error)
        self.oldhead_replay=error

    def delta(self,local,idx):
        raw=torch.cat([local.reshape(len(idx),8,12),self.context[idx]],-1)
        return self.head(torch.where(self.active,(raw-self.mu)/self.sd,0.))

    def forward(self,idx,gate,diagnostic=False):
        idx=torch.as_tensor(idx,device='cuda');ref=self.refs[idx]
        q,s=self.x[idx],self.x[ref];qe,se=self.e[idx],self.e[ref]
        a,b=self.mass[idx],self.mass[ref]
        groups=[];diagnostics=[]
        for r in REGIONS:
            qi=q[:,:,r][:,None,None,:,:,None,:];sj=s[...,r,:][...,None,:,:]
            product=qi*sj
            eq=qe[:,:,r][:,None,None,:,:,None,:];es=se[...,r,:][...,None,:,:]
            weight=gate(eq,es)
            raw=product.sum(-1);change=.25*(product*weight).sum(-1);sim=raw+change
            qa=a[:,None,None,:,r];sa=b[...,r]
            t,_,_,_=sinkhorn(sim,qa,sa)
            groups.append(pair_score(t,sim,qa,sa))
            if diagnostic:
                with torch.no_grad():
                    t0,_,_,_=sinkhorn(raw,qa,sa)
                    keep=(qa>0)[...,None]&(sa>0)[...,None,:]
                    diagnostics.append(torch.stack([(t-t0).abs().sum((-1,-2)),
                        (change*keep).abs().sum((-1,-2))/keep.sum((-1,-2)).clamp_min(1),
                        (t*change).sum((-1,-2))],-1))
        local=torch.stack(groups,-2).mean(2)
        score=self.anchor[idx]+self.delta(local,idx)-self.old_delta[idx]
        if diagnostic:return score,local,torch.stack(diagnostics,-2).mean(2)
        return score

    @torch.no_grad()
    def predict(self,idx,gate,diagnostic=False):
        out=[];locals_=[];details=[]
        for first in range(0,len(idx),BATCH):
            result=self.forward(idx[first:first+BATCH],gate,diagnostic)
            if diagnostic:
                score,local,detail=result;out.append(score.cpu().numpy());locals_.append(local.cpu().numpy());details.append(detail.cpu().numpy())
            else:out.append(result.cpu().numpy())
        if diagnostic:return np.concatenate(out),np.concatenate(locals_),np.concatenate(details)
        return np.concatenate(out)

def checks(reader):
    torch.manual_seed(17);model=Gate().cuda()
    tr=np.flatnonzero(reader.table.role.eq('utility_train'));idx=tr[:3]
    score,local,_=reader.predict(idx,model,True)
    replay=float(abs(score-reader.strong[idx]).max());assert replay<3e-5,replay
    local_error=float(abs(local-reader.oldlocal[idx].cpu().numpy()).max());assert local_error<2e-5,local_error
    y=torch.tensor(reader.table.iloc[idx].class_id.to_numpy(),device='cuda')
    loss=nn.functional.cross_entropy(reader.forward(idx,model),y);loss.backward()
    gradient=model.net[-1].weight.grad;flat=int(gradient.abs().argmax());auto=float(gradient.flatten()[flat]);assert abs(auto)>1e-7,auto
    parameter=model.net[-1].weight
    values=[];eps=.01
    with torch.no_grad():
        for sign in (1,-1):
            parameter.flatten()[flat]=sign*eps
            values.append(float(nn.functional.cross_entropy(reader.forward(idx,model),y)))
        parameter.flatten()[flat]=0
    numeric=(values[0]-values[1])/(2*eps)
    assert abs(numeric-auto)<max(1e-4,abs(auto)*.08),(auto,numeric)
    # Symmetric conditioning must not depend on which side is called query.
    u=torch.randn(2,3,24,device='cuda');v=torch.randn(2,3,24,device='cuda')
    with torch.no_grad():
        model.net[-1].weight.normal_(0,.1)
        symmetry=float(abs(model(u,v)-model(v,u)).max());assert symmetry==0
    torch.cuda.synchronize()
    return dict(initial_replay=replay,local_replay=local_error,oldhead_replay=reader.oldhead_replay,
        gradient_auto=auto,gradient_finite_difference=numeric,condition_symmetry=symmetry,parameters=sum(p.numel() for p in model.parameters()))

def fit(reader,arm,seed):
    start=time.perf_counter();table=reader.table
    tr=np.flatnonzero(table.role.eq('utility_train'));va=np.flatnonzero(table.role.eq('selection'))
    y=table.class_id.to_numpy(int);yt=torch.tensor(y,device='cuda')
    counts=np.bincount(y[tr],minlength=8)
    weights=.5/len(tr)+.5/(8*counts[y[tr]])
    wt=torch.tensor(weights,dtype=torch.float32,device='cuda')
    torch.manual_seed(seed);model=Gate().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    best=exact_j(y[va],reader.strong[va].argmax(1));chosen=0
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    curve=[dict(seed=seed,arm=arm,step=0,train_loss=None,**metrics(y[va],reader.strong[va]))]
    for step in range(1,101):
        opt.zero_grad(set_to_none=True);total=0.
        for first in range(0,len(tr),BATCH):
            batch=tr[first:first+BATCH]
            score=reader.forward(batch,model)
            loss=(nn.functional.cross_entropy(score,yt[batch],reduction='none')*wt[first:first+BATCH]).sum()
            assert torch.isfinite(loss)
            loss.backward();total+=float(loss.detach())
        opt.step()
        if step in CHECKPOINTS:
            s=reader.predict(va,model);m=metrics(y[va],s);j=exact_j(y[va],s.argmax(1))
            curve.append(dict(seed=seed,arm=arm,step=step,train_loss=total,**m))
            if j>best:
                best=j;chosen=step;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print('FIT',seed,arm,step,'train',round(total,5),'selection_J',round(m['J'],5),'seconds',round(time.perf_counter()-start,1),flush=True)
    model.load_state_dict(state)
    receipt=dict(seed=seed,arm=arm,selected_step=chosen,parameters=680,steps=100,batch=BATCH,seconds=time.perf_counter()-start)
    torch.save(dict(state=state,scaler=reader.scaler,receipt=receipt),HERE/'models'/f'{arm}_{seed}.pt')
    (HERE/'results'/f'{arm}_{seed}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    pd.DataFrame(curve).to_csv(HERE/'results'/f'curve_{arm}_{seed}.csv',index=False)
    return receipt

def main(check_only=False):
    start=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    (HERE/'models').mkdir(exist_ok=True);(HERE/'results').mkdir(exist_ok=True)
    reader=Reader('content',17);verification=checks(reader)
    (HERE/'checks.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');print('CHECKS',verification,flush=True)
    if check_only:return
    del reader;torch.cuda.empty_cache()
    receipts=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed)
            receipts.append(fit(reader,arm,seed))
            del reader;torch.cuda.empty_cache()
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    # Freeze every model/source selection before loading evaluation labels into metrics.
    results=[]
    for seed in SEEDS:
        for arm in ARMS:
            reader=Reader(arm,seed);model=Gate().cuda()
            ck=torch.load(HERE/'models'/f'{arm}_{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['state'])
            idx=np.arange(len(reader.ids));score,local,diagnostic=reader.predict(idx,model,True)
            np.savez_compressed(HERE/'results'/f'{arm}_{seed}.npz',score=score,local=local,diagnostic=diagnostic,ids=reader.ids)
            for role in ('utility_train','selection','evaluation'):
                take=np.flatnonzero(reader.table.role.eq(role));y=reader.table.iloc[take].class_id.to_numpy(int)
                results.append(dict(seed=seed,arm=arm,role=role,**metrics(y,score[take])))
                if arm=='content':
                    results.append(dict(seed=seed,arm='strong',role=role,**metrics(y,reader.strong[take])))
                    results.append(dict(seed=seed,arm='full',role=role,**metrics(y,reader.base[take])))
            del reader,model;torch.cuda.empty_cache()
    pd.DataFrame(results).to_csv(HERE/'metrics.csv',index=False)
    (HERE/'complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,fits=6,checks=verification),indent=2),encoding='utf-8')
    print(pd.DataFrame(results).query("role=='evaluation'").groupby('arm')[['OA','BA','J']].mean().to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();main(a.check_only)
