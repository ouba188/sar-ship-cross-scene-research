"""Paired scale matching and source-selected equal-capacity residual heads."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
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
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('scale_extract',HERE/'extract.py')
ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
pixel=ex.pixel
from train_readout import metrics,exact_j,joint_loss
ARMS=('repeat','multiscale','difference','mix')
SEEDS=(17,29,43)

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(51,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        d=self.net(x).squeeze(-1)
        return d-d.mean(-1,keepdim=True)

def main():
    start=time.perf_counter();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    f=pd.read_csv(HERE/'table.csv',dtype={'mmsi':str})
    data=np.load(HERE/'views.npz');y=data['y'];ids=data['ids'];mass=data['mass']
    old=np.load(ROOT/'scores_layer1.npz')
    inv=np.full(len(old['refs']),-1);inv[ids]=np.arange(len(ids))
    refs=inv[old['refs'][ids]];assert (refs>=0).all()
    q,_=np.linalg.qr(np.random.default_rng(1024).normal(size=(4,4)))
    q=q.astype(np.float32)
    np.testing.assert_allclose(q.T@q,np.eye(4),atol=2e-7)
    np.save(HERE/'mix_matrix.npy',q)
    norms=[];local={};replay=None;matching=[]
    for arm in ARMS:
        tick=time.perf_counter()
        if arm=='repeat': v=np.repeat(y[:,:,:1],4,axis=2)
        elif arm=='multiscale': v=y.copy()
        elif arm=='difference':
            v=np.stack([y[:,:,0]-y[:,:,1],y[:,:,1]-y[:,:,2],y[:,:,2]-y[:,:,3],y[:,:,3]],2)
            np.testing.assert_allclose(v.sum(2),y[:,:,0],atol=1e-5)
        else: v=np.einsum('ab,npbtd->npatd',q,y)
        for k in range(4):
            norm=np.linalg.norm(v[:,:,k],axis=-1)
            for role in ('utility_train','selection','evaluation'):
                use=f.role.eq(role).to_numpy()[:,None,None] & (mass>0)
                z=norm[use]
                norms.append(dict(arm=arm,channel=k,role=role,n=len(z),minimum=float(z.min()),p01=float(np.quantile(z,.01)),median=float(np.median(z)),tiny_fraction=float((z<1e-6).mean())))
        channels=[]
        for k in range(1 if arm=='repeat' else 4):
            x=pixel.unit(v[:,:,k]).astype(np.float32)
            channels.append(pixel.independent(x,mass,x[refs],mass[refs]))
        if arm=='repeat':
            replay=float(abs(channels[0]-old['independent'][ids]).max())
            assert replay<2e-5,replay
            channels=channels*4
        local[arm]=np.stack(channels,2)
        np.savez_compressed(HERE/f'local_{arm}.npz',local=local[arm])
        matching.append(dict(arm=arm,seconds=time.perf_counter()-tick))
        print('MATCH',arm,round(matching[-1]['seconds'],1),flush=True)
    pd.DataFrame(norms).to_csv(HERE/'norms.csv',index=False)
    pd.DataFrame(matching).to_csv(HERE/'matching_cost.csv',index=False)
    tr=np.flatnonzero(f.role.eq('utility_train'));va=np.flatnonzero(f.role.eq('selection'));te=np.flatnonzero(f.role.eq('evaluation'))
    yt=torch.tensor(f.iloc[tr].class_id.to_numpy());yv=f.iloc[va].class_id.to_numpy()
    models=HERE/'models';models.mkdir(exist_ok=True)
    results=HERE/'results';results.mkdir(exist_ok=True)
    curves=[];receipts=[];outputs={}
    for seed in SEEDS:
        base=np.load(pixel.UP/f'strategies_seed{seed}.npz')['logits'][ids,2].astype(np.float32)
        outputs[(seed,'full')]=base
        outputs[(seed,'old_independent')]=np.load(ROOT/'results'/f'layer1_independent_{seed}.npz')['score'][ids]
        z=base-base.mean(-1,keepdims=True)
        rival=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
        for arm in ARMS:
            tick=time.perf_counter()
            raw=np.concatenate([local[arm].reshape(len(ids),8,48),z[...,None],rival[...,None],(z-rival)[...,None]],-1)
            mu=raw[tr].mean((0,1),keepdims=True);sd=raw[tr].std((0,1),keepdims=True)
            active=sd>1e-6;sd=np.where(active,sd,1)
            x=np.where(active,(raw-mu)/sd,0).astype(np.float32)
            xt=torch.tensor(x[tr]);xv=torch.tensor(x[va]);bt=torch.tensor(base[tr])
            torch.manual_seed(seed);model=Head();opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            best=exact_j(yv,base[va].argmax(1));selected=0
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
            curves.append(dict(seed=seed,arm=arm,step=0,train_loss=None,**metrics(yv,base[va])))
            for step in range(1,201):
                opt.zero_grad(set_to_none=True);loss=joint_loss(bt+model(xt),yt)
                assert torch.isfinite(loss)
                loss.backward();opt.step()
                if step%5==0:
                    with torch.no_grad(): sv=base[va]+model(xv).numpy()
                    j=exact_j(yv,sv.argmax(1))
                    curves.append(dict(seed=seed,arm=arm,step=step,train_loss=float(loss.detach()),**metrics(yv,sv)))
                    if j>best:
                        best=j;selected=step;state={k:v.detach().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(state)
            torch.save(dict(state=state,mu=mu,scale=sd,active=active,selected_step=selected),models/f'{arm}_{seed}.pt')
            receipt=dict(seed=seed,arm=arm,selected_step=selected,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-tick)
            (results/f'{arm}_{seed}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            receipts.append(receipt)
            with torch.no_grad(): outputs[(seed,arm)]=base+model(torch.tensor(x)).numpy()
            print('FIT',seed,arm,'selected',selected,flush=True)
    # All models and source-selection receipts are frozen before evaluation.
    pd.DataFrame(receipts).to_csv(HERE/'training_receipts.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)
    evaluation=[]
    for (seed,arm),score in outputs.items():
        np.savez_compressed(results/f'{arm}_{seed}.npz',score=score)
        for role,idx in [('utility_train',tr),('selection',va),('evaluation',te)]:
            evaluation.append(dict(seed=seed,arm=arm,role=role,**metrics(f.iloc[idx].class_id.to_numpy(),score[idx])))
    pd.DataFrame(evaluation).to_csv(HERE/'metrics.csv',index=False)
    receipt=dict(seconds=time.perf_counter()-start,raw_match_replay=replay,models=12,parameters=637,queries=len(te),role_counts=f.role.value_counts().to_dict())
    (HERE/'complete.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(pd.DataFrame(evaluation).query("role=='evaluation'").groupby('arm')[['OA','BA','NLL','Brier']].mean().to_string(),flush=True)

if __name__=='__main__':main()
