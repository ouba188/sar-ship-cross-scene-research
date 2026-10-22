"""OOF-trained shared-class corrections; only source validation selects checkpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
CACHE=HERE
sys.path.insert(0,str(HERE.parent/'contrastive_spatial_pool_20261023'))
from run_pool import EXP,NEW,metrics
ARMS=['linear_distribution','conditional_std','conditional_signed']
SEEDS=[17,29,43]


def descriptors(base,mom):
    centered=base-base.mean(1,keepdims=True)
    sigma,signed=mom[...,0],mom[...,1]
    return np.stack([centered,sigma,sigma-sigma.mean(1,keepdims=True),signed,signed-signed.mean(1,keepdims=True)],axis=-1)


class Correction(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        if arm=='scalar_signed':
            self.coefficient=nn.Parameter(torch.zeros(()))
            return
        self.cols=[0,2,4] if arm=='linear_distribution' else [0,1,2] if arm=='conditional_std' else list(range(5))
        out=2 if arm=='conditional_signed' else 1
        if arm=='linear_distribution':self.net=nn.Linear(3,1,bias=False)
        else:self.net=nn.Sequential(nn.Linear(len(self.cols),16),nn.GELU(),nn.Linear(16,out))
        last=self.net if isinstance(self.net,nn.Linear) else self.net[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:nn.init.zeros_(last.bias)

    def forward(self,x,mom):
        if self.arm=='scalar_signed':
            delta=self.coefficient.tanh()*mom[...,1]
            return delta-delta.mean(1,keepdim=True)
        value=self.net(x[...,self.cols])
        if self.arm=='linear_distribution':delta=value.squeeze(-1)
        elif self.arm=='conditional_std':delta=value.tanh().squeeze(-1)*mom[...,0]
        elif self.arm=='signed_only':delta=value.tanh().squeeze(-1)*mom[...,1]
        else:delta=(value.tanh()*mom).sum(-1)
        return delta-delta.mean(1,keepdim=True)


def balanced_loss(score,y):
    loss=nn.functional.cross_entropy(score,y,reduction='none')
    return torch.stack([loss[y==c].mean() for c in range(3)]).mean()


def main():
    start=time.perf_counter();torch.set_num_threads(2)
    t=pd.read_csv(EXP/'readout/targets.csv',dtype={'mmsi':str})
    y=t.class_id.to_numpy();yy=torch.tensor(y,dtype=torch.long)
    cache=np.load(CACHE/'signed_features.npz')
    np.testing.assert_array_equal(cache['instance_ids'],t.instance_id)
    names=list(cache['reader_names']);moment=cache['moments']
    (HERE/'models').mkdir(exist_ok=True)
    frames=[];selected=[];curves=[];checks=[]
    for fold in range(6):
        d=dict(np.load(NEW/f'fold{fold}_inputs.npz'))
        tr,va,te=[d[k] for k in ['train','validation','test']]
        outer=moment[names.index(f'fold{fold}_outer')]
        oof=np.zeros_like(outer);coverage=np.zeros(len(t),int)
        for j,name in enumerate(names):
            if name.startswith(f'fold{fold}_inner'):
                held=np.load(NEW/'origins'/f'{name}.npz')['held']
                oof[held]=moment[j,held];coverage[held]+=1
        assert (coverage[tr]==1).all() and coverage.sum()==len(tr)
        xx=descriptors(d['oof'],oof);testx=descriptors(d['base'],outer)
        mu=xx[tr].mean((0,1));scale=xx[tr].std((0,1)).clip(1e-8)
        x=torch.tensor((xx-mu)/scale,dtype=torch.float32)
        xv=torch.tensor((testx-mu)/scale,dtype=torch.float32)
        m=torch.tensor(oof,dtype=torch.float32);mv=torch.tensor(outer,dtype=torch.float32)
        b=torch.tensor(d['oof'],dtype=torch.float32);bv=torch.tensor(d['base'],dtype=torch.float32)
        for arm in ARMS:
            for seed in [-1] if arm in ['linear_distribution','scalar_signed'] else SEEDS:
                torch.manual_seed(0 if seed<0 else seed)
                model=Correction(arm)
                with torch.no_grad():
                    assert model(xv[va],mv[va]).abs().max()==0
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best=float(balanced_loss(bv[va],yy[va]));best_epoch=0
                saved={k:v.detach().clone() for k,v in model.state_dict().items()}
                for epoch in range(1,201):
                    model.train();opt.zero_grad()
                    loss=balanced_loss(b[tr]+model(x[tr],m[tr]),yy[tr])
                    loss.backward();opt.step()
                    if epoch%5==0:
                        model.eval()
                        with torch.no_grad():risk=float(balanced_loss(bv[va]+model(xv[va],mv[va]),yy[va]))
                        curves.append(dict(fold=fold,arm=arm,seed=seed,epoch=epoch,train_loss=float(loss.detach()),validation_loss=risk))
                        if risk<best-1e-8:
                            best,best_epoch=risk,epoch
                            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(saved);model.eval()
                with torch.no_grad():
                    delta=model(xv[te],mv[te]).numpy().astype(float)
                    perm=torch.tensor([2,0,1]);inv=torch.argsort(perm)
                    permuted=model(xv[te][:,perm],mv[te][:,perm])[:,inv].numpy()
                    error=float(abs(permuted-delta).max())
                    assert error<2e-6
                score=d['base'][te].astype(float)+delta
                pred=score.argmax(1);bp=d['base'][te].argmax(1)
                g=t.iloc[te][['instance_id','class_id','mmsi','port','pass_group']].copy()
                g['fold']=fold;g['arm']=arm;g['seed']=seed;g['epoch']=best_epoch
                g['prediction']=pred;g['base_prediction']=bp
                g['net']=(pred==y[te]).astype(int)-(bp==y[te]).astype(int)
                g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),pred==y[te]],['rescued','harmed','CC'],default='WW')
                for c in range(3):
                    g[f's{c}']=score[:,c];g[f'b{c}']=d['base'][te,c];g[f'delta{c}']=delta[:,c]
                    g[f'std{c}']=outer[te,c,0];g[f'signed{c}']=outer[te,c,1]
                frames.append(g)
                selected.append(dict(fold=fold,arm=arm,seed=seed,epoch=best_epoch,validation_loss=best,
                    parameters=sum(p.numel() for p in model.parameters()),net=int(g.net.sum()),**metrics(y[te],score)))
                path=HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(state=saved,arm=arm,mu=mu,scale=scale,epoch=best_epoch),path)
                clone=Correction(arm);clone.load_state_dict(torch.load(path,weights_only=False)['state']);clone.eval()
                with torch.no_grad():replay=clone(xv[te],mv[te]).numpy()
                np.testing.assert_allclose(replay,delta,atol=1e-7,rtol=1e-7)
                checks.append(dict(fold=fold,arm=arm,seed=seed,permutation_error=error,replay_error=float(abs(replay-delta).max())))
                print(f'fold={fold} {arm} seed={seed} epoch={best_epoch} net={g.net.sum()}',flush=True)
    p=pd.concat(frames,ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    rows=[]
    for (arm,seed),g in p.groupby(['arm','seed']):
        assert len(g)==len(t) and g.instance_id.is_unique
        rows.append(dict(arm=arm,seed=seed,rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),net=int(g.net.sum()),
                         **metrics(g.class_id.to_numpy(),g[['s0','s1','s2']].to_numpy())))
    pd.DataFrame(rows).to_csv(HERE/'pooled.csv',index=False)
    pd.DataFrame(selected).to_csv(HERE/'selections.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'checks.csv',index=False)
    p.groupby(['arm','seed','class_id']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'class_effects.csv')
    p.groupby(['arm','seed','pass_group']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'pass_effects.csv')
    summary=dict(models=len(selected),seconds=time.perf_counter()-start,observations=len(t),source_only_selection=True,backbone_fits=0)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
