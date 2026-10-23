"""Same residual classifier and source roles for every local matching operator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
sys.dont_write_bytecode=True
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
UP=HERE.parent/'regional_utility_pilot/upstream'
sys.path.insert(0,str(UP))
from pipeline import joint_loss, exact_j, metrics
from match_readout import ARMS
SEEDS=(17,29,43)


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(15,12),nn.GELU(),nn.Linear(12,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,x):
        delta=self.net(x).squeeze(-1)
        return delta-delta.mean(-1,keepdim=True)


def inputs(local,base,train,score_only=False):
    z=base-base.mean(-1,keepdims=True)
    competitor=np.stack([np.delete(z,c,axis=1).max(1) for c in range(8)],1)
    if score_only:
        others=np.stack([np.sort(np.delete(z,c,axis=1),axis=1) for c in range(8)],1)
        raw=np.concatenate([z[...,None],others,z[...,None]-others],-1)
    else:
        raw=np.concatenate([local.reshape(len(local),8,12),z[...,None],competitor[...,None],(z-competitor)[...,None]],-1)
    mu=raw[train].mean((0,1),keepdims=True)
    sd=raw[train].std((0,1),keepdims=True)
    active=sd>1e-6
    sd=np.where(active,sd,1.)
    x=np.where(active,(raw-mu)/sd,0).astype(np.float32)
    return x,mu,sd,active


def main(controls=False,factorized=False):
    tick=time.perf_counter()
    torch.set_num_threads(2)
    table=pd.read_csv(UP/'table.csv',dtype={'mmsi':str})
    tr=np.flatnonzero(table.role.eq('utility_train'))
    va=np.flatnonzero(table.role.eq('selection'))
    te=np.flatnonzero(table.role.eq('evaluation'))
    for a,b in [(tr,va),(tr,te),(va,te)]:
        for col in ['mmsi','pass_group']:
            assert not set(table.iloc[a][col])&set(table.iloc[b][col])
    y=torch.tensor(table.iloc[tr].class_id.to_numpy())
    yv=table.iloc[va].class_id.to_numpy()
    models=HERE/'models';models.mkdir(exist_ok=True)
    results=HERE/'results';results.mkdir(exist_ok=True)
    receipts=[]; allcurves=[]; evaluation=[]; baseline=[]
    for seed in SEEDS:
        strategies=np.load(UP/f'strategies_seed{seed}.npz')['logits']
        base=strategies[:,2].astype(np.float32)
        # Frozen full regional chain; do not choose a better baseline on evaluation.
        for k,name in enumerate(['base','body','full']):
            for role,idx in [('selection',va),('evaluation',te)]:
                baseline.append(dict(seed=seed,arm=name,role=role,**metrics(table.iloc[idx].class_id.to_numpy(),strategies[idx,k])))
        layers=('layer1',) if factorized else (('none','stem','layer1') if controls else ('stem','layer1'))
        for layer in layers:
            file=f'scores_{layer}_factorized.npz' if factorized else f'scores_{layer}.npz'
            data=np.load(HERE/file) if layer!='none' else None
            arms=('factorized',) if factorized else ((('score_only',) if layer=='none' else ('body_match',)) if controls else ARMS)
            for arm in arms:
                start=time.perf_counter()
                local=None if data is None else data['anchored' if arm=='body_match' else arm].copy()
                if arm=='body_match':
                    local[:,:,:,1:]=0
                x,mu,sd,active=inputs(local,base,tr,score_only=arm=='score_only')
                xt,xv=torch.tensor(x[tr]),torch.tensor(x[va])
                bt=torch.tensor(base[tr])
                torch.manual_seed(seed)
                model=Head()
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                best=exact_j(yv,base[va].argmax(1))
                selected=0
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
                curves=[dict(seed=seed,layer=layer,arm=arm,step=0,train_loss=None,**metrics(yv,base[va]))]
                for step in range(1,201):
                    opt.zero_grad(set_to_none=True)
                    loss=joint_loss(bt+model(xt),y)
                    assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    if step%5==0:
                        with torch.no_grad():
                            sv=base[va]+model(xv).numpy()
                        j=exact_j(yv,sv.argmax(1))
                        curves.append(dict(seed=seed,layer=layer,arm=arm,step=step,train_loss=float(loss.detach()),**metrics(yv,sv)))
                        if j>best:
                            best=j;selected=step
                            state={k:v.detach().clone() for k,v in model.state_dict().items()}
                model.load_state_dict(state)
                path=models/f'{layer}_{arm}_{seed}.pt'
                torch.save(dict(state=state,mu=mu,scale=sd,active=active,selected_step=selected),path)
                with torch.no_grad():
                    delta=np.concatenate([model(torch.tensor(x[i:i+256])).numpy() for i in range(0,len(x),256)])
                score=base+delta
                # Freeze the choice before using evaluation labels.
                receipt=dict(seed=seed,layer=layer,arm=arm,selected_step=selected,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()))
                (results/f'{layer}_{arm}_{seed}.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
                np.savez_compressed(results/f'{layer}_{arm}_{seed}.npz',score=score,delta=delta)
                for role,idx in [('utility_train',tr),('selection',va),('evaluation',te)]:
                    evaluation.append(dict(**receipt,role=role,**metrics(table.iloc[idx].class_id.to_numpy(),score[idx])))
                receipts.append(receipt);allcurves+=curves
                print(layer,arm,seed,'step',selected,'eval',evaluation[-1]['OA'],evaluation[-1]['BA'],flush=True)
    prefix='factorized_' if factorized else ('controls_' if controls else '')
    pd.DataFrame(receipts).to_csv(HERE/f'{prefix}training_receipts.csv',index=False)
    pd.DataFrame(allcurves).to_csv(HERE/f'{prefix}curves.csv',index=False)
    pd.DataFrame(evaluation).to_csv(HERE/f'{prefix}metrics.csv',index=False)
    pd.DataFrame(baseline).to_csv(HERE/f'{prefix}baseline_metrics.csv',index=False)
    (HERE/f'{prefix}training_complete.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick,models=len(receipts),seeds=SEEDS,roles=dict(train=len(tr),selection=len(va),evaluation=len(te))),indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--controls',action='store_true')
    parser.add_argument('--factorized',action='store_true')
    args=parser.parse_args()
    main(args.controls,args.factorized)
