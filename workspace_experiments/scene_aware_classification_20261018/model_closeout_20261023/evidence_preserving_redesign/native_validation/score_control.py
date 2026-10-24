"""Predeclared score-only head control; no new image evidence or geometry."""
import time
import json
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import train_native as tr

def raw_scores(scores):
    centered=scores-scores.mean(1,keepdim=True)
    rival=scores[:,None].expand(-1,8,-1).masked_fill(torch.eye(8,dtype=torch.bool)[None],-torch.inf).max(-1).values
    zeros=torch.zeros(len(scores),8,126)
    return torch.cat((zeros,centered[:,:,None],(scores-rival)[:,:,None]),-1)

def main():
    torch.set_num_threads(2)
    data=tr.Data();y=data.table.class_id.to_numpy(int);reader=dict(np.load(tr.HERE/'reader.npz'))
    rows=[];receipts=[]
    for seed in tr.SEEDS:
        tick=time.perf_counter();torch.manual_seed(seed)
        head=tr.Model('moment',reader).head
        base=torch.tensor(data.baselines[seed]);raw=raw_scores(base)
        mu=raw[data.tr].flatten(0,1).mean(0);scale=raw[data.tr].flatten(0,1).std(0,unbiased=False).clamp_min(.05)
        x=(raw-mu)/scale
        opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.01)
        rng=np.random.default_rng(seed);draws=rng.integers(len(data.tr),size=(80,32))
        count=np.bincount(y[data.tr],minlength=8);weight=.5+.5*len(data.tr)/(8*count[y[data.tr]])
        best=tr.metrics(y[data.va],base[data.va].numpy())['J'];selected=0
        state={k:v.detach().clone() for k,v in head.state_dict().items()}
        curve=[]
        for step in range(1,81):
            opt.zero_grad(set_to_none=True);take=draws[step-1];ids=data.tr[take]
            delta=head(x[ids]).squeeze(-1);delta-=delta.mean(1,keepdim=True)
            loss=(F.cross_entropy(base[ids]+delta,torch.tensor(y[ids]),reduction='none')*torch.tensor(weight[take])).mean()
            loss.backward();opt.step()
            if step in tr.CHECKPOINTS:
                with torch.no_grad():
                    delta=head(x[data.va]).squeeze(-1);delta-=delta.mean(1,keepdim=True)
                    score=(base[data.va]+delta).numpy()
                met=tr.metrics(y[data.va],score);curve.append(dict(step=step,loss=float(loss.detach()),**met))
                if met['J']>best+1e-12:
                    best=met['J'];selected=step;state={k:v.detach().clone() for k,v in head.state_dict().items()}
        head.load_state_dict(state)
        with torch.no_grad():
            delta=head(x).squeeze(-1);delta-=delta.mean(1,keepdim=True)
            score=(base+delta).numpy()
        receipt=dict(arm='score_only',seed=seed,selected_step=selected,seconds=time.perf_counter()-tick,nominal_parameters=sum(p.numel() for p in head.parameters()),input_active_dimensions=2)
        receipts.append(receipt)
        torch.save(dict(state=state,mu=mu,scale=scale,receipt=receipt),tr.HERE/'models'/f'score_only_{seed}.pt')
        pd.DataFrame(curve).to_csv(tr.HERE/'results'/f'curve_score_only_{seed}.csv',index=False)
        np.savez_compressed(tr.HERE/'results'/f'score_only_{seed}.npz',score=score,up_index=data.up)
        for role,ids in (('utility_train',data.tr),('selection',data.va),('evaluation',data.te)):
            rows.append(dict(seed=seed,arm='score_only',role=role,**tr.metrics(y[ids],score[ids])))
    pd.DataFrame(rows).to_csv(tr.HERE/'score_control_metrics.csv',index=False)
    (tr.HERE/'score_control_receipts.json').write_text(json.dumps(receipts,indent=2))
    print(pd.DataFrame(rows).query("role=='evaluation'").to_string(index=False),flush=True)

if __name__=='__main__':main()
