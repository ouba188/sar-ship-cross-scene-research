"""Does context gain require per-target neighbor content, beyond presence prior?"""
import json
import time
import numpy as np
import pandas as pd
import torch
from train_expanded import HERE,Relation,inputs_for,SEEDS,weights,risk,metrics

def main():
    torch.set_num_threads(2);t=pd.read_csv(HERE/'readout/targets.csv',dtype={'mmsi':str});y=t.class_id.to_numpy()
    ids=t.mmsi.to_numpy();records=[];rows=[];beg=time.perf_counter()
    for fold in range(6):
        d=dict(np.load(HERE/f'fold{fold}_inputs.npz'));train=d['train'];val=d['validation'];test=d['test']
        x=inputs_for('split_unit',d,'cuda');has=d['m'].any(1);source=train[has[train]]
        node_mean=d['split'][train][d['m'][train]].mean(0)
        constant=[a.clone() for a in x]
        constant[1][:]=torch.as_tensor(node_mean,device='cuda')
        constant[3][:]=torch.as_tensor(d['stats'][source].mean(0),device='cuda')
        yy=torch.tensor(y,device='cuda');oof=torch.tensor(d['oof'],device='cuda',dtype=torch.float32)
        weight=torch.tensor(weights(y[train],ids[train]),device='cuda',dtype=torch.float32)
        for seed in SEEDS:
            fixed=Relation('split_unit').cuda().eval()
            fixed.load_state_dict(torch.load(HERE/'models'/f'fold{fold}_split_unit_{seed}.pt',weights_only=True,map_location='cpu'))
            with torch.no_grad():fixed_score=d['base']+fixed(*constant).cpu().numpy()
            torch.manual_seed(seed);model=Relation('split_unit').cuda()
            opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};br=risk(y[val],d['base'][val]);epoch_best=0
            for epoch in range(1,201):
                opt.zero_grad();score=oof[train]+model(*[a[train] for a in constant])
                loss=(torch.nn.functional.cross_entropy(score,yy[train],reduction='none')*weight).mean();loss.backward();opt.step()
                if epoch%5==0:
                    with torch.no_grad():s=d['base'][val]+model(*[a[val] for a in constant]).cpu().numpy()
                    r=risk(y[val],s)
                    if r<br-1e-6:br=r;epoch_best=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            model.load_state_dict(best)
            with torch.no_grad():fitted_score=d['base']+model(*constant).cpu().numpy()
            torch.save(best,HERE/'models'/f'fold{fold}_context_constant_{seed}.pt')
            for arm,s in [('fixed_split_mean',fixed_score),('trained_context_constant',fitted_score)]:
                bp=d['base'][test].argmax(1);pred=s[test].argmax(1)
                records.append(dict(fold=fold,seed=seed,arm=arm,epoch=epoch_best if arm.startswith('trained') else -1,
                    **metrics(y[test],s[test]),rescued=int(((bp!=y[test])&(pred==y[test])).sum()),harmed=int(((bp==y[test])&(pred!=y[test])).sum())))
                for i in test:rows.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                    prediction=int(s[i].argmax()),base_prediction=int(d['base'][i].argmax()),**{f's{c}':float(s[i,c]) for c in range(3)}))
        print('constant fold',fold,'done',flush=True)
    pd.DataFrame(records).to_csv(HERE/'content_control_metrics.csv',index=False)
    p=pd.DataFrame(rows);p.to_csv(HERE/'content_control_predictions.csv',index=False);pooled=[]
    for (arm,seed),q in p.groupby(['arm','seed']):
        pooled.append(dict(arm=arm,seed=seed,**metrics(q.class_id.to_numpy(),q[['s0','s1','s2']].to_numpy()),
            net=int((q.prediction==q.class_id).sum()-(q.base_prediction==q.class_id).sum())))
    pd.DataFrame(pooled).to_csv(HERE/'content_control_pooled.csv',index=False)
    print(pd.DataFrame(pooled).to_string(index=False));print('seconds',time.perf_counter()-beg)

if __name__=='__main__':main()
