"""Can fixed class offsets reproduce apparent conditional evidence gains?"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax,logsumexp

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
sys.path.insert(0,str(PARENT))
from run_anchored import EXP,weights as old_weights


def weights(f,ix,c):
    part=f.iloc[ix]
    return len(ix)/(c*part.groupby('mmsi').mmsi.transform('size').to_numpy()*part.groupby('class_id').mmsi.transform('nunique').to_numpy())


def risk(y,score,c):
    loss=logsumexp(score,axis=1)-score[np.arange(len(y)),y]
    return float(np.mean([loss[y==k].mean() for k in range(c)]))


def metrics(y,score,base,c):
    pred=score.argmax(1)
    p=softmax(score,axis=1)
    nll=logsumexp(score,axis=1)-score[np.arange(len(y)),y]
    brier=((p-np.eye(c)[y])**2).sum(1)
    return dict(OA=float((pred==y).mean()*100),BA=float(np.mean([(pred[y==k]==k).mean() for k in range(c)])*100),
        NLL=float(nll.mean()),balanced_NLL=risk(y,score,c),Brier=float(brier.mean()),
        net=int((pred==y).sum()-(base.argmax(1)==y).sum()))


def main():
    started=time.perf_counter()
    torch.set_num_threads(2)
    allrows,records,curves=[] ,[],[]
    for cohort in ['old','new','eight']:
        root=PARENT if cohort=='old' else PARENT/'partition_replication'
        if cohort=='eight':root=PARENT.parent/'eight_class_evidence_20261023/baseline'
        f=pd.read_csv(EXP/'readout/targets.csv' if cohort!='eight' else root.parent/'cohort.csv',dtype={'mmsi':str})
        y=f.class_id.to_numpy()
        c=len(np.unique(y))
        for fold in range(3 if cohort=='eight' else 6):
            d=dict(np.load(root/f'fold{fold}_inputs.npz'))
            train,val,test=[d[k] for k in ['train','validation','test']]
            w=weights(f,train,c)
            if c==3:np.testing.assert_allclose(w,old_weights(y[train],f.iloc[train].mmsi.to_numpy()))
            x=torch.tensor(d['oof'][train],dtype=torch.float32)
            yy=torch.tensor(y[train],dtype=torch.long)
            ww=torch.tensor(w,dtype=torch.float32)
            for arm in ['bias_only','scale_bias']:
                b=torch.nn.Parameter(torch.zeros(c))
                t=torch.nn.Parameter(torch.zeros(()))
                opt=torch.optim.AdamW([b] if arm=='bias_only' else [b,t],lr=.003,weight_decay=.01)
                best=(risk(y[val],d['base'][val],c),0,np.zeros(c),1.)
                for step in range(1,201):
                    opt.zero_grad()
                    scale=1+.5*torch.tanh(t) if arm=='scale_bias' else 1.
                    logits=scale*x+b-b.mean()
                    loss=(torch.nn.functional.cross_entropy(logits,yy,reduction='none')*ww).mean()
                    loss.backward()
                    opt.step()
                    if step%5==0:
                        bc=(b-b.mean()).detach().numpy().copy()
                        sc=float((1+.5*torch.tanh(t)).detach()) if arm=='scale_bias' else 1.
                        rr=risk(y[val],sc*d['base'][val]+bc,c)
                        curves.append(dict(cohort=cohort,fold=fold,arm=arm,step=step,loss=float(loss.detach()),validation_NLL=rr))
                        if rr<best[0]-1e-6:best=(rr,step,bc,sc)
                rr,step,bc,sc=best
                score=d['base'][test]*sc+bc
                records.append(dict(cohort=cohort,fold=fold,arm=arm,step=step,scale=sc,source_NLL=rr,**metrics(y[test],score,d['base'][test],c)))
                np.savez_compressed(HERE/f'{cohort}_fold{fold}_{arm}.npz',bias=bc,scale=sc,step=step)
                for j,i in enumerate(test):
                    allrows.append(dict(cohort=cohort,fold=fold,arm=arm,instance_id=f.iloc[i].instance_id,class_id=int(y[i]),
                        prediction=int(score[j].argmax()),base_prediction=int(d['base'][i].argmax()),
                        **{f's{k}':float(score[j,k]) for k in range(c)},**{f'b{k}':float(d['base'][i,k]) for k in range(c)}))
            print(cohort,fold,'complete',flush=True)
    p=pd.DataFrame(allrows)
    p.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(records).to_csv(HERE/'metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)
    pooled=[]
    for (cohort,arm),part in p.groupby(['cohort','arm']):
        c=8 if cohort=='eight' else 3
        pooled.append(dict(cohort=cohort,arm=arm,**metrics(part.class_id.to_numpy(),part[[f's{k}' for k in range(c)]].to_numpy(),part[[f'b{k}' for k in range(c)]].to_numpy(),c)))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    p['net']=p.prediction.eq(p.class_id).astype(int)-p.base_prediction.eq(p.class_id).astype(int)
    p.groupby(['cohort','arm','class_id']).agg(n=('net','size'),net=('net','sum')).to_csv(HERE/'class_effects.csv')
    (HERE/'summary.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,fits=len(records),deterministic=True,source_only=True,
        selection='source validation balanced NLL, including zero step',test_selection=False),indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)


if __name__=='__main__':main()
