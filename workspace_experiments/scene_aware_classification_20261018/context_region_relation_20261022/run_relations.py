"""Matched region/normalization/relation pilots on the frozen scene cohort."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import copy
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'context_relation_pilot_20261022'
sys.path.insert(0,str(OLD))
from run_ports24 import Baseline, weights, risk, metrics

ARMS=['whole_raw','whole_unit','split_unit','attention','conditional','conditional_constant','conditional_sham']
SEEDS=[17,29,43]


def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)*math.sqrt(x.shape[-1])


def summary_geometry(g,m):
    count=m.sum(1).clip(1)
    mean=(g*m[:,:,None]).sum(1)/count[:,None]
    var=((g-mean[:,None])**2*m[:,:,None]).sum(1)/count[:,None]
    return np.c_[mean,var,m.sum(1)/m.shape[1]]


def shuffled_geometry(g,m):
    out=g.copy()
    for i in range(len(m)):
        ix=np.flatnonzero(m[i])
        if len(ix)>1:
            out[i,ix]=g[i,np.roll(ix,1)]
    return out


class Relation(torch.nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        if arm.startswith('whole') or arm=='split_unit':
            self.bag=torch.nn.Sequential(torch.nn.Linear(79,32),torch.nn.ReLU())
            self.head=torch.nn.Linear(32,3)
        elif arm=='attention':
            self.query=torch.nn.Linear(32,32)
            self.key=torch.nn.Linear(64,32)
            self.value=torch.nn.Linear(64,32)
            self.geo=torch.nn.Sequential(torch.nn.Linear(7,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
            self.head=torch.nn.Linear(47,3)
        else:
            self.query=torch.nn.Linear(32,32)
            self.phi=torch.nn.Sequential(torch.nn.Linear(71,32),torch.nn.Tanh())
            self.head=torch.nn.Linear(47,3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self,q,n,g,stats,m):
        count=m.sum(1).clamp_min(1)
        if self.arm.startswith('whole') or self.arm=='split_unit':
            bag=(n*m[:,:,None]).sum(1)/count[:,None]
            z=self.bag(torch.cat([stats,bag],1))
        elif self.arm=='attention':
            qq=torch.tanh(self.query(q))
            kk=torch.tanh(self.key(n))
            logit=(qq[:,None]*kk).sum(-1)/math.sqrt(32)+self.geo(g).squeeze(-1)
            a=torch.softmax(logit.masked_fill(~m,-1e4),dim=1)*m
            a=a/a.sum(1,keepdim=True).clamp_min(1e-8)
            z=torch.cat([(a[:,:,None]*torch.tanh(self.value(n))).sum(1),stats],1)
        else:
            phi=self.phi(torch.cat([n,g],-1))
            z=(phi*torch.tanh(self.query(q))[:,None]*m[:,:,None]).sum(1)/count[:,None]
            z=torch.cat([z,stats],1)
        return self.head(z)*m.any(1)[:,None]


def load_data():
    t=pd.read_csv(OLD/'ports24/targets.csv',dtype={'mmsi':str})
    f=np.load(OLD/'ports24/features.npz')
    r=np.load(HERE/'readout/regions.npz')
    assert np.array_equal(r['mask'],f['mask'])
    assert np.array_equal(r['instance_ids'].astype(str),t.instance_id.to_numpy())
    return t,f,r


def prepare_fold(fold,t,f,r):
    split=pd.read_csv(OLD/'ports24/splits.csv')
    split=split[split.fold.eq(fold)]
    indices={role:np.flatnonzero(t.instance_id.isin(split[split.role.eq(role)].instance_id)) for role in ['train','validation','test']}
    train=indices['train'];val=indices['validation'];test=indices['test']
    y=t.class_id.to_numpy();ids=t.mmsi.to_numpy();groups=t.pass_group.to_numpy();h=f['target'];m=f['mask']
    diag=pd.read_csv(OLD/'ports24/training_diagnostics.csv')
    alpha=float(diag[diag.fold.eq(fold)].base_alpha.iloc[0])
    base=Baseline(h[train],y[train],ids[train],alpha)
    score=base.predict(h)
    ref=pd.read_csv(OLD/'ports24/predictions.csv')
    ref=ref[ref.fold.eq(fold)&ref.arm.eq('A')&ref.seed.eq(17)].set_index('instance_id').loc[t.iloc[test].instance_id]
    error=float(np.abs(score[test]-ref[['s0','s1','s2']].to_numpy()).max())
    assert error<1e-5,error
    oof=np.zeros_like(score);rt=[]
    for group in sorted(set(groups[train])):
        held=train[groups[train]==group]
        fit=train[(groups[train]!=group)&~np.isin(ids[train],ids[held])]
        if set(y[fit])==set(range(3)) and len(fit)>=5:
            oof[held]=Baseline(h[fit],y[fit],ids[fit],alpha).predict(h[held]);rt.extend(held.tolist())
    stats=summary_geometry(f['geometry'],m)
    scaler=StandardScaler().fit(stats[train])
    stats=scaler.transform(stats).astype(np.float32)
    edge_scaler=StandardScaler().fit(f['geometry'][train][m[train]])
    g=edge_scaler.transform(f['geometry'].reshape(-1,7)).reshape(len(t),16,7).astype(np.float32)
    visual={}
    for key in ['whole','fg','bg']:
        z=base.project(r[key].reshape(-1,256)).reshape(len(t),16,32)
        visual[key]=z.astype(np.float32)
    whole=np.concatenate([visual['whole'],visual['whole']],-1)/math.sqrt(2)
    bounded=np.concatenate([unit(visual['whole']),unit(visual['whole'])],-1)/math.sqrt(2)
    regions=np.concatenate([unit(visual['fg']),unit(visual['bg'])],-1)/math.sqrt(2)
    q=unit(base.project(h)).astype(np.float32)
    return dict(**indices,rt=np.asarray(sorted(rt)),base=score,oof=oof,q=q,qconstant=np.broadcast_to(q[train].mean(0),q.shape).copy(),
                whole=whole,whole_unit=bounded,split=regions,g=g,gsham=shuffled_geometry(g,m),stats=stats,m=m,
                base_replay_error=error,base_alpha=alpha)


def tensors_for(arm,d):
    n=d['whole'] if arm=='whole_raw' else d['whole_unit'] if arm=='whole_unit' else d['split']
    q=d['qconstant'] if arm=='conditional_constant' else d['q']
    g=d['gsham'] if arm=='conditional_sham' else d['g']
    return [torch.tensor(x,dtype=torch.bool if key=='m' else torch.float32) for key,x in [('q',q),('n',n),('g',g),('stats',d['stats']),('m',d['m'])]]


def main():
    torch.set_num_threads(2)
    HERE.mkdir(exist_ok=True);(HERE/'models').mkdir(exist_ok=True)
    t,f,r=load_data();y=t.class_id.to_numpy();ids=t.mmsi.to_numpy()
    predictions=[];records=[];curves=[];checks=[]
    started=time.perf_counter()
    for fold in range(t.pass_group.nunique()):
        d=prepare_fold(fold,t,f,r);rt=d['rt'];val=d['validation'];test=d['test']
        np.savez_compressed(HERE/f'fold{fold}_inputs.npz',**d)
        checks.append(dict(fold=fold,base_replay_error=d['base_replay_error'],ntrain=len(d['train']),noof=len(rt),nval=len(val),ntest=len(test)))
        for seed in SEEDS:
            for arm in ARMS:
                torch.manual_seed(seed)
                model=Relation(arm);inputs=tensors_for(arm,d)
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                weight=torch.tensor(weights(y[rt],ids[rt]),dtype=torch.float32)
                truth=torch.tensor(y,dtype=torch.long);oof=torch.tensor(d['oof'],dtype=torch.float32)
                best=copy.deepcopy(model.state_dict());best_risk=risk(y[val],d['base'][val]);best_epoch=0
                start=time.perf_counter()
                for epoch in range(1,201):
                    model.train();opt.zero_grad()
                    logits=oof[rt]+model(*[x[rt] for x in inputs])
                    loss=(torch.nn.functional.cross_entropy(logits,truth[rt],reduction='none')*weight).mean()
                    loss.backward();opt.step()
                    if epoch%5==0:
                        model.eval()
                        with torch.no_grad():
                            pred=d['base'][val]+model(*[x[val] for x in inputs]).numpy()
                        value=risk(y[val],pred)
                        curves.append(dict(fold=fold,seed=seed,arm=arm,epoch=epoch,train_loss=float(loss.detach()),val_nll=value))
                        if value<best_risk-1e-6:
                            best_risk=value;best_epoch=epoch;best=copy.deepcopy(model.state_dict())
                model.load_state_dict(best);model.eval()
                with torch.no_grad():
                    delta=model(*inputs).numpy()
                assert np.array_equal(delta[~d['m'].any(1)],np.zeros_like(delta[~d['m'].any(1)]))
                scores=d['base']+delta
                torch.save(best,HERE/'models'/f'fold{fold}_{arm}_{seed}.pt')
                bp=d['base'][test].argmax(1);pp=scores[test].argmax(1)
                records.append(dict(fold=fold,seed=seed,arm=arm,epoch=best_epoch,val_nll=best_risk,parameters=sum(p.numel() for p in model.parameters()),
                    train_seconds=time.perf_counter()-start,n=len(test),**metrics(y[test],scores[test]),
                    rescued=int(((bp!=y[test])&(pp==y[test])).sum()),harmed=int(((bp==y[test])&(pp!=y[test])).sum())))
                for i in test:
                    predictions.append(dict(fold=fold,seed=seed,arm=arm,instance_id=t.iloc[i].instance_id,class_id=int(y[i]),
                        prediction=int(scores[i].argmax()),base_prediction=int(d['base'][i].argmax()),neighbors=int(d['m'][i].sum()),
                        **{f's{c}':float(scores[i,c]) for c in range(3)},**{f'b{c}':float(d['base'][i,c]) for c in range(3)}))
            print('fold',fold,'seed',seed,'finished',flush=True)
    pd.DataFrame(records).to_csv(HERE/'metrics.csv',index=False)
    p=pd.DataFrame(predictions);p.to_csv(HERE/'predictions.csv',index=False)
    pd.DataFrame(curves).to_csv(HERE/'curves.csv',index=False)
    pd.DataFrame(checks).to_csv(HERE/'baseline_replay.csv',index=False)
    pooled=[]
    for (seed,arm),q in p.groupby(['seed','arm']):
        for scope,part in [('all',q),('context',q[q.neighbors.gt(0)])]:
            pooled.append(dict(seed=seed,arm=arm,scope=scope,n=len(part),**metrics(part.class_id.to_numpy(),part[['s0','s1','s2']].to_numpy()),
                rescued=int(((part.base_prediction!=part.class_id)&(part.prediction==part.class_id)).sum()),
                harmed=int(((part.base_prediction==part.class_id)&(part.prediction!=part.class_id)).sum())))
    pooled=pd.DataFrame(pooled);pooled.to_csv(HERE/'pooled.csv',index=False)
    stats=dict(seconds=time.perf_counter()-started,heads=len(records),targets=len(t),edges=int(f['mask'].sum()),
               context_targets=int(f['mask'].any(1).sum()),sham_effective_targets=int((f['mask'].sum(1)>1).sum()),
               sham_effective_edges=int(f['mask'][f['mask'].sum(1)>1].sum()))
    (HERE/'run_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(pooled[pooled.scope.eq('all')].groupby('arm')[['OA','BA','macro_F1','rescued','harmed']].agg(['mean','std']).to_string(),flush=True)


if __name__=='__main__':
    main()
