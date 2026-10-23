"""Two fixed-budget standard readouts of existing polarization evidence."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'2')
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import run_polarization as ref
parent=ref.parent
ARMS=['free_shared','static_pol']


class Head(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        if arm=='free_shared':
            self.net=nn.Sequential(nn.Linear(11,16),nn.GELU(),nn.Linear(16,1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.coefficient=nn.Parameter(torch.zeros(8,2))

    def forward(self,x,carrier):
        raw=self.net(x).squeeze(-1) if self.arm=='free_shared' else (self.coefficient.tanh()*carrier).sum(-1)
        return raw-raw.mean(1,keepdim=True)


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=Head(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck,model


def main():
    tick=time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    (HERE/'models').mkdir(exist_ok=True)
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    valid=cache['valid'].astype(bool)
    y=table.class_id.to_numpy();yy=torch.tensor(y,dtype=torch.long)
    selected=[];curves=[];costs=[];checks=[]
    for fold in range(3):
        d,osum,isum=parent.load_fold(fold,table,cache)
        outer,oof=ref.load_pol(fold,table,cache,d)
        tr,va=d['train'],d['validation']
        for seed in ref.SEEDS:
            base,train_base,_=parent.calibrated(fold,seed,d,osum,isum)
            x,carrier,mu,scale=ref.inputs(train_base[tr],oof[tr],valid[tr],'pol_preserved')
            xv,cv,_,_=ref.inputs(base[va],outer[va],valid[va],'pol_preserved',mu,scale)
            b,bv=[torch.tensor(v,dtype=torch.float32) for v in [train_base[tr],base[va]]]
            for arm in ARMS:
                start=time.perf_counter();torch.manual_seed(seed)
                model=Head(arm)
                parameters=sum(p.numel() for p in model.parameters())
                assert parameters=={'free_shared':209,'static_pol':16}[arm]
                with torch.no_grad():
                    assert not model(xv,cv).any()
                best=parent.frozen_state(model);epoch_best=0
                best_j=parent.exact_j(y[va],bv.numpy().argmax(1))
                meta=dict(fold=fold,seed=seed,arm=arm)
                trace=[dict(**meta,epoch=0,train_loss=np.nan,**parent.assessment(y[va],bv.numpy().astype(float)))]
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                for epoch in range(1,201):
                    opt.zero_grad()
                    loss=parent.joint_loss(b+model(x,carrier),yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    if epoch%5:
                        continue
                    with torch.no_grad():
                        score=(bv+model(xv,cv)).numpy().astype(float)
                    j=parent.exact_j(y[va],score.argmax(1))
                    trace.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**parent.assessment(y[va],score)))
                    if j>best_j:
                        best_j,epoch_best,best=j,epoch,parent.frozen_state(model)
                path=HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta,state=best,mu=mu,scale=scale,epoch=epoch_best,parameters=parameters),path)
                ck,replay=restore(path)
                with torch.inference_mode():
                    delta=replay(xv,cv).numpy().astype(float)
                replay_j=parent.exact_j(y[va],(bv.numpy()+delta).argmax(1))
                assert replay_j==best_j
                assert parent.exact_j(y[va],(base[va]+delta).argmax(1))==best_j
                selected.append(dict(**meta,epoch=epoch_best,parameters=parameters,validation_J=float(best_j),checkpoint=str(path)))
                curves.extend(trace)
                costs.append(dict(**meta,steps=200,seconds=time.perf_counter()-start))
                checks.append(dict(**meta,zero_head_error=0,validation_replay_J=float(replay_j)))
                print(fold,seed,arm,'epoch',epoch_best,'J',float(best_j),flush=True)
    for name,rows in [('selections',selected),('selection_curves',curves),('costs',costs),('checks',checks)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=len(selected),test_evaluated=False)),encoding='utf-8')
    reference=pd.read_csv(HERE.parent/'predictions.csv')
    key=['fold','seed','role','instance_id']
    reference=reference[reference.arm.eq('pol_preserved')].set_index(key)
    frames=[];replays=[]
    for fold in range(3):
        d,osum,isum=parent.load_fold(fold,table,cache)
        outer,_=ref.load_pol(fold,table,cache,d)
        for seed in ref.SEEDS:
            base=parent.calibrated(fold,seed,d,osum,isum)[0]
            for arm in ARMS:
                ck,model=restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation','test']:
                    ix=d[role]
                    x,carrier,_,_=ref.inputs(base[ix],outer[ix],valid[ix],'pol_preserved',ck['mu'],ck['scale'])
                    with torch.inference_mode():
                        delta=model(x,carrier).numpy().astype(float)
                    score=base[ix]+delta
                    assert np.isfinite(score).all()
                    g=table.iloc[ix][['instance_id','class_id','class_name','mmsi','product','port','pass_group']].copy()
                    for k,v in dict(fold=fold,seed=seed,arm=arm,role=role,epoch=ck['epoch']).items():
                        g[k]=v
                    g['valid_vv'],g['valid_vh']=valid[ix].T
                    g['prediction'],g['base_prediction']=score.argmax(1),base[ix].argmax(1)
                    g['net']=(g.prediction==g.class_id).astype(int)-(g.base_prediction==g.class_id).astype(int)
                    g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                    for c in range(8):
                        g[f's{c}'],g[f'b{c}'],g[f'delta{c}']=score[:,c],base[ix,c],delta[:,c]
                    expected=reference.loc[pd.MultiIndex.from_frame(g[key])]
                    error=float(abs(base[ix]-expected[[f'b{c}' for c in range(8)]].to_numpy()).max())
                    assert error<2e-6 and np.array_equal(g.base_prediction,expected.base_prediction)
                    replays.append(dict(fold=fold,seed=seed,arm=arm,role=role,error=error))
                    frames.append(g)
    p=pd.concat(frames,ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    pooled=[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        if role=='test':
            assert len(g)==4744 and g.instance_id.is_unique
        before=parent.assessment(g.class_id.to_numpy(),g[[f'b{c}' for c in range(8)]].to_numpy())
        after=parent.assessment(g.class_id.to_numpy(),g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role,arm=arm,seed=seed,unique_observations=g.instance_id.nunique(),
            net=int(g.net.sum()),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),**after,
            delta_OA=after['OA']-before['OA'],delta_BA=after['BA']-before['BA']))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    pd.DataFrame(replays).to_csv(HERE/'baseline_replays.csv',index=False)
    summary=dict(heads=len(selected),steps=sum(c['steps'] for c in costs),
        train_seconds=sum(c['seconds'] for c in costs),total_seconds=time.perf_counter()-tick,
        reader_fits=0,extractions=0,zero_epoch_heads=sum(s['epoch']==0 for s in selected))
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled).to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
