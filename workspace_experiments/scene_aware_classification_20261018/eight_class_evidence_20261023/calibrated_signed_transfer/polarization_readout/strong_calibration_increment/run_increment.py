"""Frozen full-model score calibration followed by two residual readouts."""
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'full_model_integration'))
import run_integration as previous
parent,ref,OLD=previous.parent,previous.ref,previous.OLD
ARMS=['local','extra_score']
SEED_PAIRS=previous.SEED_PAIRS


def model_for(arm):
    return previous.model_for('free_real' if arm=='local' else 'score_only')


def inputs(arm,base,moments,valid,mu=None,scale=None):
    return previous.inputs('free_real' if arm=='local' else 'score_only',base,moments,valid,mu,scale)


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=model_for(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck,model


def calibrated(fold,oldseed,seed,table,d):
    raw,raw_oof=previous.load_old(fold,oldseed,table,d)
    path=previous.HERE/'models'/f'fold{fold}_score_only_seed{seed}.pt'
    ck,model=previous.restore(path)
    assert not any(p.requires_grad for p in model.parameters())
    base,train_base=np.zeros_like(raw),np.zeros_like(raw)
    # Match the original evaluation batch, retaining its exact frozen baseline.
    for role in ['train','validation','test']:
        ix=d[role]
        s=raw_oof[ix] if role=='train' else raw[ix]
        x,_,_,_=previous.inputs('score_only',s,None,None,ck['mu'],ck['scale'])
        with torch.inference_mode():
            value=s+model(x,None).numpy().astype(float)
        if role=='train':train_base[ix]=value
        else:base[ix]=value
    return base,train_base,ck['epoch']


def main():
    started=time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    valid=cache['valid'].astype(bool)
    y=table.class_id.to_numpy();yy=torch.tensor(y,dtype=torch.long)
    reference=pd.read_csv(previous.HERE/'predictions.csv')
    reference=reference[reference.arm.eq('score_only')].set_index(['fold','seed','role','instance_id'])
    selected,curves,costs,checks,baseline_checks=[],[],[],[],[]
    for fold in range(3):
        d,_,_=parent.load_fold(fold,table,cache)
        outer,oof=ref.load_pol(fold,table,cache,d)
        tr,va=d['train'],d['validation']
        for oldseed,seed in SEED_PAIRS:
            base,train_base,tepoch=calibrated(fold,oldseed,seed,table,d)
            # Baseline score reproduction uses no target labels or model selection.
            for role in ['validation','test']:
                ix=d[role]
                keys=pd.MultiIndex.from_arrays([[fold]*len(ix),[seed]*len(ix),[role]*len(ix),table.iloc[ix].instance_id])
                expected=reference.loc[keys,[f's{c}' for c in range(8)]].to_numpy()
                error=float(np.abs(base[ix]-expected).max())
                assert error<1e-12
                np.testing.assert_array_equal(base[ix].argmax(1),expected.argmax(1))
                baseline_checks.append(dict(fold=fold,oldseed=oldseed,seed=seed,role=role,error=error,top1_changes=0))
            for arm in ARMS:
                tick=time.perf_counter()
                x,carrier,mu,scale=inputs(arm,train_base[tr],oof[tr],valid[tr])
                xv,cv,_,_=inputs(arm,base[va],outer[va],valid[va],mu,scale)
                b=torch.tensor(train_base[tr],dtype=torch.float32)
                torch.manual_seed(seed)
                model=model_for(arm)
                params=sum(p.numel() for p in model.parameters())
                assert params=={'local':209,'extra_score':280}[arm]
                with torch.no_grad():zero=model(xv,cv).numpy()
                assert not zero.any() and np.array_equal(base[va]+zero,base[va])
                best,chosen=parent.frozen_state(model),0
                best_j=parent.exact_j(y[va],base[va].argmax(1))
                meta=dict(fold=fold,oldseed=oldseed,seed=seed,arm=arm)
                trace=[dict(**meta,epoch=0,train_loss=np.nan,**parent.assessment(y[va],base[va]))]
                opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
                for epoch in range(1,201):
                    opt.zero_grad()
                    loss=parent.joint_loss(b+model(x,carrier),yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    if epoch%5:continue
                    with torch.no_grad():delta=model(xv,cv).numpy().astype(float)
                    score=base[va]+delta
                    j=parent.exact_j(y[va],score.argmax(1))
                    trace.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**parent.assessment(y[va],score)))
                    if j>best_j:best_j,chosen,best=j,epoch,parent.frozen_state(model)
                path=HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta,state=best,mu=mu,scale=scale,epoch=chosen,parameters=params,
                    frozen_T_epoch=tepoch,calibration_is_oof=False),path)
                ck,replay=restore(path)
                with torch.inference_mode():delta=replay(xv,cv).numpy().astype(float)
                assert parent.exact_j(y[va],(base[va]+delta).argmax(1))==best_j
                selected.append(dict(**meta,epoch=chosen,frozen_T_epoch=tepoch,parameters=params,
                    validation_J=float(best_j),checkpoint=str(path)))
                curves.extend(trace)
                checks.append(dict(**meta,zero_head_error=0,selected_replay_J=float(best_j),T_frozen=True))
                costs.append(dict(**meta,steps=200,seconds=time.perf_counter()-tick))
                print(fold,seed,arm,'epoch',chosen,'J',float(best_j),flush=True)
    for name,rows in [('selections',selected),('selection_curves',curves),('checks',checks),
                      ('costs',costs),('baseline_replays',baseline_checks)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    assert len(selected)==18
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=18,test_evaluated=False,
        baseline_scores_replayed=True,test_labels_used_for_selection=False)),encoding='utf-8')
    frames=[]
    for fold in range(3):
        d,_,_=parent.load_fold(fold,table,cache)
        outer,_=ref.load_pol(fold,table,cache,d)
        for oldseed,seed in SEED_PAIRS:
            base,_,_=calibrated(fold,oldseed,seed,table,d)
            for arm in ARMS:
                ck,model=restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation','test']:
                    ix=d[role]
                    x,carrier,_,_=inputs(arm,base[ix],outer[ix],valid[ix],ck['mu'],ck['scale'])
                    with torch.inference_mode():delta=model(x,carrier).numpy().astype(float)
                    score=base[ix]+delta
                    assert np.isfinite(score).all()
                    g=table.iloc[ix][['instance_id','obs_id','class_id','class_name','mmsi','product','port','pass_group']].copy()
                    for k,v in dict(fold=fold,oldseed=oldseed,seed=seed,arm=arm,role=role,epoch=ck['epoch'],T_epoch=ck['frozen_T_epoch']).items():g[k]=v
                    g['valid_vv'],g['valid_vh']=valid[ix].T
                    g['prediction'],g['base_prediction']=score.argmax(1),base[ix].argmax(1)
                    g['net']=(g.prediction==g.class_id).astype(int)-(g.base_prediction==g.class_id).astype(int)
                    g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                    br,sr=base[ix].copy(),score.copy()
                    br[np.arange(len(ix)),y[ix]]=sr[np.arange(len(ix)),y[ix]]=-np.inf
                    g['old_rival'],g['final_rival'],g['final_winner']=br.argmax(1),sr.argmax(1),score.argmax(1)
                    for c in range(8):g[f'b{c}'],g[f's{c}'],g[f'delta{c}']=base[ix,c],score[:,c],delta[:,c]
                    frames.append(g)
    p=pd.concat(frames,ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    pooled=[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        if role=='test':assert len(g)==4744 and g.instance_id.is_unique
        before=parent.assessment(g.class_id.to_numpy(),g[[f'b{c}' for c in range(8)]].to_numpy())
        after=parent.assessment(g.class_id.to_numpy(),g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role,arm=arm,seed=seed,oldseed=int(g.oldseed.iloc[0]),unique_observations=g.instance_id.nunique(),
            net=int(g.net.sum()),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),**after,
            **{f'base_{k}':v for k,v in before.items() if k!='n'},delta_OA=after['OA']-before['OA'],
            delta_BA=after['BA']-before['BA'],delta_NLL=after['NLL']-before['NLL'],delta_Brier=after['Brier']-before['Brier']))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    summary=dict(heads=18,steps=3600,calibrator_fits=0,reader_fits=0,extractions=0,
        train_seconds=sum(r['seconds'] for r in costs),total_seconds=time.perf_counter()-started,
        zero_epoch_heads=sum(r['epoch']==0 for r in selected),baseline_max_error=max(r['error'] for r in baseline_checks),
        baseline_top1_changes=0,calibration_is_oof=False,test_rows_per_seed=4744)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled)[['role','arm','seed','net','OA','BA','delta_OA','delta_BA','NLL','Brier']].to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
