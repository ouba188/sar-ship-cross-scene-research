"""Mean-only auxiliary GAP control on the unchanged frozen strong baseline."""
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
sys.path.insert(0,str(HERE.parent/'strong_calibration_increment'))
import run_increment as strong
previous=strong.previous
parent,ref,OLD=previous.parent,previous.ref,previous.OLD
ARMS=['mean_only']
SEED_PAIRS=previous.SEED_PAIRS


def model_for(arm):
    return strong.model_for('local')


def inputs(arm,base,moments,valid,mu=None,scale=None):
    return strong.inputs('local',base,moments,valid,mu,scale)


def restore(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    model=model_for(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck,model


calibrated=strong.calibrated


def check_means(table, cache):
    means, valid = cache['mean_pol'], cache['valid'].astype(bool)
    assert means.shape == (15, len(table), 2, 8)
    assert valid.shape == (len(table), 2) and valid.any(1).all()
    assert np.isfinite(means).all() and not means[:, ~valid].any()
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    names = list(cache['reader_names'])
    checks = []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        train, test = set(d['train']), set(d['test'])
        count = np.zeros(len(table), int)
        for suffix in ['outer'] + [f'inner{k}' for k in range(4)]:
            name = f'fold{fold}_{suffix}'
            with np.load(parent.PARENT/'baseline/models'/f'{name}.npz') as z:
                fit, intercept = z['fit'], z['intercept']
                if suffix == 'outer':
                    np.testing.assert_array_equal(fit, d['train'])
                    ix, expected = np.arange(len(table)), d['base']
                    held = np.array([], int)
                else:
                    held = z['held']
                    assert set(held) <= train
                    assert not set(table.iloc[fit].mmsi) & set(table.iloc[held].mmsi)
                    count[held] += 1
                    ix, expected = held, d['oof']
            assert set(fit) <= train and not set(fit) & test
            scores = means[names.index(name), ix].sum(1) + intercept
            error = float(np.abs(scores - expected[ix]).max())
            flips = int(np.sum(scores.argmax(1) != expected[ix].argmax(1)))
            assert error < 1e-4 and flips == 0
            checks.append(dict(reader=name, fit_n=len(fit), held_n=len(held),
                checked_n=len(ix), mean_A_error=error, top1_changes=flips,
                query_mmsi_overlap=0, outer_test_fit_overlap=0))
        assert (count[d['train']] == 1).all() and count.sum() == len(train)
    pd.DataFrame(checks).to_csv(HERE/'data_checks.csv', index=False)
    summary = dict(shape=list(means.shape), valid_vv=int(valid[:, 0].sum()),
        valid_vh=int(valid[:, 1].sum()), invalid_value_max=0,
        max_mean_A_error=max(r['mean_A_error'] for r in checks),
        top1_changes=0, source_only_inner=True, instance_ids_aligned=True)
    (HERE/'data_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('preflight', json.dumps(summary), flush=True)


def main():
    started=time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table=parent.load_table()
    cache=dict(np.load(parent.HERE/'signed_features.npz'))
    check_means(table,cache)
    cache['moments_pol']=np.repeat(cache['mean_pol'][...,None],2,axis=-1)
    np.testing.assert_array_equal(cache['moments_pol'][...,0],cache['moments_pol'][...,1])
    valid=cache['valid'].astype(bool)
    y=table.class_id.to_numpy();yy=torch.tensor(y,dtype=torch.long)
    reference=pd.read_csv(strong.HERE/'predictions.csv')
    reference=reference[reference.arm.eq('local')].set_index(['fold','seed','role','instance_id'])
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
                expected=reference.loc[keys,[f'b{c}' for c in range(8)]].to_numpy()
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
                initial=parent.frozen_state(model)
                torch.manual_seed(seed)
                local_initial=strong.model_for('local')
                assert all(torch.equal(value,local_initial.state_dict()[key]) for key,value in initial.items())
                params=sum(p.numel() for p in model.parameters())
                assert params==209
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
                    frozen_T_epoch=tepoch,calibration_is_oof=False,initial_state=initial),path)
                ck,replay=restore(path)
                with torch.inference_mode():delta=replay(xv,cv).numpy().astype(float)
                assert parent.exact_j(y[va],(base[va]+delta).argmax(1))==best_j
                selected.append(dict(**meta,epoch=chosen,frozen_T_epoch=tepoch,parameters=params,
                    validation_J=float(best_j),checkpoint=str(path)))
                curves.extend(trace)
                checks.append(dict(**meta,zero_head_error=0,selected_replay_J=float(best_j),T_frozen=True,initial_state_exact=True))
                costs.append(dict(**meta,steps=200,seconds=time.perf_counter()-tick))
                print(fold,seed,arm,'epoch',chosen,'J',float(best_j),flush=True)
    for name,rows in [('selections',selected),('selection_curves',curves),('checks',checks),
                      ('costs',costs),('baseline_replays',baseline_checks)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    assert len(selected)==9
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=9,test_evaluated=False,
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
    summary=dict(heads=9,steps=1800,calibrator_fits=0,reader_fits=0,extractions=0,
        train_seconds=sum(r['seconds'] for r in costs),total_seconds=time.perf_counter()-started,
        zero_epoch_heads=sum(r['epoch']==0 for r in selected),baseline_max_error=max(r['error'] for r in baseline_checks),
        baseline_top1_changes=0,calibration_is_oof=False,test_rows_per_seed=4744)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled)[['role','arm','seed','net','OA','BA','delta_OA','delta_BA','NLL','Brier']].to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
