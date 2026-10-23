"""Direct residual controls on frozen old A+D, without an intermediate C."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '2')
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'standard_controls'))
import run_controls as controls
ref, parent = controls.ref, controls.parent
OLD = parent.PARENT.parents[1]/'same_vessel_gallery_20261019'
ARMS = ['free_real', 'free_aggregate', 'score_only']
SEED_PAIRS = [(0, 17), (1, 29), (2, 43)]


def model_for(arm):
    return parent.Head('score_only') if arm == 'score_only' else controls.Head('free_shared')


def inputs(arm, base, moments, valid, mu=None, scale=None):
    if arm != 'score_only':
        return ref.inputs(base, moments, valid,
            'aggregate_repeated' if arm == 'free_aggregate' else 'pol_preserved', mu, scale)
    raw = base-base.mean(1, keepdims=True)
    if mu is None:
        mu, scale = raw.mean(0), raw.std(0).clip(1e-8)
    return torch.tensor((raw-mu)/scale, dtype=torch.float32), None, mu, scale


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = model_for(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def load_old(fold, oldseed, table, d):
    old = pd.read_csv(OLD/'conditional_evidence_pilot/cohort.csv', dtype={'mmsi':str})
    mapping = old.reset_index().set_index('obs_id')['index'].loc[table.obs_id].to_numpy()
    np.testing.assert_array_equal(old.iloc[mapping].instance_id, table.instance_id)
    np.testing.assert_array_equal(old.iloc[mapping].class_id, table.class_id)
    with np.load(OLD/'geometry_evidence_gate_20261021'/f'inputs_f{fold}_s{oldseed}.npz') as z:
        full = z['base']+z['residual']
    base = full[mapping].astype(np.float64)
    with np.load(OLD/'held_vessel_gate_20261022'/f'oof_f{fold}.npz') as z:
        rows, assignment = z['rows'], z['inner']
        source = (z['base'][oldseed]+z['residual'][oldseed]).astype(np.float64)
    np.testing.assert_array_equal(old.iloc[rows].instance_id, table.iloc[d['train']].instance_id)
    for k in range(3):
        assert not set(old.iloc[rows[assignment==k]].mmsi)&set(old.iloc[rows[assignment!=k]].mmsi)
    train_base = np.zeros_like(base)
    train_base[d['train']] = source
    return base, train_base


def main():
    started = time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = parent.load_table()
    cache = dict(np.load(parent.HERE/'signed_features.npz'))
    valid = cache['valid'].astype(bool)
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    selected, curves, checks, costs = [], [], [], []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        outer, oof = ref.load_pol(fold, table, cache, d)
        tr, va = d['train'], d['validation']
        for oldseed, seed in SEED_PAIRS:
            base, train_base = load_old(fold, oldseed, table, d)
            b = torch.tensor(train_base[tr], dtype=torch.float32)
            bv = torch.tensor(base[va], dtype=torch.float32)
            initial_free = None
            for arm in ARMS:
                tick = time.perf_counter()
                x, carrier, mu, scale = inputs(arm, train_base[tr], oof[tr], valid[tr])
                xv, cv, _, _ = inputs(arm, base[va], outer[va], valid[va], mu, scale)
                torch.manual_seed(seed)
                model = model_for(arm)
                params = sum(p.numel() for p in model.parameters())
                assert params == (280 if arm == 'score_only' else 209)
                if arm == 'free_real':
                    initial_free = parent.frozen_state(model)
                if arm == 'free_aggregate':
                    for k, v in model.state_dict().items():
                        assert torch.equal(v, initial_free[k])
                with torch.no_grad():
                    zero = model(xv, cv).numpy()
                assert not zero.any() and np.array_equal(base[va]+zero, base[va])
                best, chosen = parent.frozen_state(model), 0
                best_j = parent.exact_j(y[va], base[va].argmax(1))
                meta = dict(fold=fold, oldseed=oldseed, seed=seed, arm=arm)
                trace = [dict(**meta, epoch=0, train_loss=np.nan, **parent.assessment(y[va],base[va]))]
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = parent.joint_loss(b+model(x,carrier), yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch%5:
                        continue
                    with torch.no_grad():
                        delta = model(xv,cv).numpy().astype(float)
                    score = base[va]+delta
                    j = parent.exact_j(y[va], score.argmax(1))
                    trace.append(dict(**meta,epoch=epoch,train_loss=float(loss.detach()),**parent.assessment(y[va],score)))
                    if j > best_j:
                        best_j, chosen, best = j, epoch, parent.frozen_state(model)
                path = HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta,state=best,mu=mu,scale=scale,epoch=chosen,parameters=params),path)
                ck, replay = restore(path)
                with torch.inference_mode():
                    delta = replay(xv,cv).numpy().astype(float)
                replay_j = parent.exact_j(y[va],(base[va]+delta).argmax(1))
                assert replay_j == best_j
                selected.append(dict(**meta,epoch=chosen,parameters=params,validation_J=float(best_j),checkpoint=str(path)))
                checks.append(dict(**meta,zero_head_error=0,selected_replay_J=float(replay_j),shared_initialization=True))
                costs.append(dict(**meta,steps=200,seconds=time.perf_counter()-tick))
                curves.extend(trace)
                print(fold,oldseed,seed,arm,'epoch',chosen,'J',float(best_j),flush=True)
    for name, rows in [('selections',selected),('selection_curves',curves),('checks',checks),('costs',costs)]:
        pd.DataFrame(rows).to_csv(HERE/f'{name}.csv',index=False)
    assert len(selected)==27
    (HERE/'selection_frozen.json').write_text(json.dumps(dict(heads=27,test_evaluated=False)),encoding='utf-8')
    frames, replays = [], []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold,table,cache)
        outer,_ = ref.load_pol(fold,table,cache,d)
        for oldseed,seed in SEED_PAIRS:
            base,_ = load_old(fold,oldseed,table,d)
            with np.load(OLD/'neighborhood_level_split_20261021'/f'scores_f{fold}_separated_s{oldseed}.npz') as z:
                expected = z['scores']
                old = pd.read_csv(OLD/'conditional_evidence_pilot/cohort.csv')
                np.testing.assert_array_equal(old.iloc[z['rows']].instance_id,table.iloc[d['test']].instance_id)
            np.testing.assert_array_equal(base[d['test']],expected)
            for arm in ARMS:
                ck,model = restore(HERE/'models'/f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation','test']:
                    ix=d[role]
                    x,carrier,_,_ = inputs(arm,base[ix],outer[ix],valid[ix],ck['mu'],ck['scale'])
                    with torch.inference_mode():
                        delta=model(x,carrier).numpy().astype(float)
                    score=base[ix]+delta
                    assert np.isfinite(score).all()
                    g=table.iloc[ix][['instance_id','obs_id','class_id','class_name','mmsi','product','port','pass_group']].copy()
                    for k,v in dict(fold=fold,oldseed=oldseed,seed=seed,arm=arm,role=role,epoch=ck['epoch']).items():g[k]=v
                    g['valid_vv'],g['valid_vh']=valid[ix].T
                    g['prediction'],g['base_prediction']=score.argmax(1),base[ix].argmax(1)
                    g['net']=(g.prediction==g.class_id).astype(int)-(g.base_prediction==g.class_id).astype(int)
                    g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],['rescued','harmed','CC'],default='WW')
                    br,sr=base[ix].copy(),score.copy()
                    br[np.arange(len(ix)),y[ix]]=sr[np.arange(len(ix)),y[ix]]=-np.inf
                    g['old_rival'],g['final_rival'],g['final_winner']=br.argmax(1),sr.argmax(1),score.argmax(1)
                    for c in range(8):g[f'b{c}'],g[f's{c}'],g[f'delta{c}']=base[ix,c],score[:,c],delta[:,c]
                    frames.append(g)
                    replays.append(dict(fold=fold,oldseed=oldseed,seed=seed,arm=arm,role=role,baseline_error=0))
    p=pd.concat(frames,ignore_index=True)
    p.to_csv(HERE/'predictions.csv',index=False)
    pooled=[]
    for (role,arm,seed),g in p.groupby(['role','arm','seed']):
        if role=='test':assert len(g)==4744 and g.instance_id.is_unique
        before=parent.assessment(g.class_id.to_numpy(),g[[f'b{c}' for c in range(8)]].to_numpy())
        after=parent.assessment(g.class_id.to_numpy(),g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role,arm=arm,seed=seed,oldseed=int(g.oldseed.iloc[0]),unique_observations=g.instance_id.nunique(),
            net=int(g.net.sum()),rescued=int(g.net.eq(1).sum()),harmed=int(g.net.eq(-1).sum()),**after,
            **{f'base_{k}':v for k,v in before.items() if k!='n'},
            delta_OA=after['OA']-before['OA'],delta_BA=after['BA']-before['BA']))
    pd.DataFrame(pooled).to_csv(HERE/'pooled.csv',index=False)
    pd.DataFrame(replays).to_csv(HERE/'baseline_replays.csv',index=False)
    summary=dict(heads=27,steps=5400,calibrators=0,reader_fits=0,extractions=0,
        train_seconds=sum(r['seconds'] for r in costs),total_seconds=time.perf_counter()-started,
        zero_epoch_heads=sum(r['epoch']==0 for r in selected),test_rows_per_seed=4744)
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(pooled)[['role','arm','seed','net','OA','BA','delta_OA','delta_BA','NLL','Brier']].to_string(index=False),flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
