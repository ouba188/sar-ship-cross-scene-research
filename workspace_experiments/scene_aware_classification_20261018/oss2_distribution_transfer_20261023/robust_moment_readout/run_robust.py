"""One fixed source-median asinh control; original models remain untouched."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
OSS = HERE.parent
sys.path.insert(0, str(OSS))
import train_heads as original

ARMS = ['local', 'additive']
SEEDS = [17, 29, 43]


def compress(moments, scale):
    s = scale[None, :, None, None]
    value = s*np.arcsinh(moments/s)
    assert np.isfinite(value).all()
    assert np.array_equal(np.sign(value), np.sign(moments))
    assert np.all(np.abs(value[..., 1]) <= value[..., 0]+1e-9)
    assert np.all(np.abs(value) <= np.abs(moments)+1e-12)
    return value


def load_fold(fold, table, cache):
    d, mean, moment, om, od = original.load_fold(fold, table, cache)
    scale = np.median(od[d['train'], :, :, 0], axis=(0, 2))
    assert scale.shape == (2,) and np.isfinite(scale).all() and (scale > 0).all()
    return d, mean, compress(moment, scale), om, compress(od, scale), scale


def train(fold, seed, arm, btr, bva, mtr, mva, rtr, rva, vtr, vva, ytr, yva, scale):
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = original.make_model(arm)
    x, carrier, mu, sd = original.descriptor(arm, btr, mtr, rtr, vtr)
    xv, cv, _, _ = original.descriptor(arm, bva, mva, rva, vva, mu, sd)
    with torch.no_grad():
        assert not original.delta(model, arm, xv, cv).numpy().any()
    count = sum(p.numel() for p in model.parameters())
    assert count == {'local': 209, 'additive': 266}[arm]
    y = torch.tensor(ytr, dtype=torch.long)
    b = torch.tensor(btr, dtype=torch.float32)
    best, chosen = original.parent.frozen_state(model), 0
    best_j = original.parent.exact_j(yva, bva.argmax(1))
    meta = dict(fold=fold, seed=seed, arm=arm)
    curves = [dict(**meta, epoch=0, train_loss=np.nan, **original.parent.assessment(yva, bva))]
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    for epoch in range(1, 201):
        optimizer.zero_grad()
        loss = original.parent.joint_loss(b+original.delta(model, arm, x, carrier), y)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if epoch % 5:
            continue
        with torch.no_grad():
            score = bva+original.delta(model, arm, xv, cv).numpy().astype(float)
        j = original.parent.exact_j(yva, score.argmax(1))
        curves.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()),
                           **original.parent.assessment(yva, score)))
        if j > best_j:
            best_j, chosen, best = j, epoch, original.parent.frozen_state(model)
    path = HERE/'heads'/f'fold{fold}_{arm}_seed{seed}.pt'
    torch.save(dict(**meta, state=best, mu=mu, scale=sd, moment_scale=scale,
        epoch=chosen, parameters=count, calibration_is_oof=False,
        moment_transform='source_median_sigma_times_asinh'), path)
    replay, ck = original.predict(path, bva, mva, rva, vva)
    assert original.parent.exact_j(yva, replay.argmax(1)) == best_j
    assert np.array_equal(ck['moment_scale'], scale)
    selection = dict(**meta, epoch=chosen, parameters=count, validation_J=float(best_j),
        source_rows=len(ytr), validation_rows=len(yva), seconds=time.perf_counter()-started,
        zero_residual_verified=True, checkpoint_replay=True, path=str(path))
    print(fold, seed, arm, chosen, float(best_j), flush=True)
    return selection, curves


def main():
    started = time.perf_counter()
    assert (HERE/'PROTOCOL_ZH.md').exists()
    (HERE/'heads').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = pd.read_csv(OSS/'cohort.csv', dtype={'mmsi': str})
    with np.load(OSS/'moments.npz') as z:
        cache = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(cache['sample_ids'], table.sample_id)
    valid, y = cache['valid'].astype(bool), table.class_id.to_numpy()
    assert valid.all()
    selected, curves, scales = [], [], []
    for fold in range(6):
        d, mean, moment, om, od, scale = load_fold(fold, table, cache)
        tr, va = d['train'], d['validation']
        for p, pol in enumerate(['vv', 'vh']):
            scales.append(dict(fold=fold, pol=pol, source_n=len(tr),
                sigma_values=len(tr)*8, median_sigma=scale[p], formula='s*asinh(m/s)'))
        for seed in SEEDS:
            with np.load(OSS/'calibrated'/f'fold{fold}_seed{seed}.npz') as z:
                np.testing.assert_array_equal(z['sample_ids'], table.sample_id)
                bva, btr = z['base'][va], z['train_base'][tr]
            for arm in ARMS:
                result, trace = train(fold, seed, arm, btr, bva, om[tr], mean[va],
                    od[tr], moment[va], valid[tr], valid[va], y[tr], y[va], scale)
                selected.append(result)
                curves.extend(trace)
    pd.DataFrame(selected).to_csv(HERE/'selections.csv', index=False)
    pd.DataFrame(curves).to_csv(HERE/'selection_curves.csv', index=False)
    pd.DataFrame(scales).to_csv(HERE/'scales.csv', index=False)
    assert len(selected) == 36
    frozen = dict(heads=36, test_labels_used_for_selection=False, test_evaluated=False,
                  selection_seconds=time.perf_counter()-started)
    (HERE/'selection_frozen.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    print('All 36 selections frozen; starting fixed evaluation.', flush=True)
    frames = []
    for fold in range(6):
        d, mean, moment, _, _, scale = load_fold(fold, table, cache)
        for seed in SEEDS:
            with np.load(OSS/'calibrated'/f'fold{fold}_seed{seed}.npz') as z:
                base = z['base']
            for arm in ARMS:
                path = HERE/'heads'/f'fold{fold}_{arm}_seed{seed}.pt'
                for role in ['validation', 'test']:
                    ix = d[role]
                    score, ck = original.predict(path, base[ix], mean[ix], moment[ix], valid[ix])
                    np.testing.assert_array_equal(ck['moment_scale'], scale)
                    if role == 'validation':
                        expected = next(s['validation_J'] for s in selected
                            if s['fold']==fold and s['seed']==seed and s['arm']==arm)
                        assert float(original.parent.exact_j(y[ix], score.argmax(1))) == expected
                    g = table.iloc[ix][['sample_id','class_id','class_name','mmsi','scene_id','pass_group']].copy()
                    g = g.rename(columns={'sample_id':'instance_id','scene_id':'product'})
                    for k,v in dict(fold=fold,seed=seed,arm=arm,role=role,epoch=ck['epoch']).items():g[k]=v
                    g['valid_vv'],g['valid_vh'] = valid[ix].T
                    g['base_prediction'],g['prediction'] = base[ix].argmax(1),score.argmax(1)
                    g['net']=g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
                    g['outcome']=np.select([g.net.eq(1),g.net.eq(-1),g.prediction.eq(g.class_id)],
                                          ['rescued','harmed','CC'],default='WW')
                    for c in range(8):g[f'a{c}'],g[f'b{c}'],g[f's{c}']=d['base'][ix,c],base[ix,c],score[:,c]
                    frames.append(g)
    predictions = pd.concat(frames,ignore_index=True)
    predictions.to_csv(HERE/'predictions.csv',index=False)
    pooled=[]
    for (role,arm,seed),g in predictions.groupby(['role','arm','seed']):
        if role=='test':assert len(g)==2746 and g.instance_id.is_unique
        before=original.parent.assessment(g.class_id.to_numpy(),g[[f'b{c}' for c in range(8)]].to_numpy())
        after=original.parent.assessment(g.class_id.to_numpy(),g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role,arm=arm,seed=seed,net=int(g.net.sum()),**after,
            **{f'base_{k}':v for k,v in before.items() if k!='n'},
            **{f'delta_{k}':after[k]-before[k] for k in ['OA','BA','NLL','Brier']}))
    pooled=pd.DataFrame(pooled)
    pooled.to_csv(HERE/'pooled.csv',index=False)
    old=pd.read_csv(OSS/'predictions.csv')
    old=old[old.arm.isin(ARMS)]
    keys=['role','arm','seed','fold','instance_id']
    paired=predictions.merge(old[keys+['prediction','class_id']+[f'b{c}' for c in range(8)]],
        on=keys,suffixes=('_robust','_original'),validate='one_to_one')
    assert len(paired)==len(predictions)
    for c in range(8):
        np.testing.assert_allclose(paired[f'b{c}_robust'],paired[f'b{c}_original'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(paired.class_id_robust,paired.class_id_original)
    pairs=paired[keys+['class_id_robust','mmsi','pass_group','prediction_robust','prediction_original']].rename(columns={'class_id_robust':'class_id'})
    pairs['net_vs_original']=pairs.prediction_robust.eq(pairs.class_id).astype(int)-pairs.prediction_original.eq(pairs.class_id).astype(int)
    pairs.to_csv(HERE/'paired_original.csv',index=False)
    oldpool=pd.read_csv(OSS/'pooled.csv')
    comparison=pooled.merge(oldpool,on=['role','arm','seed'],suffixes=('_robust','_original'),validate='one_to_one')
    for key in ['net','OA','BA','NLL','Brier']:
        comparison['change_vs_original_'+key]=comparison[key+'_robust']-comparison[key+'_original']
    comparison.to_csv(HERE/'comparison_original.csv',index=False)
    info=dict(heads=36,trained_steps=7200,seeds=SEEDS,arms=ARMS,device='cpu',
        total_seconds=time.perf_counter()-started,fit_seconds=sum(s['seconds'] for s in selected),
        zero_epoch_heads=sum(s['epoch']==0 for s in selected),
        original_A_T_W_unchanged=True,original_90_heads_unchanged=True,
        original_B_max_error=0.,all_selections_before_test=True,calibration_is_oof=False)
    (HERE/'summary.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(pooled[pooled.role.eq('test')][['arm','seed','net','OA','BA','delta_OA','delta_BA']].to_string(index=False),flush=True)
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
