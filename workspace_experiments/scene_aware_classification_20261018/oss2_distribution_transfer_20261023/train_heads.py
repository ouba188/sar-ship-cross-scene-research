"""Frozen-method OSS2 replication; all source selections precede test evaluation."""
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
POL = HERE.parent / 'eight_class_evidence_20261023/calibrated_signed_transfer/polarization_readout'
sys.path.insert(0, str(POL / 'joint_distribution_readout'))
import run_joint as joint
strong, parent = joint.strong, joint.parent
SEEDS = [17, 29, 43]
ARMS = ['mean_only', 'local', 'additive', 'extra_score']


def make_model(arm):
    return joint.Head('additive') if arm == 'additive' else strong.model_for(
        'extra_score' if arm in ['calibration', 'extra_score'] else 'local')


def descriptor(arm, base, mean, moment, valid, mu=None, scale=None):
    if arm == 'additive':
        x, mu, scale = joint.inputs(base, mean, moment, valid, mu, scale)
        return x, None, mu, scale
    if arm in ['calibration', 'extra_score']:
        return strong.inputs('extra_score', base, None, None, mu, scale)
    data = np.repeat(mean[..., None], 2, axis=-1) if arm == 'mean_only' else moment
    return strong.inputs('local', base, data, valid, mu, scale)


def delta(model, arm, x, carrier):
    return model(x) if arm == 'additive' else model(x, carrier)


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = make_model(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def train(fold, seed, arm, btr, bva, mtr, mva, rtr, rva, vtr, vva, ytr, yva):
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = make_model(arm)
    x, carrier, mu, scale = descriptor(arm, btr, mtr, rtr, vtr)
    xv, cv, _, _ = descriptor(arm, bva, mva, rva, vva, mu, scale)
    with torch.no_grad():
        assert not delta(model, arm, xv, cv).numpy().any()
    count = sum(p.numel() for p in model.parameters())
    assert count == {'calibration': 280, 'mean_only': 209, 'local': 209,
                     'additive': 266, 'extra_score': 280}[arm]
    yy, bb = torch.tensor(ytr, dtype=torch.long), torch.tensor(btr, dtype=torch.float32)
    best, chosen = parent.frozen_state(model), 0
    best_j = parent.exact_j(yva, bva.argmax(1))
    meta = dict(fold=fold, seed=seed, arm=arm)
    curves = [dict(**meta, epoch=0, train_loss=np.nan, **parent.assessment(yva, bva))]
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    for epoch in range(1, 201):
        opt.zero_grad()
        loss = parent.joint_loss(bb + delta(model, arm, x, carrier), yy)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step()
        if epoch % 5:
            continue
        with torch.no_grad():
            score = bva + delta(model, arm, xv, cv).numpy().astype(float)
        j = parent.exact_j(yva, score.argmax(1))
        curves.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()),
            **parent.assessment(yva, score)))
        if j > best_j:
            best_j, chosen, best = j, epoch, parent.frozen_state(model)
    path = HERE / 'heads' / f'fold{fold}_{arm}_seed{seed}.pt'
    torch.save(dict(**meta, state=best, mu=mu, scale=scale, epoch=chosen,
        parameters=count, calibration_is_oof=False), path)
    ck, model = restore(path)
    with torch.inference_mode():
        replay = bva + delta(model, arm, xv, cv).numpy().astype(float)
    assert parent.exact_j(yva, replay.argmax(1)) == best_j
    selection = dict(**meta, epoch=chosen, parameters=count, validation_J=float(best_j),
        source_rows=len(ytr), validation_rows=len(yva), seconds=time.perf_counter()-started,
        zero_residual_verified=True, checkpoint_replay=True, path=str(path))
    print(fold, seed, arm, chosen, float(best_j), flush=True)
    return selection, curves


def predict(path, base, mean=None, moment=None, valid=None):
    ck, model = restore(path)
    x, carrier, _, _ = descriptor(ck['arm'], base, mean, moment, valid, ck['mu'], ck['scale'])
    with torch.inference_mode():
        value = delta(model, ck['arm'], x, carrier).numpy().astype(float)
    return base + value, ck


def load_fold(fold, table, cache):
    with np.load(HERE / 'readers' / f'fold{fold}_inputs.npz') as z:
        d = {k: z[k] for k in z.files}
    np.testing.assert_array_equal(d['sample_ids'], table.sample_id)
    names = list(cache['reader_names'])
    mean = cache['mean_pol'][names.index(f'fold{fold}_outer')]
    moment = cache['moments_pol'][names.index(f'fold{fold}_outer')]
    om, od = np.zeros_like(mean), np.zeros_like(moment)
    count = np.zeros(len(table), int)
    for inner in range(4):
        name = f'fold{fold}_inner{inner}'
        with np.load(HERE / 'readers/models' / f'{name}.npz') as z:
            ix, fit = z['held'], z['fit']
            assert not set(table.iloc[ix].mmsi) & set(table.iloc[fit].mmsi)
            assert not set(table.iloc[ix].pass_group) & set(table.iloc[fit].pass_group)
        om[ix], od[ix] = cache['mean_pol'][names.index(name), ix], cache['moments_pol'][names.index(name), ix]
        count[ix] += 1
    np.testing.assert_array_equal(np.flatnonzero(count), d['train'])
    assert (count[d['train']] == 1).all()
    return d, mean, moment, om, od


def main():
    started = time.perf_counter()
    assert (HERE / 'MODEL_PROTOCOL_ZH.md').exists()
    (HERE / 'heads').mkdir(exist_ok=True)
    (HERE / 'calibrated').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = pd.read_csv(HERE / 'cohort.csv', dtype={'mmsi': str})
    cache = dict(np.load(HERE / 'moments.npz'))
    np.testing.assert_array_equal(cache['sample_ids'], table.sample_id)
    valid, y = cache['valid'].astype(bool), table.class_id.to_numpy()
    selected, curves = [], []
    for fold in range(6):
        d, mean, moment, om, od = load_fold(fold, table, cache)
        tr, va = d['train'], d['validation']
        for seed in SEEDS:
            result, trace = train(fold, seed, 'calibration', d['oof'][tr], d['base'][va],
                None, None, None, None, None, None, y[tr], y[va])
            selected.append(result)
            curves.extend(trace)
            path = HERE / 'heads' / f'fold{fold}_calibration_seed{seed}.pt'
            bt = np.zeros_like(d['base'])
            bt[tr], _ = predict(path, d['oof'][tr])
            base, _ = predict(path, d['base'])
            np.savez_compressed(HERE / 'calibrated' / f'fold{fold}_seed{seed}.npz',
                sample_ids=table.sample_id.to_numpy(str), base=base, train_base=bt)
    (HERE / 'calibration_selection_frozen.json').write_text(json.dumps(dict(heads=18,
        test_evaluated=False)), encoding='utf-8')
    for fold in range(6):
        d, mean, moment, om, od = load_fold(fold, table, cache)
        tr, va = d['train'], d['validation']
        for seed in SEEDS:
            with np.load(HERE / 'calibrated' / f'fold{fold}_seed{seed}.npz') as z:
                base, bt = z['base'], z['train_base']
            for arm in ARMS:
                result, trace = train(fold, seed, arm, bt[tr], base[va], om[tr], mean[va],
                    od[tr], moment[va], valid[tr], valid[va], y[tr], y[va])
                selected.append(result)
                curves.extend(trace)
    pd.DataFrame(selected).to_csv(HERE / 'selections.csv', index=False)
    pd.DataFrame(curves).to_csv(HERE / 'selection_curves.csv', index=False)
    assert len(selected) == 90
    (HERE / 'selection_frozen.json').write_text(json.dumps(dict(heads=90,
        test_labels_used_for_selection=False, test_evaluated=False)), encoding='utf-8')
    frames = []
    for fold in range(6):
        d, mean, moment, om, od = load_fold(fold, table, cache)
        for seed in SEEDS:
            with np.load(HERE / 'calibrated' / f'fold{fold}_seed{seed}.npz') as z:
                base = z['base']
            for arm in ARMS:
                path = HERE / 'heads' / f'fold{fold}_{arm}_seed{seed}.pt'
                for role in ['validation', 'test']:
                    ix = d[role]
                    score, ck = predict(path, base[ix], mean[ix], moment[ix], valid[ix])
                    g = table.iloc[ix][['sample_id', 'class_id', 'class_name', 'mmsi', 'scene_id', 'pass_group']].copy()
                    g = g.rename(columns={'sample_id': 'instance_id', 'scene_id': 'product'})
                    for k, v in dict(fold=fold, seed=seed, arm=arm, role=role, epoch=ck['epoch']).items():
                        g[k] = v
                    g['valid_vv'], g['valid_vh'] = valid[ix].T
                    g['base_prediction'], g['prediction'] = base[ix].argmax(1), score.argmax(1)
                    g['net'] = g.prediction.eq(g.class_id).astype(int)-g.base_prediction.eq(g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)],
                        ['rescued', 'harmed', 'CC'], default='WW')
                    for c in range(8):
                        g[f'a{c}'], g[f'b{c}'], g[f's{c}'] = d['base'][ix, c], base[ix, c], score[:, c]
                    frames.append(g)
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE / 'predictions.csv', index=False)
    pooled = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        if role == 'test':
            assert len(g) == 2746 and g.instance_id.is_unique
        before = parent.assessment(g.class_id.to_numpy(), g[[f'b{c}' for c in range(8)]].to_numpy())
        after = parent.assessment(g.class_id.to_numpy(), g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role, arm=arm, seed=seed, net=int(g.net.sum()), **after,
            **{f'base_{k}': v for k, v in before.items() if k != 'n'},
            **{f'delta_{k}': after[k]-before[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
    pd.DataFrame(pooled).to_csv(HERE / 'pooled.csv', index=False)
    info = dict(heads=90, steps=18000, seeds=SEEDS, total_seconds=time.perf_counter()-started,
        fit_seconds=sum(r['seconds'] for r in selected),
        zero_epoch_heads=sum(r['epoch']==0 for r in selected), calibration_is_oof=False,
        selection='Frozen before test evaluation; source jointCE/J; repeated historical public dataset, not blind test')
    (HERE / 'training_summary.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled)[['role', 'arm', 'seed', 'net', 'OA', 'BA', 'delta_OA', 'delta_BA']].to_string(index=False), flush=True)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
