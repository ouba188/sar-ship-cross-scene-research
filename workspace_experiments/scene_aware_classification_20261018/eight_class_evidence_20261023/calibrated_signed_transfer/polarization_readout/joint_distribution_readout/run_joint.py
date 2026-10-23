"""Matched information, joint versus separable class-coordinate readout."""
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
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'strong_calibration_increment'))
import run_increment as strong
parent, ref, OLD = strong.parent, strong.ref, strong.OLD
ARMS = ['joint', 'additive']
MEAN_COLUMNS = [0, 1, 2, 7, 8, 13, 14]
LOCAL_COLUMNS = [0, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]


def branch(dim, width):
    net = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class Head(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == 'joint':
            self.net = branch(15, 16)
        else:
            self.mean = branch(7, 12)
            self.local = branch(11, 12)

    def forward(self, x):
        if self.arm == 'joint':
            value = self.net(x).squeeze(-1)
        else:
            value = (self.mean(x[..., MEAN_COLUMNS]) +
                     self.local(x[..., LOCAL_COLUMNS])).squeeze(-1)
        return value - value.mean(1, keepdim=True)


def inputs(base, means, moments, valid, mu=None, scale=None):
    cols = [base - base.mean(1, keepdims=True)]
    for pol in range(2):
        for value in [means[:, pol], moments[:, pol, :, 0], moments[:, pol, :, 1]]:
            cols.extend([value, value - value.mean(1, keepdims=True)])
    cols.extend([np.broadcast_to(valid[:, p, None], base.shape) for p in range(2)])
    raw = np.stack(cols, axis=-1)
    assert raw.shape == (len(base), 8, 15) and np.isfinite(raw).all()
    if mu is None:
        mu = raw.mean((0, 1), keepdims=True)
        scale = np.maximum(raw.std((0, 1), keepdims=True), 1e-8)
    return torch.tensor((raw - mu) / scale, dtype=torch.float32), mu, scale


def load_features(fold, table, cache, d):
    outer, oof = ref.load_pol(fold, table, cache, d)
    # Use the same reader assembly for the two cached information types.
    means_cache = dict(cache)
    means_cache['moments_pol'] = np.repeat(cache['mean_pol'][..., None], 2, axis=-1)
    outer_m, oof_m = ref.load_pol(fold, table, means_cache, d)
    np.testing.assert_array_equal(outer_m[..., 0], outer_m[..., 1])
    np.testing.assert_array_equal(oof_m[..., 0], oof_m[..., 1])
    return outer_m[..., 0], oof_m[..., 0], outer, oof


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = Head(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def main():
    started = time.perf_counter()
    assert (HERE / 'PROTOCOL_ZH.md').exists()
    (HERE / 'models').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    table = parent.load_table()
    cache = dict(np.load(parent.HERE / 'signed_features.npz'))
    np.testing.assert_array_equal(cache['instance_ids'], table.instance_id)
    valid = cache['valid'].astype(bool)
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    previous = pd.read_csv(strong.HERE / 'predictions.csv')
    previous = previous[previous.arm.eq('local')].set_index(['fold', 'seed', 'role', 'instance_id'])
    selections, curves, checks, costs = [], [], [], []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        mean, mean_oof, moments, moments_oof = load_features(fold, table, cache, d)
        tr, va = d['train'], d['validation']
        for oldseed, seed in strong.SEED_PAIRS:
            base, train_base, tepoch = strong.calibrated(fold, oldseed, seed, table, d)
            for role in ['validation', 'test']:
                ix = d[role]
                keys = pd.MultiIndex.from_arrays([[fold]*len(ix), [seed]*len(ix),
                    [role]*len(ix), table.iloc[ix].instance_id])
                expected = previous.loc[keys, [f'b{c}' for c in range(8)]].to_numpy()
                assert np.abs(base[ix] - expected).max() < 1e-12
            x, mu, scale = inputs(train_base[tr], mean_oof[tr], moments_oof[tr], valid[tr])
            xv, _, _ = inputs(base[va], mean[va], moments[va], valid[va], mu, scale)
            b = torch.tensor(train_base[tr], dtype=torch.float32)
            for arm in ARMS:
                tick = time.perf_counter()
                torch.manual_seed(seed)
                model = Head(arm)
                count = sum(p.numel() for p in model.parameters())
                assert count == {'joint': 273, 'additive': 266}[arm]
                with torch.no_grad():
                    assert not model(xv).numpy().any()
                state, epoch_best = parent.frozen_state(model), 0
                best_j = parent.exact_j(y[va], base[va].argmax(1))
                meta = dict(fold=fold, oldseed=oldseed, seed=seed, arm=arm)
                curves.append(dict(**meta, epoch=0, train_loss=np.nan,
                    **parent.assessment(y[va], base[va])))
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = parent.joint_loss(b + model(x), yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        s = base[va] + model(xv).numpy().astype(float)
                    j = parent.exact_j(y[va], s.argmax(1))
                    curves.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()),
                        **parent.assessment(y[va], s)))
                    if j > best_j:
                        best_j, epoch_best, state = j, epoch, parent.frozen_state(model)
                path = HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta, state=state, mu=mu, scale=scale, epoch=epoch_best,
                    parameters=count, frozen_T_epoch=tepoch, calibration_is_oof=False), path)
                ck, replay = restore(path)
                with torch.inference_mode():
                    s = base[va] + replay(xv).numpy().astype(float)
                assert parent.exact_j(y[va], s.argmax(1)) == best_j
                selections.append(dict(**meta, epoch=epoch_best, parameters=count,
                    validation_J=float(best_j), checkpoint=str(path)))
                checks.append(dict(**meta, zero_residual=True, source_scaler_n=len(tr),
                    baseline_max_error=float(np.abs(base[d['test']] - previous.loc[
                        pd.MultiIndex.from_arrays([[fold]*len(d['test']), [seed]*len(d['test']),
                        ['test']*len(d['test']), table.iloc[d['test']].instance_id]),
                        [f'b{c}' for c in range(8)]].to_numpy()).max()), selected_replay=True))
                costs.append(dict(**meta, steps=200, seconds=time.perf_counter()-tick))
                print(fold, seed, arm, epoch_best, float(best_j), flush=True)
    for name, rows in [('selections', selections), ('selection_curves', curves),
                       ('checks', checks), ('costs', costs)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    assert len(selections) == 18
    (HERE / 'selection_frozen.json').write_text(json.dumps(dict(heads=18,
        test_labels_used_for_selection=False, test_evaluated=False)), encoding='utf-8')
    frames = []
    for fold in range(3):
        d, _, _ = parent.load_fold(fold, table, cache)
        mean, _, moments, _ = load_features(fold, table, cache, d)
        for oldseed, seed in strong.SEED_PAIRS:
            base, _, tepoch = strong.calibrated(fold, oldseed, seed, table, d)
            for arm in ARMS:
                ck, model = restore(HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation', 'test']:
                    ix = d[role]
                    x, _, _ = inputs(base[ix], mean[ix], moments[ix], valid[ix], ck['mu'], ck['scale'])
                    with torch.inference_mode():
                        delta = model(x).numpy().astype(float)
                    score = base[ix] + delta
                    g = table.iloc[ix][['instance_id', 'obs_id', 'class_id', 'class_name',
                        'mmsi', 'product', 'port', 'pass_group']].copy()
                    for k, v in dict(fold=fold, oldseed=oldseed, seed=seed, arm=arm,
                            role=role, epoch=ck['epoch'], T_epoch=tepoch).items():
                        g[k] = v
                    g['valid_vv'], g['valid_vh'] = valid[ix].T
                    g['prediction'], g['base_prediction'] = score.argmax(1), base[ix].argmax(1)
                    g['net'] = (g.prediction == g.class_id).astype(int) - (g.base_prediction == g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)],
                        ['rescued', 'harmed', 'CC'], default='WW')
                    br, sr = base[ix].copy(), score.copy()
                    br[np.arange(len(ix)), y[ix]] = sr[np.arange(len(ix)), y[ix]] = -np.inf
                    g['old_rival'], g['final_rival'], g['final_winner'] = br.argmax(1), sr.argmax(1), score.argmax(1)
                    for c in range(8):
                        g[f'b{c}'], g[f's{c}'], g[f'delta{c}'] = base[ix, c], score[:, c], delta[:, c]
                    frames.append(g)
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE / 'predictions.csv', index=False)
    pooled = []
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        if role == 'test':
            assert len(g) == 4744 and g.instance_id.is_unique
        before = parent.assessment(g.class_id.to_numpy(), g[[f'b{c}' for c in range(8)]].to_numpy())
        after = parent.assessment(g.class_id.to_numpy(), g[[f's{c}' for c in range(8)]].to_numpy())
        pooled.append(dict(role=role, arm=arm, seed=seed, net=int(g.net.sum()), **after,
            **{f'base_{k}': v for k, v in before.items() if k != 'n'},
            **{f'delta_{k}': after[k]-before[k] for k in ['OA', 'BA', 'NLL', 'Brier']}))
    pd.DataFrame(pooled).to_csv(HERE / 'pooled.csv', index=False)
    (HERE / 'summary.json').write_text(json.dumps(dict(heads=18, steps=3600,
        train_seconds=sum(v['seconds'] for v in costs), total_seconds=time.perf_counter()-started,
        zero_epoch_heads=sum(v['epoch']==0 for v in selections), extractions=0,
        calibration_is_oof=False), indent=2), encoding='utf-8')
    print(pd.DataFrame(pooled)[['role', 'arm', 'seed', 'net', 'OA', 'BA', 'delta_OA', 'delta_BA']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
