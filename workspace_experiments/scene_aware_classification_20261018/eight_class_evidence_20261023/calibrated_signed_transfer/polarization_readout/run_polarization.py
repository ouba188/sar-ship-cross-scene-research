"""Matched polarization-preserving versus aggregate residual readouts."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(name, '2')
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_training as parent

ARMS = ['pol_preserved', 'aggregate_repeated', 'pol_preserved_capped']
SEEDS = [17, 29, 43]


def organized(moments, valid, arm):
    if arm != 'aggregate_repeated':
        return moments
    mean = moments.sum(1, keepdims=True) / valid.sum(1)[:, None, None, None]
    return mean * valid[:, :, None, None]


def descriptor(base, moments, valid):
    centered = base - base.mean(1, keepdims=True)
    cols = [centered]
    for pol in range(2):
        sigma, kappa = moments[:, pol, :, 0], moments[:, pol, :, 1]
        cols += [sigma, sigma-sigma.mean(1, keepdims=True), kappa, kappa-kappa.mean(1, keepdims=True)]
    cols += [np.broadcast_to(valid[:, p, None], base.shape) for p in range(2)]
    return np.stack(cols, -1)


class PolarizationHead(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        self.net = nn.Sequential(nn.Linear(11, 16), nn.GELU(), nn.Linear(16, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def contributions(self, x, carrier):
        contributions = self.net(x).tanh() * carrier
        if self.arm == 'pol_preserved_capped':
            factor = carrier.sum(-1).abs() / carrier.abs().sum(-1).clamp_min(1e-12)
            contributions = contributions * factor.unsqueeze(-1)
        return contributions

    def forward(self, x, carrier):
        raw = self.contributions(x, carrier).sum(-1)
        return raw - raw.mean(1, keepdim=True)


def load_pol(fold, table, cache, d):
    names = list(cache['reader_names'])
    outer = cache['moments_pol'][names.index(f'fold{fold}_outer')]
    oof = np.zeros_like(outer)
    count = np.zeros(len(table), int)
    for inner in range(4):
        name = f'fold{fold}_inner{inner}'
        with np.load(parent.PARENT / 'baseline/models' / f'{name}.npz') as z:
            held, fit = z['held'], z['fit']
        assert not set(table.iloc[fit].mmsi) & set(table.iloc[held].mmsi)
        oof[held] = cache['moments_pol'][names.index(name), held]
        count[held] += 1
    assert (count[d['train']] == 1).all() and count.sum() == len(d['train'])
    return outer, oof


def inputs(base, moments, valid, arm, mu=None, scale=None):
    moments = organized(moments, valid, arm)
    raw = descriptor(base, moments, valid)
    if mu is None:
        mu, scale = raw.mean((0, 1)), raw.std((0, 1)).clip(1e-8)
    return (torch.tensor((raw-mu)/scale, dtype=torch.float32),
            torch.tensor(moments[..., 1].transpose(0, 2, 1), dtype=torch.float32), mu, scale)


def restore(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = PolarizationHead(ck['arm']).eval().requires_grad_(False)
    model.load_state_dict(ck['state'])
    return ck, model


def structural_checks():
    # Missing VH and opposite signed carriers exercise the two actual failure risks.
    m = np.zeros((2, 2, 8, 2))
    m[0, 0, :, 1], m[0, 1, :, 1] = 1, -1
    m[1, 0, :, 1] = 2
    valid = np.array([[True, True], [True, False]])
    agg = organized(m, valid, 'aggregate_repeated')
    np.testing.assert_array_equal(agg.sum(1), m.sum(1))
    assert not agg[1, 1].any()
    carrier = torch.tensor(m[..., 1].transpose(0, 2, 1), dtype=torch.float32)
    x = torch.zeros((2, 8, 11))
    uncapped, capped = PolarizationHead('pol_preserved'), PolarizationHead('pol_preserved_capped')
    with torch.no_grad():
        for model in [uncapped, capped]:
            model.net[-1].bias[:] = torch.tensor([np.arctanh(.5), -np.arctanh(.5)])
        parts, capped_parts = uncapped.contributions(x, carrier), capped.contributions(x, carrier)
    torch.testing.assert_close(parts[0].sum(-1), torch.ones(8))
    assert not capped_parts[0].any()
    torch.testing.assert_close(parts[1], capped_parts[1])


def main():
    start = time.perf_counter()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    structural_checks()
    (HERE / 'models').mkdir(parents=True, exist_ok=True)
    table = parent.load_table()
    cache = dict(np.load(parent.HERE / 'signed_features.npz'))
    valid = cache['valid'].astype(bool)
    assert valid.any(1).all()
    assert not cache['moments_pol'][:, ~valid].any()
    y = table.class_id.to_numpy()
    yy = torch.tensor(y, dtype=torch.long)
    curves, selected, checks, costs = [], [], [], []
    for fold in range(3):
        d, outer_sum, oof_sum = parent.load_fold(fold, table, cache)
        outer, oof = load_pol(fold, table, cache, d)
        np.testing.assert_allclose(outer.sum(1), outer_sum, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(oof.sum(1), oof_sum, atol=1e-12, rtol=1e-12)
        tr, va = d['train'], d['validation']
        for seed in SEEDS:
            base, train_base, _ = parent.calibrated(fold, seed, d, outer_sum, oof_sum)
            b, bv = [torch.tensor(a, dtype=torch.float32) for a in [train_base[tr], base[va]]]
            for arm in ARMS:
                tick = time.perf_counter()
                x, carrier, mu, scale = inputs(train_base[tr], oof[tr], valid[tr], arm)
                xv, cv, _, _ = inputs(base[va], outer[va], valid[va], arm, mu, scale)
                torch.manual_seed(seed)
                model = PolarizationHead(arm)
                assert sum(p.numel() for p in model.parameters()) == 226
                with torch.no_grad():
                    assert not model(xv, cv).any()
                best, epoch_best = parent.frozen_state(model), 0
                best_j = parent.exact_j(y[va], bv.numpy().argmax(1))
                meta = dict(fold=fold, seed=seed, arm=arm)
                trace = [dict(**meta, epoch=0, train_loss=np.nan, **parent.assessment(y[va], bv.numpy()))]
                opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
                for epoch in range(1, 201):
                    opt.zero_grad()
                    loss = parent.joint_loss(b + model(x, carrier), yy[tr])
                    assert torch.isfinite(loss)
                    loss.backward()
                    opt.step()
                    if epoch % 5:
                        continue
                    with torch.no_grad():
                        scores = (bv + model(xv, cv)).numpy().astype(float)
                    j = parent.exact_j(y[va], scores.argmax(1))
                    trace.append(dict(**meta, epoch=epoch, train_loss=float(loss.detach()), **parent.assessment(y[va], scores)))
                    if j > best_j:
                        best_j, epoch_best, best = j, epoch, parent.frozen_state(model)
                path = HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt'
                torch.save(dict(**meta, state=best, mu=mu, scale=scale, epoch=epoch_best), path)
                ck, saved = restore(path)
                with torch.inference_mode():
                    delta = saved(xv, cv)
                    scores = (bv+delta).numpy()
                    perm = torch.tensor([3, 0, 7, 1, 6, 4, 2, 5])
                    perm_error = float((saved(xv[:, perm], cv[:, perm])[:, torch.argsort(perm)]-delta).abs().max())
                assert parent.exact_j(y[va], scores.argmax(1)) == best_j
                assert perm_error < 2e-6
                curves.extend(trace)
                selected.append(dict(**meta, epoch=epoch_best, parameters=226, validation_J=float(best_j)))
                checks.append(dict(**meta, zero_error=0, permutation_error=perm_error, replay_J=float(best_j)))
                costs.append(dict(**meta, steps=200, seconds=time.perf_counter()-tick))
                print(fold, seed, arm, 'epoch', epoch_best, 'J', round(float(best_j), 6), flush=True)
    for name, rows in [('selection_curves', curves), ('selections', selected), ('checks', checks), ('costs', costs)]:
        pd.DataFrame(rows).to_csv(HERE / f'{name}.csv', index=False)
    (HERE / 'selection_frozen.json').write_text(json.dumps(dict(heads=len(selected), test_evaluated=False)), encoding='utf-8')
    frames, effects = [], []
    reference = pd.read_csv(parent.HERE / 'increment/predictions.csv')
    reference = reference[reference.arm.eq('signed_only')].set_index(['fold', 'seed', 'role', 'instance_id'])
    replays = []
    for fold in range(3):
        d, outer_sum, oof_sum = parent.load_fold(fold, table, cache)
        outer, _ = load_pol(fold, table, cache, d)
        for seed in SEEDS:
            base = parent.calibrated(fold, seed, d, outer_sum, oof_sum)[0]
            for arm in ARMS:
                ck, model = restore(HERE / 'models' / f'fold{fold}_{arm}_seed{seed}.pt')
                for role in ['validation', 'test']:
                    ix = d[role]
                    x, carrier, _, _ = inputs(base[ix], outer[ix], valid[ix], arm, ck['mu'], ck['scale'])
                    with torch.inference_mode():
                        parts = model.contributions(x, carrier).numpy().astype(float)
                    raw_delta = parts.sum(-1)
                    delta = raw_delta-raw_delta.mean(1, keepdims=True)
                    score = base[ix]+delta
                    if role == 'validation':
                        final_j = parent.exact_j(y[ix], score.argmax(1))
                        chosen_j = next(r['validation_J'] for r in selected
                                        if r['fold'] == fold and r['seed'] == seed and r['arm'] == arm)
                        assert abs(float(final_j)-chosen_j) < 1e-15
                    g = table.iloc[ix][['instance_id', 'class_id', 'class_name', 'mmsi', 'product', 'port', 'pass_group']].copy()
                    for key, value in dict(stage='polarization_readout', fold=fold, seed=seed, arm=arm, role=role, epoch=ck['epoch']).items():
                        g[key] = value
                    g['valid_vv'], g['valid_vh'] = valid[ix].T
                    g['prediction'], g['base_prediction'] = score.argmax(1), base[ix].argmax(1)
                    g['net'] = (g.prediction == g.class_id).astype(int)-(g.base_prediction == g.class_id).astype(int)
                    g['outcome'] = np.select([g.net.eq(1), g.net.eq(-1), g.prediction.eq(g.class_id)], ['rescued', 'harmed', 'CC'], default='WW')
                    for c in range(8):
                        g[f's{c}'], g[f'b{c}'] = score[:, c], base[ix, c]
                        g[f'vv_delta{c}'], g[f'vh_delta{c}'] = parts[:, c, 0], parts[:, c, 1]
                    expected = reference.loc[pd.MultiIndex.from_frame(g[['fold', 'seed', 'role', 'instance_id']])]
                    error = float(abs(expected[[f'b{c}' for c in range(8)]].to_numpy()-base[ix]).max())
                    assert error < 2e-6 and np.array_equal(expected.base_prediction, g.base_prediction)
                    replays.append(dict(fold=fold, seed=seed, arm=arm, role=role, baseline_error=error))
                    frames.append(g)
    p = pd.concat(frames, ignore_index=True)
    p.to_csv(HERE / 'predictions.csv', index=False)
    for (role, arm, seed), g in p.groupby(['role', 'arm', 'seed']):
        if role == 'test':
            assert len(g) == 4744 and g.instance_id.is_unique
        effects.append(dict(role=role, arm=arm, seed=seed, net=int(g.net.sum()),
                            rescued=int(g.net.eq(1).sum()), harmed=int(g.net.eq(-1).sum()),
                            **parent.assessment(g.class_id.to_numpy(), g[[f's{c}' for c in range(8)]].to_numpy())))
    pd.DataFrame(effects).to_csv(HERE / 'pooled.csv', index=False)
    pd.DataFrame(replays).to_csv(HERE / 'baseline_replays.csv', index=False)
    summary = dict(heads=len(selected), train_seconds=sum(c['seconds'] for c in costs),
                   total_seconds=time.perf_counter()-start, test_evaluated=True, reader_fits=0, extractions=0)
    (HERE / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(effects).query("role == 'test'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
