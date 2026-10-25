"""Grouped gate-level cross-selection on historical source selection predictions."""
import common as common
from common import HERE, PREVIOUS, BASE, LOCAL, SEEDS, Data, metrics, write_json, load
import time
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('constant', 'score', 'score_wide', 'spatial', 'spatial_wide', 'frequency')
COMPARATORS = ('old_full', 'prior_late', 'hybrid_equal', 'hybrid_source', 'base')
STEPS = (0, 10, 20, 40, 60, 80, 100)
DIM = dict(constant=11, score=11, score_wide=11, spatial=41, spatial_wide=41, frequency=185)
WIDTH = dict(score=12, score_wide=161, spatial=12, spatial_wide=51, frequency=12)
A_PASSES = {'S1A_058088_072C54', 'S1A_058168_072F9F', 'S1A_059032_075271',
            'S1A_059738_076B1D', 'S1A_060443_07839A', 'S1A_060465_078475', 'S1C_003745_0077B1'}


def folds(data):
    mask = data.table.iloc[data.va].pass_group.isin(A_PASSES).to_numpy()
    a, b = data.va[mask], data.va[~mask]
    assert (len(a), len(b)) == (432, 180)
    for key in ('mmsi', 'pass_group'):
        assert not set(data.table.iloc[a][key]) & set(data.table.iloc[b][key])
    frame = data.table.iloc[data.va][['obs_id', 'up_index', 'mmsi', 'pass_group', 'class_id', 'class_name']].copy()
    frame['fold'] = np.where(mask, 'A', 'B')
    frame.to_csv(HERE / 'source_folds.csv', index=False)
    return a, b


def normalize(raw, ids):
    mu, sd = raw[ids].mean((0, 1)), raw[ids].std((0, 1))
    scale = np.where(sd > 1e-6, sd, 1)
    value = np.where(sd > 1e-6, (raw-mu)/scale, 0).astype(np.float32)
    return torch.tensor(value), dict(mu=mu, scale=scale, active=sd > 1e-6)


class Gate(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == 'constant':
            self.logit = nn.Parameter(torch.zeros(2))
        else:
            self.net = nn.Sequential(nn.Linear(DIM[arm], WIDTH[arm]), nn.GELU(), nn.Linear(WIDTH[arm], 2))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, available):
        if self.arm == 'constant':
            logits = self.logit[None].expand(len(x), -1)
        else:
            logits = self.net(x[..., :DIM[self.arm]]).mean(1)
        return logits.sigmoid() * available


def model_for(arm, seed):
    torch.manual_seed(seed)
    return Gate(arm)


def prediction(model, x, z, ids):
    g = model(x[ids], z['available'][ids])
    s = z['base'][ids] + g[:, :1]*z['delta_vv'][ids] + g[:, 1:]*z['delta_vh'][ids]
    return s, g


def inputs(data, seed):
    with np.load(HERE / 'results' / f'features_{seed}.npz') as f:
        np.testing.assert_array_equal(f['up_index'], data.up[:data.n])
        raw = f['raw'].copy()
        z = {k: torch.tensor(f[k].astype(np.float32)) for k in ('base', 'delta_vv', 'delta_vh', 'available')}
    return raw, z


def train_steps(arm, seed, raw, z, y, tr, va, total=100):
    x, scaler = normalize(raw, tr)
    model = model_for(arm, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    count = np.bincount(y[tr], minlength=8)
    assert (count > 0).all()
    wt = torch.tensor(.5/len(tr) + .5/(8*count[y[tr]]), dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.long)
    snapshots, curve = {}, []
    for step in range(total+1):
        if step:
            opt.zero_grad(set_to_none=True)
            score, _ = prediction(model, x, z, tr)
            loss = (F.cross_entropy(score, yt, reduction='none')*wt).sum()
            assert torch.isfinite(loss)
            loss.backward()
            opt.step()
        if step in STEPS or step == total:
            with torch.no_grad():
                s, g = prediction(model, x, z, va)
                snapshots[step] = (s.numpy().copy(), g.numpy().copy())
            curve.append(dict(step=step, train_loss=None if step == 0 else float(loss.detach()),
                              **metrics(y[va], snapshots[step][0])))
    return model, scaler, snapshots, curve


def checks(data, raw, z):
    ids = data.va[:16]
    x, _ = normalize(raw, data.va)
    model = model_for('frequency', 17)
    s, g = prediction(model, x, z, ids)
    expected = z['base'][ids] + .5*z['delta_vv'][ids] + .5*z['delta_vh'][ids]
    error = float((s-expected).abs().max().detach())
    assert error < 2e-6
    y = torch.tensor(data.table.iloc[ids].class_id.to_numpy(), dtype=torch.long)
    F.cross_entropy(s, y).backward()
    grad = model.net[-1].weight.grad
    ij = np.unravel_index(int(grad.abs().argmax()), grad.shape)
    auto = float(grad[ij]); assert abs(auto) > 1e-7
    values = []
    with torch.no_grad():
        for sign in (1, -1):
            model.net[-1].weight[ij] = sign*.001
            values.append(float(F.cross_entropy(prediction(model, x, z, ids)[0], y)))
        model.net[-1].weight[ij] = 0
        gm = model(x, z['available'])
        assert float(gm[~z['available'].bool()].abs().max()) == 0
        assert float(gm.min()) >= 0 and float(gm.max()) <= 1
    numeric = (values[0]-values[1])/.002
    assert abs(auto-numeric) < max(.0003, .05*abs(auto)), (auto, numeric)
    return dict(initial_replay_error=error, gradient_auto=auto, gradient_numeric=numeric,
        invalid_view_weight=0, parameters={a: sum(p.numel() for p in model_for(a, 17).parameters()) for a in ARMS})


def source_space(data, z, seed):
    ids = data.va; y = data.table.iloc[ids].class_id.to_numpy(int)
    base, dv, dh = (z[k][ids].numpy() for k in ('base', 'delta_vv', 'delta_vh'))
    endpoint = np.stack([base+a*dv+b*dh for a in (0., .5, 1.) for b in (0., .5, 1.)], 1)
    correct = endpoint.argmax(-1) == y[:, None]
    baseline = correct[:, 4]
    return dict(seed=seed, n=len(ids), equal_correct=int(baseline.sum()),
        rescue_possible=int((~baseline & correct.any(1)).sum()),
        harm_possible=int((baseline & ~correct.all(1)).sum()),
        interpretation='Source-label action-space diagnostic only, not a deployable selection result.')


def main():
    torch.set_num_threads(2)
    for folder in ('models', 'results', 'analysis'):
        (HERE / folder).mkdir(exist_ok=True)
    data = Data(); a, b = folds(data)
    y = data.table.iloc[:data.n].class_id.to_numpy(int)
    # Never use evaluation labels for a fit, normalization, checkpoint or arm choice.
    receipts, cv_rows, fold_rows, spaces, fixed_rows = [], [], [], [], []
    all_checks = {}
    for seed in SEEDS:
        raw, z = inputs(data, seed)
        all_checks[seed] = checks(data, raw, z)
        spaces.append(source_space(data, z, seed))
        for arm in ARMS:
            tick = time.perf_counter()
            oof = {step: np.zeros((data.n, 8), np.float32) for step in STEPS}
            for fold, tr, va in (('A_to_B', a, b), ('B_to_A', b, a)):
                model, scaler, snapshots, curve = train_steps(arm, seed, raw, z, y, tr, va)
                for step, (s, g) in snapshots.items():
                    oof[step][va] = s
                    fold_rows.append(dict(seed=seed, arm=arm, fold=fold, step=step, **metrics(y[va], s)))
                pd.DataFrame(curve).to_csv(HERE / 'results' / f'curve_{arm}_{seed}_{fold}.csv', index=False)
            selected = 0
            best = common.exact_j(y[data.va], oof[0][data.va].argmax(1))
            for step in STEPS:
                m = metrics(y[data.va], oof[step][data.va])
                j = common.exact_j(y[data.va], oof[step][data.va].argmax(1))
                cv_rows.append(dict(seed=seed, arm=arm, step=step, **m))
                if j > best:
                    selected, best = step, j
            model, scaler, snapshots, curve = train_steps(arm, seed, raw, z, y, data.va, data.va, total=selected)
            receipt = dict(seed=seed, arm=arm, selected_step=selected, source_cv_J=float(best),
                           parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-tick)
            torch.save(dict(state=model.state_dict(), scaler=scaler, receipt=receipt), HERE / 'models' / f'{arm}_{seed}.pt')
            np.savez_compressed(HERE / 'results' / f'source_cv_{arm}_{seed}.npz',
                **{f'score_{step}': oof[step][data.va] for step in STEPS}, up_index=data.up[data.va])
            receipts.append(receipt)
            print('FIT', receipt, flush=True)
        for name, tr, va in (('A_to_B', a, b), ('B_to_A', b, a)):
            alpha, _ = common.previous.select((z['base']+z['delta_vv'])[tr].numpy(),
                (z['base']+z['delta_vh'])[tr].numpy(), y[tr])
            pred = (z['base']+float(alpha)*z['delta_vv']+(1-float(alpha))*z['delta_vh'])[va].numpy()
            fixed_rows.append(dict(seed=seed, fold=name, alpha=float(alpha), **metrics(y[va], pred)))
    pd.DataFrame(receipts).to_csv(HERE / 'training_receipts.csv', index=False)
    pd.DataFrame(cv_rows).to_csv(HERE / 'source_cv.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(HERE / 'source_cv_folds.csv', index=False)
    pd.DataFrame(fixed_rows).to_csv(HERE / 'source_fixed_folds.csv', index=False)
    write_json(HERE / 'source_space.json', spaces)
    write_json(HERE / 'checks.json', all_checks)
    write_json(HERE / 'source_choices.json', dict(choices=receipts, developer_metrics_computed=False,
        limitation='Gate-level grouped cross-selection on historically selected source predictions; not fully untouched OOF.'))
    # All learned states are now frozen. Only now compute development metrics.
    rows, diagnostics = [], []
    for seed in SEEDS:
        raw, z = inputs(data, seed)
        for arm in ARMS:
            model, x, ck = restore(seed, arm, raw)
            with torch.no_grad():
                s, g = prediction(model, x, z, np.arange(data.n))
                score, weight = s.numpy(), g.numpy()
            np.savez_compressed(HERE / 'results' / f'{arm}_{seed}.npz', score=score, weight=weight, up_index=data.up[:data.n])
            for role, ids in (('source_refit', data.va), ('evaluation', data.te)):
                rows.append(dict(seed=seed, arm=arm, role=role, **metrics(y[ids], score[ids])))
            if arm == 'frequency':
                with torch.no_grad():
                    nofreq = x.clone(); nofreq[..., 41:] = 0
                    s0, g0 = prediction(model, nofreq, z, np.arange(data.n))
                    # Source means within availability strata, so missing views do not dilute means.
                    means = np.array([weight[data.va, p][z['available'][data.va, p].numpy() > 0].mean() for p in range(2)])
                    gm = z['available']*torch.tensor(means)[None]
                    sm = z['base']+gm[:, :1]*z['delta_vv']+gm[:, 1:]*z['delta_vh']
                np.savez_compressed(HERE / 'results' / f'diagnostics_{seed}.npz', no_frequency=s0.numpy(),
                    no_frequency_weights=g0.numpy(), mean_weights=sm.numpy(), source_mean=means, up_index=data.up[:data.n])
                for name, diag in (('no_frequency', s0.numpy()), ('mean_weights', sm.numpy())):
                    diagnostics.append(dict(seed=seed, mode=name, **metrics(y[data.te], diag[data.te])))
        for arm, score in common.scores(data, seed).items():
            if arm not in COMPARATORS:
                continue
            np.savez_compressed(HERE / 'results' / f'{arm}_{seed}.npz', score=score, up_index=data.up[:data.n])
            for role, ids in (('source_refit', data.va), ('evaluation', data.te)):
                rows.append(dict(seed=seed, arm=arm, role=role, **metrics(y[ids], score[ids])))
    pd.DataFrame(rows).to_csv(HERE / 'metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(HERE / 'diagnostic_metrics.csv', index=False)
    write_json(HERE / 'complete.json', dict(status='complete', cv_fits=36, final_refits=18,
        source=612, development=485, parameters_frozen_before_development=True))
    print(pd.DataFrame(rows).query("role=='evaluation'").groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].mean().to_string(), flush=True)


def restore(seed, arm, raw):
    ck = torch.load(HERE / 'models' / f'{arm}_{seed}.pt', map_location='cpu', weights_only=False)
    model = model_for(arm, seed)
    model.load_state_dict(ck['state']); model.eval()
    sc = ck['scaler']
    x = torch.tensor(np.where(sc['active'], (raw-sc['mu'])/sc['scale'], 0).astype(np.float32))
    return model, x, ck


if __name__ == '__main__':
    main()
