"""Source-selected raw/summary baselines; freeze predictions without target truth."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]/'sar_m123_standalone'))
from sar_m123 import train_aligned as ta, train_ops as op

SEEDS = (17, 29, 43)
ALPHAS = (.01, .1, 1., 10., 100., 1000.)
BUDGETS = {
    'fit_only': (('fit_train',), ('fit_validation',)),
    'pooled': (('fit_train', 'utility_train'), ('fit_validation', 'selection')),
}


def load_inputs(bundle):
    table = pd.read_csv(bundle/'table.csv', dtype={'mmsi': str, 'pass_group': str},
                        usecols=['obs_id', 'mmsi', 'pass_group', 'role', 'class_id'])
    table.loc[table.role.eq('evaluation'), 'class_id'] = -1
    inputs = []
    for filename in ('features.npz', 'external.npz'):
        with np.load(bundle/filename, allow_pickle=False) as z:
            inputs.append({key: z[key].copy() for key in ('obs_id', 'raw', 'aux')})
    source, external = inputs
    np.testing.assert_array_equal(source['obs_id'].astype(str), table.obs_id.to_numpy().astype(str))
    for data in inputs:
        n = len(data['obs_id'])
        if data['raw'].shape != (n, 1538) or data['aux'].shape != (n, 176):
            raise ValueError('Eight-class baselines require raw1538 and aux176.')
    return table, source, external


def budget_rows(table, budget):
    fit_roles, val_roles = BUDGETS[budget]
    fit = np.flatnonzero(table.role.isin(fit_roles))
    val = np.flatnonzero(table.role.isin(val_roles))
    evaluation = np.flatnonzero(table.role.eq('evaluation'))
    for ids in (fit, val):
        if set(table.iloc[ids].class_id) != set(range(8)):
            raise ValueError(f'{budget} fit and validation need all eight classes.')
    for key in ('mmsi', 'pass_group'):
        if set(table.iloc[fit][key]) & set(table.iloc[val][key]):
            raise ValueError(f'{budget}: {key} overlaps fitting and validation.')
        if set(table.iloc[np.r_[fit, val]][key]) & set(table.iloc[evaluation][key]):
            raise ValueError(f'{budget}: {key} overlaps source evaluation.')
    return fit, val


def sample_weights(table, fit):
    f = table.iloc[fit]
    ships = f.groupby('class_id').mmsi.transform('nunique').to_numpy()
    repeat = f.groupby('mmsi').mmsi.transform('size').to_numpy()
    return len(f)/(f.class_id.nunique()*ships*repeat)


def train_ridge(x, xe, table, fit, val, out):
    y = table.class_id.to_numpy(int)
    classes = int(y[fit].max())+1
    target = 2*np.eye(classes)[y[fit]]-1
    weights = sample_weights(table, fit)
    best, chosen, rows = None, None, []
    tick = time.perf_counter()
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver='lsqr', tol=1e-6, max_iter=10000)
        model.fit(x[fit], target, sample_weight=weights)
        scores = model.predict(x[val])
        j = op.exact_j(y[val], scores)
        rows.append(dict(alpha=alpha, **op.metrics(y[val], scores)))
        if best is None or j > best:
            best, chosen = j, model
    source, external = chosen.predict(x), chosen.predict(xe)
    np.savez_compressed(out/'ridge_model.npz', coef=chosen.coef_, intercept=chosen.intercept_)
    pd.DataFrame(rows).to_csv(out/'ridge_selection.csv', index=False)
    return source, external, dict(alpha=float(chosen.alpha), selection_J=float(best),
        seconds=time.perf_counter()-tick, parameters=int(chosen.coef_.size+chosen.intercept_.size))


def train_mlp(x, xe, table, fit, val, seed, out, steps=200):
    torch.manual_seed(seed)
    y = table.class_id.to_numpy(int)
    classes = int(y[fit].max())+1
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, classes))
    xt, yt = torch.tensor(x), torch.tensor(y[fit])
    weights = torch.tensor(sample_weights(table, fit), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    best, chosen, selected, curve = None, None, 0, []
    tick = time.perf_counter()
    for step in range(1, steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = (nn.functional.cross_entropy(model(xt[fit]), yt, reduction='none')*weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite source loss.')
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            with torch.inference_mode():
                scores = model(xt[val]).numpy()
            j = op.exact_j(y[val], scores)
            curve.append(dict(step=step, loss=float(loss.detach()), **op.metrics(y[val], scores)))
            if best is None or j > best:
                best, chosen, selected = j, op.state(model), step
    model.load_state_dict(chosen)
    with torch.inference_mode():
        source, external = model(xt).numpy(), model(torch.tensor(xe)).numpy()
    torch.save(dict(state=chosen, step=selected, input_dim=x.shape[1], classes=classes),
               out/f'mlp_seed{seed}.pt')
    pd.DataFrame(curve).to_csv(out/f'mlp_seed{seed}_selection.csv', index=False)
    return source, external, dict(step=selected, selection_J=float(best),
        seconds=time.perf_counter()-tick, parameters=sum(p.numel() for p in model.parameters()))


def run(bundle, output, budgets=tuple(BUDGETS), seeds=SEEDS):
    torch.set_num_threads(2)
    table, source, external = load_inputs(Path(bundle))
    output = Path(output)
    tick = time.perf_counter()
    completions = []
    for budget in budgets:
        fit, val = budget_rows(table, budget)
        dest = output/budget
        dest.mkdir(parents=True, exist_ok=True)
        outputs, receipts = {}, []
        for name in ('raw', 'raw_summary'):
            folder = dest/name
            folder.mkdir(exist_ok=True)
            x, xe = source['raw'], external['raw']
            if name == 'raw_summary':
                x, xe = np.c_[x, source['aux']], np.c_[xe, external['aux']]
            scaler = ta.source_scaler(x, fit)
            np.savez_compressed(folder/'scaler.npz', **scaler)
            x, xe = ta.scale_input(x, scaler), ta.scale_input(xe, scaler)
            s, e, receipt = train_ridge(x, xe, table, fit, val, folder)
            outputs[f'{name}_ridge'] = (s, e)
            receipts.append(dict(method=f'{name}_ridge', seed=None, **receipt))
            print('FROZEN', budget, name, 'ridge', receipt, flush=True)
            for seed in seeds:
                s, e, receipt = train_mlp(x, xe, table, fit, val, seed, folder)
                outputs[f'{name}_mlp_seed{seed}'] = (s, e)
                receipts.append(dict(method=f'{name}_mlp', seed=seed, **receipt))
                print('FROZEN', budget, name, 'mlp', seed, receipt, flush=True)
        np.savez_compressed(dest/'source_predictions.npz', obs_id=source['obs_id'],
                            class_id=table.class_id.to_numpy(int),
                            **{key: value[0] for key, value in outputs.items()})
        np.savez_compressed(dest/'external_predictions.npz', obs_id=external['obs_id'],
                            class_id=np.full(len(external['obs_id']), -1, dtype=np.int64),
                            **{key: value[1] for key, value in outputs.items()})
        pd.DataFrame(receipts).to_csv(dest/'training_receipts.csv', index=False)
        receipt = dict(budget=budget, fit=len(fit), validation=len(val),
            source=len(table), source_evaluation=int(table.role.eq('evaluation').sum()),
            external=len(external['obs_id']), classes=8, seeds=list(seeds), alphas=list(ALPHAS),
            mlp_steps=200, mlp_initialization='ordinary 128-wide MLP; no raw-only warm initialization',
            validation_criterion='exact (OA+BA)/2; retain first candidate on ties',
            target_labels_used=False, evaluation_performed=False,
            methods=list(outputs), cpu_threads=2)
        op.write_json(dest/'complete.json', receipt)
        completions.append(receipt)
    op.write_json(output/'complete.json', dict(budgets=completions, seconds=time.perf_counter()-tick,
        target_labels_used=False, evaluation_performed=False))
    return completions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=HERE/'bundle')
    parser.add_argument('--output', type=Path, default=HERE/'baselines')
    parser.add_argument('--budgets', nargs='+', choices=tuple(BUDGETS), default=list(BUDGETS))
    parser.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS))
    args = parser.parse_args()
    run(args.bundle, args.output, args.budgets, args.seeds)


if __name__ == '__main__':
    main()
