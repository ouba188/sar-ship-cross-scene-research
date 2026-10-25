"""Frozen-score polar fusion; source-only scalar selection before evaluation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
from fractions import Fraction
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'null_centered_readout_20261025'
BASE = HERE.parents[1]
LOCAL = BASE / 'local_evidence_transport'
SEEDS = (17, 29, 43)
NEW = ('hybrid_equal', 'hybrid_source', 'ordinary_source',
       'ordinary_at_hybrid_weight', 'hybrid_at_ordinary_weight')
COMPARATORS = ('prior_late', 'center_late', 'prior_vv', 'prior_vh', 'center_vh', 'base', 'strong')
ARMS = NEW + COMPARATORS
GRID = tuple(Fraction(k, 10) for k in range(11))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


previous = load('hybrid_previous_train', PREVIOUS / 'train.py')
Data, metrics, write_json = previous.Data, previous.metrics, previous.write_json


def blend(vv, vh, alpha):
    a = float(alpha)
    return a * vv + (1 - a) * vh


def exact_j(y, prediction):
    correct = prediction == y
    oa = Fraction(int(correct.sum()), len(y))
    ba = sum(Fraction(int(correct[y == c].sum()), int((y == c).sum())) for c in np.unique(y)) / len(np.unique(y))
    return (oa + ba) / 2


def select(vv, vh, y):
    candidates = []
    for alpha in GRID:
        s = blend(vv, vh, alpha)
        j = exact_j(y, s.argmax(1))
        candidates.append((alpha, j, metrics(y, s)))
    best = min(candidates, key=lambda row: (-row[1], abs(row[0] - Fraction(1, 2)), row[0]))
    return best[0], [m | dict(alpha=float(a), J=float(j), selected=a == best[0]) for a, j, m in candidates]


def read_scores(data):
    scores = {}
    for seed in SEEDS:
        for arm in ('old_full', *COMPARATORS):
            with np.load(PREVIOUS / 'results' / f'{arm}_{seed}.npz') as z:
                np.testing.assert_array_equal(z['up_index'], data.up[:data.n])
                s = z['score'].copy()
            assert s.shape == (data.n, 8) and np.isfinite(s).all()
            scores[seed, arm] = s
    return scores


def main():
    (HERE / 'results').mkdir(exist_ok=True)
    data = Data()
    scores = read_scores(data)
    checks = []
    grid_rows = []
    choices = []
    yva = data.table.iloc[data.va].class_id.to_numpy(int)
    # Complete all source choices before reading evaluation labels or metrics.
    for seed in SEEDS:
        vv, vh, new_vh, base = (scores[seed, a] for a in ('prior_vv', 'prior_vh', 'center_vh', 'base'))
        equal = blend(vv, vh, .5)
        error = float(abs(equal - scores[seed, 'prior_late']).max())
        assert error < 2e-6, error
        np.testing.assert_array_equal(blend(vv, new_vh, 1), vv)
        np.testing.assert_array_equal(blend(vv, new_vh, 0), new_vh)
        residual_error = float(abs(blend(vv, new_vh, .5) - (base + .5 * (vv - base) + .5 * (new_vh - base))).max())
        assert residual_error < 3e-6, residual_error
        checks.append(dict(seed=seed, old_equal_replay_error=error, base_once_error=residual_error))
        picked = {}
        for family, right in (('hybrid', new_vh), ('ordinary', vh)):
            alpha, rows = select(vv[data.va], right[data.va], yva)
            picked[family] = float(alpha)
            grid_rows.extend(dict(seed=seed, family=family, **r) for r in rows)
        choices.append(dict(seed=seed, **picked))
    write_json(HERE / 'source_choices.json', dict(grid=[float(x) for x in GRID], choices=choices,
        selection_n=len(data.va), evaluation_metrics_computed=False,
        criterion='exact (OA+BA)/2; ties nearest 0.5 then lower alpha; source selection only'))
    pd.DataFrame(grid_rows).to_csv(HERE / 'source_grid.csv', index=False)
    write_json(HERE / 'checks.json', checks)
    for row in choices:
        seed = row['seed']
        vv, vh, new_vh = (scores[seed, a] for a in ('prior_vv', 'prior_vh', 'center_vh'))
        for arm, right, alpha in (
            ('hybrid_equal', new_vh, .5), ('hybrid_source', new_vh, row['hybrid']),
            ('ordinary_source', vh, row['ordinary']),
            ('ordinary_at_hybrid_weight', vh, row['hybrid']),
            ('hybrid_at_ordinary_weight', new_vh, row['ordinary'])):
            scores[seed, arm] = blend(vv, right, alpha)
    rows = []
    for seed in SEEDS:
        for arm in ('old_full', *ARMS):
            s = scores[seed, arm]
            np.savez_compressed(HERE / 'results' / f'{arm}_{seed}.npz', score=s, up_index=data.up[:data.n])
            for role, ids in (('utility_train', data.tr), ('selection', data.va), ('evaluation', data.te)):
                y = data.table.iloc[ids].class_id.to_numpy(int)
                rows.append(dict(seed=seed, arm=arm, role=role) | metrics(y, s[ids]))
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / 'metrics.csv', index=False)
    write_json(HERE / 'complete.json', dict(status='complete', refits=0, choices=choices, configs=list(SEEDS)))
    print('SOURCE CHOICES', choices, flush=True)
    print(frame[frame.role.eq('evaluation')].groupby('arm')[['OA', 'BA', 'NLL', 'Brier']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
