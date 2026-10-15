"""Link frozen relation changes to continuous full-system decision margins."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT / 'class_contrast_scope_validation_20261014'
sys.path.insert(0, str(ROOT / 'class_reference_external_20261014'))
import experiment


def margin(p, y):
    other = p.copy()
    other[np.arange(len(y)), y] = -np.inf
    return p[np.arange(len(y)), y] - other.max(axis=1)


def main():
    plans = json.loads((OLD / 'ENROLLMENT.json').read_text(encoding='utf-8'))
    paired = pd.read_csv(OUT / 'ship_outcome_link.csv', dtype={'ship': str})
    rows = []
    for name, budget in [('f0921', 64), ('oss2', 64), ('ports', 32)]:
        meta = experiment.source_metadata() if name == 'ports' else pd.read_csv(
            ROOT / 'external_scene_transfer_20261013' / f'{name}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
        selected = [p for p in plans if p['dataset'] == name and p['budget'] == budget and p['cell'] == 'other_acquisition']
        for t in selected:
            tag = f"{name}_{t['episode']:03d}_{budget}_{t['draw']}_other_acquisition.npz"
            with np.load(OLD / 'predictions' / tag) as archive:
                q = archive['query_rows']
                y = archive['labels']
                np.testing.assert_array_equal(q, t['query'])
                np.testing.assert_array_equal(y, meta.loc[q].class_id.to_numpy(int))
                probs = archive['full']
                arms = list(archive['arms'])
                baseline = probs[arms.index('class_fixed')]
                m0 = margin(baseline, y)
                onehot = np.eye(8)[y]
                b0 = np.sum((baseline - onehot) ** 2, axis=1)
                for seed in range(3):
                    cand = probs[arms.index(f'class_utility_{seed}')]
                    dm = margin(cand, y) - m0
                    db = b0 - np.sum((cand - onehot) ** 2, axis=1)
                    rows.extend(dict(dataset=name, budget=budget, ship=str(ship), class_id=int(label),
                                     episode=t['episode'], draw=t['draw'], seed=seed,
                                     margin_gain=float(x), brier_improvement=float(v))
                                for ship, label, x, v in zip(meta.loc[q].identity, y, dm, db))
        print('READ', name, len(selected), flush=True)
    d = pd.DataFrame(rows)
    ship = d.groupby(['dataset', 'budget', 'ship'], as_index=False)[['margin_gain', 'brier_improvement']].mean()
    ship = paired.merge(ship, on=['dataset', 'budget', 'ship'], how='left', validate='one_to_one')
    assert ship.margin_gain.notna().all()
    ship.to_csv(OUT / 'ship_margin_link.csv', index=False)
    quartile = ship.groupby(['dataset', 'budget', 'quartile']).agg(ships=('ship', 'size'),
        relative_shift=('relative_shift', 'median'), margin_gain=('margin_gain', 'mean'),
        brier_improvement=('brier_improvement', 'mean'), ba_contribution=('ba_contribution', 'sum')).reset_index()
    quartile.to_csv(OUT / 'margin_quartiles.csv', index=False)
    association = []
    for name, g in ship.groupby('dataset'):
        for field in ('margin_gain', 'brier_improvement', 'ba_contribution'):
            corr = spearmanr(g.relative_shift, g[field])
            association.append(dict(dataset=name, ships=len(g), outcome=field,
                                    rho=float(corr.statistic), p_exploratory=float(corr.pvalue),
                                    outcome_mean=float(g[field].mean())))
    pd.DataFrame(association).to_csv(OUT / 'margin_associations.csv', index=False)
    print(pd.DataFrame(association).to_string(index=False), flush=True)
    print(quartile.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
