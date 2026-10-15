"""Ship-cluster intervals and pair-graph information count for root diagnosis."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE/'condition_failure_root_20261010'


def main():
    pred = pd.read_csv(OUT/'classification_predictions.csv')
    pred = pred[pred.split == 'test_all'].copy()
    pred['correct'] = (pred['true'] == pred.pred).astype(float)
    base = pred[pred.arm == 'state'].sort_values('obs_id')
    ships, inverse = np.unique(base.mmsi, return_inverse=True)
    y = base['true'].to_numpy()
    counts = np.zeros((len(ships), 8))
    np.add.at(counts, (inverse, y), 1)
    groups = {'state': ['state'], 'state_matched': ['state_matched'], 'old_condition': ['state_condition'],
        'new_task': ['state_scene_task_20261031', 'state_scene_task_20261032']}
    correct = {}
    for name, names in groups.items():
        values = []
        for arm in names:
            p = pred[pred.arm == arm].sort_values('obs_id')
            np.testing.assert_array_equal(p.obs_id, base.obs_id)
            values.append(p.correct.to_numpy())
        matrix = np.zeros_like(counts)
        np.add.at(matrix, (inverse, y), np.mean(values, axis=0))
        correct[name] = matrix
    draws = np.random.default_rng(20261010).integers(len(ships), size=(2000, len(ships)))
    denominator = counts[draws].sum(1)
    records = []
    for arm in ('old_condition', 'new_task'):
        for reference in ('state', 'state_matched'):
            difference = correct[arm]-correct[reference]
            sampled = difference[draws].sum(1)
            distributions = {'OA': 100*sampled.sum(1)/denominator.sum(1),
                'BA': 100*(sampled/denominator).mean(1)}
            estimates = {'OA': 100*difference.sum()/counts.sum(),
                'BA': 100*(difference.sum(0)/counts.sum(0)).mean()}
            for metric in ('OA', 'BA'):
                low, high = np.quantile(distributions[metric], [.025, .975])
                records.append(dict(arm=arm, reference=reference, metric=metric, delta=estimates[metric],
                    ci_low=low, ci_high=high))
    pd.DataFrame(records).to_csv(OUT/'classification_bootstrap.csv', index=False)
    hold = pd.read_csv(OUT/'observation_holdout/predictions.csv')
    hold['delta_error'] = (hold.prediction-hold.target)**2-(hold.base-hold.target)**2
    rows = []
    for (group, arm, task), p in hold.groupby(['group', 'arm', 'task']):
        values = p.groupby('mmsi').delta_error.mean().to_numpy()
        ix = np.random.default_rng(20261010).integers(len(values), size=(2000, len(values)))
        lo, hi = np.quantile(values[ix].mean(1), [.025, .975])
        rows.append(dict(group=group, arm=arm, task=task, delta_MSE=values.mean(), ci_low=lo, ci_high=hi))
    pd.DataFrame(rows).to_csv(OUT/'observation_holdout/ship_bootstrap.csv', index=False)
    pairs = pd.read_csv(BASE/'condition_error_link_20261010/train_pairs.csv')
    nodes = np.unique(np.r_[pairs.a, pairs.b])
    parent = {int(i): int(i) for i in nodes}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in zip(pairs.a, pairs.b):
        parent[root(a)] = root(b)
    components = len({root(i) for i in nodes})
    result = dict(pair_edges=len(pairs), observations=len(nodes), connected_components=components,
        independent_scalar_differences=len(nodes)-components,
        note='Graph-incidence rank counts algebraic independence of endpoint differences, not a full statistical effective sample-size estimate.')
    (OUT/'pair_information_count.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(pd.DataFrame(records).to_string(index=False))
    print(pd.DataFrame(rows).query("group == 'new_observations_same_ships'").to_string(index=False))
    print(result)


if __name__ == '__main__':
    main()
