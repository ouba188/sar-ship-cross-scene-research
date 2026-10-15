"""Post-readout reliability diagnostics; no parameter fitting or rule changes."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json

import numpy as np
import pandas as pd

from frequency_reliability_run import OUT, EXP, ROOT, SOURCE, DATASETS, load_npz
from frequency_reliability_core import point_ce


def main():
    source_y = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    signs, cohorts = [], []
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        folder = marker.parent
        info = json.loads((folder/'PREPARED.json').read_text())
        p = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        y = source_y[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        with np.load(folder/'policies.npz') as saved:
            ix = [list(saved['modes']).index(m) for m in ('baseline', 'selective')]
            score = saved['scores'][:, ix].astype(float)
        for draw, ev in enumerate(p['eval_masks']):
            meta = dict(dataset=info['dataset'], task=info['task'], draw=draw)
            gamma = p['local_gamma'][draw]
            utility = point_ce(score[draw, 0], y)-point_ce(score[draw, 1], y)
            mask = ev & (utility != 0)
            positive = utility[mask] > 0
            predicted = gamma[mask] > .5
            pos_recall = float(predicted[positive].mean()) if positive.any() else np.nan
            neg_recall = float((~predicted[~positive]).mean()) if (~positive).any() else np.nan
            signs.append(dict(**meta, n=int(mask.sum()),
                sign_accuracy=100*float(np.mean(positive == predicted)),
                always_positive_accuracy=100*float(positive.mean()),
                balanced_sign_accuracy=50*(pos_recall+neg_recall),
                positive_recall=100*pos_recall, negative_recall=100*neg_recall,
                gamma_positive=float(gamma[mask][positive].mean()),
                gamma_negative=float(gamma[mask][~positive].mean()),
                pearson=float(np.corrcoef(gamma[mask], utility[mask])[0, 1])))
            for stage, key in [('base', 'base_predictions'), ('full', 'predictions')]:
                old, new, local = p[key][draw, 0], p[key][draw, 1], p[key][draw, 4]
                groups = {
                    'learned_rescue': ev & (old != y) & (new == y),
                    'learned_harm': ev & (old == y) & (new != y),
                    'local_lost_rescue': ev & (old != y) & (new == y) & (local != y),
                    'local_repaired_harm': ev & (old == y) & (new != y) & (local == y)}
                for group, selected in groups.items():
                    cohorts.append(dict(**meta, stage=stage, group=group, n=int(selected.sum()),
                        gamma_sum=float(gamma[selected].sum()),
                        gamma_mean=float(gamma[selected].mean()) if selected.any() else np.nan))
    s = pd.DataFrame(signs)
    s.to_csv(OUT/'utility_sign_metrics.csv', index=False)
    fields = ['sign_accuracy', 'always_positive_accuracy', 'balanced_sign_accuracy',
              'positive_recall', 'negative_recall', 'gamma_positive', 'gamma_negative', 'pearson']
    summary = s.groupby(['dataset', 'task'])[fields].mean().groupby('dataset').mean()
    summary.to_csv(OUT/'utility_sign_summary.csv')
    c = pd.DataFrame(cohorts)
    c.to_csv(OUT/'gamma_cohorts.csv', index=False)
    cs = c.groupby(['dataset', 'stage', 'group'])[['n', 'gamma_sum']].sum()
    cs['gamma_event_weighted'] = cs.gamma_sum / cs.n
    cs.to_csv(OUT/'gamma_cohort_summary.csv')
    source = pd.read_csv(OUT/'source_selection.csv')
    tasks = pd.read_csv(OUT/'task_means.csv')
    target = tasks[tasks['mode'].eq('local')].pivot(index=['dataset', 'task'], columns='stage', values='delta_ba')
    source.merge(target, on=['dataset', 'task']).to_csv(OUT/'source_transfer.csv', index=False)
    print(summary.to_string(), flush=True)
    print(cs.to_string(), flush=True)


if __name__ == '__main__':
    main()
