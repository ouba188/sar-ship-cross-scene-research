"""Separate paired support variability from seed variability and class effects."""
import json
import numpy as np
import pandas as pd
from run_controls import OUT, run, probe, SEEDS


def main():
    analysis = probe.analysis
    draws = pd.read_csv(OUT / 'draw_metrics.csv')
    records = []
    for keys, g in draws.groupby(analysis.KEY):
        for metric in analysis.VALUES:
            p = g.pivot(index='draw', columns='arm', values=metric)
            ours = p[[f'class_utility_{i}' for i in range(3)]].mean(axis=1)
            for name, prefix in [('cluster', 'cluster_edge'), ('pca', 'pca_metric'), ('free_class', 'class_metric')]:
                competitor = p[[f'{prefix}_{i}' for i in range(3)]].mean(axis=1)
                delta = ours - competitor
                records.append(dict(zip(analysis.KEY, keys), competitor=name, metric=metric,
                                    ours_minus_competitor=delta.mean(), sd_support=delta.std(),
                                    positive_draws=int((delta > 1e-10).sum()), negative_draws=int((delta < -1e-10).sum())))
    pd.DataFrame(records).to_csv(OUT / 'ours_advantage_by_support.csv', index=False)
    meta = {'ports': run.old.source_metadata()}
    for name in ('f0921', 'oss2'):
        meta[name] = pd.read_csv(run.old.OLD / f'{name}_metadata.csv', dtype={'identity': str}, keep_default_na=False)
    tasks = json.loads((OUT / 'EVALUATION_ENROLLMENT.json').read_text(encoding='utf-8'))
    class_rows, events, checks = [], [], []
    for task in tasks:
        name = task['dataset']; q = np.array(task['query'])
        y = meta[name].loc[q].class_id.to_numpy(int)
        tag = f"{name}_{task['episode']:03d}_{task['budget']}_{task['draw']}_{task['cell']}"
        archive = run.load_npz(OUT / 'predictions' / f'{tag}.npz')
        np.testing.assert_array_equal(archive['query_rows'], q)
        predictions = {a: p.argmax(1) for a, p in zip(archive['arms'], archive['full'])}
        nc = len(np.unique(y))
        key = {k: task[k] for k in ['dataset', 'episode', 'port', 'cohort', 'budget', 'draw', 'cell', 'study']}
        for i, seed in enumerate(SEEDS):
            ours = predictions[f'class_utility_{i}']
            for competitor, prefix in [('cluster', 'cluster_edge'), ('pca', 'pca_metric'), ('free_class', 'class_metric')]:
                other = predictions[f'{prefix}_{i}']
                total = 0.
                for c in range(8):
                    mask = y == c
                    value = 100 * ((ours[mask] == c).mean() - (other[mask] == c).mean()) / nc if mask.any() else 0.
                    total += value
                    class_rows.append(dict(**key, seed=seed, competitor=competitor, class_id=c, ba_contribution=value))
                expected = np.mean([100 * ((ours[y == c] == c).mean() - (other[y == c] == c).mean()) for c in np.unique(y)])
                np.testing.assert_allclose(total, expected, atol=1e-10)
                checks.append(abs(total - expected))
                for status, mask in [('ours_only', (ours == y) & (other != y)), ('competitor_only', (ours != y) & (other == y))]:
                    for j in np.flatnonzero(mask):
                        events.append(dict(**key, seed=seed, competitor=competitor, status=status,
                                           row=int(q[j]), ship=str(meta[name].loc[q[j]].identity), class_id=int(y[j])))
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(OUT / 'class_ba_contributions.csv', index=False)
    class_frame = analysis.add_scopes(class_frame)
    ck = ['dataset', 'cohort', 'budget', 'cell', 'competitor', 'class_id']
    summary = class_frame.groupby(ck + ['seed', 'draw', 'port']).ba_contribution.mean().groupby(ck).mean().reset_index()
    summary.to_csv(OUT / 'class_contribution_summary.csv', index=False)
    event = pd.DataFrame(events)
    event.to_csv(OUT / 'pairwise_events.csv', index=False)
    event.groupby(['dataset', 'budget', 'cell', 'competitor', 'seed', 'status', 'class_id']).agg(
        exposures=('row', 'size'), ships=('ship', 'nunique'), scenes=('episode', 'nunique')).reset_index().to_csv(
            OUT / 'pairwise_event_groups.csv', index=False)
    check_data = json.loads((OUT / 'CHECKS.json').read_text())
    final = dict(class_reconstructions=len(checks), maximum_error=max(checks),
                 sklearn_checks=sum(c['kind'] == 'sklearn' for c in check_data),
                 gradient_checks=sum(c['kind'] == 'gradient' for c in check_data),
                 same_geometry_support_replays=sum(c['kind'] == 'same_geometry_metric_replay' for c in check_data))
    (OUT / 'ANALYSIS_COMPLETE.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
    print(pd.DataFrame(records).query("mode=='full' and metric=='ba' and cell=='other_acquisition' and cohort in ['external','all48']").to_string(index=False))
    print(summary.query("cell=='other_acquisition' and competitor=='free_class' and cohort in ['external','all48']").to_string(index=False))


if __name__ == '__main__':
    main()
