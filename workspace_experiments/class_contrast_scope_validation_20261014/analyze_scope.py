"""Matched full-system effects, support variability, and vessel contributions."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
KEY = ['dataset', 'cohort', 'budget', 'cell']
VALUES = ['oa', 'ba', 'brier', 'macro_brier', 'nll', 'macro_nll']


def scopes(d):
    all_rows = d.copy(); all_rows['cohort'] = 'all'
    return pd.concat([d, all_rows], ignore_index=True)


def main():
    d = scopes(pd.read_csv(OUT / 'metrics.csv'))
    port = d.groupby(KEY + ['port', 'draw', 'arm'])[VALUES].mean().reset_index()
    draw = port.groupby(KEY + ['draw', 'arm'])[VALUES].mean().reset_index()
    draw.to_csv(OUT / 'draw_metrics.csv', index=False)
    differences, summary, scenes = [], [], []
    for key, g in draw.groupby(KEY):
        for value in VALUES:
            p = g.pivot(index='draw', columns='arm', values=value)
            for comp in ('class_mean', 'class_fixed', 'free_class', 'original'):
                a = np.column_stack([p[f'class_utility_{i}'] for i in range(3)])
                b = np.column_stack([p[f'class_metric_{i}' if comp == 'free_class' else comp] for i in range(3)])
                delta = a - b
                dv, sv = delta.mean(1), delta.mean(0)
                direction = 1 if value in ('oa', 'ba') else -1
                for j in range(5):
                    for i in range(3):
                        differences.append(dict(zip(KEY, key), metric=value, comparator=comp, draw=j, seed=i, delta=delta[j, i]))
                summary.append(dict(zip(KEY, key), metric=value, comparator=comp, baseline=b.mean(), ours=a.mean(), delta=delta.mean(),
                                    sd_draw=dv.std(ddof=1), sd_seed=sv.std(ddof=1), positive_draws=int((direction * dv > 1e-10).sum()),
                                    positive_seeds=int((direction * sv > 1e-10).sum()), worst_draw=dv[np.argmin(direction * dv)],
                                    without_best_draw=np.delete(dv, np.argmax(direction * dv)).mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'effects_summary.csv', index=False)
    pd.DataFrame(differences).to_csv(OUT / 'paired_draw_seed_effects.csv', index=False)
    for keys, g in d.groupby(KEY + ['port', 'episode']):
        p = g.groupby('arm')[VALUES].mean()
        for comp in ('class_mean', 'class_fixed', 'free_class', 'original'):
            ours = p.loc[[f'class_utility_{i}' for i in range(3)]].mean()
            base = p.loc[[f'class_metric_{i}' for i in range(3)]].mean() if comp == 'free_class' else p.loc[comp]
            scenes.append(dict(zip(KEY + ['port', 'episode'], keys), comparator=comp,
                               **{k: float(ours[k] - base[k]) for k in VALUES}))
    scene = pd.DataFrame(scenes)
    scene.to_csv(OUT / 'scene_effects.csv', index=False)
    scene.groupby(KEY + ['comparator']).agg(scenes=('ba', 'size'), positive=('ba', lambda x: int((x > 1e-10).sum())),
        negative=('ba', lambda x: int((x < -1e-10).sum())), minimum=('ba', 'min'), maximum=('ba', 'max')).reset_index().to_csv(OUT / 'scene_robustness.csv', index=False)
    scene.groupby(KEY + ['port', 'comparator'])[VALUES].mean().reset_index().to_csv(OUT / 'port_effects.csv', index=False)
    # Fixed scene/port weights reconstruct every paired metric before interpreting event counts.
    tasks = pd.DataFrame(json.loads((OUT / 'ENROLLMENT.json').read_text(encoding='utf-8')))
    count = scopes(tasks).drop_duplicates(KEY + ['port', 'episode']).groupby(KEY + ['port']).size().rename('nscene').reset_index()
    count['nport'] = count.groupby(KEY).port.transform('size')
    events = scopes(pd.read_csv(OUT / 'events.csv', dtype={'ship': str}))
    events['comparator'] = events.comparator.str.replace(r'class_metric_\d', 'free_class', regex=True)
    events = events.merge(count, on=KEY + ['port'], validate='many_to_one')
    events['weighted_ba'] = events.ba_contribution / (15 * events.nscene * events.nport)
    events['weighted_oa'] = events.oa_contribution / (15 * events.nscene * events.nport)
    total = events.groupby(KEY + ['comparator'])[['weighted_ba', 'weighted_oa']].sum().reset_index()
    checks = []
    for metric in ('ba', 'oa'):
        reference = summary[summary.metric == metric][KEY + ['comparator', 'delta']]
        joined = reference.merge(total, on=KEY + ['comparator'], how='left').fillna(0)
        np.testing.assert_allclose(joined.delta, joined['weighted_' + metric], atol=1e-9)
        checks.append(dict(kind='event_reconstruction', metric=metric, count=len(joined), max_error=float(abs(joined.delta - joined['weighted_' + metric]).max())))
    contributions = events.groupby(KEY + ['comparator', 'class_id']).agg(
        ba_contribution=('weighted_ba', 'sum'), oa_contribution=('weighted_oa', 'sum'),
        rescue_exposures=('net_hit', lambda x: int((x > 0).sum())), harm_exposures=('net_hit', lambda x: int((x < 0).sum())),
        affected_ships=('ship', 'nunique'), affected_scenes=('episode', 'nunique')).reset_index()
    contributions.to_csv(OUT / 'class_contributions.csv', index=False)
    ships = events.groupby(KEY + ['comparator', 'ship', 'class_id']).agg(
        ba_contribution=('weighted_ba', 'sum'), oa_contribution=('weighted_oa', 'sum'), net_hits=('net_hit', 'sum'),
        scenes=('episode', 'nunique'), observations=('row', 'nunique')).reset_index()
    ships.to_csv(OUT / 'ship_contributions.csv', index=False)
    sensitivity = []
    for keys, g in ships.groupby(KEY + ['comparator']):
        pos = g[g.ba_contribution > 1e-12].sort_values('ba_contribution', ascending=False)
        neg = g[g.ba_contribution < -1e-12]
        sensitivity.append(dict(zip(KEY + ['comparator'], keys), ships_net_positive=len(pos), ships_net_negative=len(neg),
                                total=g.ba_contribution.sum(), top_positive_ship=pos.ba_contribution.iloc[0] if len(pos) else 0,
                                without_top_ship=g.ba_contribution.sum() - (pos.ba_contribution.iloc[0] if len(pos) else 0),
                                without_top3_ships=g.ba_contribution.sum() - pos.ba_contribution.head(3).sum()))
    pd.DataFrame(sensitivity).to_csv(OUT / 'ship_concentration.csv', index=False)
    timing = pd.read_csv(OUT / 'timings.csv')
    mt = pd.read_csv(OUT / 'mapping_timings.csv')
    mt['seconds_per_row'] = mt.seconds / mt.rows
    mapcost = mt.groupby(['dataset', 'spec']).seconds_per_row.median().to_dict()
    timing['mapping_seconds'] = [mapcost.get((r.dataset, 'class_edge' if r.arm.startswith('class_utility') or r.arm == 'class_fixed' else 'class_metric' if r.arm.startswith('class_metric') else 'class_mean' if r.arm == 'class_mean' else ''), 0) * (r.query_n + r.support_n) for r in timing.itertuples()]
    timing['total_ms'] = 1000 * (timing.common_seconds + timing.operator_seconds + timing.mapping_seconds)
    tk = ['dataset', 'episode', 'budget', 'draw', 'cell']
    base = timing[timing.arm == 'class_mean'][tk + ['total_ms']].rename(columns={'total_ms': 'core_ms'})
    timing = timing.merge(base, on=tk, validate='many_to_one')
    timing['extra_ms'] = timing.total_ms - timing.core_ms
    timing['family'] = timing.arm.str.replace(r'_\d$', '', regex=True)
    cost = timing.groupby(['dataset', 'budget', 'cell', 'family'])[['total_ms', 'core_ms', 'extra_ms']].median().reset_index()
    cost.to_csv(OUT / 'cached_cost_summary.csv', index=False)
    (OUT / 'ANALYSIS_CHECKS.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, dataset in zip(axes, ('f0921', 'oss2', 'ports')):
        for comp, label in [('class_mean', 'vs fixed reference'), ('class_fixed', 'vs equal pair weights'), ('free_class', 'vs free metric')]:
            z = summary[(summary.dataset == dataset) & (summary.cohort == 'all') & (summary.cell == 'other_acquisition') & (summary.metric == 'ba') & (summary.comparator == comp)].sort_values('budget')
            ax.errorbar(z.budget, z.delta, yerr=z.sd_draw, marker='o', capsize=3, label=label)
        ax.axhline(0, color='gray', lw=.8); ax.set_title(dataset); ax.set_xticks([16, 32, 64]); ax.set_xlabel('Support-label budget')
    axes[0].set_ylabel('Paired BA change (pp)'); axes[0].legend(fontsize=7)
    fig.suptitle('Frozen models; expanded acquisitions; bars: support-draw SD')
    fig.tight_layout(); fig.savefig(OUT / 'budget_effects.png', dpi=180); plt.close(fig)
    selected = summary[(summary.cohort == 'all') & (summary.cell == 'other_acquisition') & (summary.metric.isin(['oa', 'ba'])) & (summary.comparator.isin(['class_mean', 'class_fixed', 'free_class']))]
    print(selected.to_string(index=False))


if __name__ == '__main__':
    main()
