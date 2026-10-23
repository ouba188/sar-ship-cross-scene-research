"""Apply pre-outcome source size bins and plot the complete paired comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_results import paired, SCORE, BASE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    config = json.loads((HERE / 'strata_frozen.json').read_text())
    geoms = []
    for fold in range(3):
        with np.load(ROOT / 'summaries_test' / f'fold{fold}_outer.npz') as z:
            g, q = z['geometry'][:, 0], z['quality'][:, 0]
            part = pd.DataFrame(dict(instance_id=z['instance_ids'], fold=fold,
                vv_length=g[:, 0], vv_width=g[:, 1],
                vv_aspect=g[:, 0] / np.maximum(g[:, 1], 1e-8),
                vv_min_coverage=q[:, :, 1].min(1),
                all_region_valid=z['region_valid'][:, 0, 1:4].all(1),
                vh_valid=z['region_valid'][:, 1, 0]))
            geoms.append(part)
    geometry = pd.concat(geoms, ignore_index=True)
    assert not geometry.duplicated(['fold', 'instance_id']).any()
    for key, boundaries in config['boundaries'].items():
        geometry[key + '_bin'] = np.array(['small', 'middle', 'large'])[np.searchsorted(boundaries, geometry[key], side='right')]
    geometry['coverage_bin'] = np.where(geometry.vv_min_coverage >= .99, '>=99%', '<99%')
    f = pd.read_csv(ROOT / 'test_predictions.csv', dtype={'mmsi': str}, float_precision='round_trip')
    f = f.merge(geometry, on=['fold', 'instance_id'], validate='many_to_one')
    rows = []
    for seed, group in f.groupby('seed'):
        arms = {arm: part.sort_values(['fold', 'instance_id']).reset_index(drop=True) for arm, part in group.groupby('arm')}
        dyn = arms['dynamic']
        y = dyn.class_id.to_numpy(int)
        for right in ['baseline', 'fixed', 'global', 'merged']:
            other = dyn[BASE].to_numpy() if right == 'baseline' else arms[right][SCORE].to_numpy()
            for dimension in ['vv_length_bin', 'vv_width_bin', 'vv_aspect_bin', 'coverage_bin', 'vh_valid']:
                for value, idx in dyn.groupby(dimension).groups.items():
                    ix = np.asarray(idx)
                    rows.append(dict(seed=seed, left='dynamic', right=right,
                        dimension=dimension, stratum=value, classes=len(np.unique(y[ix])),
                        **paired(y[ix], dyn[SCORE].to_numpy()[ix], other[ix])))
    pd.DataFrame(rows).to_csv(HERE / 'test_strata.csv', index=False)
    geometry.to_csv(HERE / 'test_geometry_groups.csv', index=False)
    m = pd.read_csv(HERE / 'test_metrics.csv')
    c = pd.read_csv(HERE / 'test_contrasts.csv')
    order = ['baseline', 'global', 'fixed', 'dynamic', 'merged']
    labels = ['Frozen model', 'Global shallow', 'Fixed regions', 'Dynamic OBB', 'Merged regions']
    colors = ['#777777', '#287CA4', '#BC8626', '#18835E', '#9B486D']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ['OA', 'BA']):
        means = m.groupby('arm')[metric].mean().reindex(order)
        std = m.groupby('arm')[metric].std().reindex(order)
        ax.bar(np.arange(5), means, yerr=std, color=colors, capsize=4)
        for s, marker in [(17, 'o'), (29, 's'), (43, '^')]:
            vals = m.loc[m.seed.eq(s)].set_index('arm')[metric].reindex(order)
            ax.scatter(np.arange(5), vals, c='black', s=18, marker=marker, label=f'seed {s}', zorder=3)
        ax.set_xticks(np.arange(5), labels, rotation=18, ha='right')
        ax.set_ylabel(metric + ' (%)')
        ax.set_ylim(0, max(means + std) + 6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Same 4,744 historical held observations; three training seeds\nBars: mean; error bars: sample standard deviation (not confidence intervals)')
    fig.tight_layout()
    fig.savefig(HERE / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    contrasts = ['baseline', 'global', 'fixed', 'merged']
    for ax, metric in zip(axes, ['delta_OA', 'delta_BA']):
        for j, seed in enumerate([17, 29, 43]):
            part = c.loc[c.left.eq('dynamic') & c.seed.eq(seed)].set_index('right')
            ax.bar(np.arange(4) + (j-1)*.23, part.loc[contrasts, metric], width=.23, label=f'seed {seed}')
        ax.axhline(0, c='black', lw=.8)
        ax.set_xticks(np.arange(4), ['vs frozen', 'vs global', 'vs fixed', 'vs merged'])
        ax.set_ylabel(metric.replace('delta_', '') + ' difference (percentage points)')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle('Dynamic OBB: complete paired contrasts, no best-seed selection')
    fig.tight_layout()
    fig.savefig(HERE / 'paired_comparison.png', dpi=160)
    plt.close(fig)
    print(pd.DataFrame(rows).query("right=='fixed' and dimension=='vv_length_bin'").to_string(index=False))


if __name__ == '__main__':
    main()
