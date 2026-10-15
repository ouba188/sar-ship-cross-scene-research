"""Render descriptive C4 attribution figures from saved paired readouts."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_c4_conditions import OUT

DATASETS = ['self24ports', 'f0921', 'oss2']
NAMES = ['24-port LOPO', 'F_0921', 'OpenSARShip 2.0']
CLASSES = ['Bulk', 'Fishing', 'General cargo', 'Product/chemical', 'Container', 'Crude oil', 'Tug', 'Offshore supply']


def main():
    phase = pd.read_csv(OUT/'phase_summary.csv')
    full = phase[phase.branch.eq('full')].set_index(['dataset', 'phase'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    x = np.arange(3)
    for ax, metric in zip(axes, ['oa', 'ba']):
        for offset, name, color, label in [(-.26, 'query_c4', '#387f9e', 'Query C4'),
                (0, 'support_c4', '#36866f', 'C4 + support refit'),
                (.26, 'matched_c4', '#bd606c', 'Also refit source')]:
            values = [full.loc[(d, name), metric]-full.loc[(d, 'reference'), metric] for d in DATASETS]
            bars = ax.bar(x+offset, values, width=.25, color=color, label=label)
            ax.bar_label(bars, fmt='%+.2f', fontsize=8, padding=23 if name == 'support_c4' else 3)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(x, NAMES)
        ax.set_title(metric.upper()+' | full handoff model')
        ax.set_ylabel('Change from reference (percentage points)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.3)
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.savefig(OUT/'stage_effects.png', dpi=180)
    plt.close(fig)
    classes = pd.read_csv(OUT/'class_stage_summary.csv')
    classes = classes[classes.branch.eq('full')].set_index(['dataset', 'class_id'])
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 5.4), constrained_layout=True)
    for ax, column, title in zip(axes,
            ['support_total_delta_recall_pp', 'source_refit_delta_recall_pp'],
            ['C4 + support adaptation vs original', 'Additional source refit vs C4 + support adaptation']):
        values = np.array([[classes.loc[(d, c), column] for c in range(8)] for d in DATASETS])
        vmax = max(1., np.abs(values).max())
        im = ax.imshow(values, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        for r in range(3):
            for c in range(8):
                ax.text(c, r, f'{values[r,c]:+.2f}', ha='center', va='center', fontsize=9,
                        color='white' if abs(values[r,c]) > .65*vmax else 'black')
        ax.set_xticks(np.arange(8), CLASSES, fontsize=8)
        ax.set_yticks(np.arange(3), NAMES, fontsize=9)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, label='Class recall change (pp)', shrink=.9)
    fig.savefig(OUT/'class_stage_effects.png', dpi=180)
    plt.close(fig)
    for prefix in ('', 'rank_'):
        conditions = OUT/(prefix+'shared_class_condition_contrasts.csv')
        if not conditions.exists():
            continue
        table = pd.read_csv(conditions)
        table = table[(table.branch == 'full') & (table.contrast == 'support_total')]
        features = list(table.feature.unique())
        fig, ax = plt.subplots(figsize=(9, max(4, .45*len(features))), constrained_layout=True)
        lookup = table.set_index(['dataset', 'feature'])
        values = np.array([[lookup.loc[(d, f), 'high_minus_low_pp'] for d in DATASETS] for f in features])
        vmax = max(1., float(np.nanmax(np.abs(values))))
        im = ax.imshow(values, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')
        for r, f in enumerate(features):
            for c, d in enumerate(DATASETS):
                classes_n = int(lookup.loc[(d, f), 'common_classes'])
                text = f'{values[r,c]:+.2f} (C={classes_n})' if np.isfinite(values[r,c]) else 'insufficient'
                ax.text(c, r, text, ha='center', va='center', fontsize=9,
                        color='white' if abs(values[r,c]) > .65*vmax else 'black')
        ax.set_xticks(np.arange(3), NAMES)
        ax.set_yticks(np.arange(len(features)), [f.replace('_', ' ') for f in features], fontsize=9)
        bin_label = 'within-dataset ranks' if prefix else 'source-defined bins'
        ax.set_title('High vs low feature values: difference in C4 benefit\n'
                     f'Same classes across datasets; {bin_label}; >=20 images per class/bin', fontsize=10)
        fig.colorbar(im, ax=ax, label='Difference in recall benefit (pp)', shrink=.8)
        fig.savefig(OUT/(prefix+'condition_effects.png'), dpi=180)
        plt.close(fig)
    print('FIGURES COMPLETE', flush=True)


if __name__ == '__main__':
    main()
