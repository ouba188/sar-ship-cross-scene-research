"""Plot the two source-selected condition/evidence candidates without reselection."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'condition_evidence_20261011'


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    metrics = pd.read_csv(OUT/'held_subsets.csv')
    chosen = metrics[(metrics.name == 'center_border_log_ratio') & (metrics['mode'] == 'magnitude')]
    arms = ['anchor', 'capacity', 'condition', 'sham']
    colors = ['#74838A', '#A38560', '#20866D', '#805F99']
    for j, (subset, title) in enumerate([('all', 'All pairs\n189 pairs / 43 ships'),
                                        ('gap2', 'Gap >= 2 deg\n53 pairs / 18 ships')]):
        table = chosen[chosen.subset == subset].set_index('arm')
        for k, arm in enumerate(arms):
            axes[0].bar(j+(k-1.5)*.17, table.loc[arm, 'mse'], width=.16,
                        color=colors[k], label=['Anchor only', 'Capacity control',
                                               'Real condition', 'Shuffled condition'][k] if j == 0 else None)
    axes[0].set_xticks([0, 1], ['All pairs\n189 pairs / 43 ships', 'Gap >= 2 deg\n53 pairs / 18 ships'])
    axes[0].set_ylabel('Mean squared prediction error (lower is better)')
    axes[0].set_title('A. Center/periphery volatility prediction')
    axes[0].set_ylim(0, .083)
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    relations = pd.read_csv(OUT/'condition_relations.csv')
    labels = ['near', 'moderate', 'large']
    for role, color, marker in [('fit', '#397DB2', 'o'), ('held', '#C66B47', 's')]:
        table = relations[(relations.name == 'center_border_log_ratio') &
                          (relations['mode'] == 'magnitude') & (relations.role == role)].set_index('gap_bin')
        ys = table.loc[labels, 'outcome_mean'].to_numpy()
        axes[1].plot(range(3), ys, marker=marker, color=color, label=role, linewidth=2)
        for j, y in enumerate(ys):
            offset = 10 if role == 'fit' else -16
            axes[1].annotate(f"{int(table.loc[labels[j], 'ships'])} ships", (j, y),
                             xytext=(0, offset), textcoords='offset points', ha='center', fontsize=8)
        slopes = relations[(relations.name == 'middle_energy_share') &
                           (relations['mode'] == 'signed') & (relations.role == role)].set_index('gap_bin')
        axes[2].plot(range(3), slopes.loc[labels, 'angle_slope'], marker=marker,
                     color=color, linewidth=2, label=role)
    for ax in axes[1:]:
        ax.set_xticks(range(3), ['< 2 deg', '2-5 deg', '>= 5 deg'])
        ax.set_xlabel('Absolute incidence-angle difference')
        ax.legend(frameon=False)
    axes[1].set_title('B. Observed spatial volatility')
    axes[1].set_ylabel('Mean absolute change in log activation ratio')
    axes[1].set_ylim(.12, .45)
    axes[2].axhline(0, color='#858585', linewidth=.8)
    axes[2].set_title('C. Middle-band direction does not transfer')
    axes[2].set_ylabel('Descriptive share-change slope per degree')
    fig.suptitle('Condition-related CNN volatility: a fragile local candidate, not classification gain\n'
                 'Removing one held ship reverses the aggregate benefit; no causal isolation established', fontsize=12)
    fig.savefig(OUT/'condition_evidence_summary.png')
    fig.savefig(OUT/'condition_evidence_summary.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
