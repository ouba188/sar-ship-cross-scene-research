"""Compact numerical readout for the learned-frequency development experiment."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

OUT = Path('E:/Hermes/paper_tracking/sensitivity/learned_frequency_residual_20261009')


def main():
    data = pd.read_csv(OUT/'summary.csv')
    modes = ['unsplit', 'fixed', 'selective', 'selective_uniform']
    labels = ['Unsplit learned branch', 'Separate bands, equal fusion',
              'Separate bands, learned fusion', 'Learned model, forced equal fusion']
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharey=True)
    for i, stage in enumerate(['base', 'full']):
        for j, dataset in enumerate(['self24ports', 'f0921', 'oss2']):
            ax = axes[i, j]
            part = data[data.dataset.eq(dataset) & data.stage.eq(stage)].set_index('mode').loc[modes]
            pos = np.arange(len(modes))
            ax.barh(pos-.17, part.delta_oa, .32, color='#237A78', label='OA')
            ax.barh(pos+.17, part.delta_ba, .32, color='#B34463', label='BA')
            ax.set_yticks(pos, labels)
            ax.axvline(0, color='#555555', linewidth=.8)
            ax.grid(axis='x', alpha=.2)
            ax.set_title(dataset+' / '+stage)
            ax.set_xlabel('Change from prior mean-correction baseline (pp)')
    axes[0, 0].invert_yaxis()
    axes[0, -1].legend()
    fig.suptitle('Source-trained frequency residual; same64 labels and 5% expert handoff')
    fig.tight_layout()
    fig.savefig(OUT/'frequency_increment.png', dpi=160)
    plt.close(fig)
    profiles = []
    for p in sorted((OUT/'tasks').rglob('gate_statistics.csv')):
        df = pd.read_csv(p)
        row = df[df['mode'].eq('selective')].set_index('band')['mean'].to_dict()
        row['task'] = p.parent.name
        profiles.append(row)
    table = pd.DataFrame(profiles).set_index('task')
    fig, ax = plt.subplots(figsize=(10, 9))
    left = np.zeros(len(table))
    colors = ['#276DAB', '#218477', '#B34D65', '#A08C21']
    for band, color in zip(['LL', 'LH', 'HL', 'HH'], colors):
        ax.barh(np.arange(len(table)), table[band], left=left, label=band, color=color)
        left += table[band].to_numpy()
    ax.set_yticks(np.arange(len(table)), table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Mean target-cohort neural fusion weight')
    ax.set_title('Frequency preference is not proof of sample-dependent selection')
    ax.legend(ncol=4, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT/'gate_profiles.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
