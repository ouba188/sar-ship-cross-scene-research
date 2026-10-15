"""Compact plots of fixed-checkpoint information removal diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('E:/Hermes/paper_tracking/sensitivity/frequency_information_content_20261009')
NAMES = {'self24ports': '24-port (23 tasks)', 'f0921': 'F_0921', 'oss2': 'OpenSARShip 2.0'}
MODES = ['unsigned_bands', 'no_local_band_layout', 'no_channel_identity', 'no_quadrant_layout']
LABELS = ['Remove local\nsign', 'Flatten local\nbands', 'Average\nchannels', 'Average 2x2\nlayout']
COLORS = ['#236f9c', '#b24942', '#268067']


def main():
    data = pd.read_csv(OUT/'summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    for ax, stage, title in zip(axes, ['base', 'full'], ['Base classifier', 'Full model, rerouted <=5%']):
        values = data[data.stage.eq(stage)].pivot(index='mode', columns='dataset', values='ba')
        x = np.arange(len(MODES))
        for j, (dataset, label) in enumerate(NAMES.items()):
            delta = np.array([values.loc['actual', dataset]-values.loc[m, dataset] for m in MODES])
            bars = ax.bar(x+(j-1)*.24, delta, width=.22, label=label, color=COLORS[j])
            for bar, val in zip(bars, delta):
                ax.annotate(f'{val:.3f}', (bar.get_x()+bar.get_width()/2, val),
                            xytext=(0, 3 if val >= 0 else -11), textcoords='offset points',
                            ha='center', fontsize=7)
        ax.axhline(0, color='#555555', lw=.8)
        ax.set_xticks(x, LABELS, fontsize=9)
        ax.set_ylim(-.13, .62)
        ax.set_title(title, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('BA drop after information removal (percentage points)')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, .93), ncol=3)
    fig.suptitle('What information does the frequency residual use?', fontsize=14)
    fig.text(.5, .015, 'Frozen learned head; original gates; matched support/query intervention; same 64 support labels.\n'
             'Negative bars mean removal helped. Dependence of this checkpoint, not identified physical factors.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=(0, .09, 1, .86))
    fig.savefig(OUT/'information_removal.png', dpi=180)
    fig.savefig(OUT/'information_removal.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
