"""Plot measured stability/discrimination tradeoffs; no selected target winners."""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent/'paired_evidence_factors_20261011'


def main():
    source = pd.read_csv(OUT/'source_results.csv')
    geometry = pd.read_csv(OUT/'geometry.csv')
    geo = geometry[(geometry.role == 'held') & (geometry.scope == 'cross_port') &
                   (geometry.part.isin(['candidate', 'full']))].groupby('method').normalized_drift.mean()
    accuracy = source.groupby('variant').BA.mean()
    classes = pd.read_csv(OUT/'mechanism/source_classes.csv')
    table = classes[(classes.role == 'held') & (classes.part == 'candidate')].groupby(['class_id', 'method']).recall.mean().unstack('method')
    delta = table.paired-table.class_only
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={'width_ratios': [1.05, 1]})
    settings = [('paired', 'Paired evidence 7D', '#087f8c', (8, -16)),
                ('class_only', 'Class-only 7D', '#cf5c36', (8, 10)),
                ('stable_only', 'Stability-only 7D', '#699934', (-5, -22)),
                ('head', 'Head span 7D', '#8e5ca2', (-70, 10)),
                ('pca', 'PCA 7D', '#cc9b17', (-15, 10)),
                ('full', 'Full 32D probe', '#555555', (-45, 20))]
    for method, label, color, offset in settings:
        value = accuracy['full_refit' if method == 'full' else method+'_probe7']
        ax[0].scatter(geo[method], value, s=70, color=color, zorder=3)
        ax[0].annotate(label, (geo[method], value), xytext=offset, textcoords='offset points', fontsize=9)
    ax[0].set(xlabel='Held cross-port drift / total variance (lower is better)',
              ylabel='Held-vessel classification BA (%)', xlim=(.65, 1.95), ylim=(20, 43),
              title='A. Stability is not sufficient for classification')
    ax[0].grid(alpha=.18)
    names = ['Bulk', 'Fishing', 'General', 'Chemical', 'Container', 'Crude', 'Tug', 'Offshore']
    ax[1].barh(names, delta.to_numpy(), color=['#087f8c' if v > 0 else '#cf5c36' for v in delta], height=.65)
    ax[1].axvline(0, color='#444444', linewidth=.8)
    ax[1].invert_yaxis()
    ax[1].set(xlabel='Recall change: paired 7D minus class-only 7D (pp)',
              title='B. What additional stability costs', xlim=(-16, 8))
    for j, value in enumerate(delta):
        ax[1].text(value+(.3 if value >= 0 else -.3), j, f'{value:+.2f}', va='center',
                   ha='left' if value >= 0 else 'right', fontsize=9)
    fig.suptitle('Explicit factor separation: source held-vessel evidence', fontsize=14)
    fig.text(.5, .015, 'Means over two existing representations and two seeds; same reused development cohort, not four independent datasets.',
             ha='center', fontsize=8, color='#555555')
    fig.tight_layout(rect=(0, .055, 1, .94))
    fig.savefig(OUT/'FACTOR_EVIDENCE.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
