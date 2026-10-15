"""Plot completed budget study without metric or parameter selection."""

from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


OUT = Path(__file__).resolve().parent/'support_budget_20261012'
data = pd.read_csv(OUT/'budget_summary.csv')
target = data[data.family == 'target']
styles = {'raw': ('#454545', 'Original'), 'corrected': ('#008571', 'Learned repair'),
          'simple': ('#a34482', 'Simple attenuation')}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
panels = [
    ('ba_mean', 'Full system BA (%)', False),
    ('oa_mean', 'Full system OA (%)', False),
    ('ba_std', 'Mean within-port BA SD (pp)', False),
    ('portclass_risk_gain_mean', 'Expert class-balanced risk gain', True),
    ('portclass_risk_gain_mean', 'Full class-balanced risk gain', False),
    ('delta_brier_mean', 'Full row-weighted risk gain', False)]
for i, (ax, (metric, title, expert)) in enumerate(zip(axes.flat, panels)):
    for suffix, (color, label) in styles.items():
        if i >= 3 and suffix == 'raw':
            continue
        arm = ('expert_' if expert else 'fusion_')+suffix
        view = target[target.arm == arm].sort_values('budget')
        if expert:
            label = 'Kernel replacement: U_K' if suffix == 'corrected' else 'Readout attenuation (not U_K)'
        ax.plot(view.budget, view[metric], '-o', color=color, label=label, linewidth=1.8, markersize=5)
    ax.set_title(f'({chr(97+i)}) {title}', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks([16, 64, 128], ['16', '64', '128'])
    ax.set_xlim(13, 155)
    ax.set_xlabel('Labeled support cases B')
    ax.grid(axis='y', alpha=.18)
    if i >= 3:
        ax.axhline(0, color='#777777', linewidth=.8, linestyle=':')
        ax.set_ylabel('Old risk - new risk; higher is better')
    if i == 0:
        ax.legend(frameon=False, fontsize=8)
    if i == 3:
        ax.legend(frameon=False, fontsize=7, loc='lower left')
fig.suptitle('Three target ports: fixed queries and nested support budgets', fontsize=14)
fig.savefig(OUT/'BUDGET_EVIDENCE.png', dpi=180)
fig.savefig(OUT/'BUDGET_EVIDENCE.pdf')
plt.close(fig)
print('SAVED', OUT/'BUDGET_EVIDENCE.png')
