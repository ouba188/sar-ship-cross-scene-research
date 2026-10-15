"""Display mechanism interventions separately from the actual support64 probe."""

from reverse_analysis import OUT, TASKS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

m = pd.read_csv(OUT/'mean_metrics.csv')
p = m.pivot(index='task', columns='intervention', values='ba').loc[list(TASKS)]
s = pd.read_csv(OUT/'support64/mean_metrics.csv').pivot(index='task', columns='arm', values='ba').loc[list(TASKS)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
x = np.arange(4)
colors = ['#24887f', '#b55476', '#4d75ad', '#8d8435']
for j, (key, label) in enumerate([('bias_only', 'Class bias only'), ('old_score_remix_bias', 'Old-score remix + bias'),
                                 ('new_evidence_bias', 'Remaining features + bias'), ('full', 'Full target-experience change')]):
    axes[0].bar(x+(j-1.5)*.19, p[key]-p.reference, width=.18, color=colors[j], label=label)
for j, (key, label) in enumerate([('global64_only', 'Global refit only'), ('global64_residual64', 'Global refit + residual64')]):
    axes[1].bar(x+(j-.5)*.3, s[key]-s.original_residual64, width=.28, color=colors[j], label=label)
for ax in axes:
    ax.set_xticks(x, ['Los Angeles', 'Port Klang', 'Santos', 'OpenSARShip2'])
    ax.tick_params(axis='x', labelsize=9)
    ax.axhline(0, color='#444444', linewidth=.8)
    ax.set_ylabel('Change in balanced accuracy (percentage points)')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(-4, 6.5)
axes[0].set_title('Mechanism diagnosis: extra-label teacher', fontsize=12)
axes[1].set_title('Actual fitting: the same 64 target labels', fontsize=12)
fig.suptitle('Frozen features; two predefined developmental splits; no neural training', fontsize=12)
fig.savefig(OUT/'reverse_findings.png', dpi=180)
plt.close(fig)
