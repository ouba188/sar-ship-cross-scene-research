"""Plot predeclared paired comparisons without selecting experiments."""

from experiment import OUT, TASKS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

frame = pd.read_csv(OUT/'mean_metrics.csv')
frame = frame.loc[frame['mode'].eq('fusion')]
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True, layout='constrained')
for ax, task in zip(axes, TASKS):
    part = frame.loc[frame.task.eq(task)]
    for offset, stage, color, label in [(-.17, 'frozen', '#26827a', 'Frozen features'),
                                       (.17, 'updated', '#aa4d71', 'R18 layer4 updated')]:
        vals = [float(part.loc[part.stage.eq(stage) & part.arm.eq(a), 'ba'].iloc[0])
                for a in ('S', 'S_U', 'S_T')]
        bars = ax.bar(np.arange(3)+offset, vals, width=.32, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=3)
    ax.set_title(task, fontsize=12)
    ax.set_xticks(range(3), ['S', 'S + source', 'S + target'])
    ax.tick_params(axis='x', labelsize=9)
    ax.set_ylim(0, 52)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Balanced accuracy (%)')
axes[-1].legend(loc='upper left', fontsize=8)
fig.suptitle('Same queries and support64; equal-size, class-matched additions\nCompact dual-readout diagnostic, mean of two predefined splits', fontsize=12)
fig.savefig(OUT/'matched_comparisons.png', dpi=180)
plt.close(fig)
