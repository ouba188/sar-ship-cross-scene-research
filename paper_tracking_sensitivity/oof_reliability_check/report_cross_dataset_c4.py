"""Paired descriptive readout; no parameter selection from target outcomes."""

import json
import numpy as np
import pandas as pd
from cross_dataset_c4 import OUT, ROOT, DATASETS, ARMS, CLASSES


def main():
    metrics = pd.read_csv(OUT/'metrics.csv')
    classes = pd.read_csv(OUT/'class_metrics.csv')
    reference = metrics.loc[metrics.arm.eq('reference'), ['dataset', 'draw', 'branch', 'oa', 'ba']]
    paired = metrics.merge(reference, on=['dataset', 'draw', 'branch'], suffixes=('', '_reference'), validate='many_to_one')
    paired['delta_oa_pp'] = paired.oa-paired.oa_reference
    paired['delta_ba_pp'] = paired.ba-paired.ba_reference
    paired.to_csv(OUT/'paired_draws.csv', index=False)
    deltas = paired.groupby(['dataset', 'branch', 'arm'], sort=False).agg(
        oa=('oa', 'mean'), ba=('ba', 'mean'), delta_oa_pp=('delta_oa_pp', 'mean'),
        delta_ba_pp=('delta_ba_pp', 'mean'), oa_positive_draws=('delta_oa_pp', lambda v: int((v > 1e-9).sum())),
        ba_positive_draws=('delta_ba_pp', lambda v: int((v > 1e-9).sum())))
    deltas.to_csv(OUT/'paired_summary.csv')
    c = classes.groupby(['dataset', 'branch', 'class_id', 'class_name', 'arm'], sort=False)[['n', 'recall', 'recovered', 'harmed']].mean().reset_index()
    ref = c.loc[c.arm.eq('reference'), ['dataset', 'branch', 'class_id', 'recall']]
    c = c.merge(ref, on=['dataset', 'branch', 'class_id'], suffixes=('', '_reference'), validate='many_to_one')
    c['delta_recall_pp'] = c.recall-c.recall_reference
    c.to_csv(OUT/'class_summary.csv', index=False)
    transitions = []
    counts, checks = [], []
    for name in metrics.dataset.unique():
        directory = ROOT/DATASETS[name][0]
        frame = pd.read_csv(directory/'target_manifest.csv', keep_default_na=False)
        truth = frame.class_id.to_numpy(int)
        for k, n in frame.class_id.value_counts().sort_index().items():
            counts.append(dict(dataset=name, class_id=k, class_name=CLASSES[k], samples=n))
        checks.append(dict(dataset=name, **json.loads((OUT/name/'FROZEN.json').read_text()),
                           encoder=json.loads((OUT/name/'encoder_replay.json').read_text())))
        with np.load(OUT/name/'policies.npz') as p:
            for draw, ev in enumerate(p['eval_masks']):
                for arm in ('c4_matched', 'fixed_class_mix'):
                    candidate = p['predictions'][draw, ARMS.index(arm)]
                    previous = p['predictions'][draw, 0]
                    changed = ev & (candidate != previous)
                    triples, n = np.unique(np.c_[truth[changed], previous[changed], candidate[changed]], axis=0, return_counts=True)
                    for (y, a, b), count in zip(triples, n):
                        transitions.append(dict(dataset=name, draw=draw, arm=arm, true_class=CLASSES[y],
                            previous_prediction=CLASSES[a], new_prediction=CLASSES[b], count=count,
                            event='recovered' if y == b else 'harmed' if y == a else 'wrong_to_wrong'))
    pd.DataFrame(counts).to_csv(OUT/'class_counts.csv', index=False)
    pd.DataFrame(transitions).to_csv(OUT/'prediction_transitions.csv', index=False)
    (OUT/'verification_summary.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(metrics.dataset.unique())
    labels = {'oss1': 'OpenSARShip 1.0', 'oss2': 'OpenSARShip 2.0', 'fusar': 'FUSAR', 'f0921': 'F_0921'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    for ax, metric in zip(axes, ('oa', 'ba')):
        for offset, arm, color, label in [(-.19, 'c4_matched', '#247a79', 'C4 matched'),
                                          (.19, 'fixed_class_mix', '#bd5261', 'C4 + fixed class mix')]:
            values = [deltas.loc[(n, 'full', arm), f'delta_{metric}_pp'] for n in names]
            bars = ax.bar(x+offset, values, width=.36, color=color, label=label)
            ax.bar_label(bars, fmt='%+.2f', fontsize=8, padding=3)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set_xticks(x, [labels[n] for n in names], rotation=18, ha='right')
        ax.set_ylabel('Change vs matched reference (percentage points)')
        ax.set_title(metric.upper()+' | full handoff model')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('SAR cross-dataset transfer: same 64 labels, at most 5% handoff')
    fig.savefig(OUT/'cross_dataset_deltas.png', dpi=180)
    plt.close(fig)
    diagnostic = OUT/'support_refit_diagnostic_summary.csv'
    if diagnostic.exists():
        table = pd.read_csv(diagnostic).set_index(['dataset', 'branch', 'arm'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        modes = [(-.26, 'c4_query_only', '#477eab', 'Query averaging only'),
                 (0., 'c4_support_refit', '#267d60', 'Also refit support (diagnostic)'),
                 (.26, 'c4_matched', '#bb5869', 'Also refit source head')]
        for ax, metric in zip(axes, ('oa', 'ba')):
            for offset, arm, color, label in modes:
                values = [table.loc[(n, 'full', arm), metric]-table.loc[(n, 'full', 'reference'), metric] for n in names]
                bars = ax.bar(x+offset, values, width=.25, color=color, label=label)
                ax.bar_label(bars, fmt='%+.2f', fontsize=7,
                             padding=13 if arm == 'c4_support_refit' else 3)
            ax.axhline(0, color='black', linewidth=.8)
            ax.set_xticks(x, [labels[n] for n in names], rotation=18, ha='right')
            ax.set_ylabel('Change vs matched reference (percentage points)')
            ax.set_title(metric.upper()+' | full handoff model')
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
            ax.margins(y=.25)
        axes[0].legend(frameon=False, fontsize=7, loc='lower left')
        fig.suptitle('Stage diagnosis: which refit changes transfer?')
        fig.savefig(OUT/'stage_diagnosis.png', dpi=180)
        plt.close(fig)
    print(deltas.to_string())


if __name__ == '__main__':
    main()
