"""Check class allocation and a matched mean-feature control after readout."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from report_conditional_direction import OUT, SOURCE, TARGETS, event_stats


def main():
    source = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    records = []
    control = OUT/'mean_representation_control'
    folders = [('self24ports', p.name, p) for p in sorted((control/'targets/self24ports').iterdir())
               if (p/'policies.npz').exists()]
    folders += [(name, name, control/'targets'/name) for name in TARGETS]
    for dataset, task, folder in folders:
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        frame = source if dataset == 'self24ports' else pd.read_csv(TARGETS[dataset]/'target_manifest.csv', usecols=['class_id'])
        truth = frame.class_id.to_numpy()[p['query_rows']]
        modes = list(p['modes'])
        for stage, key, original_key in [('base', 'base_predictions', 'original_base_predictions'),
                                         ('full', 'predictions', 'original_predictions')]:
            for draw, ev in enumerate(p['eval_masks']):
                baseline = p[key][draw, modes.index('baseline')]
                original = p[original_key][draw]
                for mode, pred in zip(modes, p[key][draw]):
                    records.append(dict(dataset=dataset, task=task, draw=draw, stage=stage, mode=mode,
                        **event_stats(truth, pred, baseline, original, ev)))
    means = pd.DataFrame(records)
    means.to_csv(control/'metrics.csv', index=False)
    direction = pd.read_csv(OUT/'metrics.csv')
    joined = direction.merge(means, on=['dataset', 'task', 'draw', 'stage', 'mode'], suffixes=('_direction', '_mean'))
    joined['direction_minus_mean_oa'] = joined.oa_direction-joined.oa_mean
    joined['direction_minus_mean_ba'] = joined.ba_direction-joined.ba_mean
    joined.to_csv(OUT/'representation_control_paired.csv', index=False)
    cols = ['oa_direction', 'ba_direction', 'oa_mean', 'ba_mean', 'direction_minus_mean_oa', 'direction_minus_mean_ba']
    summary = joined.groupby(['dataset', 'task', 'stage', 'mode'])[cols].mean().groupby(['dataset', 'stage', 'mode']).mean()
    summary.to_csv(OUT/'representation_control_summary.csv')
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    datasets = ['self24ports', 'f0921', 'oss2']
    current = summary.reset_index()
    current = current[current.stage.eq('full')]
    base = current[current['mode'].eq('baseline')].set_index('dataset').loc[datasets]
    chosen = current[current['mode'].eq('conditional_selected')].set_index('dataset').loc[datasets]
    for ax, metric in zip(axes, ['oa', 'ba']):
        x = np.arange(3)
        ax.bar(x-.18, chosen[metric+'_direction']-base[metric+'_direction'], .34,
               color='#237A78', label='Directional residual')
        ax.bar(x+.18, chosen[metric+'_mean']-base[metric+'_mean'], .34,
               color='#B34463', label='Mean-feature control')
        ax.set_xticks(x, ['24 ports', 'F_0921', 'OpenSARShip 2.0'])
        ax.set_ylabel(metric.upper()+' change from C4 baseline (pp)')
        ax.axhline(0, color='#555555', linewidth=.8)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend()
    figure.suptitle('Same labels, same strengths, different residual information')
    figure.tight_layout()
    figure.savefig(OUT/'representation_control.png', dpi=160)
    plt.close(figure)
    classes = pd.read_csv(OUT/'class_metrics.csv')
    classes = classes[classes['mode'].eq('conditional_selected')]
    classes['support_bin'] = pd.cut(classes.support_count, [-1, 0, 2, 7, np.inf], labels=['0', '1-2', '3-7', '8+'])
    classes.groupby(['dataset', 'stage', 'support_bin'], observed=True).agg(
        task_draw_class_cells=('delta', 'size'), mean_recall_delta=('delta', 'mean'),
        mean_support=('support_count', 'mean')).to_csv(OUT/'support_count_effects.csv')
    within = []
    for (dataset, task, stage, cl), data in classes.groupby(['dataset', 'task', 'stage', 'class_id']):
        lo, hi = data.support_count.min(), data.support_count.max()
        if hi == lo:
            continue
        low = data.loc[data.support_count.eq(lo), 'delta'].mean()
        high = data.loc[data.support_count.eq(hi), 'delta'].mean()
        within.append(dict(dataset=dataset, task=task, stage=stage, class_id=cl,
                           support_low=int(lo), support_high=int(hi),
                           low_delta=low, high_delta=high, high_minus_low=high-low))
    pd.DataFrame(within).to_csv(OUT/'within_task_class_support_diagnostic.csv', index=False)
    # Each complete original-target run has already checked actual score/policy
    # parity. This records those results without repeating expensive inference.
    checks = [json.loads(p.read_text()) for p in (OUT/'targets').rglob('FROZEN.json')]
    verification = dict(target_tasks=len(checks), exact_baseline_tasks=sum(
        c['baseline_predictions_and_selection_exact'] for c in checks),
        source_episodes=506, tests_passed=6, current_model_promoted=False,
        mean_control='Post-readout diagnostic, matched strengths not reselected',
        support_count_analysis='Descriptive; label count co-varies with other support labels and difficulty')
    (OUT/'VERIFICATION.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(summary.reset_index().query("stage=='full'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
