"""Matched readout and error-flow diagnosis for directional residuals."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probe_display_rotation import ROOT, SOURCE

OUT = ROOT/'conditional_direction_residual_20261009'
TARGETS = {'f0921': ROOT/'f0921_detector_recrop_20261007',
           'oss2': ROOT/'opensarship2_transfer_20261007'}


def event_stats(truth, pred, baseline, original, ev):
    correct, was, initial = pred == truth, baseline == truth, original == truth
    c4_rescue = ~initial & was & ev
    c4_harm = initial & ~was & ev
    return dict(n=int(ev.sum()), oa=100*correct[ev].mean(),
        ba=100*np.mean([correct[ev & (truth == c)].mean() for c in np.unique(truth[ev])]),
        rescues=int((correct & ~was & ev).sum()), harms=int((~correct & was & ev).sum()),
        changes=int(((pred != baseline) & ev).sum()),
        c4_rescues=int(c4_rescue.sum()), c4_rescues_destroyed=int((c4_rescue & ~correct).sum()),
        c4_harms=int(c4_harm.sum()), c4_harms_repaired=int((c4_harm & correct).sum()))


def main():
    source = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id'])
    records, classes, contrasts, strata = [], [], [], []
    folders = [('self24ports', p.name, p) for p in sorted((OUT/'targets/self24ports').iterdir())
               if (p/'policies.npz').exists()]
    folders += [(name, name, OUT/'targets'/name) for name in TARGETS]
    for dataset, task, folder in folders:
        with np.load(folder/'policies.npz') as f:
            p = {k: f[k] for k in f.files}
        frame = source if dataset == 'self24ports' else pd.read_csv(TARGETS[dataset]/'target_manifest.csv')
        truth = frame.class_id.to_numpy()[p['query_rows']]
        modes = list(p['modes'])
        conditions = p['conditions']
        for stage, key, original_key in [('base', 'base_predictions', 'original_base_predictions'),
                                         ('full', 'predictions', 'original_predictions')]:
            for draw, ev in enumerate(p['eval_masks']):
                baseline = p[key][draw, modes.index('baseline')]
                original = p[original_key][draw]
                for mode, pred in zip(modes, p[key][draw]):
                    record = dict(dataset=dataset, task=task, draw=draw, stage=stage, mode=mode)
                    records.append(dict(**record, **event_stats(truth, pred, baseline, original, ev)))
                    for c in np.unique(truth[ev]):
                        mask = ev & (truth == c)
                        classes.append(dict(**record, class_id=int(c), n=int(mask.sum()),
                            support_count=int((truth[p['support_rows'][draw]] == c).sum()),
                            recall=100*np.mean(pred[mask] == c),
                            delta=100*(np.mean(pred[mask] == c)-np.mean(baseline[mask] == c))))
                for candidate, reference in [('conditional_selected', 'unconditional_selected'),
                        ('conditional_fixed1', 'unconditional_fixed1'),
                        ('conditional_fixed1', 'uniform_condition_fixed1')]:
                    pred, ref = p[key][draw, modes.index(candidate)], p[key][draw, modes.index(reference)]
                    m = event_stats(truth, pred, ref, original, ev)
                    r = event_stats(truth, ref, ref, original, ev)
                    contrasts.append(dict(dataset=dataset, task=task, draw=draw, stage=stage,
                        comparison=candidate+' minus '+reference, oa=m['oa']-r['oa'],
                        ba=m['ba']-r['ba'], rescues=m['rescues'], harms=m['harms'], changes=m['changes']))
                # Conditions are precomputed without labels. Report within-class
                # high/low effects rather than conflating different class mixes.
                for mode in ('unconditional_fixed1', 'conditional_fixed1', 'uniform_condition_fixed1'):
                    pred = p[key][draw, modes.index(mode)]
                    change = (pred == truth).astype(float)-(baseline == truth)
                    for j, feature in enumerate(('elongation', 'center_excess')):
                        for level, group in [('low', conditions[:, j] < 1/3), ('high', conditions[:, j] >= 2/3)]:
                            for c in np.unique(truth[ev]):
                                mask = ev & group & (truth == c)
                                if mask.any():
                                    strata.append(dict(dataset=dataset, task=task, draw=draw, stage=stage,
                                        mode=mode, feature=feature, level=level, class_id=int(c),
                                        n=int(mask.sum()), delta=100*change[mask].mean()))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    pd.DataFrame(strata).to_csv(OUT/'condition_class_strata.csv', index=False)
    port = metrics.groupby(['dataset', 'task', 'stage', 'mode'])[['oa', 'ba']].mean().reset_index()
    port.to_csv(OUT/'task_means.csv', index=False)
    summary = port.groupby(['dataset', 'stage', 'mode'])[['oa', 'ba']].mean().reset_index()
    baseline = summary[summary['mode'].eq('baseline')][['dataset', 'stage', 'oa', 'ba']]
    summary = summary.merge(baseline, on=['dataset', 'stage'], suffixes=('', '_baseline'))
    summary['delta_oa'] = summary.oa-summary.oa_baseline
    summary['delta_ba'] = summary.ba-summary.ba_baseline
    summary.to_csv(OUT/'summary.csv', index=False)
    contrast = pd.DataFrame(contrasts)
    contrast.to_csv(OUT/'paired_contrasts.csv', index=False)
    contrast.groupby(['dataset', 'task', 'stage', 'comparison'])[['oa', 'ba']].mean().groupby(
        ['dataset', 'stage', 'comparison']).mean().to_csv(OUT/'contrast_summary.csv')
    events = metrics.groupby(['dataset', 'stage', 'mode'])[
        ['n', 'rescues', 'harms', 'changes', 'c4_rescues', 'c4_rescues_destroyed', 'c4_harms', 'c4_harms_repaired']].sum()
    events['change_precision'] = events.rescues/(events.rescues+events.harms).replace(0, np.nan)
    events['c4_rescue_retention'] = 1-events.c4_rescues_destroyed/events.c4_rescues.replace(0, np.nan)
    events['c4_harm_repair_rate'] = events.c4_harms_repaired/events.c4_harms.replace(0, np.nan)
    events.to_csv(OUT/'event_summary.csv')
    # Five draw deltas, retaining the equal-task definition used for source ports.
    draws = metrics.groupby(['dataset', 'draw', 'stage', 'mode'])[['oa', 'ba']].mean().reset_index()
    ref = draws[draws['mode'].eq('baseline')].drop(columns='mode')
    draws = draws.merge(ref, on=['dataset', 'draw', 'stage'], suffixes=('', '_baseline'))
    draws['delta_oa'], draws['delta_ba'] = draws.oa-draws.oa_baseline, draws.ba-draws.ba_baseline
    draws.to_csv(OUT/'draw_metrics.csv', index=False)
    modes = ['unconditional_selected', 'conditional_selected', 'unconditional_fixed1',
             'conditional_fixed1', 'uniform_condition_fixed1']
    labels = ['Direction / selected', 'Conditional / selected', 'Direction / eta=1',
              'Conditional / eta=1', 'Uniform damping / eta=1']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    for ax, dataset in zip(axes, ['self24ports', 'f0921', 'oss2']):
        part = summary[summary.dataset.eq(dataset) & summary.stage.eq('full')].set_index('mode').loc[modes]
        yy = np.arange(len(modes))
        ax.barh(yy-.17, part.delta_oa, .32, label='OA', color='#237A78')
        ax.barh(yy+.17, part.delta_ba, .32, label='BA', color='#B34463')
        ax.axvline(0, color='#555555', linewidth=.8)
        ax.set_yticks(yy, labels)
        ax.set_title(dataset)
        ax.set_xlabel('Change from C4 + same64 baseline (pp)')
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(OUT/'incremental_effects.png', dpi=160)
    plt.close(fig)
    print(summary.to_string(index=False), flush=True)
    (OUT/'READOUT_COMPLETE.json').write_text(json.dumps(dict(tasks=len(folders),
        historical_development=True, query_labels_used_only_for_readout=True,
        event_counts='Five repeated support draws; not independent ships or metric weights'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
