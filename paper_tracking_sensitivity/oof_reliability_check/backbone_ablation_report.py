"""Post-freeze readout: matched contrasts, per-class rescues, and timing scope."""

import json
import time

import numpy as np
import pandas as pd

from backbone_ablation_core import OUT, MODES, SOURCE, ROOT, folders, info_for, load_npz, write
from prepare_learned_frequency import DATASETS

CONTRASTS = tuple((factor, f'{a}_{head}', f'{b}_{head}')
    for factor, a, b in [('r18_increment', 'dual_refit', 'r50'),
                         ('source_prior_refit', 'dual_refit', 'dual_original'),
                         ('r50_replacement', 'r50', 'dual_original')]
    for head in ('expert', 'late', 'unified')) + (
    ('r50_head_structure', 'r50_unified', 'r50_late'),)


def ba(y, p):
    return 100*float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def report():
    tasks = folders()
    expected = json.loads((OUT/'PROTOCOL.json').read_text())['tasks']
    if [info_for(f)[1] for f in tasks] != expected or len(tasks) != 25:
        raise ValueError('Frozen task list differs')
    if not all((OUT/'targets'/info_for(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Finish all25 predictions before target query evaluation')
    marker = OUT/'ALL_PREDICTIONS_SAVED.json'
    if not marker.exists():
        write(marker, dict(tasks=25, draws=125, modes=MODES,
            saved_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), query_metrics_not_yet_computed=True))
    metrics, classes, effects, class_effects, timings, fits = [], [], [], [], [], []
    for folder in tasks:
        info, suffix = info_for(folder)
        dst = OUT/'targets'/suffix
        p = load_npz(dst/'predictions.npz')
        details = json.loads((dst/'COMPLETE.json').read_text())
        with np.load(folder/'target.npz') as z:
            support_labels = z['support_labels']
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy(int)
        np.testing.assert_array_equal(truth[p['support']], support_labels)
        common = dict(dataset=info['dataset'], task=info['task'])
        fits.append(dict(**common, source_n=details['source_n'], r50_width=details['r50_width'],
            dual_width=details['dual_width'], source_fit_seconds=details['source_fit_seconds'],
            shared_both_expert_kernel_build_seconds=details['shared_both_expert_kernel_build_seconds'],
            total_seconds=details['total_seconds'],
            maximum_original_replay_error=max(r['original_expert_max_error'] for r in details['replay'])))
        timings.extend(dict(**common, **row) for row in details['timing'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            present = np.unique(y)
            preds = {str(mode): p['predictions'][draw, mi, ev] for mi, mode in enumerate(p['modes'])}
            for mode, pred in preds.items():
                metrics.append(dict(**common, draw=draw, mode=mode, query_n=len(y),
                                    oa=100*np.mean(pred == y), ba=ba(y, pred)))
                for c in present:
                    mask = y == c
                    classes.append(dict(**common, draw=draw, mode=mode, class_id=int(c),
                        n=int(mask.sum()), support_n=int(np.sum(support_labels[draw] == c)),
                        recall=100*np.mean(pred[mask] == c)))
            for factor, a, b in CONTRASTS:
                pa, pb = preds[a], preds[b]
                rescue, harm = (pa == y) & (pb != y), (pa != y) & (pb == y)
                effects.append(dict(**common, draw=draw, factor=factor, candidate=a, control=b,
                    delta_oa=100*np.mean((pa == y).astype(float)-(pb == y)),
                    delta_ba=ba(y, pa)-ba(y, pb), rescues=int(rescue.sum()), harms=int(harm.sum()),
                    net_correct=int(rescue.sum()-harm.sum()), changed=int(np.sum(pa != pb)), query_n=len(y)))
                for c in present:
                    mask = y == c
                    delta = 100*np.mean((pa[mask] == c).astype(float)-(pb[mask] == c))
                    class_effects.append(dict(**common, draw=draw, factor=factor, candidate=a, control=b,
                        class_id=int(c), n=int(mask.sum()), delta_recall=delta,
                        ba_contribution=delta/len(present), rescues=int(rescue[mask].sum()),
                        harms=int(harm[mask].sum()), net_correct=int(rescue[mask].sum()-harm[mask].sum())))
    tables = dict(metrics=pd.DataFrame(metrics), class_metrics=pd.DataFrame(classes),
        effects=pd.DataFrame(effects), class_effects=pd.DataFrame(class_effects),
        timing=pd.DataFrame(timings), task_cost=pd.DataFrame(fits))
    for name, table in tables.items():
        table.to_csv(OUT/f'{name}.csv', index=False)
    task_means = tables['metrics'].groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean().reset_index()
    task_means.to_csv(OUT/'task_means.csv', index=False)
    summary = task_means.groupby(['dataset', 'mode'])[['oa', 'ba']].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    ekeys = ['factor', 'candidate', 'control']
    values = ['delta_oa', 'delta_ba', 'rescues', 'harms', 'net_correct', 'changed']
    task_effects = tables['effects'].groupby(['dataset', 'task', *ekeys])[values].mean().reset_index()
    task_effects.to_csv(OUT/'task_effects.csv', index=False)
    effect_summary = task_effects.groupby(['dataset', *ekeys])[values].mean().reset_index()
    effect_summary.to_csv(OUT/'effect_summary.csv', index=False)
    draw_effects = tables['effects'].groupby(['dataset', 'draw', *ekeys])[values].mean().reset_index()
    draw_effects.to_csv(OUT/'draw_effects.csv', index=False)
    # Absent classes contribute zero to a task's BA difference; divide by ALL tasks.
    ce = tables['class_effects'].copy()
    task_counts = task_means.groupby('dataset').task.nunique()
    class_summary = ce.groupby(['dataset', *ekeys, 'class_id']).agg(
        ba_contribution=('ba_contribution', 'sum'), rescues=('rescues', 'sum'),
        harms=('harms', 'sum'), net_correct=('net_correct', 'sum'), query_exposures=('n', 'sum')).reset_index()
    class_summary['ba_contribution'] /= class_summary.dataset.map(task_counts)*5
    class_summary.to_csv(OUT/'class_contributions.csv', index=False)
    for _, row in effect_summary.iterrows():
        subset = class_summary[(class_summary.dataset == row.dataset) &
            (class_summary.factor == row.factor) & (class_summary.candidate == row.candidate)]
        np.testing.assert_allclose(subset.ba_contribution.sum(), row.delta_ba, atol=1e-10)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, modes=MODES, evaluated_rows=len(metrics),
        all_predictions_saved_before_query_metrics=True, original_predictions_exact=True,
        maximum_original_expert_replay_error=max(r['maximum_original_replay_error'] for r in fits),
        no_target_parameter_selection=True, cpu_threads=2, independent_confirmation=False))
    render_report(summary, effect_summary, class_summary, draw_effects, tables)
    print(summary.to_string(index=False), flush=True)
    print(effect_summary.to_string(index=False), flush=True)


def render_report(summary, effects, classes, draws, tables):
    ds_order = ['self24ports', 'f0921', 'oss2']
    ds_labels = dict(self24ports='24 ports (23 tasks)', f0921='F_0921', oss2='OSS2')
    lines = ['# Fixed Backbone Necessity Ablation', '',
        'Completed all 25 tasks, five original 64-label support draws each. All 1,250 task/draw/mode predictions were frozen before query evaluation.', '',
        'Metrics below are OA / BA (%). Ports use an equal mean over 23 tasks, then five draws. No parameter was chosen using these results.', '',
        '| Mode | 24 ports (23 tasks) | F_0921 | OSS2 |', '| --- | --- | --- | --- |']
    for mode in MODES:
        values = []
        for ds in ds_order:
            row = summary[(summary.dataset == ds) & (summary['mode'] == mode)].iloc[0]
            values.append(f'{row.oa:.5f} / {row.ba:.5f}')
        lines.append('| '+mode+' | '+' | '.join(values)+' |')
    lines += ['', '## Matched R18 Increment', '',
        'Positive values favor dual backbones over R50-only, with both source heads refitted on the same permitted source cohort and current frozen representation. Rescue/harm counts below are summed across five overlapping draws, not unique images.', '',
        '| Dataset | Head | Delta OA (pp) | Delta BA (pp) | Rescued | Harmed | Net | BA-positive draws |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for ds in ds_order:
        for head in ('expert', 'late', 'unified'):
            mode = 'dual_refit_'+head
            row = effects[(effects.dataset == ds) & (effects.factor == 'r18_increment') & (effects.candidate == mode)].iloc[0]
            detail = tables['effects'][(tables['effects'].dataset == ds) &
                (tables['effects'].factor == 'r18_increment') & (tables['effects'].candidate == mode)]
            dsdraw = draws[(draws.dataset == ds) & (draws.factor == 'r18_increment') & (draws.candidate == mode)]
            lines.append(f'| {ds_labels[ds]} | {head} | {row.delta_oa:+.5f} | {row.delta_ba:+.5f} | '
                f'{detail.rescues.sum()} | {detail.harms.sum()} | {detail.net_correct.sum():+d} | {int((dsdraw.delta_ba > 0).sum())}/5 |')
    lines += ['', '## Prior Refit Control', '',
        'Dual refit minus original dual separates the source-prior update from SSL18 removal. Port priors replay exactly; external priors move from pre-tail source fitting to the cached joint-tail deployment representation.', '',
        '| Dataset | Head | Delta OA (pp) | Delta BA (pp) |', '| --- | --- | --- | --- |']
    for ds in ds_order:
        for head in ('expert', 'late', 'unified'):
            row = effects[(effects.dataset == ds) & (effects.factor == 'source_prior_refit') & (effects.candidate == 'dual_refit_'+head)].iloc[0]
            lines.append(f'| {ds_labels[ds]} | {head} | {row.delta_oa:+.5f} | {row.delta_ba:+.5f} |')
    lines += ['', '## Per-Class Explanation', '',
        'The following are each class contribution to the equal-task BA increment of adding R18 in late fusion. An absent class contributes zero for that task. Full per-class recalls, support counts, rescues and harms are in class_metrics.csv, class_effects.csv, and class_contributions.csv.', '',
        '| Dataset | Class ID | BA contribution (pp) | Rescued | Harmed | Net |', '| --- | --- | --- | --- | --- | --- |']
    selected = classes[(classes.factor == 'r18_increment') & (classes.candidate == 'dual_refit_late')]
    for ds in ds_order:
        for row in selected[selected.dataset == ds].sort_values('ba_contribution').itertuples():
            lines.append(f'| {ds_labels[ds]} | {row.class_id} | {row.ba_contribution:+.5f} | {row.rescues} | {row.harms} | {row.net_correct:+d} |')
    cost = tables['task_cost']
    timing = tables['timing']
    lines += ['', '## Timing and Resource Boundary', '',
        f'- Two CPU threads per process. Total measured task time: {cost.total_seconds.sum():.2f} s; source input preparation plus both head fits: {cost.source_fit_seconds.sum():.2f} s; shared construction of both expert kernels: {cost.shared_both_expert_kernel_build_seconds.sum():.2f} s.',
        '- Per-draw cached-kernel adaptation times include solving and scoring all target rows; the unified timing also includes weighted kernel/prior assembly. They exclude backbone/image forward, image loading, extraction, source fitting, and feature-to-support kernel construction. Concurrent other experiments can affect wall time. This is not an isolated hardware throughput benchmark.',
        f'- Mean R50 two-head solve/score/readout: {1000*timing.r50_two_head_solve_score_seconds.mean():.3f} ms; mean R50 unified solve/score: {1000*timing.r50_unified_solve_score_seconds.mean():.3f} ms.',
        '- R50 expert retains 128 PCA and 2048 complementary coordinates. The removed SSL branch has 64 PCA and 512 complementary coordinates. Original divisors sqrt(192), sqrt(2560), source stable scaling, and all surviving feature scales are held fixed. This is a drop-only metric, not a search over single-backbone normalization.',
        '- Each source coefficient matrix has 8 output columns. Each support residual coefficient matrix is 64x8; two heads solve two such systems, one unified head solves one. No backbone parameter/FLOP reduction or end-to-end acceleration was measured.',
        '- One R50 backbone does NOT mean one forward: the unchanged full base retains C4 four-rotation forwards and the frozen frequency representation. Removing the SSL18 branch does not establish half the total compute. The concurrent C4 input ablation is a separate intervention.',
        '- Prediction and small coefficient archives are saved. No feature copies, new model downloads, or old-result modifications.', '',
        '## Verification and Limits', '',
        '- Exact base, original expert, original equal-probability fusion and original unified predictions replay on every task and draw. Original expert scores match the existing tolerance of 3e-5 absolute / 1e-6 relative.',
        '- Source rows equal the base source cohort; target ports or previously purged external identity rows are excluded. Source heads are balanced ridge1 with unpenalized intercept; target residuals use the original class-balanced ridge8 and equal fusion.',
        '- ExpertContext helpers load the full source manifest, including labels for excluded rows. The fit receives only the permitted source slice. Target fits receive the stored 64 support labels. No claim is made that all label-containing files remained unopened.',
        '- No source-episode hyperparameter selection or new source OOF gate was needed: rho=.5 and lambda=8 were frozen. Geometry and backbone weights remain inherited.',
        '- The R50-only expert uses the original R50 directional and complementary information. The full base still uses its complete frozen condition/frequency representation. This experiment does not retrain a new end-to-end single model or exhaust single-model architectures.',
        '- F_0921 was historically exposed to the SSL pretraining corpus; source head identity exclusion does not undo that exposure. Overlapping support draws are not independent replications.',
        '- The concurrently supplied fusion_module_ablation_20261010 result motivates reading redundancy/blocking together with these contrasts; it did not change parameters, modes, or task selection. This ablation alone cannot identify the mechanism behind any R18 gain or loss.', '']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    report()
