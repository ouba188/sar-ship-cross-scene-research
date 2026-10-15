"""Post-freeze frequency-pair readout; target labels never select a policy."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frequency_pair_run import OUT, PREV, EXP, MODES
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_band_report import BANDS, markdown_table
from frequency_reliability_run import statistics

BASELINES = ('previous_selected', 'smooth_parent', 'learned')
RULES = ('previous_selected', 'class_local', 'pair_local')
THRESHOLDS = {'smooth': 0., 'smooth02': .02, 'smooth05': .05}
STAGES = (('base', 'base_predictions'), ('full', 'predictions'),
          ('frozen_previous_route', 'frozen_predictions'))
VALUES = ['oa', 'ba', 'macro_f1_8']
DELTAS = [f'delta_{v}_vs_{b}' for b in BASELINES for v in VALUES]


def diagnostic(name, records, keys, values, counts=()):
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / f'{name}.csv', index=False)
    group = ['dataset', *keys]
    task = frame.groupby([*group, 'task'])[values].mean().reset_index()
    summary = task.groupby(group)[values].mean()
    if counts:
        summary = summary.join(frame.groupby(group)[list(counts)].sum())
    summary.reset_index().to_csv(OUT / f'{name}_summary.csv', index=False)
    return summary.reset_index()


def report():
    folders = [p.parent for p in sorted((EXP / 'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p / 'PREPARED.json').read_text()) for p in folders]
    expected = {(i['dataset'], i['task']) for i in infos}
    markers = {(p.parent.parent.name, p.parent.name)
               for p in (OUT / 'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or markers != expected:
        raise ValueError('Freeze all25 matching target tasks before any target-label reads')
    sources = []
    for info in infos:
        suffix = Path(info['dataset']) / info['task']
        if not (OUT / 'targets' / suffix / 'policies.npz').is_file():
            raise ValueError(f'Missing frozen predictions: {suffix}')
        source = json.loads((OUT / 'source' / suffix / 'SELECTION.json').read_text())
        previous = json.loads((PREV / 'source' / suffix / 'SELECTION.json').read_text())
        gains, old_gains = source['source_delta_ba'], previous['source_delta_ba']
        parent = max(THRESHOLDS, key=lambda r: old_gains[r])
        if (not all(np.isfinite(gains[r]) for r in RULES)
                or source['rule'] != max(RULES, key=lambda r: gains[r])
                or source['parent_rule'] != parent or source['threshold'] != THRESHOLDS[parent]
                or source['previous_rule'] != previous['rule']):
            raise ValueError(f'Source-only rule/parent/threshold mismatch: {suffix}')
        sources.append(source)

    metrics, classes, selections, captures, routes, utilities, coverage = [], [], [], [], [], [], []
    for folder, info, source in zip(folders, infos, sources):
        meta = dict(dataset=info['dataset'], task=info['task'])
        suffix = Path(info['dataset']) / info['task']
        p, old = [load_npz(root / 'targets' / suffix / 'policies.npz') for root in (OUT, PREV)]
        original = load_npz(folder / 'target.npz')
        modes, old_modes = list(p['modes']), list(old['modes'])
        if tuple(modes) != tuple(MODES):
            raise ValueError(f'Unexpected mode order: {suffix}')
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], original[key])
            np.testing.assert_array_equal(p[key], old[key])
        for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected', 'choices'):
            if p[key].shape != (5, len(MODES), len(p['rows'])):
                raise ValueError(f'Unexpected {key} shape: {suffix}')
            np.testing.assert_array_equal(p[key][:, modes.index('source_selected')],
                                          p[key][:, modes.index(source['rule'])])
            if key != 'frozen_predictions':
                for mode, prior in (('previous_selected', 'source_selected'),
                                    ('smooth_parent', source['parent_rule']), ('learned', 'learned')):
                    np.testing.assert_array_equal(p[key][:, modes.index(mode)], old[key][:, old_modes.index(prior)])
        for key, shape in (('candidates', (5, len(p['rows']), 4)),
                           ('utilities', (5, 3, len(p['rows']), 4)), ('coverage', (5, len(p['rows']), 4))):
            if p[key].shape != shape:
                raise ValueError(f'Unexpected {key} shape: {suffix}')
        np.testing.assert_array_equal(p['candidates'], old['candidates'])
        previous_mask = old['selected'][:, old_modes.index('source_selected')]
        fixed = np.where(previous_mask[:, None, :], original['expert'].argmax(2)[:, None, :], p['base_predictions'])
        np.testing.assert_array_equal(p['frozen_predictions'], fixed)
        np.testing.assert_array_equal(fixed[:, modes.index('previous_selected')], old['predictions'][:, old_modes.index('source_selected')])
        manifest, id_column = ((SOURCE / 'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT / DATASETS[info['dataset']][0] / 'target_manifest.csv', 'sample_id'))
        aligned = pd.read_csv(manifest, usecols=[id_column, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(aligned[id_column].to_numpy(str), original['ids'])
        labels = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(labels[p['support']], original['support_labels'])
        selections.append(dict(**meta, **{k: source[k] for k in ('rule', 'parent_rule', 'threshold', 'previous_rule')},
            **{f'source_delta_ba_{k}': v for k, v in source['source_delta_ba'].items()},
            source_selected_delta_ba_vs_previous_selected=source['source_delta_ba'][source['rule']] - source['source_delta_ba']['previous_selected']))
        for draw, ev in enumerate(p['eval_masks']):
            y = labels[ev]
            for stage, key in STAGES:
                predictions = p[key][draw][:, ev]
                stats = {mode: statistics(y, predictions[j]) for j, mode in enumerate(modes)}
                for j, mode in enumerate(modes):
                    pred = predictions[j]
                    admissions = p['selected'][draw, j, ev].sum() if stage == 'full' else previous_mask[draw, ev].sum() if stage == 'frozen_previous_route' else 0
                    row = dict(**meta, draw=draw, stage=stage, mode=mode, n=len(y), admissions=int(admissions), **dict(zip(VALUES, stats[mode])))
                    for baseline in BASELINES:
                        reference = predictions[modes.index(baseline)]
                        row.update({f'delta_{v}_vs_{baseline}': stats[mode][vi] - stats[baseline][vi] for vi, v in enumerate(VALUES)})
                        row.update({f'changes_vs_{baseline}': int(np.sum(pred != reference)),
                            f'rescues_vs_{baseline}': int(np.sum((pred == y) & (reference != y))),
                            f'harms_vs_{baseline}': int(np.sum((pred != y) & (reference == y)))})
                    metrics.append(row)
                    for c in np.unique(y):
                        mask = y == c
                        recall = 100 * float(np.mean(pred[mask] == c))
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c), n=int(mask.sum()),
                            support_n=int(np.sum(original['support_labels'][draw] == c)), recall=recall,
                            **{f'delta_recall_vs_{b}': recall - 100 * float(np.mean(predictions[modes.index(b), mask] == c)) for b in BASELINES}))
            main_correct = p['base_predictions'][draw, modes.index('learned'), ev] == y
            bands = p['candidates'][draw, ev]
            opportunity = ~main_correct & (bands == y[:, None]).any(1)
            for j, mode in enumerate(modes):
                correct = p['base_predictions'][draw, j, ev] == y
                captures.append(dict(**meta, draw=draw, stage='base', mode=mode, opportunities=int(opportunity.sum()),
                    captured=int(np.sum(opportunity & correct)), harms=int(np.sum(main_correct & ~correct)),
                    capture_rate=float(correct[opportunity].mean()) if opportunity.any() else np.nan))
                a, b, c = [statistics(y, z) for z in (p['predictions'][draw, modes.index('previous_selected'), ev],
                    p['frozen_predictions'][draw, j, ev], p['predictions'][draw, j, ev])]
                row = dict(**meta, draw=draw, mode=mode)
                for vi, value in enumerate(VALUES):
                    row.update({f'previous_full_{value}': a[vi], f'frozen_previous_route_{value}': b[vi],
                        f'full_{value}': c[vi], f'fixed_previous_route_delta_{value}': b[vi] - a[vi],
                        f'changed_route_delta_{value}': c[vi] - b[vi], f'full_delta_{value}': c[vi] - a[vi]})
                routes.append(row)
            for bi, band in enumerate(BANDS):
                rho = p['coverage'][draw, ev, bi]
                coverage.append(dict(**meta, draw=draw, band=band, n=len(y), covered=int(np.sum(rho > 0)),
                    coverage_fraction=float(np.mean(rho > 0)), mean_rho=float(rho.mean())))
                rescue, harm = ~main_correct & (bands[:, bi] == y), main_correct & (bands[:, bi] != y)
                events = rescue | harm
                for ui, utility in enumerate(('smooth', 'class', 'pair')):
                    value = p['utilities'][draw, ui, ev, bi]
                    utilities.append(dict(**meta, draw=draw, band=band, utility=utility,
                        rescue_events=int(rescue.sum()), harm_events=int(harm.sum()), events=int(events.sum()),
                        auc_defined=int(rescue.any() and harm.any()),
                        auc=float(roc_auc_score(rescue[events], value[events])) if rescue.any() and harm.any() else np.nan))
        print(f'PAIR REPORT {info["dataset"]}/{info["task"]}: aligned', flush=True)

    m, selection = pd.DataFrame(metrics), pd.DataFrame(selections)
    m.to_csv(OUT / 'metrics.csv', index=False)
    contrasts, join = [], ['dataset', 'task', 'draw', 'stage']
    for mode, baseline in (('class_local', 'smooth_parent'), ('pair_local', 'class_local')):
        left = m.loc[m['mode'].eq(mode), [*join, *VALUES]]
        right = m.loc[m['mode'].eq(baseline), [*join, *VALUES]]
        contrast = left.merge(right, on=join, suffixes=('', '_reference'), validate='one_to_one')
        for value in VALUES:
            contrast[f'delta_{value}'] = contrast[value] - contrast[f'{value}_reference']
        contrasts.append(contrast.assign(mode=mode, baseline=baseline))
    pd.concat(contrasts, ignore_index=True).to_csv(OUT / 'local_contrasts.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_metrics.csv', index=False)
    selection.to_csv(OUT / 'source_selection.csv', index=False)
    values, group = VALUES + DELTAS, ['dataset', 'stage', 'mode']
    task = m.groupby([*group, 'task'])[values].mean()
    draws = m.groupby([*group, 'draw'])[values].mean()
    summary = task.reset_index().groupby(group)[values].mean()
    summary['tasks'] = task.reset_index().groupby(group).size()
    summary['draws'], summary['task_draws'] = draws.groupby(group).size(), m.groupby(group).size()
    for baseline in BASELINES:
        for value in VALUES:
            delta, count = f'delta_{value}_vs_{baseline}', f'positive_draws_{value}_vs_{baseline}'
            task[count] = m[delta].gt(0).groupby([m[k] for k in [*group, 'task']]).sum()
            summary[count] = draws[delta].gt(0).groupby(group).sum()
            summary[f'positive_task_draws_{value}_vs_{baseline}'] = m[delta].gt(0).groupby([m[k] for k in group]).sum()
            draws[f'positive_tasks_{value}_vs_{baseline}'] = m[delta].gt(0).groupby([m[k] for k in [*group, 'draw']]).sum()
    task.reset_index().to_csv(OUT / 'task_means.csv', index=False)
    draws.reset_index().to_csv(OUT / 'draw_metrics.csv', index=False)
    summary = summary.reset_index()
    summary.to_csv(OUT / 'summary.csv', index=False)
    counts = ['n', 'admissions', *[f'{v}_vs_{b}' for b in BASELINES for v in ('changes', 'rescues', 'harms')]]
    m.groupby(group)[counts].sum().reset_index().to_csv(OUT / 'events.csv', index=False)
    m.groupby([*group, 'draw'])[counts].sum().reset_index().to_csv(OUT / 'draw_events.csv', index=False)
    capture = diagnostic('capture_probe', captures, ['stage', 'mode'], ['capture_rate'], ['opportunities', 'captured', 'harms'])
    route_values = [k for k in routes[0] if k.endswith(('_oa', '_ba', '_macro_f1_8'))]
    route = diagnostic('route_probe', routes, ['mode'], route_values)
    auc_counts = ['rescue_events', 'harm_events', 'events', 'auc_defined']
    auc = diagnostic('utility_probe', utilities, ['utility'], ['auc'], auc_counts)
    diagnostic('utility_band_probe', utilities, ['utility', 'band'], ['auc'], auc_counts)
    cov = diagnostic('coverage_probe', coverage, ['band'], ['coverage_fraction', 'mean_rho'], ['n', 'covered'])
    task.reset_index().query("mode == 'source_selected'").merge(selection, on=['dataset', 'task'],
        validate='many_to_one').to_csv(OUT / 'source_target_comparison.csv', index=False)
    primary = summary.loc[summary.stage.eq('full') & summary['mode'].eq('source_selected'),
        ['dataset', 'tasks', 'ba', 'draws', *[k for b in BASELINES for k in (f'delta_ba_vs_{b}', f'positive_draws_ba_vs_{b}')]]]
    notes = ['# 频率类别/类别对选择器冻结报告',
        '全部25个匹配任务完成冻结后才读取目标标签，按原始manifest样本ID、rows、support和eval_masks对齐。目标标签仅用于事后评估和诊断，不参与阈值或策略选择。根因与研究结论由主线另写。',
        '## Source-selected full', markdown_table(primary),
        'previous_selected精确重放PREV source_selected；smooth_parent仅按PREV源域验证均值从smooth/smooth02/smooth05选择，继承固定阈值。class_local与pair_local继承该阈值。当前策略按相同源域episode base BA均值从previous_selected/class_local/pair_local选择，同分旧策略优先。',
        '## 口径',
        'OA、BA、固定八类macro F1单位为百分数，差值为百分点。summary先平均每任务五次draw，再在各数据集内等权平均任务，self24ports为task23；不混合all25。draw_metrics固定draw后等权平均任务。positive_draws统计五个均值中严格正差次数；positive_task_draws累计任务/draw正差次数。五次support抽样不是独立训练重复。',
        'base不交接；full使用自身路由；frozen_previous_route全部模式使用PREV source_selected admission mask。各阶段均与同阶段previous_selected、smooth_parent、learned比较。',
        '## Capture', markdown_table(capture),
        '机会为learned base错误且至少一个preserved频带正确的并集；captured为其中被当前base救回的数量，harms为learned正确却被改错的数量。计数累计query-draw事件，重复样本不是独立样本。',
        '## 精确冻结路由分解', markdown_table(route[['dataset', 'mode', 'fixed_previous_route_delta_ba', 'changed_route_delta_ba', 'full_delta_ba']]),
        '逐任务/draw计算当前frozen减previous full，以及当前full减当前frozen，两项之和等于full差值；这是预测算术分解，不能直接解释为根因。',
        '## Utility与coverage', markdown_table(auc), markdown_table(cov),
        'AUC仅在各频带相对learned base的rescue/harm事件上计算，rescue为正类；缺少任一类留空。先平均任务内有效band/draw，再等权平均有效任务，auc_defined记录有效项数。coverage在全部评估query上计算rho>0比例和rho均值，逐任务/draw/band保存在coverage_probe.csv。',
        'rho=min(neff_a,neff_b)/(min(neff_a,neff_b)+2)，相同8近邻中任一query预测类缺失时为0，singleton不超过1/3。local_contrasts.csv直接保存逐任务/draw/stage的class_local减smooth_parent与pair_local减class_local。',
        'source_target_comparison连接冻结源域选择与目标三阶段任务均值，仅用于描述。metrics/class_metrics/task_means/summary/draw_metrics/events/draw_events保存指标与事件；capture_probe/route_probe/utility_probe及逐频带表保存诊断。无图。']
    (OUT / 'REPORT_ZH.md').write_text('\n\n'.join(notes) + '\n', encoding='utf-8')
    (OUT / 'REPORT_COMPLETE.json').write_text(json.dumps(dict(tasks=len(folders), draws=len(folders) * 5,
        stages=[s for s, _ in STAGES], modes=list(MODES), labels_aligned=True, source_selected_aligned=True,
        baseline_replay_exact=True, frozen_previous_route_exact=True, root_diagnosis_pending=True,
        target_labels_only_postfreeze_reporting_and_diagnosis=True, target_labels_used_for_selection=False,
        primary_comparison='source_selected vs previous_selected, full'), indent=2), encoding='utf-8')
    print(primary.to_string(index=False), flush=True)
    return summary


if __name__ == '__main__':
    report()
