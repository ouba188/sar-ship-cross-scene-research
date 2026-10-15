"""Post-freeze reporting only; never selects a policy using target labels."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frequency_margin_run import OUT, PREV, MODES, EXP
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_band_report import BANDS, markdown_table
from frequency_reliability_run import statistics

RULES = ('hard', 'smooth', 'smooth02', 'smooth05')
BASELINES = ('hard', 'learned', 'standalone_local')
STAGES = (('base', 'base_predictions'), ('full', 'predictions'),
          ('frozen_hard_route', 'frozen_predictions'))
VALUES = ['oa', 'ba', 'macro_f1_8']
DELTAS = [f'delta_{v}_vs_{b}' for b in BASELINES for v in VALUES]


def _all25(frame):
    return pd.concat([frame, frame.assign(dataset='all25')], ignore_index=True)


def _diagnostic_output(output, name, records, keys, counts, values):
    frame = pd.DataFrame(records)
    frame.to_csv(output / (name + '.csv'), index=False)
    group = ['dataset', *keys]
    task = frame.groupby([*group, 'task'])[values].mean().reset_index()
    summary = _all25(task).groupby(group)[values].mean()
    if counts:
        summary = summary.join(_all25(frame).groupby(group)[counts].sum())
    summary = summary.reset_index()
    summary.to_csv(output / (name + '_summary.csv'), index=False)
    return summary


def report():
    output = Path(OUT)
    folders = [p.parent for p in sorted((EXP / 'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p / 'PREPARED.json').read_text()) for p in folders]
    expected = {(i['dataset'], i['task']) for i in infos}
    markers = {(p.parent.parent.name, p.parent.name)
               for p in (output / 'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or markers != expected:
        raise ValueError('Freeze all25 matching target tasks before any target-label reads')
    sources = []
    for info in infos:
        suffix = Path(info['dataset']) / info['task']
        if not (output / 'targets' / suffix / 'policies.npz').is_file():
            raise ValueError(f'Missing frozen predictions: {suffix}')
        source = json.loads((output / 'source' / suffix / 'SELECTION.json').read_text())
        gains = source['source_delta_ba']
        if not all(np.isfinite(gains[r]) for r in RULES):
            raise ValueError(f'Nonfinite source validation mean: {suffix}')
        if source['rule'] != max(RULES, key=lambda r: gains[r]):
            raise ValueError(f'Source rule disagrees with frozen source means: {suffix}')
        sources.append(source)

    # All25 markers and all source rules have been checked before labels enter.
    metrics, classes, selections, captures, routes, utilities = [], [], [], [], [], []
    for folder, info, source in zip(folders, infos, sources):
        meta = dict(dataset=info['dataset'], task=info['task'])
        suffix = Path(info['dataset']) / info['task']
        p = load_npz(output / 'targets' / suffix / 'policies.npz')
        old = load_npz(PREV / 'targets' / suffix / 'policies.npz')
        original = load_npz(folder / 'target.npz')
        modes, old_modes = list(p['modes']), list(old['modes'])
        if tuple(modes) != tuple(MODES):
            raise ValueError(f'Unexpected mode order: {suffix}')
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], original[key])
            np.testing.assert_array_equal(p[key], old[key])
        n = len(p['rows'])
        for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected', 'choices'):
            if p[key].shape != (5, len(MODES), n):
                raise ValueError(f'Unexpected {key} shape: {suffix}')
            np.testing.assert_array_equal(p[key][:, modes.index('source_selected')],
                                          p[key][:, modes.index(source['rule'])])
        if p['utilities'].shape != (5, 2, n, 4) or p['candidates'].shape != (5, n, 4):
            raise ValueError(f'Unexpected utility/candidate shape: {suffix}')
        for current, previous in (('learned', 'learned'),
                                  ('standalone_local', 'standalone_support_local'),
                                  ('hard', 'preserved_support_local')):
            for key in ('base_predictions', 'predictions', 'selected'):
                np.testing.assert_array_equal(p[key][:, modes.index(current)],
                                              old[key][:, old_modes.index(previous)])
        hard_mask = old['selected'][:, old_modes.index('preserved_support_local')]
        expert = original['expert'].argmax(2)
        for j in range(len(modes)):
            np.testing.assert_array_equal(p['frozen_predictions'][:, j],
                np.where(hard_mask, expert, p['base_predictions'][:, j]))
        np.testing.assert_array_equal(p['frozen_predictions'][:, modes.index('hard')],
                                      p['predictions'][:, modes.index('hard')])
        if info['dataset'] == 'self24ports':
            manifest, id_column = SOURCE / 'manifest.csv', 'instance_id'
        else:
            manifest = ROOT / DATASETS[info['dataset']][0] / 'target_manifest.csv'
            id_column = 'sample_id'
        aligned = pd.read_csv(manifest, usecols=[id_column, 'class_id'],
                              keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(aligned[id_column].to_numpy(str), original['ids'])
        labels = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(labels[p['support']], original['support_labels'])
        selections.append(dict(**meta, rule=source['rule'],
            **{'source_delta_ba_' + k: v for k, v in source['source_delta_ba'].items()},
            source_selected_delta_ba_vs_hard=(source['source_delta_ba'][source['rule']]
                                             - source['source_delta_ba']['hard'])))
        for draw, ev in enumerate(p['eval_masks']):
            y = labels[ev]
            for stage, key in STAGES:
                predictions = p[key][draw][:, ev]
                stats = {mode: statistics(y, predictions[j]) for j, mode in enumerate(modes)}
                for j, mode in enumerate(modes):
                    pred = predictions[j]
                    row = dict(**meta, draw=draw, stage=stage, mode=mode, n=len(y),
                               **dict(zip(VALUES, stats[mode])))
                    row['admissions'] = (int(p['selected'][draw, j, ev].sum()) if stage == 'full'
                        else int(hard_mask[draw, ev].sum()) if stage == 'frozen_hard_route' else 0)
                    for baseline in BASELINES:
                        reference = predictions[modes.index(baseline)]
                        correct = reference == y
                        row.update({f'delta_{v}_vs_{baseline}': stats[mode][vi] - stats[baseline][vi]
                                    for vi, v in enumerate(VALUES)})
                        row.update({f'changes_vs_{baseline}': int(np.sum(pred != reference)),
                                    f'rescues_vs_{baseline}': int(np.sum((pred == y) & ~correct)),
                                    f'harms_vs_{baseline}': int(np.sum((pred != y) & correct))})
                    metrics.append(row)
                    for c in np.unique(y):
                        mask = y == c
                        recall = 100 * float(np.mean(pred[mask] == c))
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode,
                            class_id=int(c), n=int(mask.sum()),
                            support_n=int(np.sum(original['support_labels'][draw] == c)), recall=recall,
                            **{f'delta_recall_vs_{b}': recall - 100 * float(np.mean(
                                predictions[modes.index(b), mask] == c)) for b in BASELINES}))

            main = p['base_predictions'][draw, modes.index('learned'), ev]
            main_correct = main == y
            bands = p['candidates'][draw, ev]
            standalone = old['base_predictions'][draw,
                [old_modes.index('standalone_' + b) for b in BANDS]][:, ev].T
            opportunity = ~main_correct & (bands == y[:, None]).any(1)
            old_opportunity = ~main_correct & (standalone == y[:, None]).any(1)
            added = opportunity & ~old_opportunity
            for mode in (*RULES, 'source_selected'):
                j = modes.index(mode)
                base = p['base_predictions'][draw, j, ev]
                choice = p['choices'][draw, j, ev]
                correct = base == y
                opportunities, added_n = int(opportunity.sum()), int(added.sum())
                captured = int(np.sum(opportunity & correct))
                captured_added = int(np.sum(added & correct))
                captures.append(dict(**meta, draw=draw, stage='base', mode=mode,
                    opportunities=opportunities, captured=captured,
                    harms=int(np.sum(main_correct & ~correct)),
                    standalone_opportunities=int(old_opportunity.sum()), added_opportunities=added_n,
                    captured_added=captured_added,
                    added_keep_main=int(np.sum(added & (choice == 0))),
                    added_wrong_band=int(np.sum(added & (choice != 0) & ~correct)),
                    capture_rate=captured / opportunities if opportunities else np.nan,
                    added_capture_rate=captured_added / added_n if added_n else np.nan))
                hard_full = p['predictions'][draw, modes.index('hard'), ev]
                fixed = p['frozen_predictions'][draw, j, ev]
                new_full = p['predictions'][draw, j, ev]
                a, b, c = [statistics(y, z) for z in (hard_full, fixed, new_full)]
                row = dict(**meta, draw=draw, mode=mode)
                for vi, value in enumerate(VALUES):
                    row.update({f'hard_full_{value}': a[vi], f'fixed_hard_route_{value}': b[vi],
                        f'new_full_{value}': c[vi], f'fixed_hard_route_delta_{value}': b[vi] - a[vi],
                        f'changed_route_delta_{value}': c[vi] - b[vi],
                        f'full_delta_{value}': c[vi] - a[vi]})
                routes.append(row)
            for bi, band in enumerate(BANDS):
                rescue = ~main_correct & (bands[:, bi] == y)
                harm = main_correct & (bands[:, bi] != y)
                events = rescue | harm
                for ui, utility in enumerate(('hard', 'smooth')):
                    value = p['utilities'][draw, ui, ev, bi]
                    utilities.append(dict(**meta, draw=draw, band=band, utility=utility,
                        rescue_events=int(rescue.sum()), harm_events=int(harm.sum()),
                        events=int(events.sum()), nonzero_events=int(np.sum(value[events] != 0)),
                        auc=float(roc_auc_score(rescue[events], value[events]))
                            if rescue.any() and harm.any() else np.nan,
                        auc_defined=int(rescue.any() and harm.any()),
                        nonzero=float(np.mean(value[events] != 0)) if events.any() else np.nan))
        print(f'MARGIN REPORT {info["dataset"]}/{info["task"]}: aligned', flush=True)

    m = pd.DataFrame(metrics)
    m.to_csv(output / 'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(output / 'class_metrics.csv', index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(output / 'source_selection.csv', index=False)
    values = VALUES + DELTAS
    group = ['dataset', 'stage', 'mode']
    task = m.groupby([*group, 'task'])[values].mean()
    expanded = _all25(m)
    draws = expanded.groupby([*group, 'draw'])[values].mean()
    summary = _all25(task.reset_index()).groupby(group)[values].mean()
    summary['tasks'] = _all25(task.reset_index()).groupby(group).size()
    summary['task_draws'] = expanded.groupby(group).size()
    summary['draws'] = draws.groupby(group).size()
    for baseline in BASELINES:
        for value in VALUES:
            delta = f'delta_{value}_vs_{baseline}'
            count = f'positive_draws_{value}_vs_{baseline}'
            task[count] = m[delta].gt(0).groupby([m[k] for k in [*group, 'task']]).sum()
            summary[count] = draws[delta].gt(0).groupby(group).sum()
            summary[f'positive_task_draws_{value}_vs_{baseline}'] = expanded[delta].gt(0).groupby(
                [expanded[k] for k in group]).sum()
            draws[f'positive_tasks_{value}_vs_{baseline}'] = expanded[delta].gt(0).groupby(
                [expanded[k] for k in [*group, 'draw']]).sum()
    task.reset_index().to_csv(output / 'task_means.csv', index=False)
    draws.reset_index().to_csv(output / 'draw_metrics.csv', index=False)
    summary = summary.reset_index()
    summary.to_csv(output / 'summary.csv', index=False)
    counts = ['n', 'admissions', *[f'{v}_vs_{b}' for b in BASELINES
                                 for v in ('changes', 'rescues', 'harms')]]
    expanded.groupby(group)[counts].sum().reset_index().to_csv(output / 'events.csv', index=False)
    capture_summary = _diagnostic_output(output, 'capture_probe', captures, ['stage', 'mode'],
        ['opportunities', 'captured', 'harms', 'standalone_opportunities', 'added_opportunities',
         'captured_added', 'added_keep_main', 'added_wrong_band'], ['capture_rate', 'added_capture_rate'])
    route_values = [k for k in routes[0] if k.endswith(('_oa', '_ba', '_macro_f1_8'))]
    route_summary = _diagnostic_output(output, 'route_probe', routes, ['mode'], [], route_values)
    utility_summary = _diagnostic_output(output, 'utility_probe', utilities, ['utility'],
        ['rescue_events', 'harm_events', 'events', 'nonzero_events', 'auc_defined'], ['auc', 'nonzero'])
    _diagnostic_output(output, 'utility_band_probe', utilities, ['utility', 'band'],
        ['rescue_events', 'harm_events', 'events', 'nonzero_events', 'auc_defined'], ['auc', 'nonzero'])
    comparison = task.reset_index().query("mode == 'source_selected'").merge(
        selection[['dataset', 'task', 'rule', 'source_selected_delta_ba_vs_hard']],
        on=['dataset', 'task'], validate='many_to_one')
    comparison.to_csv(output / 'source_target_comparison.csv', index=False)
    primary = summary.loc[summary.stage.eq('full') & summary['mode'].eq('source_selected'),
        ['dataset', 'ba', 'delta_ba_vs_hard', 'positive_draws_ba_vs_hard',
         'delta_ba_vs_standalone_local', 'positive_draws_ba_vs_standalone_local', 'draws']]
    notes = [
        '# 连续 margin 选择器冻结报告',
        '本报告描述冻结预测与事后诊断；机制解释和结论边界由主线单独写入 FINDINGS_ZH.md。全部25个匹配任务的 COMPLETE.json '
        '及全部源域规则在读取目标标签前检查；样本ID、rows、support、eval_masks、support标签直接对齐。',
        '## 主要比较：source_selected 对 hard（full）', markdown_table(primary),
        'source_selected 仅从 hard、smooth、smooth02、smooth05 中按源域验证 episode 平均 base BA '
        '选择，同分按固定规则顺序。source_delta_ba 相对原 learned；报告只核对冻结规则，不重选策略。'
        '源域获胜不意味着目标增益，目标差值仅来自冻结预测评估。源域端口与表征训练共享，'
        '不能称为独立表征验证；目标数据属于此前使用过的开发数据。',
        '## 指标口径',
        'base 无专家交接；full 使用每模式自身路由；frozen_hard_route 对全部模式统一使用 PREV '
        'preserved_support_local admission mask。learned、standalone_local、hard 的 base/full/selected '
        '逐项核对旧基线；新 frozen 阶段统一重放 hard 路由，不与旧 learned 路由阶段混用。',
        'OA、BA、macro_f1_8 为百分数，delta 为百分点。BA 使用评估集合实际出现的类别，F1固定八类。'
        '所有三阶段指标提供 vs_hard、vs_learned、vs_standalone_local。summary 先平均任务内五次draw，'
        '再等权平均任务；all25 同样等权平均25任务。draw_metrics 在固定draw内等权平均任务。'
        'positive_draws 统计五个任务等权draw均值中严格正差的次数；positive_task_draws 统计任务/draw正差次数。'
        '五次draw是support抽样，不是训练随机种子或独立显著性重复。',
        '## Capture', markdown_table(capture_summary),
        'capture_probe 只针对 base，以 learned base 错误且至少一个 preserved 频带正确为机会；'
        'added 是这些机会中 PREV standalone 四频带均无法救回的部分。captured_added、added_keep_main '
        '与 added_wrong_band 分别表示救回、保留主分支、选错频带。harms 为 learned正确却被选择器改错。'
        '所有计数累加 query-draw 事件，同样本跨draw重复，不表示独立样本数。capture率先任务内均值再任务等权均值。',
        '## 精确路由分解', markdown_table(route_summary[['dataset', 'mode',
            'fixed_hard_route_delta_ba', 'changed_route_delta_ba', 'full_delta_ba']]),
        '每任务/draw依次比较 hard full、当前模式 frozen_hard_route、当前模式 full；'
        '固定路由变化加改变路由变化严格分解 full 差值。此为冻结预测算术分解，不是根因结论。',
        '## Utility 区分度', markdown_table(utility_summary),
        'utility_probe 仅在每频带相对 learned base 的 rescue 或 harm 事件上计算AUC；rescue为正类，'
        'utility越大越倾向rescue。缺少任一类别时AUC为空，不填0.5。nonzero 是相同事件中utility非零比例。'
        '先在每任务内平均有效band/draw值，再等权平均有效任务；auc_defined给出有效band/draw数。'
        '事件计数是query-band-draw，同query可能在多个频带重复。utility_band_probe另存逐频带汇总。'
        'smooth02和smooth05共享smooth utility，因此不重复AUC。',
        'metrics、class_metrics、task_means、summary、draw_metrics、events 为三阶段CSV；'
        'source_selection保留冻结源域规则和全部源域差值；source_target_comparison按任务连接源域差值和目标三阶段差值，'
        '只用于描述，不用于选择。capture_probe、route_probe、utility_probe及其summary保存诊断。无图。'
    ]
    (output / 'REPORT_ZH.md').write_text('\n\n'.join(notes) + '\n', encoding='utf-8')
    (output / 'REPORT_COMPLETE.json').write_text(json.dumps(dict(tasks=25, draws=125,
        stages=[s for s, _ in STAGES], modes=list(MODES), labels_aligned=True,
        source_selected_aligned=True, baseline_replay_exact=True, frozen_hard_route_exact=True,
        target_labels_only_postfreeze_reporting_and_diagnosis=True, target_labels_used_for_selection=False,
        primary_comparison='source_selected vs hard, full', root_diagnosis_pending=True), indent=2),
        encoding='utf-8')
    print(primary.to_string(index=False), flush=True)
    return summary


if __name__ == '__main__':
    report()
