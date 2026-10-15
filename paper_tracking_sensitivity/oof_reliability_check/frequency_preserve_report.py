"""Frozen preservation readouts; training/evaluation and final FINDINGS belong to the main run."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import frequency_specialist_report as specialist
from frequency_band_report import BANDS, STAGES, markdown_table
from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz

OUT = ROOT/'frequency_preserve_20261010'
FAMILIES = ('old', 'independent', 'standalone', 'preserved')
MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *BANDS)),
         *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')
SUFFIXES = ('equal', *BANDS, 'support_static', 'support_local')
VALUES = ['oa', 'ba', 'macro_f1_8']
INDEX = ['dataset', 'task', 'draw', 'stage']
GROUP = ['dataset', 'stage', 'suffix', 'comparison']
DELTAS = ['delta_'+v for v in VALUES]


def matched_report(output):
    metrics = pd.read_csv(output/'metrics.csv', dtype={'dataset': str, 'task': str})
    records = []
    for suffix in SUFFIXES:
        frames = {f: metrics.loc[metrics['mode'].eq(f'{f}_{suffix}'), INDEX+['n', *VALUES]]
                  .set_index(INDEX, verify_integrity=True).sort_index() for f in FAMILIES[1:]}
        for family, baseline in (('standalone', 'independent'), ('preserved', 'independent'),
                                 ('preserved', 'standalone')):
            frame, reference = frames[family], frames[baseline]
            if not frame.index.equals(reference.index) or not frame['n'].equals(reference['n']):
                raise ValueError(f'Unmatched task/draw/stage: {family}-{baseline}/{suffix}')
            delta = (frame[VALUES]-reference[VALUES]).set_axis(DELTAS, axis=1)
            records.append(delta.reset_index().assign(suffix=suffix, comparison=family+'-'+baseline))
    paired = pd.concat(records, ignore_index=True)
    task = paired.groupby([*GROUP, 'task'])[DELTAS].mean()
    summary = task.groupby(GROUP)[DELTAS].mean()
    draws = paired.groupby([*GROUP, 'draw'])[DELTAS].mean()
    task['draws'] = paired.groupby([*GROUP, 'task']).size()
    if not task['draws'].eq(5).all() or set(paired['draw']) != set(range(5)):
        raise ValueError('Matched readouts require five support draws per task')
    summary['tasks'] = task.groupby(GROUP).size()
    summary['draws'] = draws.groupby(GROUP).size()
    summary['task_draws'] = paired.groupby(GROUP).size()
    draws['tasks'] = paired.groupby([*GROUP, 'draw']).size()
    for value in VALUES:
        positive = 'positive_'+value
        paired[positive] = paired['delta_'+value] > 0
        task['positive_draws_'+value] = paired.groupby([*GROUP, 'task'])[positive].sum()
        draws['positive_tasks_'+value] = paired.groupby([*GROUP, 'draw'])[positive].sum()
        summary['positive_task_draws_'+value] = paired.groupby(GROUP)[positive].sum()
        summary['positive_draws_'+value] = (draws['delta_'+value] > 0).groupby(GROUP).sum()
    for name, frame in (('metrics', paired), ('task_means', task.reset_index()),
                        ('draw_metrics', draws.reset_index()), ('summary', summary.reset_index())):
        frame.to_csv(output/f'matched_{name}.csv', index=False)
    return summary.reset_index()


def retention(predictions, modes, labels):
    """Rows are queries; fix anchors to the independent teacher, regardless of later uniqueness."""
    y, modes = np.asarray(labels), list(modes)
    wrong = predictions[:, modes.index('learned')] != y
    correct = {f: predictions[:, [modes.index(f'{f}_{b}') for b in BANDS]] == y[:, None]
               for f in FAMILIES[1:]}
    anchor = correct['independent'] & (wrong & (correct['independent'].sum(1) == 1))[:, None]
    union = {f: wrong & c.any(1) for f, c in correct.items()}
    baseline, alone, preserved = (union[f] for f in FAMILIES[1:])
    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    weights = 100/(len(counts)*counts[inverse])
    result = dict(n=len(y), baseline_anchor_count=int(anchor.sum()),
        baseline_union_count=int(baseline.sum()),
        standalone_lost_baseline_union_rescued_by_preserved=int((baseline & ~alone & preserved).sum()),
        standalone_union_lost_by_preserved=int((alone & ~preserved).sum()),
        preserved_union_new_vs_standalone=int((preserved & ~alone).sum()),
        preserved_oracle_delta_ba_vs_standalone=float(weights @ (preserved.astype(float)-alone)))
    for b, band in enumerate(BANDS):
        result['baseline_anchor_'+band] = int(anchor[:, b].sum())
    for family in ('standalone', 'preserved'):
        retained = anchor & correct[family]
        result.update({family+'_anchor_retained': int(retained.sum()),
            family+'_union_count': int(union[family].sum()),
            family+'_union_new_vs_independent': int((union[family] & ~baseline).sum()),
            family+'_union_lost_vs_independent': int((baseline & ~union[family]).sum()),
            family+'_oracle_delta_ba_vs_independent': float(weights @ (union[family].astype(float)-baseline))})
        result.update({family+'_anchor_retained_'+band: int(retained[:, b].sum())
                       for b, band in enumerate(BANDS)})
    return result


def retention_report(folders, output):
    # Called only after specialist.report has frozen/validated all25 and aligned target labels.
    records = []
    for folder in folders:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'], stage='base')
        suffix = Path(info['dataset'])/info['task']
        source = output/'source'/suffix
        paths = sorted(source.glob('query_scores_*.npz'))
        selection = json.loads((source/'SELECTION.json').read_text())
        if not paths or len(paths) != selection['source_episodes']:
            raise ValueError(f'Incomplete source validation query scores: {source}')
        for path in paths:
            z = load_npz(path)
            if z['scores'].shape != (len(z['labels']), len(z['modes']), 8):
                raise ValueError(f'Expected source query scores N x M x 8: {path}')
            records.append(dict(**meta, split='source_validation', unit=int(path.stem.split('_')[-1]),
                **retention(z['scores'].argmax(2), z['modes'], z['labels'])))
        p, original = load_npz(output/'targets'/suffix/'policies.npz'), load_npz(folder/'target.npz')
        if info['dataset'] == 'self24ports':
            manifest, id_column = SOURCE/'manifest.csv', 'instance_id'
        else:
            manifest, id_column = ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'
        aligned = pd.read_csv(manifest, usecols=[id_column, 'class_id'],
                              keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(aligned[id_column].to_numpy(str), original['ids'])
        y = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(y[p['support']], original['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            records.append(dict(**meta, split='target_postfreeze', unit=draw,
                **retention(p['base_predictions'][draw][:, ev].T, p['modes'], y[ev])))
    frame = pd.DataFrame(records)
    keys = ['split', 'dataset', 'stage']
    values = [c for c in frame if c not in [*keys, 'task', 'unit']]
    deltas = [c for c in values if 'oracle_delta_ba' in c]
    counts = [c for c in values if c not in deltas]
    task = frame.groupby([*keys, 'task']).agg({c: 'mean' if c in deltas else 'sum' for c in values})
    task['units'] = frame.groupby([*keys, 'task']).size()
    summary = task.groupby(keys)[deltas].mean().join(task.groupby(keys)[[*counts, 'units']].sum())
    summary['tasks'] = task.groupby(keys).size()
    frame.to_csv(output/'retention.csv', index=False)
    task.reset_index().to_csv(output/'retention_task_summary.csv', index=False)
    summary.reset_index().to_csv(output/'retention_summary.csv', index=False)
    return summary.reset_index()


def report(folders=None, output=OUT):
    output = Path(output)
    folders = ([p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
               if folders is None else [Path(p) for p in folders])
    saved = specialist.OUT, specialist.FAMILIES, specialist.MODES
    try:
        specialist.OUT, specialist.FAMILIES, specialist.MODES = output, FAMILIES, MODES
        specialist.report(folders=folders, output=output)
    finally:
        specialist.OUT, specialist.FAMILIES, specialist.MODES = saved
    notes = [
        '# 频带保留实验冻结读出 / Frozen frequency preservation readouts',
        '家族为 old、independent、standalone、preserved；learned 为保留的主分支，共 30 模式。'
        '训练、评估、训练诊断与最终中文 FINDINGS 由主流程负责。本文件只生成描述性报告，不作因果推断或目标域赢家选择。',
        'Main-run protocol: preserved objective = standalone objective + 0.25 * retain_hinge. '
        'Training anchors use SOURCE TRAIN queries only: the retained learned main is wrong and '
        'exactly one of four frozen independent teacher bands is correct. The target margin is '
        'the teacher true-class-minus-best-other-class margin clamped to [0, 0.1]. '
        'Other bands gaining correctness are not penalized. These are protocol definitions, not training verification.',
        '训练锚点仅来自源域训练 query；目标 margin 为 teacher margin 截断到 [0, 0.1]。'
        '源域验证 probe 与训练锚点分开，验证或目标标签不会用于建立训练锚点。'
        'retention 使用冻结部署频带读出，不是 standalone-only ridge 读出或训练 hinge 的重算。',
        'All 25 matching target COMPLETE markers and policies are required; every source SELECTION '
        'rule is loaded before any target-label read. The reused report checks source_selected '
        'against its frozen rule, rows, IDs, supports and support labels. Globals are restored afterward. '
        'Labels use the existing manifest paths and row alignment; no predictions or selections are changed.',
        'OA/BA/macro_f1_8 为百分数，delta 为百分点。BA 使用当前评估集合出现的真实类别，macro F1 固定八类。'
        '目标指标先平均任务内五个 draw，再等权平均任务。base 无专家交接；full 使用各模式自己的路由；'
        'frozen_route 统一使用 learned admission mask。原 metrics、task_means、summary、draw_metrics、'
        'class_metrics、events CSV 的 delta 相对同阶段 learned；source_selection.csv 保留冻结源域选择。',
        'Complementarity, band_events and rescue_overlaps retain base/frozen_route diagnostics. '
        'Label-assisted oracles preserve learned-correct queries and rescue main-wrong queries. '
        'Best-single-band oracles optimize OA and BA separately within task/draw only as upper bounds; '
        'union_minus_best_single_oracle measures the extra union gain. They are not deployable policies.',
    ]
    markdown = output/'REPORT_ZH.md'
    markdown.write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    matched, retained = matched_report(output), retention_report(folders, output)
    summary = pd.read_csv(output/'summary.csv')
    controls = ['learned', *(f'{f}_equal' for f in FAMILIES), 'source_selected']
    show = summary[summary['mode'].isin(controls)].pivot(
        index=['dataset', 'mode'], columns='stage', values='ba').reindex(
        columns=[s for s, _ in STAGES]).reset_index()
    notes.extend([
        '## 固定对照 BA / Fixed controls', markdown_table(show),
        '## 全部配对差值 / All matched differences',
        markdown_table(matched[[*GROUP, *DELTAS, 'positive_draws_ba', 'draws']]),
        'matched_* CSV 比较 standalone-independent、preserved-independent、preserved-standalone，'
        '固定保留 equal、四单频带、support_static、support_local 七种读出与三个阶段。'
        '先按同 dataset/task/draw/stage/suffix 相减，再汇总；不按目标结果筛选模式。'
        'positive_draws 严格计数五个 support draw 中 delta > 0 的次数；summary 使用每 draw 的任务等权均值，'
        'task_means 使用任务内 draw；positive_task_draws 计数各 task/draw 正差值。OA、BA、F1 均保存。',
        '## 保留诊断 / Retention diagnostics',
        markdown_table(retained[['split', 'dataset', 'baseline_anchor_count',
            'standalone_anchor_retained', 'preserved_anchor_retained',
            'standalone_lost_baseline_union_rescued_by_preserved', 'standalone_union_lost_by_preserved',
            'standalone_oracle_delta_ba_vs_independent', 'preserved_oracle_delta_ba_vs_independent',
            'preserved_oracle_delta_ba_vs_standalone']]),
        'retention.csv 的 source_validation unit 是 query_scores_XX.npz 的 episode；target_postfreeze unit '
        '是五次 support draw，仅使用 base_predictions 的 eval_mask query。源域验证不是训练锚点统计；'
        '同源域端口可能参与表征训练，不能称为表征独立验证。训练诊断以主流程记录为准。',
        'Baseline anchors fix learned-main-wrong AND exactly-one-independent-band-correct. Retention '
        'requires that same anchor band to remain correct, even if other bands become correct too. '
        'CSV includes LL/LH/HL/HH anchor denominators and retained counts. Union rescues include all '
        'main-wrong queries with any correct band, including non-unique rescues. New/lost union counts '
        'compare each family with independent. Recovery counts baseline union rescues lost by standalone '
        'but recovered by preserved; standalone_union_lost_by_preserved includes every standalone '
        'union rescue lost by preserved, whether or not it belonged to the independent baseline.',
        'Oracle BA delta weights each query by 100 / (K * n_true_class), using all true classes in '
        'that task/draw or source episode, never only rescue/anchor classes. Main-correct contributions '
        'cancel in the difference. retention_task_summary averages episode/draw BA deltas within task; '
        'retention_summary averages tasks equally. Counts sum query events, including repeated queries '
        'across draws/episodes/tasks, and are not independent replications or significance tests.',
    ])
    markdown.write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    print(f'PRESERVE REPORT: {output}', flush=True)


def main():
    report()


if __name__ == '__main__':
    main()
