"""Frozen-target readout; label-assisted oracles never feed a selector."""

import json
from itertools import combinations

import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz
from frequency_reliability_run import statistics
from frequency_band_core import BANDS, oracle_predictions

OUT = ROOT/'frequency_band_complementarity_20261009'
DYNAMIC = ('dynamic_scores', 'dynamic_support')
STAGES = (('base', 'base_predictions'), ('full', 'predictions'),
          ('frozen_route', 'frozen_predictions'))


def complementarity(reference, bands, y):
    correct = bands == y[:, None]
    rescue = correct & (reference != y)[:, None]
    oracle = oracle_predictions(reference, bands, y)
    roa, rba, _ = statistics(y, reference)
    oa, ba, _ = statistics(y, oracle)
    singles = np.array([statistics(y, bands[:, b])[:2] for b in range(4)])
    single_oracles = np.array([statistics(y, oracle_predictions(
        reference, bands[:, b:b+1], y))[:2] for b in range(4)])
    bo, bb = singles.argmax(0)
    oo, ob = single_oracles.argmax(0)
    return dict(n=len(y), reference_oa=roa, reference_ba=rba,
        oracle_oa=oa, oracle_ba=ba, oracle_delta_oa=oa-roa, oracle_delta_ba=ba-rba,
        opportunities=int(rescue.any(1).sum()),
        best_single_oa_band=BANDS[bo], best_single_oa=float(singles[bo, 0]),
        best_single_ba_band=BANDS[bb], best_single_ba=float(singles[bb, 1]),
        union_minus_best_single_oa=oa-singles[bo, 0],
        union_minus_best_single_ba=ba-singles[bb, 1],
        best_single_oracle_oa_band=BANDS[oo], best_single_oracle_oa=float(single_oracles[oo, 0]),
        best_single_oracle_ba_band=BANDS[ob], best_single_oracle_ba=float(single_oracles[ob, 1]),
        union_minus_best_single_oracle_oa=oa-single_oracles[oo, 0],
        union_minus_best_single_oracle_ba=ba-single_oracles[ob, 1]), rescue


def markdown_table(frame):
    lines = ['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    for row in frame.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)


def report():
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    infos = [json.loads((p/'PREPARED.json').read_text()) for p in folders]
    expected = {(d['dataset'], d['task']) for d in infos}
    markers = {(p.parent.parent.name, p.parent.name) for p in (OUT/'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or markers != expected:
        raise ValueError('Freeze all25 matching target tasks before evaluation-label reads')
    # Labels enter only after every expected target task is frozen.
    source_manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id'])
    metrics, classes, selections, complements, band_events, overlaps, captures = [], [], [], [], [], [], []
    for folder, info in zip(folders, infos):
        meta = dict(dataset=info['dataset'], task=info['task'])
        target_dir = OUT/'targets'/info['dataset']/info['task']
        p = load_npz(target_dir/'policies.npz')
        original = load_npz(folder/'target.npz')
        np.testing.assert_array_equal(p['rows'], original['rows'])
        np.testing.assert_array_equal(p['support'], original['support'])
        np.testing.assert_array_equal(p['eval_masks'], original['eval_masks'])
        if info['dataset'] == 'self24ports':
            manifest, id_column = source_manifest, 'instance_id'
        else:
            manifest = pd.read_csv(ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv',
                                   usecols=['sample_id', 'class_id'], keep_default_na=False)
            id_column = 'sample_id'
        aligned = manifest.iloc[p['rows']]
        np.testing.assert_array_equal(aligned[id_column].to_numpy(str), original['ids'])
        y_all = aligned.class_id.to_numpy()
        np.testing.assert_array_equal(y_all[p['support']], original['support_labels'])
        source = json.loads((OUT/'source'/info['dataset']/info['task']/'SELECTION.json').read_text())
        selections.append(dict(**meta, rule=source['rule'], static_rule=source['static_rule'],
            source_queries=source['source_queries'],
            **{'delta_ba_'+k: v for k, v in source['source_delta_ba'].items()}))
        modes = list(p['modes'])
        reference_index = modes.index('learned')
        query = dict(rows=p['rows'], eval_masks=p['eval_masks'], choices=p['choices'])
        for stage, _ in (STAGES[0], STAGES[2]):
            query[stage+'_rescue_band_count'] = np.zeros((5, 2, len(y_all)), np.int8)
            query[stage+'_dynamic_captured'] = np.zeros((5, 2, 2, len(y_all)), bool)
        for draw, ev in enumerate(p['eval_masks']):
            y = y_all[ev]
            for stage, key in STAGES:
                predictions = p[key][draw][:, ev]
                reference = predictions[reference_index]
                roa, rba, _ = statistics(y, reference)
                for j, mode in enumerate(modes):
                    pred = predictions[j]
                    oa, ba, f1 = statistics(y, pred)
                    admissions = (int(p['selected'][draw, j, ev].sum()) if stage == 'full' else
                        int(p['selected'][draw, reference_index, ev].sum()) if stage == 'frozen_route' else 0)
                    metrics.append(dict(**meta, draw=draw, stage=stage, mode=mode, n=len(y),
                        oa=oa, ba=ba, macro_f1_8=f1, delta_oa=oa-roa, delta_ba=ba-rba,
                        changes=int(np.sum(pred != reference)),
                        rescues=int(np.sum((pred == y) & (reference != y))),
                        harms=int(np.sum((pred != y) & (reference == y))), admissions=admissions))
                    for c in np.unique(y):
                        mask = y == c
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c),
                            n=int(mask.sum()), support_n=int(np.sum(original['support_labels'][draw] == c)),
                            recall=100*float(np.mean(pred[mask] == c)),
                            delta_recall=100*float(np.mean(pred[mask] == c)-np.mean(reference[mask] == c))))
                if stage == 'full':
                    continue
                for fi, family in enumerate(('fixed', 'selective')):
                    bands = predictions[[modes.index(f'{family}_{b}') for b in BANDS]].T
                    values, rescue = complementarity(reference, bands, y)
                    complements.append(dict(**meta, draw=draw, stage=stage, family=family, **values))
                    count = rescue.sum(1)
                    opportunity = count > 0
                    query[stage+'_rescue_band_count'][draw, fi, ev] = count
                    for b, band in enumerate(BANDS):
                        band_events.append(dict(**meta, draw=draw, stage=stage, family=family, band=band,
                            rescues=int(rescue[:, b].sum()),
                            harms=int(np.sum((bands[:, b] != y) & (reference == y))),
                            unique_rescues=int(np.sum(rescue[:, b] & (count == 1)))))
                    for a, b in combinations(range(4), 2):
                        overlaps.append(dict(**meta, draw=draw, stage=stage, family=family,
                            band_a=BANDS[a], band_b=BANDS[b],
                            intersection=int(np.sum(rescue[:, a] & rescue[:, b])),
                            union=int(np.sum(rescue[:, a] | rescue[:, b]))))
                    for mode in ('fixed_equal', 'source_static', *DYNAMIC, 'source_selected'):
                        pred = predictions[modes.index(mode)]
                        captured = opportunity & (pred == y)
                        record = dict(**meta, draw=draw, stage=stage, family=family, mode=mode,
                            n=len(y), opportunities=int(opportunity.sum()), captured=int(captured.sum()),
                            changes=int(np.sum(pred != reference)),
                            rescues=int(np.sum((pred == y) & (reference != y))),
                            harms=int(np.sum((pred != y) & (reference == y))))
                        if mode in DYNAMIC:
                            di = DYNAMIC.index(mode)
                            choices = p['choices'][draw, di, ev]
                            record['selected_band'] = int(np.sum(choices != 0))
                            record['selected_on_opportunity'] = int(np.sum((choices != 0) & opportunity))
                            for idx, name in ((0, 'learned'), (2, 'LL'), (3, 'LH'), (4, 'HL'), (5, 'HH')):
                                record['choice_'+name] = int(np.sum(choices == idx))
                            query[stage+'_dynamic_captured'][draw, fi, di, ev] = captured
                        captures.append(record)
        np.savez_compressed(target_dir/'report_opportunities.npz', **query)
        print('BAND REPORT '+info['task']+': label IDs and support labels aligned', flush=True)
    m = pd.DataFrame(metrics)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(OUT/'source_selection.csv', index=False)
    values = ['oa', 'ba', 'macro_f1_8', 'delta_oa', 'delta_ba']
    task = m.groupby(['dataset', 'task', 'stage', 'mode'])[values].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[values].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'stage', 'mode'])[values].mean().reset_index().to_csv(OUT/'draw_metrics.csv', index=False)
    m.groupby(['dataset', 'stage', 'mode'])[['changes', 'rescues', 'harms', 'admissions']].sum().reset_index().to_csv(OUT/'events.csv', index=False)
    comp = pd.DataFrame(complements)
    comp.to_csv(OUT/'complementarity.csv', index=False)
    comp_values = [c for c in comp.columns if c.endswith(('_oa', '_ba'))]
    comp_summary = comp.groupby(['dataset', 'task', 'stage', 'family'])[comp_values].mean().groupby(
        ['dataset', 'stage', 'family']).mean().reset_index()
    comp_summary.to_csv(OUT/'complementarity_summary.csv', index=False)
    for name, rows, keys, fields in (
        ('band_events', band_events, ['family', 'band'], ['rescues', 'harms', 'unique_rescues']),
        ('rescue_overlaps', overlaps, ['family', 'band_a', 'band_b'], ['intersection', 'union'])):
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT/(name+'.csv'), index=False)
        totals = frame.groupby(['dataset', 'stage', *keys])[fields].sum().reset_index()
        totals.to_csv(OUT/(name+'_summary.csv'), index=False)
    capture = pd.DataFrame(captures)
    capture['capture_proportion'] = capture.captured/capture.opportunities.replace(0, np.nan)
    capture.to_csv(OUT/'capture_metrics.csv', index=False)
    counts = ['n', 'opportunities', 'captured', 'changes', 'rescues', 'harms', 'selected_band',
              'selected_on_opportunity', 'choice_learned', *('choice_'+b for b in BANDS)]
    capture_summary = capture.groupby(['dataset', 'stage', 'family', 'mode'])[counts].sum(min_count=1).reset_index()
    capture_summary['capture_proportion'] = capture_summary.captured/capture_summary.opportunities.replace(0, np.nan)
    capture_summary.to_csv(OUT/'capture_summary.csv', index=False)
    controls = ['learned', 'fixed_equal', 'source_static', *DYNAMIC, 'source_selected']
    show = summary[summary.stage.eq('base') & summary['mode'].isin(controls)]
    oracle_show = comp_summary[['dataset', 'stage', 'family', 'oracle_delta_oa', 'oracle_delta_ba',
                                'union_minus_best_single_oracle_oa', 'union_minus_best_single_oracle_ba']]
    capture_show = capture_summary[capture_summary.stage.eq('base') & capture_summary.family.eq('fixed')][
        ['dataset', 'mode', 'opportunities', 'captured', 'harms', 'capture_proportion']]
    notes = [
        '# 频带互补性结果',
        '25 个任务、125 个 support draw 的预测全部冻结后读取目标标签；样本 ID、行索引、support 标签均已对应核验。',
        '这是冻结特征及预测层面的描述性比较，不构成物理频率或因果解释。Oracle 使用目标真值，只是不可部署的上界；没有用它选择部署策略。',
        '选择器在源域按 episode 港口留出评估，但表征训练共享源港口，不是独立表征留港验证。动态策略是硬候选选择；选择多样性或权重变化不能证明学到了有效注意力。',
        '## 部署策略与静态对照（base）', markdown_table(show[['dataset', 'mode', 'oa', 'ba', 'delta_oa', 'delta_ba']]),
        '## Oracle 互补上界',
        'Oracle 保留 learned 的所有正确预测，只在 learned 错误且候选正确时修复。最佳单频带及最佳 learned+单频带 oracle 均在每个 task/draw 上用标签描述性选择，OA 和 BA 分别取最大值；全四频带相对最佳单频带 oracle 的增益才衡量额外多频带多样性。',
        markdown_table(oracle_show),
        '只计算 base 与固定 learned admission mask 的 frozen_route oracle；不拼接独立重路由的 full 预测，避免突破 5% 专家预算。',
        '## 机会捕获与代价（fixed 家族，base）', markdown_table(capture_show),
        'capture_proportion = 汇总 captured / 汇总 opportunities，不是各任务比例的平均。harms 是 learned 正确而策略错误的数量；捕获机会不等于净收益。selective 家族的捕获仅表示机会集合重叠，动态策略实际候选仍是 fixed 家族。',
        '## 源域选择', markdown_table(selection.groupby(['dataset', 'rule']).size().reset_index(name='tasks')),
        '## 输出口径',
        'OA/BA/F1 为百分数，增益为百分点。summary 先平均任务内五个 draw，再等权平均任务；计数按 query-draw 累加，同一样本跨 draw 会重复，不能视为独立样本。BA 使用当前评估集合出现的类别，macro F1 固定八类。',
        'metrics.csv、task_means.csv、summary.csv、draw_metrics.csv、class_metrics.csv、events.csv 给出三种路由阶段；complementarity.csv 给出每个任务/draw 的 oracle；band_events.csv 含独有救回；rescue_overlaps.csv 含两两交并计数；capture_summary.csv 用总计数计算捕获率。',
        '各 target 下 report_opportunities.npz 保存 rows、eval_masks、choices、每 query 的两个频带家族救回频带数及两个动态策略捕获标记。家族轴为 fixed/selective，动态轴为 dynamic_scores/dynamic_support；非评估行必须用 eval_masks 排除。这些均为报告产物，不得回流选择器。'
    ]
    (OUT/'REPORT_ZH.md').write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    (OUT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(tasks=25, draws=125,
        labels_aligned=True, target_labels_only_in_report=True, oracle_deployable=False,
        independent_representation_validation=False), indent=2), encoding='utf-8')
    print(show.to_string(index=False), flush=True)


if __name__ == '__main__':
    report()
