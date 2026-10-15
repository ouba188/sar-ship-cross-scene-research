"""Post-freeze specialist readout; target-assisted oracles never select a policy."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, DATASETS, load_npz
from frequency_reliability_run import statistics
from frequency_band_report import BANDS, STAGES, complementarity, markdown_table

OUT = ROOT/'frequency_specialists_20261009'
FAMILIES = ('old', 'continued', 'independent', 'specialist')
MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *BANDS)),
         *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')
VALUES = ['oa', 'ba', 'macro_f1_8', 'delta_oa', 'delta_ba']


def report(folders=None, output=OUT):
    output = Path(output)
    folders = ([p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
               if folders is None else [Path(p) for p in folders])
    infos = [json.loads((p/'PREPARED.json').read_text()) for p in folders]
    expected = {(d['dataset'], d['task']) for d in infos}
    markers = {(p.parent.parent.name, p.parent.name)
               for p in (output/'targets').glob('*/*/COMPLETE.json')}
    if len(folders) != 25 or len(expected) != 25 or markers != expected:
        raise ValueError('Freeze all25 matching target tasks before any label reads')
    # Snapshot every final source rule before opening any target-label manifest.
    sources = []
    for info in infos:
        suffix = Path(info['dataset'])/info['task']
        if not (output/'targets'/suffix/'policies.npz').is_file():
            raise ValueError(f'Missing frozen policies: {suffix}')
        source = json.loads((output/'source'/suffix/'SELECTION.json').read_text())
        if source['rule'] not in MODES[:-1] or source['rule'] not in source['source_delta_ba']:
            raise ValueError(f'Invalid frozen source selection: {suffix}')
        sources.append(source)
    source_manifest = pd.read_csv(SOURCE/'manifest.csv', usecols=['instance_id', 'class_id'])
    metrics, classes, selections, complements, band_events, overlaps = [], [], [], [], [], []
    for folder, info, source in zip(folders, infos, sources):
        meta = dict(dataset=info['dataset'], task=info['task'])
        p = load_npz(output/'targets'/info['dataset']/info['task']/'policies.npz')
        original = load_npz(folder/'target.npz')
        for key in ('rows', 'support', 'eval_masks'):
            np.testing.assert_array_equal(p[key], original[key])
        modes = list(p['modes'])
        if len(modes) != len(MODES) or set(modes) != set(MODES):
            raise ValueError(f'Unexpected specialist modes: {meta}')
        reference_index = modes.index('learned')
        chosen_index, rule_index = modes.index('source_selected'), modes.index(source['rule'])
        for key in ('base_predictions', 'predictions', 'frozen_predictions', 'selected'):
            if p[key].shape != (5, len(modes), len(p['rows'])):
                raise ValueError(f'Unexpected {key} shape: {meta}')
            np.testing.assert_array_equal(p[key][:, chosen_index], p[key][:, rule_index])
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
        selections.append(dict(**meta, rule=source['rule'],
            **{'delta_ba_'+k: v for k, v in source['source_delta_ba'].items()}))
        for draw, ev in enumerate(p['eval_masks']):
            y = y_all[ev]
            for stage, key in STAGES:
                predictions = p[key][draw][:, ev]
                reference = predictions[reference_index]
                correct = reference == y
                roa, rba, _ = statistics(y, reference)
                for j, mode in enumerate(modes):
                    pred = predictions[j]
                    oa, ba, f1 = statistics(y, pred)
                    admissions = (int(p['selected'][draw, j, ev].sum()) if stage == 'full' else
                        int(p['selected'][draw, reference_index, ev].sum()) if stage == 'frozen_route' else 0)
                    metrics.append(dict(**meta, draw=draw, stage=stage, mode=mode, n=len(y),
                        oa=oa, ba=ba, macro_f1_8=f1, delta_oa=oa-roa, delta_ba=ba-rba,
                        changes=int(np.sum(pred != reference)), rescues=int(np.sum((pred == y) & ~correct)),
                        harms=int(np.sum((pred != y) & correct)), admissions=admissions))
                    for c in np.unique(y):
                        mask = y == c
                        classes.append(dict(**meta, draw=draw, stage=stage, mode=mode, class_id=int(c),
                            n=int(mask.sum()), support_n=int(np.sum(original['support_labels'][draw] == c)),
                            recall=100*float(np.mean(pred[mask] == c)),
                            delta_recall=100*float(np.mean(pred[mask] == c)-np.mean(correct[mask]))))
                if stage == 'full':
                    continue
                for family in FAMILIES:
                    bands = predictions[[modes.index(f'{family}_{b}') for b in BANDS]].T
                    values, rescue = complementarity(reference, bands, y)
                    complements.append(dict(**meta, draw=draw, stage=stage, family=family, **values))
                    count = rescue.sum(1)
                    for b, band in enumerate(BANDS):
                        band_events.append(dict(**meta, draw=draw, stage=stage, family=family, band=band,
                            rescues=int(rescue[:, b].sum()), harms=int(np.sum((bands[:, b] != y) & correct)),
                            unique_rescues=int(np.sum(rescue[:, b] & (count == 1)))))
                    for a, b in combinations(range(4), 2):
                        overlaps.append(dict(**meta, draw=draw, stage=stage, family=family,
                            band_a=BANDS[a], band_b=BANDS[b],
                            intersection=int(np.sum(rescue[:, a] & rescue[:, b])),
                            union=int(np.sum(rescue[:, a] | rescue[:, b]))))
        print(f'SPECIALIST REPORT {info["dataset"]}/{info["task"]}: IDs and support labels aligned', flush=True)
    m = pd.DataFrame(metrics)
    m.to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(output/'class_metrics.csv', index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(output/'source_selection.csv', index=False)
    task = m.groupby(['dataset', 'task', 'stage', 'mode'])[VALUES].mean().reset_index()
    task.to_csv(output/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'stage', 'mode'])[VALUES].mean().reset_index()
    summary.to_csv(output/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'stage', 'mode'])[VALUES].mean().reset_index().to_csv(
        output/'draw_metrics.csv', index=False)
    m.groupby(['dataset', 'stage', 'mode'])[['changes', 'rescues', 'harms', 'admissions']].sum().reset_index().to_csv(
        output/'events.csv', index=False)
    comp = pd.DataFrame(complements)
    comp.to_csv(output/'complementarity.csv', index=False)
    comp_values = [c for c in comp.columns if c.endswith(('_oa', '_ba'))]
    comp_summary = comp.groupby(['dataset', 'task', 'stage', 'family'])[comp_values].mean().groupby(
        ['dataset', 'stage', 'family']).mean().reset_index()
    comp_summary.to_csv(output/'complementarity_summary.csv', index=False)
    event_summaries = {}
    for name, rows, keys, fields in (
        ('band_events', band_events, ['family', 'band'], ['rescues', 'harms', 'unique_rescues']),
        ('rescue_overlaps', overlaps, ['family', 'band_a', 'band_b'], ['intersection', 'union'])):
        frame = pd.DataFrame(rows)
        frame.to_csv(output/(name+'.csv'), index=False)
        totals = frame.groupby(['dataset', 'stage', *keys])[fields].sum().reset_index()
        totals.to_csv(output/(name+'_summary.csv'), index=False)
        event_summaries[name] = totals
    controls = ['learned', *(f'{f}_equal' for f in FAMILIES), 'source_selected']
    show = summary[summary['mode'].isin(controls)].pivot(
        index=['dataset', 'mode'], columns='stage', values='ba').reindex(
        columns=['base', 'full', 'frozen_route']).reset_index()
    oracle_show = comp_summary[['dataset', 'stage', 'family', 'oracle_delta_ba',
                                'best_single_oracle_ba', 'union_minus_best_single_oracle_ba']]
    events = event_summaries['band_events']
    unique = events[events.stage.eq('base')].pivot(
        index=['dataset', 'family'], columns='band', values='unique_rescues').reindex(columns=BANDS).reset_index()
    notes = [
        '# 频带 specialist 冻结结果',
        '本报告仅呈现结果与口径，不给出训练有效性或失效根因结论；结论等待主流程根因诊断。',
        '全部 25 个匹配任务的 COMPLETE.json 存在后才读取标签。全部源域 SELECTION.json 在目标标签读取前载入，'
        '报告不重新选择策略；source_selected 已与源域 rule 对应预测逐项对齐，样本 ID、行索引、support 与标签沿用直接对齐核验。',
        'old 为原始等权联合训练固定 checkpoint；continued、independent、specialist 为主流程生成的三个训练家族。'
        'learned 保留原 selective 主分支。四个家族分别比较 equal 与 LL/LH/HL/HH；具体训练目标以主流程记录为准。',
        '## 主要对照 BA（百分数）', markdown_table(show),
        '完整模式含各家族 support_static/support_local，均保存在 summary.csv；增益相对同阶段 learned。'
        'base 不交接专家，full 按各模式独立路由，frozen_route 统一使用 learned admission mask。',
        '## 四频带互补上界', markdown_table(oracle_show),
        'Oracle 保留 learned 正确项，只修复 learned 错误且至少一个频带正确的项；使用目标标签，仅为不可部署上界。'
        'best_single_oracle_ba 是每个 task/draw 内最佳 learned+单频带 oracle 的 BA；'
        'union_minus_best_single_oracle_ba 才是四频带相对该上界的额外增益。OA 对应字段也保存在 CSV。'
        '最佳单频带及 oracle 按 OA/BA 各自取最优后汇总，不是源域部署选择。'
        '仅对 base 与 frozen_route 构造 oracle，不合并各自重路由的 full 预测。',
        '## 各频带独有救回（base）', markdown_table(unique),
        '独有救回指 learned 错误、当前家族恰好一个频带正确；不表示跨家族独有。'
        'band_events.csv 同时包含 rescues、harms；rescue_overlaps.csv 记录每对频带救回集合的 intersection/union。'
        '两者均含 base/frozen_route 明细及汇总。预测不同本身不证明学到了不同且有用的证据。',
        '## 冻结源域选择', markdown_table(selection.groupby(['dataset', 'rule']).size().reset_index(name='tasks')),
        'source_selection.csv 保留 rule 与全部 source_delta_ba；目标指标不回流训练或选择。'
        '源域验证是否与表征训练独立，应由主流程训练/验证记录判定。',
        '## 输出口径',
        'OA/BA/macro_f1_8 为百分数，delta 为百分点；BA 使用评估集合出现的类别，macro F1 固定八类。'
        'summary 先平均任务内五个 draw，再等权平均任务；draw_metrics 在同一 draw 内等权平均任务。'
        'class_metrics 保留每任务/draw 的类别样本数、support 数、recall 与 delta_recall。'
        '所有救回、伤害和交并计数按 query-draw 累加，同一样本跨 draw 会重复，不代表独立样本数。',
        'metrics.csv、task_means.csv、summary.csv、draw_metrics.csv、class_metrics.csv、events.csv 为三阶段指标；'
        'complementarity.csv 与 complementarity_summary.csv 为四家族 oracle 明细及等权汇总。'
        '本报告不生成图、不修改冻结预测或源域选择。'
    ]
    (output/'REPORT_ZH.md').write_text('\n\n'.join(notes)+'\n', encoding='utf-8')
    (output/'REPORT_COMPLETE.json').write_text(json.dumps(dict(tasks=25, draws=125,
        labels_aligned=True, source_selected_aligned=True, target_labels_only_in_report=True,
        oracle_deployable=False, families=FAMILIES, root_diagnosis_pending=True), indent=2), encoding='utf-8')
    print(show.to_string(index=False), flush=True)


if __name__ == '__main__':
    report()
