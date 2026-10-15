"""Report a completed fixed-joint-model budget replay; never run experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path('E:/Hermes/paper_tracking/sensitivity/handoff_budget_sweep_20261008')
TASK_COUNTS = {'self24ports': 23, 'F0921': 1, 'OSS1': 1, 'OSS2': 1,
               'pacs': 4, 'officehome': 4}
POLICIES = {'gate_000': 0., 'gate_005': .05, 'gate_010': .1,
            'gate_020': .2, 'gate_050': .5, 'gate_100': 1., 'all_expert': 1.}
EXPANDED = ('gate_010', 'gate_020', 'gate_050', 'gate_100')
LABELS = {'gate_000': '0%', 'gate_005': '5%', 'gate_010': '10%',
          'gate_020': '20%', 'gate_050': '50%', 'gate_100': '100% 门控',
          'all_expert': '全 expert'}
COUNTS = ['selected', 'evaluated', 'eligible', 'rescued', 'harmed', 'changed',
          'selection_net_correct']
VALUES = ['budget', 'acc', 'ba', *COUNTS, 'realized_handoff_rate',
          'delta_acc_vs_base_pp', 'delta_ba_vs_base_pp']
KEYS = ['dataset', 'task', 'draw', 'policy']
EPS = 1e-9


def validate_metrics(frame):
    """Reject incomplete grids before their means become scientific results."""
    missing = set(KEYS + VALUES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing metrics columns: {sorted(missing)}')
    if frame[KEYS + VALUES].isna().any().any():
        raise ValueError('Required metrics contain missing values')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate dataset/task/draw/policy rows')
    if set(frame['dataset']) != set(TASK_COUNTS):
        raise ValueError('Expected exactly the six core datasets')
    if set(frame['policy']) != set(POLICIES):
        raise ValueError('Expected exactly seven budget policies')
    if set(frame['draw']) != set(range(5)):
        raise ValueError('Expected draw identifiers 0..4')
    for dataset, count in TASK_COUNTS.items():
        part = frame.loc[frame['dataset'].eq(dataset)]
        sizes = part.groupby('task').size()
        if len(sizes) != count or not sizes.eq(35).all():
            raise ValueError(f'{dataset}: expected {count} tasks, each with 5 draws x 7 policies')
    if not np.isfinite(frame[VALUES].to_numpy(dtype=float)).all():
        raise ValueError('Required metrics must be finite numeric values')
    if not np.allclose(frame['budget'], frame['policy'].map(POLICIES), rtol=0, atol=EPS):
        raise ValueError('Budget values disagree with policy identifiers')


def records(frame):
    return json.loads(frame.to_json(orient='records', double_precision=15))


def aggregate(frame):
    validate_metrics(frame)
    work = frame[KEYS + VALUES].copy()
    pair_keys = ['dataset', 'task', 'draw']
    reference = work.loc[work['policy'].eq('gate_005'), pair_keys + ['acc', 'ba']]
    work = work.merge(reference.rename(columns={'acc': 'acc_005', 'ba': 'ba_005'}),
                      on=pair_keys, validate='many_to_one')
    work['delta_acc_vs_5_pp'] = 100 * (work['acc'] - work['acc_005'])
    work['delta_ba_vs_5_pp'] = 100 * (work['ba'] - work['ba_005'])
    work['oa_pct'] = 100 * work['acc']
    work['ba_pct'] = 100 * work['ba']
    work['realized_handoff_pct'] = 100 * work['realized_handoff_rate']
    work['net_correct'] = work['rescued'] - work['harmed']
    values = VALUES + ['delta_acc_vs_5_pp', 'delta_ba_vs_5_pp', 'oa_pct', 'ba_pct',
                       'realized_handoff_pct', 'net_correct']
    tasks = work.groupby(['dataset', 'task', 'policy'], sort=False)[values].mean().reset_index()
    datasets = tasks.groupby(['dataset', 'policy'], sort=False)[values].mean().reset_index()
    for result in (tasks, datasets):
        result['_dataset_order'] = result['dataset'].map({v: i for i, v in enumerate(TASK_COUNTS)})
        result['_policy_order'] = result['policy'].map({v: i for i, v in enumerate(POLICIES)})
        order = ['_dataset_order'] + (['task'] if 'task' in result else []) + ['_policy_order']
        result.sort_values(order, inplace=True, ignore_index=True)
        result.drop(columns=['_dataset_order', '_policy_order'], inplace=True)
    tasks['n_draws'] = 5
    datasets['n_tasks'] = datasets['dataset'].map(TASK_COUNTS)
    datasets['n_draws_per_task'] = 5
    heterogeneity = []
    for (dataset, policy), part in tasks.groupby(['dataset', 'policy'], sort=False):
        oa, ba = part['delta_acc_vs_5_pp'], part['delta_ba_vs_5_pp']
        heterogeneity.append(dict(
            dataset=dataset, policy=policy, n_tasks=len(part),
            ba_positive_tasks=int((ba > EPS).sum()),
            ba_zero_tasks=int((ba.abs() <= EPS).sum()),
            ba_negative_tasks=int((ba < -EPS).sum()),
            min_delta_ba_vs_5_pp=float(ba.min()), max_delta_ba_vs_5_pp=float(ba.max()),
            oa_up_ba_down_tasks=int(((oa > EPS) & (ba < -EPS)).sum()),
            oa_down_ba_up_tasks=int(((oa < -EPS) & (ba > EPS)).sum()),
            ba_positive_task_names=part.loc[ba > EPS, 'task'].astype(str).tolist(),
            ba_negative_task_names=part.loc[ba < -EPS, 'task'].astype(str).tolist()))
    datasets = datasets.merge(pd.DataFrame(heterogeneity).drop(
        columns=['n_tasks', 'ba_positive_task_names', 'ba_negative_task_names']),
        on=['dataset', 'policy'], validate='one_to_one')
    increments = []
    for dataset in TASK_COUNTS:
        part = tasks.loc[tasks['dataset'].eq(dataset)]
        for lower, upper in [('gate_005', 'gate_010'), ('gate_010', 'gate_020'),
                             ('gate_020', 'gate_050'), ('gate_050', 'gate_100')]:
            lo = part.loc[part['policy'].eq(lower)].set_index('task')
            hi = part.loc[part['policy'].eq(upper)].set_index('task')
            difference = hi[values] - lo[values]
            row = dict(dataset=dataset, from_policy=lower, to_policy=upper,
                       n_tasks=len(part['task'].unique()))
            for column in ['selected', 'rescued', 'harmed', 'changed', 'net_correct']:
                row['increment_' + column] = float(difference[column].mean())
            row['delta_acc_pp'] = float(difference['oa_pct'].mean())
            row['delta_ba_pp'] = float(difference['ba_pct'].mean())
            row['delta_realized_handoff_pp'] = float(difference['realized_handoff_pct'].mean())
            increments.append(row)
    best = []
    for dataset in TASK_COUNTS:
        gate = datasets.loc[datasets['dataset'].eq(dataset) & datasets['policy'].ne('all_expert')]
        row = {'dataset': dataset}
        for metric in ('oa_pct', 'ba_pct'):
            maximum = float(gate[metric].max())
            row['max_' + metric] = maximum
            row[metric + '_policies'] = gate.loc[(gate[metric] - maximum).abs() <= EPS,
                                                'policy'].tolist()
        best.append(row)
    summary = dict(
        status='complete_metrics_report', metrics_rows=len(frame), expected_tasks=TASK_COUNTS,
        expected_draws=list(range(5)), policy_budgets=POLICIES,
        aggregation='Mean of five paired draws within task, then equal task mean within dataset; no pooling.',
        units={'acc_ba_budget_realized_handoff_rate': 'fraction',
               'oa_pct_ba_pct_realized_handoff_pct': 'percent',
               'delta_fields': 'percentage points',
               'count_fields': 'mean count per task after averaging five draws, not pooled counts',
               'evaluated': 'query-label denominator per task/draw; dataset value is its equal-task mean',
               'task_sign_counts': 'number of tasks after averaging paired draw differences'},
        scope={'fixed_joint_model': True, 'retraining': False,
               'query_based_policy_selection': False, 'oracle_ranking': False,
               'gate_100': 'Unlimited cap; disagreement, observed class and score > 0 still required.',
               'all_expert': 'All evaluated queries selected; no eligibility or score filter; changed counts only disagreement.',
               'five_percent_exact_original_replay': 'Verified by root replay; not verified by this report generator.',
               'compute_reduction_claim': False,
               'best_grid_point': 'Descriptive observed maxima only; no new policy is selected.'},
        dataset_means=records(datasets), task_means=records(tasks),
        task_heterogeneity_vs_5=heterogeneity, marginal_increments=increments,
        observed_gate_grid_maxima=best)
    return tasks, datasets, summary


def markdown(summary):
    lines = [
        '# 固定 joint 模型交接预算扫描', '',
        '## 问题与口径', '',
        '本报告比较原 5% 上限与 10%、20%、50%、100% 门控上限：放宽预算能否改善最终 OA/BA，'
        '新增交接带来多少纠正与伤害，以及收益是否集中在少数任务。0% 是 base 对照；全 expert 是独立对照。', '',
        '模型与预测固定，不重新训练，不用 query 标签选择策略，也不按 oracle 排序。'
        '5% 与原 joint 结果的精确重放校验由主重放程序负责，本报告生成器不宣称独立完成该校验。', '',
        '每个任务先平均 draw 0..4，再在各数据集内对任务等权平均；'
        '任务数依次为 self24ports 23、F0921 1、OSS1 1、OSS2 1、PACS 4、OfficeHome 4。'
        '不按 query 样本数池化，也不把五次支持集抽样当作五次独立神经网络训练。', '',
        '100% 门控仅取消预算截断，仍要求 base/expert 分歧、observed-class 条件及 score > 0；'
        '因此实际交接率未必为 100%。全 expert 不使用 eligibility 或 score 过滤，selected=evaluated；'
        'changed 仅统计预测分歧，不能与 selected 混为一谈。', '',
        'OA、BA、实际交接率以 % 表示，差值以百分点（pp）表示。'
        '纠正、伤害、净纠正、selected、changed、eligible 和 evaluated 均为“每任务平均计数”，'
        '即先平均五次 draw 后再平均任务，允许小数；evaluated 是每任务 query 标签分母的均值，'
        '不是池化总标签数，不能用平均计数之比替代平均准确率或交接率。', '',
        '## 各数据集最终表现与累计效用', '',
        '纠正=base 错而最终预测对；伤害=base 对而最终预测错；净纠正=纠正−伤害。'
        '累计纠正/伤害均相对 base，Δ5% 才是相对原预算的性能变化。', '']
    for dataset in TASK_COUNTS:
        rows = [r for r in summary['dataset_means'] if r['dataset'] == dataset]
        lines += [f'### {dataset}（{TASK_COUNTS[dataset]} 个任务）', '',
                  '| 策略 | OA % | BA % | 实际交接 % | 纠正/任务 | 伤害/任务 | 净纠正/任务 | ΔOA base | ΔBA base | ΔOA 5% | ΔBA 5% |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            lines.append(f'| {LABELS[r["policy"]]} | {r["oa_pct"]:.4f} | {r["ba_pct"]:.4f} | '
                         f'{r["realized_handoff_pct"]:.4f} | {r["rescued"]:.3f} | {r["harmed"]:.3f} | '
                         f'{r["net_correct"]:+.3f} | {r["delta_acc_vs_base_pp"]:+.4f} | '
                         f'{r["delta_ba_vs_base_pp"]:+.4f} | {r["delta_acc_vs_5_pp"]:+.4f} | '
                         f'{r["delta_ba_vs_5_pp"]:+.4f} |')
        lines += ['', '| 策略 | selected/任务 | changed/任务 | eligible/任务 | evaluated 标签分母/任务 |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for r in rows:
            lines.append(f'| {LABELS[r["policy"]]} | {r["selected"]:.3f} | {r["changed"]:.3f} | '
                         f'{r["eligible"]:.3f} | {r["evaluated"]:.3f} |')
        expanded = [r for r in rows if r['policy'] in EXPANDED]
        both_up = [LABELS[r['policy']] for r in expanded
                   if r['delta_acc_vs_5_pp'] > EPS and r['delta_ba_vs_5_pp'] > EPS]
        losses = [f'{LABELS[r["policy"]]}（ΔOA {r["delta_acc_vs_5_pp"]:+.4f}、'
                  f'ΔBA {r["delta_ba_vs_5_pp"]:+.4f} pp）'
                  for r in expanded if min(r['delta_acc_vs_5_pp'], r['delta_ba_vs_5_pp']) < -EPS]
        lines += ['', '相对 5%，OA 与 BA 同时严格提高的扩展上限：' + ('、'.join(both_up) or '无') + '。',
                  '出现至少一项指标下降的扩展上限：' + ('；'.join(losses) or '无') + '。']
        tradeoffs = [LABELS[r['policy']] for r in expanded
                    if (r['delta_acc_vs_5_pp'] > EPS and r['delta_ba_vs_5_pp'] < -EPS)
                    or (r['delta_acc_vs_5_pp'] < -EPS and r['delta_ba_vs_5_pp'] > EPS)]
        lines += [('OA/BA 方向相反的上限：' + '、'.join(tradeoffs) +
                   '。总体正确率与类别等权召回存在权衡；不能仅凭 OA 判定整体改善。') if tradeoffs else
                  '数据集均值中未出现 OA 与 BA 严格反向变化；任务层面的权衡仍见下表。']
        expert = next(r for r in rows if r['policy'] == 'all_expert')
        gate = next(r for r in rows if r['policy'] == 'gate_100')
        lines += [f'全 expert 相对 5%：ΔOA {expert["delta_acc_vs_5_pp"]:+.4f}、'
                  f'ΔBA {expert["delta_ba_vs_5_pp"]:+.4f} pp；相对 100% 门控：'
                  f'ΔOA {expert["oa_pct"]-gate["oa_pct"]:+.4f}、'
                  f'ΔBA {expert["ba_pct"]-gate["ba_pct"]:+.4f} pp。两者不是同一策略。', '']
    lines += ['## 相对 5% 的任务异质性', '',
              '逐任务配对相同 draw 后平均差值，再统计任务正/零/负；'
              '零的数值容差为 1e-9 pp。正任务数不是显著性检验，五次 draw 不作为独立任务计数。', '',
              '| 数据集 | 策略 | BA 正/总任务 | BA 零 | BA 负 | ΔBA 最小/最大 pp | OA↑BA↓任务 | OA↓BA↑任务 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary['task_heterogeneity_vs_5']:
        if r['policy'] not in (*EXPANDED, 'all_expert'):
            continue
        lines.append(f'| {r["dataset"]} | {LABELS[r["policy"]]} | {r["ba_positive_tasks"]}/{r["n_tasks"]} | '
                     f'{r["ba_zero_tasks"]} | {r["ba_negative_tasks"]} | '
                     f'{r["min_delta_ba_vs_5_pp"]:+.4f}/{r["max_delta_ba_vs_5_pp"]:+.4f} | '
                     f'{r["oa_up_ba_down_tasks"]} | {r["oa_down_ba_up_tasks"]} |')
    lines += ['', '## 新增预算的边际效用', '',
              '下表对相同任务的相邻上限作差后等权平均。新增纠正/伤害是累计计数之差，'
              '不把上表的累计收益误读为新增预算收益。固定排序下嵌套选集的成员关系由重放实现负责；'
              '仅凭汇总 CSV 不能独立验证成员关系。', '',
              '| 数据集 | 上限变化 | 新增 selected/任务 | 新增纠正/任务 | 新增伤害/任务 | 新增净纠正/任务 | Δ交接 pp | ΔOA pp | ΔBA pp |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary['marginal_increments']:
        lines.append(f'| {r["dataset"]} | {LABELS[r["from_policy"]]} → {LABELS[r["to_policy"]]} | '
                     f'{r["increment_selected"]:+.3f} | {r["increment_rescued"]:+.3f} | '
                     f'{r["increment_harmed"]:+.3f} | {r["increment_net_correct"]:+.3f} | '
                     f'{r["delta_realized_handoff_pp"]:+.4f} | {r["delta_acc_pp"]:+.4f} | '
                     f'{r["delta_ba_pp"]:+.4f} |')
    lines += ['', '## 观测网格极值（仅描述）', '',
              '极值仅在七档中的六档 gate 上限内描述，包含 0% 与原 5%，不将全 expert 混入门控上限网格。'
              '并列值全部列出；这些 query 结果不用于选出新的部署策略。', '',
              '| 数据集 | 观测最高 OA % | 对应 gate 上限 | 观测最高 BA % | 对应 gate 上限 |',
              '| --- | ---: | --- | ---: | --- |']
    for r in summary['observed_gate_grid_maxima']:
        lines.append(f'| {r["dataset"]} | {r["max_oa_pct"]:.4f} | '
                     f'{", ".join(LABELS[p] for p in r["oa_pct_policies"])} | {r["max_ba_pct"]:.4f} | '
                     f'{", ".join(LABELS[p] for p in r["ba_pct_policies"])} |')
    lines += ['', '## 机会与边界', '',
              '放宽上限提供释放已有 expert 纠错能力的机会。是否值得扩展，应结合相对 5% 的最终 OA/BA、'
              '新增纠正与伤害以及正收益任务覆盖来看；均值改善不保证每个任务或每个类别改善。', '',
              '若预算继续增加而实际交接及指标不变，说明当前候选或正分筛选已饱和；'
              '若新增伤害抵消新增纠正，则扩大预算并不自动带来收益。BA 与 OA 反向时存在类别平衡权衡，'
              '这里没有逐类结果，不能把变化归因于某个具体类别。', '',
              '该实验衡量固定模型与固定打分下的交接预算敏感性，不证明新的最优策略、统计显著性或盲测泛化。'
              'selected 与实际交接率仅刻画决策交接，不能解释为实际计算量、延迟或能耗下降；'
              '本报告没有开展计算节省测量。', '',
              'task_means.csv 保留全部逐任务均值与相对 5% 差值；dataset_means.csv 保留数据集均值；'
              'summary.json 同时保存任务异质性、边际增量、单位和策略定义。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=OUT / 'metrics.csv')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics, dtype={'dataset': str, 'task': str, 'policy': str})
    tasks, datasets, summary = aggregate(frame)
    summary['metrics_source'] = str(args.metrics.resolve())
    report = markdown(summary)
    args.out.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(args.out / 'task_means.csv', index=False, encoding='utf-8-sig')
    datasets.to_csv(args.out / 'dataset_means.csv', index=False, encoding='utf-8-sig')
    (args.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   allow_nan=False) + '\n', encoding='utf-8')
    (args.out / 'BUDGET_FINDINGS_ZH.md').write_text(report, encoding='utf-8')
    print(f'Saved six-dataset budget report to {args.out}')


if __name__ == '__main__':
    main()
