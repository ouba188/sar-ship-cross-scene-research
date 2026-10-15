"""Report paired budget isolation without choosing on target labels."""

from run_support_budget_isolation import OUT, ARMS, old
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    families = [f for f in ('source', 'target') if (OUT/(f+'_evaluation')/'COMPLETE.json').exists()]
    summaries, pairs, seed_rows, spreads = [], [], [], []
    for family in families:
        dest = OUT/(family+'_evaluation')
        for name, result in [('summary', summaries), ('paired_summary', pairs), ('seed_summary', seed_rows)]:
            frame = pd.read_csv(dest/(name+'.csv'))
            frame['family'] = family
            result.append(frame)
        frame = pd.read_csv(dest/'paired.csv')
        for key, group in frame.groupby(['context', 'seed', 'budget', 'layer', 'reference']):
            spreads.append(dict(family=family, **dict(zip(['context', 'seed', 'budget', 'layer', 'reference'], key)),
                mean_delta_ba=group.delta_ba.mean(), std_delta_ba=group.delta_ba.std(ddof=0),
                wins=int((group.delta_ba > 1e-10).sum()), ties=int((abs(group.delta_ba) <= 1e-10).sum()),
                losses=int((group.delta_ba < -1e-10).sum()), minimum=group.delta_ba.min(), maximum=group.delta_ba.max()))
    summary, paired, seeds = map(lambda x: pd.concat(x, ignore_index=True), (summaries, pairs, seed_rows))
    summary.to_csv(OUT/'all_summary.csv', index=False)
    paired.to_csv(OUT/'all_paired_summary.csv', index=False)
    seeds.to_csv(OUT/'all_seed_summary.csv', index=False)
    pd.DataFrame(spreads).to_csv(OUT/'support_variation.csv', index=False)
    new = pd.read_csv(OUT/'fitting_probe.csv')
    prev = pd.read_csv(old.OUT/'fitting_probe.csv')
    prev = prev[prev.arm == 'task'].copy()
    prev['step'] = prev.step//3
    keys = ['context', 'seed', 'step', 'budget', 'fold']
    probe = new.merge(prev, on=keys, suffixes=('_isolated', '_shared'), validate='one_to_one')
    probe['ce_gain'] = probe.task_ce_shared-probe.task_ce_isolated
    probe['delta_ba'] = probe.fusion_ba_isolated-probe.fusion_ba_shared
    probe.to_csv(OUT/'matched_fitting_probe.csv', index=False)
    pg = probe.groupby(['step', 'budget'])[['ce_gain', 'delta_ba']].mean().reset_index()
    pg.to_csv(OUT/'fitting_probe_summary.csv', index=False)
    costs = pd.read_csv(OUT/'training_cost.csv')
    names = dict(raw='无修正', initial='原修正器', simple='源选简单对照', reconstruction='继续重建训练',
                 task='共享任务训练', isolated='按预算分别训练')
    text = ['# 按支持预算拆分训练：结果', '', '日期：2026-10-12。', '',
        '18个新修正器均已完成，三个预算各30步，每个几何版本/种子合计90步。未重训骨干，未搜索超参数。',
        '评价完成范围：'+', '.join(families)+'。目标结果仅在有源端任务信号后以冻结模型评估。', '',
        '## 完整模型OA与BA', '', '| 范围 | 支持标签数 | 方法 | OA (%) | BA (%) |', '|---|---:|---|---:|---:|']
    for family in families:
        for b in old.BUDGETS:
            for arm in ARMS:
                r = summary[(summary.family == family) & (summary.budget == b) &
                            (summary.layer == 'fusion') & (summary.arm == arm)].iloc[0]
                text.append(f'| {family} | {b} | {names[arm]} | {r.oa:.4f} | {r.ba:.4f} |')
    text += ['', '## 拆分新增收益与概率风险', '',
        '| 范围 | 支持标签数 | 对照 | BA变化 (百分点) | OA变化 (百分点) | 类均衡NLL收益 | 类均衡Brier收益 |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in paired[(paired.layer == 'fusion') & paired.reference.isin(['task', 'initial', 'simple'])].itertuples():
        text.append(f'| {r.family} | {r.budget} | {names[r.reference]} | {r.delta_ba:+.4f} | {r.delta_oa:+.4f} | {r.class_nll_gain:+.6f} | {r.class_brier_gain:+.6f} |')
    text += ['', '风险收益为对照风险减新风险，正值表示改善。样本均值与港类均衡风险另存，不与类均衡口径混用。', '',
        '## 两个训练抽样种子', '', '| 范围 | 种子 | 支持标签数 | 相对共享训练BA变化 | 相对原修正器BA变化 |',
        '|---|---:|---:|---:|---:|']
    for family in families:
        for seed in old.SEEDS:
            for b in old.BUDGETS:
                f = seeds[(seeds.family == family) & (seeds.seed == seed) & (seeds.budget == b) & (seeds.layer == 'fusion')]
                a, i = f[f.reference == 'task'].iloc[0], f[f.reference == 'initial'].iloc[0]
                text.append(f'| {family} | {seed} | {b} | {a.delta_ba:+.4f} | {i.delta_ba:+.4f} |')
    text += ['', '## 同曝光量的拟合probe', '',
        '拆分的第10/20/30步对应共享训练第30/60/90步，各预算训练案例曝光相同。probe只在源拟合船上，不是独立验证。', '',
        '| 每预算更新次数 | 支持标签数 | 拆分相对共享CE收益 | 拆分相对共享BA变化 |', '|---|---:|---:|---:|']
    for r in pg.itertuples():
        text.append(f'| {r.step} | {r.budget} | {r.ce_gain:+.6f} | {r.delta_ba:+.4f} |')
    text += ['', '## 解释范围与成本', '',
        '- 两组所有训练观测ID和预算内顺序一致；原五组baseline的OA/BA/NLL/Brier重放一致，全部新配对满足救回减误伤等于BA变化，阈值1e-10。',
        '- 本轮干预是取消不同预算间的参数与Adam矩估计共享，不是仅改变梯度范数。拆分还改变了每个修正器的总更新数及优化器历史，因此不能唯一归因于梯度冲突。',
        '- 三个预算仍分别使用既有dose1/.5/.25和lambda8B/64；梯度差异也包含这些已有算子的影响，不是单独由样本数造成。',
        f'- 18个新修正器训练循环合计{costs.seconds.sum():.2f}秒，含固定拟合probe，不含缓存准备、诊断重放与留出/目标预测。每个332352参数，模型库存三份，但已知预算只用一份，未新增查询骨干。',
        '- 源54个预测单元复用418条查询/156条船，三个geometry context不等于三个独立源港；目标若评估为LA/Klang/Santos三历史港、两个训练抽样种子和五组支持。不是完整23港，不是独立确认。',
        '- 同时保留专家层、完整融合、逐港逐类、支持抽样胜平负；不要求每个单元获胜，不把小退化自动叫随机噪声。',
        '- 单步梯度/Adam诊断见diagnostics目录；它定位局部更新方向，不替代正式留出效果，也不提供物理因果分离证明。', '',
        '![拆分训练的预算收益](budget_isolation.png)', '']
    (OUT/'RESULTS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    fig, axes = plt.subplots(1, len(families), figsize=(6*len(families), 4.4), squeeze=False, constrained_layout=True)
    for ax, family in zip(axes.flat, families):
        x = np.arange(len(old.BUDGETS))
        for i, (ref, label, color) in enumerate([('task', 'vs shared training', '#238a8d'),
                                               ('initial', 'vs initial corrector', '#be6176'),
                                               ('simple', 'vs simple control', '#666666')]):
            f = paired[(paired.family == family) & (paired.layer == 'fusion') & (paired.reference == ref)].sort_values('budget')
            ax.bar(x+(i-1)*.24, f.delta_ba, .24, label=label, color=color)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(x, labels=[str(b) for b in old.BUDGETS])
        ax.set_xlabel('Labeled support budget')
        ax.set_ylabel('Full-system BA change (percentage points)')
        ax.set_title('Source-held vessels' if family == 'source' else 'Historical target ports')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.16)
    fig.savefig(OUT/'budget_isolation.png', dpi=170)
    plt.close(fig)
    print('FIXED FITTING PROBE', pg.to_string(index=False), flush=True)
    print(OUT/'RESULTS_ZH.md', flush=True)


if __name__ == '__main__':
    main()
