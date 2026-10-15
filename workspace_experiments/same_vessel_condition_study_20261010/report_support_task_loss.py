"""Paired error-flow diagnosis without further training or model selection."""

from run_support_task_loss import OUT, BASE, SOURCE, PORTS, SEEDS, BUDGETS, cached
from evaluate_probability_layers import probability_rows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def diagnose():
    rows, cells = [], []
    for family in ('source', 'target'):
        path = BASE/'observations.csv' if family == 'source' else SOURCE/'manifest.csv'
        meta = pd.read_csv(path, usecols=['class_id', 'port'])
        paired = pd.read_csv(OUT/(family+'_evaluation')/'paired.csv')
        for context in PORTS:
            for seed in SEEDS:
                for draw in range(3 if family == 'source' else 5):
                    for budget in BUDGETS:
                        z = cached(OUT/family/context/str(seed)/f'{draw}_{budget}.npz')
                        qm = meta.iloc[z['query_ids']]
                        y = qm.class_id.to_numpy()
                        _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
                        w = 1/(len(counts)*counts[inv])
                        info = dict(family=family, context=context, seed=seed, draw=draw, budget=budget)
                        task = probability_rows(z['fusion_task'], y, 'probabilities')
                        for ref in ('raw', 'initial', 'reconstruction', 'simple'):
                            old = probability_rows(z['fusion_'+ref], y, 'probabilities')
                            rescue = ~old['correct'] & task['correct']
                            harm = old['correct'] & ~task['correct']
                            r, h = 100*w@rescue, 100*w@harm
                            original = paired[(paired.context == context) & (paired.seed == seed) &
                                (paired.draw == draw) & (paired.budget == budget) &
                                (paired.layer == 'fusion') & (paired.comparison == 'task_vs_'+ref)].iloc[0]
                            np.testing.assert_allclose(r-h, original.delta_ba, atol=1e-10, rtol=0)
                            rows.append(dict(**info, reference=ref, rescue_ba=r, harm_ba=h, delta_ba=r-h,
                                class_nll_gain=w@(old['nll']-task['nll']),
                                class_brier_gain=w@(old['brier']-task['brier']),
                                correct_probability_change=w@(z['fusion_task'][np.arange(len(y)), y]-
                                                               z['fusion_'+ref][np.arange(len(y)), y])))
                            frame = pd.DataFrame(dict(port=qm.port.to_numpy(), class_id=y,
                                recall_change=100*(task['correct'].astype(float)-old['correct']),
                                nll_gain=old['nll']-task['nll'], brier_gain=old['brier']-task['brier']))
                            for (port, c), group in frame.groupby(['port', 'class_id']):
                                cells.append(dict(**info, reference=ref, actual_port=port, class_id=c,
                                    n=len(group), delta_recall=group.recall_change.mean(),
                                    nll_gain=group.nll_gain.mean(), brier_gain=group.brier_gain.mean()))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'error_flow.csv', index=False)
    pd.DataFrame(cells).to_csv(OUT/'port_class_changes.csv', index=False)
    metrics = ['rescue_ba', 'harm_ba', 'delta_ba', 'class_nll_gain', 'class_brier_gain', 'correct_probability_change']
    summary = frame.groupby(['family', 'budget', 'reference'])[metrics].mean().reset_index()
    summary.to_csv(OUT/'error_flow_summary.csv', index=False)
    frame.groupby(['family', 'context', 'budget', 'reference'])[metrics].mean().reset_index().to_csv(
        OUT/'port_error_flow.csv', index=False)
    # Average the two training seeds before describing repeated support draws.
    draws = frame.groupby(['family', 'context', 'draw', 'budget', 'reference'])[metrics].mean().reset_index()
    draws.to_csv(OUT/'seed_averaged_draws.csv', index=False)
    return summary


def main():
    flows = diagnose()
    rows = []
    for family in ('source', 'target'):
        d = pd.read_csv(OUT/(family+'_evaluation')/'summary.csv')
        d = d[d.layer == 'fusion'].copy()
        d['family'] = family
        rows.append(d)
    summary = pd.concat(rows, ignore_index=True)
    summary.to_csv(OUT/'full_system_summary.csv', index=False)
    probe = pd.read_csv(OUT/'fitting_probe.csv').groupby(['arm', 'step', 'budget'])[
        ['feature_mse', 'task_ce', 'fusion_ba']].mean().reset_index()
    probe.to_csv(OUT/'fitting_probe_summary.csv', index=False)
    costs = pd.read_csv(OUT/'training_cost.csv')
    names = dict(raw='无修正', initial='原修正器', simple='源选简单对照',
                 reconstruction='继续重建训练', task='完整任务损失训练')
    text = ['# 支持修正器损失对齐：完成结果与边界', '',
        '日期：2026-10-12。12组仿射修正器训练、54组源留出和90组目标预测单元已完成；不重训骨干。', '',
        '## 结论先说', '',
        '认可明确任务范围内可重复、值得成本的净收益，不要求处处获胜。'
        '本轮保留支持修正的已有有效范围，但暂不把新任务损失升级为主方案：它学会降低拟合任务损失，'
        '却没有稳定超过原修正器及简单对照。不能把相对无修正模型的全部收益归功于改损失。', '',
        '## 完整模型结果', '', '| 评价范围 | 标签预算 | 方法 | OA (%) | BA (%) |',
        '|---|---:|---|---:|---:|']
    for family in ('source', 'target'):
        for b in BUDGETS:
            for arm in names:
                r = summary[(summary.family == family) & (summary.budget == b) & (summary.arm == arm)].iloc[0]
                text.append(f"| {'源留出' if family == 'source' else '三目标港均值'} | {b} | {names[arm]} | {r.oa:.4f} | {r.ba:.4f} |")
    text += ['', '## 新损失真正新增多少', '',
        '| 范围 | 标签预算 | 相对对照 | BA变化 (百分点) | 类均衡NLL收益 | 类均衡Brier收益 |',
        '|---|---:|---|---:|---:|---:|']
    for r in flows[flows.reference.isin(['initial', 'reconstruction', 'simple'])].itertuples():
        text.append(f"| {r.family} | {r.budget} | {names[r.reference]} | {r.delta_ba:+.4f} | {r.class_nll_gain:+.6f} | {r.class_brier_gain:+.6f} |")
    text += ['', '概率风险收益以旧风险减新风险计算，正值为改善；这里是类均衡口径，原评价目录同时保留样本均值与港口/类别均衡Brier，不能混用。', '',
        '## 学到了什么与为什么没全面兑现', '',
        '1. 实现层已核对：独立原始空间ridge、质心/能量约束、有限差分梯度、真实融合CE共4项测试通过。9个真实训练episode初始化分数与原NumPy求解最大差约1.73e-14，源/目标原修正器重放均通过1e-10阈值。没有发现断梯度或替错baseline的问题。',
        '2. 学习确实发生：下表是固定源拟合probe，不是独立验证。任务训练主动牺牲特征重建精度来改善分类概率目标，说明它不是仅仅模仿旧特征。', '',
        '| 训练目标 | 标签预算 | MSE开始→结束 | 类均衡任务CE开始→结束 |', '|---|---:|---|---|']
    for arm in ('reconstruction', 'task'):
        for b in BUDGETS:
            a = probe[(probe.arm == arm) & (probe.budget == b) & (probe.step == 0)].iloc[0]
            z = probe[(probe.arm == arm) & (probe.budget == b) & (probe.step == 90)].iloc[0]
            text.append(f'| {names[arm]} | {b} | {a.feature_mse:.6f} → {z.feature_mse:.6f} | {a.task_ce:.6f} → {z.task_ce:.6f} |')
    text += ['',
        '3. 功能层定位：只有支持端表示与重新求解后的支持系数改变，查询特征、源先验、基础分支和融合规则固定。收益或误伤通过案例关系/支持校正路径发生，不是新骨干或新物理知识的收益。',
        '4. 泛化边界：拟合CE下降不等于留出CE下降，也不等于top-1改善。源64标签对旧修正器的微弱正信号没有在目标64标签保持；这支持效用迁移受限，但还不能唯一判定是容量过拟合、源目标关系差异还是小样本抽样。没有以退化为理由无限调参。',
        '5. 抽样解释：两个训练episode种子与重复支持抽样分开报告；不能把同一查询船上的多个几何版本当作独立样本。局部负结果不是一票否决，但有重复同向退化时也不直接称为随机波动。', '',
        '## 目标端救回与误伤', '',
        '| 标签预算 | 对照 | 救回BA贡献 | 误伤BA贡献 | 净BA变化 |', '|---|---|---:|---:|---:|']
    for r in flows[(flows.family == 'target') & flows.reference.isin(['initial', 'simple'])].itertuples():
        text.append(f'| {r.budget} | {names[r.reference]} | {r.rescue_ba:.4f} | {r.harm_ba:.4f} | {r.delta_ba:+.4f} |')
    text += ['',
        '救回减误伤与每一个配对BA差逐项相等，阈值1e-10；明细见error_flow.csv和port_class_changes.csv。它解释决策变化的直接来源，不等于物理因果机制。', '',
        '## 成本与范围', '',
        f"- 每个修正器332352个参数，12个训练循环累计{costs.seconds.sum():.2f}秒；含固定拟合probe，不含缓存准备、离线预测与报告。不是端到端总耗时。",
        '- 两种训练损失的推理结构相同；修正支持向量并重求系数属于支持集适配阶段，查询仍用既有SAR特征，没有新查询骨干前向。未另测端到端延迟。',
        '- 1296条拟合观测相对旧初始化1040条的扩展同时用于两种训练目标；任务与重建对比匹配，但相对旧初始化的变化混有继续训练和数据扩展，不能单独归因于损失。',
        '- 源先验新拟合排除留出船，历史骨干/几何并非完全未见留出。三个目标港是Los Angeles、Port Klang、Santos，不是完整23港，也不是三个新数据集。目标支持/查询按历史行级排除，可能含同船不同观测。',
        '- 两种子控制训练episode抽样，不是两次骨干重训。历史研发数据、非独立确认；不能据此声称分离出了纯因果因素。', '',
        '## 本轮决策', '',
        '保留16标签源选简单衰减、64标签原修正器的各自已有证据，128标签仍以无修正为强对照；这只是研发范围总结，不事后用目标结果拼出新的已验证选择策略。'
        '新任务损失暂列局部信号，不覆盖主模型。下一步如继续训练，应先证明候选表示或修正方向在未见船上有可利用的增量，再决定是否增加模型复杂度；本轮不追加无依据的参数网格。', '',
        '![预算与训练目标对照](budget_results.png)', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    colors = dict(raw='#555555', initial='#007f86', simple='#ab4d00', reconstruction='#b34880', task='#377ac2')
    labels = dict(raw='Uncorrected', initial='Initial corrector', simple='Source-selected simple',
                  reconstruction='Reconstruction loss', task='Fusion task loss')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, family in zip(axes, ('source', 'target')):
        for arm in names:
            data = summary[(summary.family == family) & (summary.arm == arm)].sort_values('budget')
            ax.plot(data.budget, data.ba, marker='o', color=colors[arm], label=labels[arm], lw=1.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks(BUDGETS, labels=[str(x) for x in BUDGETS])
        ax.set_xlabel('Labeled support budget')
        ax.set_ylabel('Full-system BA (%)')
        ax.set_title('Source-held vessels' if family == 'source' else 'Three historical target ports')
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'budget_results.png', dpi=170)
    plt.close(fig)
    print(flows.to_string(index=False))
    print(OUT/'FINDINGS_ZH.md')


if __name__ == '__main__':
    main()
