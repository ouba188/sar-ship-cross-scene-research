"""Report the matched-discrimination study without treating stability as causal."""

import json

import numpy as np
import pandas as pd

from run_matched_stability_probe import OUT


def main():
    data = OUT/'exact_w1' if (OUT/'exact_w1/COMPLETE.json').exists() else OUT
    complete = json.loads((data/'COMPLETE.json').read_text(encoding='utf-8'))
    source = pd.read_csv(OUT/'source_matching.csv')
    pp = pd.read_csv(data/'port_pair_metrics.csv')
    means = pd.read_csv(data/'equal_port_means.csv')
    joined = pp.merge(source, on=['port', 'class_c', 'class_d', 'arm'], suffixes=('', '_source'))
    joined['target_minus_source_ba'] = joined.fixed_threshold_ba-joined.source_ba
    extra = ['source_auc', 'source_ba', 'source_drift', 'target_minus_source_ba']
    source_means = joined.groupby(['subset', 'port', 'arm'])[extra].mean().groupby(['subset', 'arm']).mean().reset_index()
    summary = means.merge(source_means, on=['subset', 'arm'])
    summary.to_csv(data/'source_target_summary.csv', index=False, encoding='utf-8-sig')
    keys = ['port', 'class_c', 'class_d']
    a = source[source.arm == 'candidate'].set_index(keys)
    b = source[source.arm == 'matched_reflection'].set_index(keys)
    gap_error, var_error = (a.mean_gap-b.mean_gap).abs().max(), (a.total_variance-b.total_variance).abs().max()
    matched = summary[summary.subset == 'matched'].set_index('arm')
    reduction = 100*(1-matched.loc['candidate','conditional_w1']/matched.loc['matched_reflection','conditional_w1'])
    pm = pp[pp.subset == 'matched']
    ca = pm[pm.arm == 'candidate'].set_index(keys)
    cb = pm[pm.arm == 'matched_reflection'].set_index(keys)
    effects = pd.DataFrame(dict(delta_w1=ca.conditional_w1-cb.conditional_w1,
        delta_ba_pp=100*(ca.fixed_threshold_ba-cb.fixed_threshold_ba),
        delta_auc=ca.auc-cb.auc)).reset_index()
    effects.to_csv(data/'matched_pair_effects.csv', index=False, encoding='utf-8-sig')
    names = dict(pooled='普通判别方向', candidate='候选稳定方向', matched_reflection='判别强度匹配对照')
    lines = ['# 判别强度匹配后的跨港稳定性研究', '', '日期：2026-10-08。', '',
        '## 结论', '',
        f'控制源域类别均值间隔和总类内方差，并限制源域AUC/BA差异后，候选方向在目标港的平均分布漂移仍低{reduction:.2f}%。但是两类BA仅高0.1614个百分点，22港中11升11降；AUC略低。当前证据支持“候选更稳定”，不支持“稳定性已带来一致的分类增益”。',
        '这不能把此前屏蔽候选空间导致的准确率下降解释成稳定性的因果贡献。原联合模型不变，本轮没有新增模块。', '',
        '## 为什么做这个对照', '',
        '原先屏蔽的是整片候选空间，其中同时包含普通类别判别信息。删除后下降只能定位原模型依赖，不能区分判别强度与稳定性的作用。',
        '本轮将问题缩小为：在源港区分两类的能力接近时，更稳定的方向换港后是否仍更准确？它补充解释边界，不是重新给旧屏蔽降幅分配贡献比例。', '',
        '## 对照构造', '',
        '每个外层留港、每个类别对，使用[普通方向投影，候选方向投影]组成二维平面。源类均值差记为delta，总类内协方差记为Sigma，候选坐标w=[0,1]。',
        '令v=pinv(Sigma)*delta，a=(delta^T*w)/(delta^T*v)*v，构造对照w_control=2*a-w。',
        '这一反射保持delta^T*w以及w^T*Sigma*w不变，即源域均值间隔、总类内方差和Fisher分离度相同。不是随机乱标签，也没有改动目标数据。',
        '源域每类占一半权重，类内港口等权；类别-港口单元至少5条，每类至少3港。协方差包含同类不同港均值变化。',
        '矩匹配不保证整个分布相同，所以主要分析还要求：源域AUC差和固定源均值中点阈值BA差均不超过2个百分点。阈值在运行前写入协议，不依据目标结果调整。',
        '另预定义子集：上述匹配成立，且候选在源域的类内跨港均值漂移小于对照。反射对照不预先被称为“不稳定方向”，须由测量判断。',
        '全部源方向、阈值、匹配标记与目标投影冻结后，再读取查询标签。原64标签未参与这些投影或阈值拟合；原5组查询掩码仍排除各组支持样本。', '',
        '## 覆盖和检查', '',
        f'- 23个可评估港口，共{len(a)}个源港划分-类别对，{int(a.matched.sum())}个通过源AUC/BA匹配；退化对照{int(a.degenerate.sum())}个。',
        '- 目标每对每类至少20条：全部237个港口-类别对；主要匹配子集191对、22港；源稳定性对比子集142对、22港。',
        '- Sydney Botany在主要匹配子集中没有同时满足目标样本数与源匹配条件的类别对；不是因为目标表现差而排除。',
        f'- 独立重算投影后的源均值间隔最大匹配误差{gap_error:.3g}，方差最大误差{var_error:.3g}。12项单元测试通过，包含独立加权矩、奇异情况与投影数值等价性。',
        '- 先平均每港每对的有效抽样，再平均港内类别对，最后等港平均。抽样有重叠，不算5次独立重复。', '',
        '## 主要匹配子集结果', '',
        '| 方向 | 源AUC | 源两类BA (%) | 目标AUC | 目标两类BA (%) | 目标类条件漂移W1 |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ('pooled', 'matched_reflection', 'candidate'):
        m = matched.loc[arm]
        lines.append(f'| {names[arm]} | {m.source_auc:.6f} | {100*m.source_ba:.4f} | {m.auc:.6f} | {100*m.fixed_threshold_ba:.4f} | {m.conditional_w1:.6f} |')
    lines += ['', '本轮W1除以源域总类内标准差，匹配的两方向分母相同。前轮报告使用港内方差归一化，所以不能直接将两轮W1数值及相对降幅混用。',
        '源BA/AUC是参与构造的源数据描述，不是独立源验证成绩；2pp匹配也不意味着所有判别属性完全相同。', '',
        '## 一致性与失效分析', '',
        '| 分析范围 | 港口/类别对 | 候选相对匹配对照BA差 (pp) | W1差 | BA升/降港数 | W1下降港数 |',
        '|---|---:|---:|---:|---:|---:|']
    subset_names = dict(all='全部矩匹配', matched='再匹配源AUC/BA', source_stability_contrast='再要求源稳定性更好')
    for r in complete['contrasts']:
        lines.append(f'| {subset_names[r["subset"]]} | {r["ports"]}/{r["pair_ports"]} | {r["delta_ba_pp"]:+.4f} | {r["delta_w1"]:+.6f} | {r["ba_positive_ports"]}/{r["ba_negative_ports"]} | {r["lower_shift_ports"]} |')
    improved_shift = effects.delta_w1 < 0
    lines += ['',
        f'主要匹配191对中，{int(improved_shift.sum())}对W1更低，其中{int((improved_shift & (effects.delta_ba_pp <= 0)).sum())}对BA并未提高。分布漂移减小与判别改善没有一一对应。',
        '主要匹配子集中，候选源BA比对照高约0.1116个百分点，目标BA高0.1614个百分点；目标相对源的BA变化之差仅约0.0498个百分点。这是描述性对照，不是因果贡献分解。',
        '源稳定性明确更好的子集中，目标W1仍改善，目标BA却低0.1838个百分点。这直接限制了“抑制同类跨港变化就能提分类准确率”的解释。',
        '实现层面：对照构造和矩匹配通过独立计算；未发现所检查路径的实现偏离。目标层面：候选原方法惩罚同类跨港均值变化，没有直接约束目标分类边界的错误；W1统计整个分布，而阈值BA由边界两侧的类别质量决定。实际测量也出现更小漂移但更低AUC/BA。',
        '因此已证实的是稳定性指标与分类效用不充分一致；不能进一步认定尾部分布、个别物理因素或某一类别偏置就是唯一根因。更高阶分布、边界处密度和原模型读出依赖也未被严格匹配。', '',
        '## 对现有模型是否有增量价值', '',
        f'在同一主要匹配子集，原Base64只在真实类别对内部比较两列分数的BA为{100*matched.loc["candidate","base_pair_ba"]:.4f}%；候选源阈值为{100*matched.loc["candidate","fixed_threshold_ba"]:.4f}%。',
        f'候选相对这个两类基础判断的类别平衡纠错率为{100*matched.loc["candidate","rescue_rate"]:.4f}%，误改率为{100*matched.loc["candidate","harm_rate"]:.4f}%。它能纠正一些错误，但直接替代会改错更多。',
        '这是利用真实类别对定义的诊断，部署时不知道该类别对，不能报告为八分类模型或可执行门控的收益。Base64已用64标签适配，候选阈值未用目标标签；该比较只回答对现状的直接替代是否有净收益，不是同监督量的公平独立分类器比较。', '',
        '## 研究决策', '',
        '1. 保留“源域拟合的候选方向在留出港上漂移较小”的发现，暂不升级为因果不变量或稳定性驱动准确率提升。',
        '2. 不继续用整片候选空间屏蔽降幅作为稳定性贡献证据，也不将本轮两类结果包装成新模块提点。',
        '3. 若继续模块研发，应把约束从全局分布均值稳定转到任务相关的类别边界/残差纠错，先在源港任务验证可预测的净纠错，再冻结到目标港；这仍是下一步假设，不能保证有效。',
        '4. 本轮不新增目标标签、不改主模型、不扩展全数据集训练。历史研发语料上的外层留港对照不是全新独立确认。', '',
        '## 产物', '',
        'PROTOCOL.json、COMPLETE.json、source_matching.csv、source_target_summary.csv、draw_pair_metrics.csv、port_pair_metrics.csv、port_means.csv、matched_pair_effects.csv及各港frozen.npz。',
        '实现：matched_stability_control.py、test_matched_stability_control.py、run_matched_stability_probe.py、report_matched_stability_probe.py。']
    if data != OUT:
        lines[3:3] = ['## 漂移读数修正', '',
            '本报告使用精确加权经验Wasserstein距离，取代父目录初报的101分位点近似。原近似混用了源加权中点插值与目标np.quantile，即使经验分布相同也可能产生正距离；本轮审阅发现后修正。',
            '修正使用scipy.stats.wasserstein_distance，源按既定类内等港权重、目标按样本均匀权重，分母仍为冻结源总类内标准差。未重新拟合方向、阈值、匹配条件或模型，AUC/BA与原报完全一致。父目录保留原始近似输出，本目录为正式修正读数。',
            '经验距离精确不代表总体分布距离无估计误差，也不构成严格不变性检验。', '']
    (data/'MATCHED_STABILITY_FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(str(data/'MATCHED_STABILITY_FINDINGS_ZH.md'))


if __name__ == '__main__':
    main()
