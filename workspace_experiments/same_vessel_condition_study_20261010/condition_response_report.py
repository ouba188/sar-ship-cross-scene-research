"""Report the shared condition-response experiment without target winner selection."""

import json
import numpy as np
import pandas as pd
from condition_response import OUT, BASE, PARENT
from channel_operator_report import table


def run():
    completed = json.loads((OUT/'COMPLETE.json').read_text())
    results = pd.read_csv(OUT/'results.csv')
    baseline = results[results.arm.eq('baseline__refit')].set_index('port')
    for metric in ('OA', 'BA'):
        results['delta_'+metric] = results[metric]-results.port.map(baseline[metric])
    results.to_csv(OUT/'paired_results.csv', index=False)
    means = results.groupby('arm', sort=False)[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
    means.to_csv(OUT/'port_macro_summary.csv', index=False)
    cv = pd.read_csv(OUT/'source_generalization.csv')
    fit = json.loads((OUT/'FIT.json').read_text())
    accounts = pd.read_csv(OUT/'gain_accounting.csv')
    primary = means[means.arm.eq('condition_gain__refit')].iloc[0]
    text = ['# 保留通道的观测条件响应校正', '', '## 结论先行', '',
        f'主检验版本为 condition_gain__refit。三港等权平均 OA 改变 {primary.delta_OA:+.3f} 个百分点，BA 改变 {primary.delta_BA:+.3f} 个百分点。不能依据表中目标港结果逐港挑选不同版本。', '',
        '本轮未获得正增益，不并入主方案。不是实现没有接入：原始预测复现、条件缺失回退和空间排列保持均通过检查。核心限制是当前同船拟合得到的统一条件响应关系跨港迁移弱；重拟合分类头与源域强度校准均未使其超过基线。这里仅否定本轮固定条件函数，不否定场景因素存在，也不要求改用船型专用规则。', '',
        '本轮没有按船型、类别混淆或目标港设置规则。候选输入为入射角、升降轨、卫星平台；同一套系数用于全部船型与港口。平台在同船同港重复组内没有变化，无法单独识别平台效应，其系数实际为零。有效校正来自入射角及升降轨项。船型标签仅用于训练识别头和评价。', '',
        '## 方法', '',
        f'从排除三个目标港及其船舶身份后的源域中，使用 {fit["observations"]} 次重复观测、{fit["vessels"]} 艘船、{fit["ports"]} 个港。每个同船同港组内去均值，拟合观测条件差异与通道对数平均响应差异的回归。组内固定的船舶结构差异不进入回归，但随时间变化的装载、背景等未测因素仍可能混杂，不能直接解释为因果效应。', '',
        '公式：log(mu_ic) = vessel_port_offset_c + phi(condition_i) B_c + residual_ic。', '',
        '推理：gain_ic = clip(exp(-0.5 * (phi(condition_i)-source_reference) B_c), 0.5, 2)。', '',
        'h_prime_icuv = gain_ic * h_icuv。所有通道保留，既可减弱也可增强，通道内相对空间排列不变。它校正的是条件相关幅度，不是从每个神经元中分离出了纯噪声。缺失条件时 gain=1。随后继续经过原始非线性残差块。', '',
        '## 对照与范围', '',
        '- baseline：不处理；固定头与重新拟合头都精确复现前轮原始查询预测。',
        '- sample_mean_shrink：不区分条件来源，统一把样本通道均值往源域均值收缩；同样只处理条件已知样本。',
        '- condition_gain：本轮条件相关响应校正，统一固定强度0.5。',
        '- permuted_channel_gain：固定一次通道对应关系置乱；每个样本的校正幅度集合与正确版本完全相同，检查对应关系是否重要。不是新增跨域数据集置乱任务。',
        '- 每个表示同时比较原分类头和源域重新拟合分类头，不增加目标标注或目标条件统计。',
        '- 分类头沿用相同939个源域标签；响应估计额外利用重复船舶身份和观测元数据。这是新增信息，不能称为完全同信息量的架构比较。',
        '- 三港共同留出；历史预训练曝光保留。不是完整主模型的64标签结果或独立确认集。', '',
        '## 源域关系是否可迁移', '',
        table(cv, ['split', 'fold', 'train', 'test', 'prediction_R2',
                   'half_correction_variation_reduction', 'positive_channels']), '',
        'prediction_R2 相对“不预测任何同船变化”；variation_reduction 测量固定半强度校正后同船变化的减少。正负结果均保留。船舶留出与港口留出回答不同问题，不能用前者代替后者。R2不是分类准确率，也不是已解释的全部物理噪声比例。', '',
        '## 三港等权指标', '',
        table(means, ['arm', 'OA', 'BA', 'delta_OA', 'delta_BA']), '',
        '## 逐港结果', '',
        table(results, ['arm', 'port', 'n', 'OA', 'BA', 'delta_OA', 'delta_BA', 'rescue', 'harm']), '',
        '## 校正覆盖与强度', '',
        table(accounts[accounts.variant.eq('condition_gain')], ['port', 'n', 'available',
            'mean_abs_log_gain', 'gain_q01', 'gain_q99', 'capped_fraction']), '',
        '## 决策敏感性', '',
        table(pd.read_csv(OUT/'sensitivity.csv'), ['arm', 'source_scene_score_energy', 'source_BA']), '',
        '## 如何判读失败', '',
        '先看接入、预测复现和校正强度，再区分两种情况：一是观测条件与响应的关系本身在换港后失效；二是关系能迁移，但被校正的变化并非全是分类有害因素。若只在同港不同船上可预测，不应宣称已找到跨港不变量。', '',
        '原分类头与重拟合头的差异用于定位训练协调问题；样本均值收缩和通道错配对照用于判断收益是否来自普遍的幅度扰动。逐类结果保存在 by_class.csv，仅用于诊断，不用于制定船型专用规则。', '',
        f'总耗时 {completed["seconds"]:.1f} 秒，CPU单线程，未使用或中断主线程GPU任务。', '']
    with np.load(PARENT/'split.npz') as split:
        test = split['test_obs_ids']
        n = len(split['train_obs_ids'])
    obs = pd.read_csv(BASE/'observations.csv').iloc[test].reset_index(drop=True)
    with np.load(OUT/'gains.npz') as saved:
        available = saved['available'][n:]
    with np.load(OUT/'baseline__fixed.npz') as saved:
        bp = saved['predictions']
    coverage = []
    for arm in ('condition_gain__fixed', 'condition_gain__refit'):
        with np.load(OUT/f'{arm}.npz') as saved:
            pred = saved['predictions']
        for known in (False, True):
            q = available == known
            y = obs.class_id.to_numpy()[q]
            before, after = bp[q] == y, pred[q] == y
            coverage.append(dict(arm=arm, metadata_known=known, n=int(q.sum()),
                predictions_changed=int((bp[q] != pred[q]).sum()),
                rescue=int((~before & after).sum()), harm=int((before & ~after).sum())))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT/'availability_diagnosis.csv', index=False)
    unknown = coverage[coverage.arm.eq('condition_gain__fixed') & ~coverage.metadata_known]
    if unknown.predictions_changed.sum():
        raise ValueError('Missing-condition fixed-head predictions changed unexpectedly')
    text += ['## 条件缺失与训练协调', '',
        table(coverage, ['arm', 'metadata_known', 'n', 'predictions_changed', 'rescue', 'harm']), '',
        '固定原分类头时，缺失条件的样本预测完全不变。重新拟合头后，这些样本也可能变化，这是共享分类头更新的影响，不是给缺失条件填了虚构校正值。', '']
    calibration = OUT/'source_calibration'
    if (calibration/'COMPLETE.json').exists():
        ci = json.loads((calibration/'CALIBRATION.json').read_text())
        cr = pd.read_csv(calibration/'results.csv')
        for metric in ('OA', 'BA'):
            cr['delta_'+metric] = cr[metric]-cr.port.map(baseline[metric])
        cr.to_csv(calibration/'paired_results.csv', index=False)
        cm = cr.groupby('head', sort=False)[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
        cm.to_csv(calibration/'port_macro_summary.csv', index=False)
        text += ['## 源港校准与负结果定位', '',
            '原港口交叉验证三个验证折样本量为6088、1150、222。为检查是否仅由划分失衡造成，追加按港口样本量贪心平衡的三折；原结果完整保留。没有用目标港结果选择划分。', '',
            table(pd.read_csv(calibration/'balanced_port_cv.csv'), ['fold', 'train', 'test', 'prediction_R2', 'half_reduction']), '',
            table(pd.read_csv(calibration/'balanced_port_breakdown.csv'), ['port', 'n', 'vessels', 'prediction_R2']), '',
            f'平衡后源港OOF整体 R2={ci["source_port_OOF_prediction_R2"]:.5f}。用这些OOF预测拟合一个全船型、全港共享的响应强度系数 k={ci["scalar"]:.5f}，不做参数搜索，不按类别、通道或目标港设置不同系数。此校准使用了OOF数据，校准后的OOF拟合误差不作为独立验证。', '',
            table(cm, ['head', 'OA', 'BA', 'delta_OA', 'delta_BA']), '',
            table(cr, ['head', 'port', 'OA', 'BA', 'delta_BA', 'rescue', 'harm']), '',
            '校准减轻损失，但未带来超过原模型的平均BA。跨港预测弱也出现在Busan、Fujairah、New York、Singapore等样本相对较多的源港，不能全部归于少数小港权重。平台组内方差与平台系数均为零，不把平台列的存在写成已学得平台校正机制。', '',
            '可支持的结论是：当前全局入射角/轨道到通道幅度的固定回归式不足以可靠描述跨港响应变化。未测观测条件、船体朝向与场景交互、回归形式不足等仍是待区分解释；不能据此直接归因于某种具体物理噪声。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print(means.to_string(index=False))


if __name__ == '__main__':
    run()
