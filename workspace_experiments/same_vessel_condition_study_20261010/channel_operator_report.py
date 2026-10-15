"""Read the completed operator pilot without picking a target-best variant."""

import json
import numpy as np
import pandas as pd
from channel_operator import OUT, PORTS, BASE


def table(frame, columns):
    rows = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append('| '+' | '.join(f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(rows)


def run():
    completion = json.loads((OUT/'COMPLETE.json').read_text())
    result = pd.read_csv(OUT/'results.csv')
    base = result[result.arm.eq('baseline')].set_index('port')
    for name in ('OA', 'BA'):
        result['delta_'+name] = result[name]-result.port.map(base[name])
    result.to_csv(OUT/'paired_results.csv', index=False)
    means = result.groupby('arm', sort=False)[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
    means.to_csv(OUT/'port_macro_summary.csv', index=False)
    classes = pd.read_csv(OUT/'by_class.csv')
    clsbase = classes[classes.arm.eq('baseline')][['port', 'class_id', 'recall']].rename(columns={'recall': 'baseline_recall'})
    classes = classes.merge(clsbase, on=['port', 'class_id'])
    classes['delta_recall'] = classes.recall-classes.baseline_recall
    classes['net_correct'] = classes.rescue-classes.harm
    classes.to_csv(OUT/'class_attribution.csv', index=False)
    sensitivity = pd.read_csv(OUT/'sensitivity.csv')
    for name in ('source_scene_score_energy', 'source_centroid_score_energy'):
        sensitivity[name+'_ratio'] = sensitivity[name]/sensitivity.loc[sensitivity.arm.eq('baseline'), name].iloc[0]
    sensitivity.to_csv(OUT/'relative_sensitivity.csv', index=False)
    rank = pd.read_csv(OUT/'internal_channel_ranking.csv')
    comparisons = []
    for arm in completion['arms'][1:]:
        with np.load(OUT/f'{arm}.npz') as saved:
            chosen = saved['selected_channels']
            gate = saved['gate']
        comparisons.append(dict(arm=arm, channels=len(chosen), gate_mean=float(gate[chosen].mean()),
            gate_min=float(gate[chosen].min()), at_floor=int((gate[chosen] <= .500001).sum()),
            at_ceiling=int((gate[chosen] >= .999999).sum()),
            mean_scene_component=float(rank.iloc[chosen].shared_scene_component.mean()),
            mean_class_variance=float(rank.iloc[chosen].class_centroid_variance.mean())))
    gates = pd.DataFrame(comparisons)
    gates.to_csv(OUT/'gate_accounting.csv', index=False)
    obs = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'split.npz') as split:
        tm = obs.iloc[split['test_obs_ids']].reset_index(drop=True)
    with np.load(OUT/'baseline.npz') as b, np.load(OUT/'targeted_learned.npz') as t:
        before, after = b['predictions'], t['predictions']
    changed = tm[['instance_id', 'port', 'product', 'mmsi', 'class_id']].copy()
    changed['baseline_prediction'], changed['gated_prediction'] = before, after
    changed['rescue'] = (before != tm.class_id.to_numpy()) & (after == tm.class_id.to_numpy())
    changed['harm'] = (before == tm.class_id.to_numpy()) & (after != tm.class_id.to_numpy())
    changed[before != after].to_csv(OUT/'targeted_changed_cases.csv', index=False)
    learned = means[means.arm.eq('targeted_learned')].iloc[0]
    text = ['# 内部通道软抑制：首轮算子级实验', '',
        '## 结论先行', '',
        f'真实内部门控已实现并完成试验。定向可学习门控的三港等权 OA 改变 {learned.delta_OA:+.3f} 个百分点，BA 改变 {learned.delta_BA:+.3f} 个百分点。优于本轮三个随机通道对照的三港平均指标，但尚未超过未门控基线的平均 BA，不能作为成熟提点模块。', '',
        '根因测量表明：选中通道的抑制确实降低了内部共同场景变化；分类头重新拟合抵消了大部分分数层降幅。补上固定场景分数约束后，平均 BA 仍未改善。因此，分类头补偿是一个已定位的中间环节，但不是退化的全部解释；当前通道统计尚未区分有害依赖和有用判别成分。', '',
        'Santos 的得失尤其明确：散货船净增9个正确样本，但成品油/化学品船净减4个、原油船减1个、拖轮减2个、海工补给船减1个。不能用多数类收益掩盖其他类的损失。', '',
        '本轮不是末端特征减法。门控位于 SAR R18 最后一个残差块之前，抑制后的响应继续经过真实卷积、BN、ReLU 和残差相加。骨干权重固定，门控与源域分类头交替优化。', '',
        '## 范围', '',
        f'- 训练 {completion["train"]} 个源港样本，三港共同留出，评估 {completion["test"]} 个样本；不是三个独立 LOPO，也不是主模型的64标签实验。',
        '- 源域排除三个目标港、目标船舶身份及场景；目标标签只在训练完成后计算指标。历史神经预训练曝光仍然存在。',
        '- 512个内部通道中选64个。选择依据为源港多船型共同场景变化与类别均值差异的相对关系，并不把它们认定为纯噪声神经元。',
        '- 三次随机对照只是随机通道集合，不是三个独立神经网络训练种子；固定三轮，不按目标结果选择版本。', '',
        '## 三港等权结果', '', table(means, ['arm', 'OA', 'BA', 'delta_OA', 'delta_BA']), '',
        'OA、BA以百分数计，delta以百分点计。三港等权均值不是把所有查询合并的OA。', '',
        '## 逐港结果', '', table(result, ['arm', 'port', 'n', 'OA', 'BA', 'delta_BA', 'rescue', 'harm']), '',
        '## 抑制是否真正发生', '', table(gates, ['arm', 'channels', 'gate_mean', 'gate_min', 'at_floor', 'at_ceiling', 'mean_scene_component', 'mean_class_variance']), '',
        table(sensitivity, ['arm', 'source_scene_score_energy_ratio', 'source_centroid_score_energy_ratio', 'source_BA', 'source_score_relative_change']), '',
        '场景能量为源域不同类别场景残差平均后的分数能量，不是已识别的纯噪声方差；类别均值能量也不等于全部船型语义。两个指标下降或保留不能单独证明跨港泛化。', '',
        '## 定向可学习版本逐类得失', '',
        table(classes[classes.arm.eq('targeted_learned')], ['port', 'class_id', 'n', 'baseline_recall', 'recall', 'delta_recall', 'net_correct']), '',
        '## 可解释性与下一步判断', '',
        '- pooled_channel_trace.csv 给出先前候选方向回溯到原始池化通道的线性载荷；internal_channel_ranking.csv 对应本轮真实内部层，二者层位不同，不混为同一神经元。',
        '- 先确认 REPLAY.json 的原始输出复现与训练梯度；再判断定向是否优于相同训练的随机抑制，不能只与不改动版本比较。',
        '- 如果门控改变而原有场景分数能量没有下降，说明通道选择与最终决策敏感性之间仍存在缺口；不能归因为门控没有接入。',
        '- 如果随机抑制也改善，当前增益可能含普通正则化效应；若定向不能超过随机，不把本轮包装成专门消除港口噪声。',
        '- 仅做最后一个残差块的门控与分类头协同，尚未微调骨干卷积，也没有证明其他层位、条件门控或更大样本训练无效。', '',
        '所有分组得失保存在 class_attribution.csv 和 targeted_changed_cases.csv。不按目标港逐一挑选获胜方法。', '']
    if (OUT/'stage_attribution.csv').exists():
        stage = pd.read_csv(OUT/'stage_attribution.csv')
        text += ['## 分层根因定位', '', table(stage, ['arm', 'input_shared_energy_ratio',
            'output_shared_energy_ratio', 'original_head_scene_energy_ratio',
            'refitted_head_scene_energy_ratio']), '',
            '所有比值均相对未门控版本，指标尺度固定。输入与输出表示只计算跨类别共同协变的正对角部分。', '',
            '同一套门控特征在原分类头与重新拟合分类头下的对比，直接定位了头部重新加权对场景分数响应的影响。它不说明分类头有意学习港口，也不把这些协变分量认定为纯噪声。', '']
    coordination = OUT/'coordination'
    if (coordination/'COMPLETE.json').exists():
        cr = pd.read_csv(coordination/'results.csv')
        cb = cr[cr.arm.eq('baseline__ordinary')].set_index('port')
        for metric in ('OA', 'BA'):
            cr['delta_'+metric] = cr[metric]-cr.port.map(cb[metric])
        cr.to_csv(coordination/'paired_results.csv', index=False)
        cm = cr.groupby('arm', sort=False)[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
        cm.to_csv(coordination/'port_macro_summary.csv', index=False)
        text += ['## 针对性协调对照', '',
            '发现分类头重新加权抵消抑制后，追加固定两因素对照：有/无内部门控 × 普通/场景约束分类头。门控沿用原权重，不重新训练；约束沿用前轮 rank=8、tau=0.1，不搜索目标港参数。', '',
            table(cm, ['arm', 'OA', 'BA', 'delta_OA', 'delta_BA']), '',
            table(cr, ['arm', 'port', 'OA', 'BA', 'delta_BA', 'rescue', 'harm']), '',
            table(pd.read_csv(coordination/'sensitivity.csv'), ['arm', 'source_scene_energy', 'source_BA']), '',
            '这是针对已观察问题的研发纠错，不是独立确认。必须与仅约束分类头比较，不能把头部自身收益归给门控。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    print(means.to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == '__main__':
    run()
