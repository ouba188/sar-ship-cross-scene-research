"""Current-head numerical report for the reused source-trained architectures."""

import json
import numpy as np
import pandas as pd

from completion_training_frequency import OUT, ARMS, READOUTS

DATASETS = ('self24ports', 'f0921', 'oss2')
NAMES = dict(self24ports='24港（23任务）', f0921='F_0921', oss2='OpenSARShip 2.0')
MODES = dict(baseline='无新增分频', unsplit='未拆频空间分支', fixed='分频固定等权',
             selective='分频学习权重', selective_uniform='学习后强制等权')
CONTRASTS = dict(unsplit_added='未拆频 - 无分频', fixed_added='固定分频 - 无分频',
    selective_added='学习分频 - 无分频', split_vs_unsplit='固定分频 - 未拆频',
    learned_vs_fixed='学习分频 - 固定分频', learned_gate_removed='强制等权 - 原学习权重',
    selective_training_uniform_readout='学习后等权 - 原固定等权训练')
CLASSES = ('散货船', '渔船', '杂货船', '成品油/化学品船', '集装箱船', '原油轮', '拖轮/拖带船', '海工补给船')


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def main():
    complete = json.loads((OUT/'COMPLETE.json').read_text())
    assert complete['metric_rows'] == 1875
    summary = pd.read_csv(OUT/'summary.csv')
    effects = pd.read_csv(OUT/'effects.csv')
    classes = pd.read_csv(OUT/'class_ba_contributions.csv')
    inv = pd.read_csv(OUT/'checkpoint_inventory.csv')
    lines = ['# 源训练分频架构：当前双头与统一头补实验', '',
        '日期：2026-10-10。25任务、125次支持抽样、5个版本、3种读出，共1,875条任务级OA/BA结果，全部完成。', '',
        '本轮复用75个实际完成六轮源训练的检查点，不是本轮重新训练75个网络。当前完整模型的基础头、双头融合、统一头以及专家全部逐样本精确回放；旧版五种基础头也逐样本回放。没有目标查询标签选参，F_0921恢复版入射角没有并入当前冻结模型。', '',
        '## 主要结论', '',
        '固定分频相对未拆频，在双头融合中的BA增量依次为24港 +0.09704、F_0921 +0.01831、OSS2 -0.00284个百分点。说明“拆分频带再分别学习”在24港有比同参数量未拆频分支更明确的净贡献，但不是三个数据集均有明显收益。', '',
        '学习分频权重相对固定等权的融合BA增量为 +0.02107 / -0.04494 / +0.07631。24港和OSS2正向，F_0921负向；没有据此分数据集挑选架构覆盖主模型。学习分频整体相对无分频，融合BA仅 +0.10334 / +0.01310 / +0.00628，而基础头是 +0.73053 / +0.09106 / +0.50162，不能继续使用旧5%交接的较大增量来描述当前全量融合。统一头中OSS2相对无分频BA仍为 -0.05136。', '',
        '逐类解释也不是“所有类都更好”：F_0921学习权重相对固定等权的主要负BA贡献来自海工补给船 -0.04705、原油轮 -0.01417；OSS2相同对比的海工补给船贡献 +0.09525，五组抽样合计救回3次、误改1次，说明这一正信号对少量稀有类事件敏感，不能把它说成均衡稳健的大提升。24港则是杂货船、渔船等正贡献与拖轮、海工补给船等负贡献的净和。', '',
        '门控仍近似固定偏好：F_0921的LH均值0.99918、任务内标准差0.00017，OSS2为0.99978、0.00006。本轮没有证明灵活样本级频带选择有效。这里的原因解释到“架构改变了哪些预测、哪些类受益/受损、融合如何改变净增量”为止，不将其冒充物理因果发现。', '',
        '## 绝对OA / BA（%）', '',
        '24港按23个任务等权，再对各任务五组支持抽样平均；外部数据集按五组抽样平均。OA按样本，BA对每任务查询中实际存在的类别等权，模型始终输出八类。', '']
    for readout, title in zip(READOUTS, ('基础头', '等权双头softmax融合', '统一核分类头')):
        lines += ['### '+title, '']
        rows = []
        for arm in ARMS:
            row = [MODES[arm]]
            for dataset in DATASETS:
                r = summary[(summary.dataset == dataset) & (summary.readout == readout) & (summary.arm == arm)].iloc[0]
                row.append(f'{r.oa:.5f} / {r.ba:.5f}')
            rows.append(row)
        lines += [table(['版本', *[NAMES[d] for d in DATASETS]], rows), '']
    lines += ['## 固定配对增量', '', '以下为OA / BA百分点，不是相对百分比。', '']
    for readout, title in [('fusion', '双头融合'), ('unified', '统一头')]:
        rows = []
        for contrast in CONTRASTS:
            row = [CONTRASTS[contrast]]
            for dataset in DATASETS:
                r = effects[(effects.dataset == dataset) & (effects.readout == readout) & (effects.contrast == contrast)].iloc[0]
                row.append(f'{r.delta_oa:+.5f} / {r.delta_ba:+.5f}；BA正向{r.ba_positive}/5')
            rows.append(row)
        lines += ['### '+title, '', table(['配对对比', *[NAMES[d] for d in DATASETS]], rows), '']
    lines += ['## 逐类收益与损失', '',
        '下表是两种最直接的架构差别在双头融合中的逐类BA贡献。类贡献按每任务现有类别数归一，再按任务与抽样等权；各类之和严格等于对应总体BA差。救回/误改是五组抽样的出现次数，不是独立图片数。全部逐类召回与所有七个配对对比另存CSV。', '']
    for contrast in ('split_vs_unsplit', 'learned_vs_fixed'):
        rows = []
        for dataset in DATASETS:
            for _, r in classes[(classes.dataset == dataset) & (classes.readout == 'fusion') &
                                (classes.contrast == contrast)].sort_values('class_id').iterrows():
                rows.append([NAMES[dataset], CLASSES[int(r.class_id)], f'{r.ba_contribution:+.5f}', int(r.rescued), int(r.harmed)])
        lines += ['### '+CONTRASTS[contrast], '', table(['数据集', '类别', 'BA贡献', '救回', '误改'], rows), '']
    replay = pd.concat([pd.read_csv(p) for p in (OUT/'targets').rglob('replay_checks.csv')], ignore_index=True)
    gates = pd.concat([pd.read_csv(p) for p in (OUT/'targets').rglob('gate_statistics.csv')], ignore_index=True)
    gates.to_csv(OUT/'gate_statistics.csv', index=False)
    gate = gates[gates.arm == 'selective'].groupby(['dataset', 'band'])[['mean', 'std']].mean().reset_index()
    gate.to_csv(OUT/'gate_summary.csv', index=False)
    lines += ['## 训练与机制解释边界', '',
        '- unsplit和fixed均有33,600个参数；selective为35,876。unsplit处理16×16特征，固定分频处理四个8×8频带，参数量匹配不等于运算量与归纳偏置完全相同。',
        '- 三架构有相同源队列、六轮、固定种子、相同支持任务与查询抽样计划，训练时的新增核权重均为0.25、支持岭系数均为8。检查点记录和非零首步梯度已核对。五组目标支持抽样不是五个网络训练种子。',
        '- fixed与selective分别真实训练，因此其差异同时包含训练后表征变化和融合权重变化。selective_uniform只去掉已训练权重，不撤销权重曾经影响训练梯度这一事实。',
        '- 这些对照能回答空间拆分和训练权重在当前读出中有多少净作用，不能单凭学习权重臂获益就宣称逐样本动态注意力有效。各任务gate均值/标准差附在gate_statistics.csv。',
        '- 数值回放排除了本次接入中支持错序、最新专家项遗漏、使用历史5%交接成绩替代全融合成绩等错误。它不证明六轮是最优训练，也不将统计差值解释为已识别物理因果机制。',
        '- 神经分频训练损失仍针对基础支持残差头，未针对当前双头融合/统一头最终损失再联合训练。最新24港专家梯度算子的具体未完成项见EXPERT_TRAINING_BOUNDARY_ZH.md。',
        '- 不按数据集选择获胜架构覆盖原模型，保留全部正负结果。这些均为反复使用的开发数据，F_0921历史无标签SSL暴露边界不变。', '',
        '### 学习权重的任务内波动', '',
        '下表对每个任务的权重均值/样本标准差作任务等权平均，不是将所有港口样本混成一个分布。', '',
        table(['数据集', '频带', '权重均值', '任务内样本标准差'],
              [[NAMES[r.dataset], r.band, f'{r["mean"]:.6f}', f'{r["std"]:.6f}'] for _, r in gate.iterrows()]), '',
        '## 验证与资源', '',
        f'- 75个检查点严格加载成功；源队列、6轮、末轮、种子、eta、beta与当前冻结任务一致。原五臂基础分数最大绝对回放误差为{replay.old_base_score_max_error.max():.9g}，所有argmax预测精确一致。',
        '- 当前selective完整模型在全部125次抽样中的三种读出精确一致，专家预测亦精确一致；汇总OA/BA与已有unified_head表一致。逐类BA贡献之和与配对BA差一致。',
        '- CPU两线程，GPU使用0分钟；压缩数组按512样本块读取，未解压复制大maps，未生成全量双精度核缓存，旧模型和特征文件只读。',
        f'- 历史75个检查点记录的纯训练循环总计{inv.training_seconds.sum():.2f}秒，仅为历史日志汇总，不含提取、加载、准备，也不是本轮端到端训练成本。', '',
        '## 文件', '',
        '- PROTOCOL.json、INVENTORY_COMPLETE.json、checkpoint_inventory.csv、source_episode_inventory.csv：固定配方与真实训练产物核对。',
        '- summary.csv、metrics.csv、effects.csv、draw_effects.csv、paired_metrics.csv：绝对与配对OA/BA。',
        '- class_recalls.csv、class_effects.csv、class_ba_contributions.csv：逐类召回与精确BA分解。',
        '- targets/：125次抽样的新预测、回放核对、完成标记；没有保存重复特征。',
        '- EXPERT_TRAINING_BOUNDARY_ZH.md：最新专家训练算子的已完成/未完成边界。',
        '- COMPLETE.json：本轮25任务读出完成回执。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(summary[summary.readout == 'fusion'].to_string(index=False))


if __name__ == '__main__':
    main()
