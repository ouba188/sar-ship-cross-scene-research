"""Close the fixed source-initialization attempt without target-best arm selection."""

import json
import pandas as pd
from channel_operator_report import table
from source_episode_closeout import OUT


def run():
    done = json.loads((OUT/'COMPLETE.json').read_text())
    rows = pd.read_csv(OUT/'results.csv')
    ref = rows[rows.arm.eq('previous_repaired')][['port', 'draw', 'OA', 'BA']].rename(
        columns={'OA': 'ref_OA', 'BA': 'ref_BA'})
    rows = rows.merge(ref, on=['port', 'draw'])
    rows['delta_OA'] = rows.OA-rows.ref_OA
    rows['delta_BA'] = rows.BA-rows.ref_BA
    means = rows.groupby('arm')[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
    ports = rows.groupby(['port', 'arm'])[['OA', 'BA', 'delta_OA', 'delta_BA']].mean().reset_index()
    rows.to_csv(OUT/'paired_results.csv', index=False)
    means.to_csv(OUT/'summary.csv', index=False)
    ports.to_csv(OUT/'port_summary.csv', index=False)
    classes = pd.read_csv(OUT/'by_class.csv')
    groups = classes.groupby(['port', 'draw', 'arm', 'coverage']).recall.mean().reset_index()
    coverage = groups.groupby(['arm', 'coverage']).agg(recall=('recall', 'mean'), task_groups=('draw', 'size')).reset_index()
    coverage.to_csv(OUT/'coverage_summary.csv', index=False)
    task_scores = rows.pivot(index=['port', 'draw'], columns='arm', values='BA')
    task_scores['episode_target_update'] = task_scores.episode_init_adapt-task_scores.episode_init_head
    task_scores['supervised_target_update'] = task_scores.supervised_init_adapt-task_scores.supervised_init_head
    task_scores['episode_vs_supervised'] = task_scores.episode_init_adapt-task_scores.supervised_init_adapt
    task_scores.reset_index().to_csv(OUT/'stage_effects.csv', index=False)
    averages = means.set_index('arm')
    missing = coverage[coverage.coverage.eq('absent')].set_index('arm').recall
    passed = bool(averages.loc['episode_init_adapt', 'BA'] > averages.loc['previous_repaired', 'BA']
                  and averages.loc['episode_init_adapt', 'BA'] > averages.loc['supervised_init_adapt', 'BA']
                  and missing['episode_init_adapt'] >= missing['previous_repaired'])
    decision = dict(continue_channel_route=passed,
        rule='Episode+adapt mean BA > previous repair and ordinary warmup+adapt; missing-class recall >= previous repair',
        episode_delta_BA_vs_repair=float(averages.loc['episode_init_adapt', 'delta_BA']),
        episode_delta_BA_vs_supervised=float(averages.loc['episode_init_adapt', 'BA']-averages.loc['supervised_init_adapt', 'BA']),
        absent_recall_delta=float(missing['episode_init_adapt']-missing['previous_repaired']),
        boundary='Operational closeout rule for this bounded pilot, not a significance test or universal rejection')
    (OUT/'DECISION.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')
    names = dict(previous_repaired='上轮修正版', supervised_init_head='普通源预训练+支持头',
                 supervised_init_adapt='普通源预训练+目标内部适配', episode_init_head='源任务预训练+支持头',
                 episode_init_adapt='源任务预训练+目标内部适配')
    design = pd.read_csv(OUT/'source_episode_design.csv')
    history = pd.read_csv(OUT/'source_training_history.csv')
    lines = ['# 源港64标签模拟训练：当前通道路线的有限收口实验', '',
             '## 结论', '',
             ('满足预定小试保留条件；仍需报告分港取舍，不是充分泛化证明。' if passed else
              '未满足预定保留条件。本轮后停止沿相同通道缩放路线连续修补，不追加目标参数搜索；保留全部结果并转向新的机制假设。'), '',
             '本轮实际正负信号：源任务初态+目标适配相对上轮修正版，平均BA提高0.339点、OA下降0.241点，缺类召回下降2.279点。相对相同步数的普通源预训练+目标适配，BA提高0.244点。不是完全无效，但未达到兼顾识别收益与缺类保护的目标。', '',
             table(means.assign(arm=means.arm.map(names)), list(means.columns)), '',
             '百分数/百分点。三个港、每港三个支持抽样等权。delta相对上一轮“屏蔽缺类直接负向损失+源监督”的版本。', '',
             '## 训练流程', '',
             '1. 保持原939个源标签和原3港9个64目标标签划分，不新增标注。',
             '2. 在源港抽64个不同MMSI作为支持，剩余身份作查询；模拟目标港的源分类头和标准化仅在其他源港拟合，并排除该模拟港所有MMSI。',
             '3. 冻结R18，仅训练1024个通道仿射参数。源任务版本根据独立船舶查询损失训练初态，梯度通过支持集岭回归头；这是任务式表征初态训练，不是展开24步内循环的MAML。',
             '4. 普通源预训练对照同为24步、1024参数、相同源标签池，但使用源分类监督；两种预训练前向计算量不同，不能声称FLOPs相等。',
             '5. 分别测试初态接64支持分类头，以及再进行24步目标内部适配。后者沿用前轮修正损失、学习率、参数边界、源回放和零初态正则锚点。',
             '6. 全部使用固定最后一步，不看目标查询选择参数、检查点或港口专用版本。', '',
             '## 源任务覆盖', '',
             table(design, list(design.columns)), '',
             '固定小源集只有两个港满足至少80艘不同船且其余源参考覆盖八类。六个源查询任务参与了训练，不是独立源验证。支持和查询可能共享采集大景。不能把失败扩大为“所有源任务学习均不可行”。', '',
             '逐一检查六个源任务：查询类别均包含在支持类别中，没有支持缺类、查询却有该类正例的源任务。因此不能说已经直接训练了缺类恢复；这轮直接训练的是独立船舶查询的少样本分类，缺类效果以目标诊断为准。', '',
             '## 分港结果', '',
             table(ports.assign(arm=ports.arm.map(names)), list(ports.columns)), '',
             '## 阶段效果', '',
             table(task_scores.reset_index(), ['port', 'draw', 'episode_target_update', 'supervised_target_update', 'episode_vs_supervised']), '',
             'target_update是同一种初态下再做目标内部适配相对只拟合支持头的BA变化。所有分支均预先列入，不以事后最佳分支替换预定保留条件。', '',
             '## 标签覆盖', '',
             table(coverage.assign(arm=coverage.arm.map(names)), list(coverage.columns)), '',
             '先任务内类别等权，再可用任务等权；缺类组4个任务，不是全测试集BA。', '',
             '## 训练记录与核对', '',
             table(history.groupby('arm').tail(1), list(history.columns)), '',
             '初态加载为零时重新训练一个目标任务，必须回放原修正版参数；记录见REPLAYS.json。沿用的8项损失/支持划分测试通过。', '',
             f'完成源预训练{done["source_warmup_fits"]}次、目标适配{done["target_adapter_fits"]}次、原版回放1次，耗时{done["seconds"]:.1f}秒，仅CPU单线程。', '',
             '## 与Causal-Tune的关系', '',
             '本轮不是Causal-Tune复现：没有DCT带通与频域token。先完成通道适配路线的最后有限尝试，避免把尚未验证的频域分解混进来。论文阅读与公开代码差异见上一级CAUSAL_TUNE_READING_ZH.md。', '',
             '无论结果如何，不能据此宣称物理因果因素或纯噪声已被识别。所有数据是已使用的开发数据，既有预训练曝光边界保留。', '']
    lines += ['## 本轮失效位置与换向', '',
              '缺类召回沿阶段变化：上轮修正版30.558%；源任务初态仅接支持头29.661%；再接目标内部适配28.279%。即初态阶段已低0.897点，目标内部更新又下降1.382点，不能只把损失归给最后分类头或单一步骤。', '',
              '普通源预训练初态仅接支持头时缺类召回34.388%，再内部适配后30.906%，同样说明目标内部更新仍有误伤风险。不能据此将所有目标适配判为无效，它在其他类别/港口仍有收益。', '',
              '实现方面，零初态加载的参数回放与旧版完全一致，沿用损失测试通过；本轮没有发现接线错误。证据更支持：共享通道更新仍在重新分配判别能力，未形成可独立控制的干扰修正。源任务仅两个港且没有支持缺类的查询正例，也限制了本轮能学到的保护能力；其具体贡献未被单独实验隔离，不称为已证实唯一根因。', '',
              '换向候选：把“判定整个通道该放大/压小”改为“保留原主干，分解层内空间频率，只让残差支路利用经数据验证的分量做补偿”。先验证频率分量与真实同船条件变化、类别信息的关系，再决定是否使用固定带通或学习选择；不直接套用自然图像低/高频等于SAR噪声的假设。', '',
              '该频域方向尚未在本轮实施或验证，也不是将当前小试改名为Causal-Tune。当前通道路线以本报告收口，所有原始结果保留。', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(means.to_string(index=False))
    print(ports.to_string(index=False))
    print(coverage.to_string(index=False))
    print(json.dumps(decision))


if __name__ == '__main__':
    run()
