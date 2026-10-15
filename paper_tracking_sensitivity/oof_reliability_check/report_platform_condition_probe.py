"""Report platform response replication and its failure boundary."""

import numpy as np
import pandas as pd
from run_platform_condition_probe import OUT


def main():
    metrics=pd.read_csv(OUT/'response_metrics.csv')
    means=pd.read_csv(OUT/'response_means.csv')
    effects=pd.read_csv(OUT/'incremental_skill.csv')
    coverage=pd.read_csv(OUT/'pair_coverage.csv')
    lines=['# 入射角之外：S1A/S1C平台条件', '',
        '2026-10-09。全23个源划分的固定同船响应试验，不读取目标查询标签、不修改主分类模型。', '',
        '## 输入与设计', '',
        '当前38091条样本中，产品编号标记S1A为32184条、S1C为5907条。本轮只用已有入射角与位置均有效的28628条候选记录；不是根据类别或表现筛选平台。',
        '与上轮角度+位置试验采用相同观测聚合、同源身份排除和留出划分。MMSI固定留出约1/3，另外留出3/6/9/12月份。给定既有表示的条件响应验证，不是全流程内层OOF验证。',
        '同港同船配对，以角度+位置为基准，分别加入平台主效应c，以及c与t、t平方的交互项（S1C=1，S1A=0，t=(angle-38)/5）。全部输入先在单次观测计算，再做同船时相差分。零截距岭回归固定0.01，不调参。',
        '平台编号可能与时间、采集位置、观测几何及样本构成相关。它是可检验条件，不预设为独立传感器噪声，更不自动等于硬件因果效应。', '',
        '## 配对覆盖', '', '| 划分 | 每折配对数 | 跨平台配对数 | 跨平台船数 | 跨平台源港数 | 跨平台类别数 |',
        '|---|---:|---:|---:|---:|---:|']
    for split,c in coverage.groupby('split'):
        fields=['pairs','platform_switch_pairs','switch_ships','switch_ports','switch_classes']
        ranges=[f'{int(c[k].min())}-{int(c[k].max())}' for k in fields]
        lines.append('| '+split+' | '+' | '.join(ranges)+' |')
    lines+=['', '## 结果', '',
        'skill=1-预测差分残差能量/原差分能量。下表是加入平台相对同数据角度+位置基准的增量，单位为skill百分点，不是分类OA/BA。', '',
        '| 空间 | 划分 | 平台主效应增量pp | 正增量源划分 | 平台及角度交互增量pp | 正增量源划分 |',
        '|---|---|---:|---:|---:|---:|']
    for (space,split),g in effects.groupby(['space','split']):
        lines.append(f'| {space} | {split} | {100*g.platform_increment.mean():+.4f} | {(g.platform_increment>0).sum()}/23 | {100*g.interaction_increment.mean():+.4f} | {(g.interaction_increment>0).sum()}/23 |')
    lines+=['', '### 新船且新月份的绝对skill', '', '| 空间 | 模型 | skill (%) |', '|---|---|---:|']
    for _,r in means[means['split']=='new_ship_month'].iterrows():
        lines.append(f'| {r.space} | {r.model} | {100*r.skill:.4f} |')
    lines+=['', '## 诊断与下一步', '',
        '两项针对性测试验证平台差分及观测级角度交互，而非错误地相乘两个差分。相同数据基准与上轮逐折重放比较见COMPLETE.json；若重放不符则停止，不把实现偏差解释为平台无效。',
        '配对覆盖和条件矩阵秩用于区分没有可辨识变化与拟合后泛化失败。拟合/留出、逐类别指标全部保留；正的训练增量而负的留出增量不能作为有效平台响应。',
        '23个源划分重叠，不能计为23个独立实验。不同空间的skill也不是条件的物理能量占比。没有进行新的平台条件分类训练，本轮不能报告平台带来的OA/BA提升。',
        '其他条件来源与升降轨探查另见../additional_conditions_20261009。保留主效应与交互负结果，不继续凭目标表现搜索角度/平台组合。']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(effects.groupby(['space','split'])[['platform_increment','interaction_increment']].agg(['mean','min','max']).to_string())


if __name__=='__main__': main()
