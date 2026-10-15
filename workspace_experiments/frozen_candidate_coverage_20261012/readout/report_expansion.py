"""Describe all frozen outcomes, including the newly covered ports."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
OLD_PORTS = ('Los Angeles', 'Port Klang', 'Santos')
POLICIES = ('original','resolve','retain_half','retain_full','cv_oa','cv_nll')
NAMES = dict(self24ports='23 ports', f0921='F_0921', oss2='OSS2', oss1='OSS1 (compact)')


def table(frame):
    return frame.to_markdown(index=False, floatfmt='.4f')


def main():
    metrics = pd.read_csv(OUT/'metrics.csv')
    summary = pd.read_csv(OUT/'summary.csv')
    aggregate = pd.read_csv(OUT/'aggregate.csv')
    full = summary.query("stage=='full' and scope=='original'").copy()
    primary = full[full.policy=='cv_oa']
    port = full[full.dataset=='self24ports'].copy()
    port['coverage'] = np.where(port.task.isin(OLD_PORTS), 'original_3_ports', 'added_20_ports')
    parts = port.groupby(['coverage','policy'],sort=False)[['oa','ba','delta_oa','delta_ba','delta_macro_brier']].mean().reset_index()
    parts.to_csv(OUT/'port_coverage_groups.csv',index=False)
    effects = []
    for (dataset,policy), f in full.groupby(['dataset','policy'],sort=False):
        effects.append(dict(dataset=dataset,policy=policy,tasks=len(f),
            ba_positive_ports=int((f.delta_ba>1e-10).sum()),
            ba_negative_ports=int((f.delta_ba < -1e-10).sum()),
            ba_tied_ports=int((f.delta_ba.abs()<=1e-10).sum()),
            delta_ba_min=f.delta_ba.min(),delta_ba_max=f.delta_ba.max()))
    spread = pd.DataFrame(effects)
    spread.to_csv(OUT/'task_spread.csv',index=False)
    choices, supports = [],[]
    receipt = json.loads((OUT/'PREDICTIONS_FROZEN.json').read_text(encoding='utf-8'))
    for task in receipt['tasks']:
        folder = OUT/'tasks'/task['dataset']/task['task']
        chosen = json.loads((folder/'SELECTION_FROZEN.json').read_text(encoding='utf-8'))['choices']
        for r in chosen:
            choices.append(dict(dataset=task['dataset'],task=task['task'],**r))
        inventory = pd.read_csv(folder/'support_inventory.csv')
        inventory['dataset'],inventory['task'] = task['dataset'],task['task']
        supports.append(inventory)
    choices = pd.DataFrame(choices)
    choices.to_csv(OUT/'choices.csv',index=False)
    inventory = pd.concat(supports,ignore_index=True)
    outcomes = metrics.query("stage=='full' and scope=='original' and policy=='cv_oa'")
    joined = inventory.merge(outcomes,on=['dataset','task','draw'],validate='one_to_one',suffixes=('_support','_query'))
    joined.to_csv(OUT/'support_outcomes.csv',index=False)
    class_frame = pd.read_csv(OUT/'class_recalls.csv').query("scope=='original' and policy=='cv_oa'")
    by_task = class_frame.groupby(['dataset','task','class_id'])[['recall','recall_delta']].mean().reset_index()
    class_effects = by_task.groupby(['dataset','class_id']).agg(
        recall=('recall','mean'),recall_delta=('recall_delta','mean'),ports_with_class=('task','nunique')).reset_index()
    class_effects.to_csv(OUT/'class_effects.csv',index=False)
    displays = aggregate.query("stage=='full' and scope=='original'")[
        ['dataset','operator','policy','oa','ba','delta_oa','delta_ba','delta_macro_brier']]
    main_rows = displays[displays.policy=='cv_oa']
    common = aggregate.query("stage=='full' and scope=='common' and policy=='cv_oa'")[
        ['dataset','oa','ba','delta_oa','delta_ba','delta_macro_brier']]
    stages = aggregate.query("scope=='original' and policy=='cv_oa'")[
        ['dataset','stage','delta_oa','delta_ba','delta_macro_brier']]
    variation = pd.read_csv(OUT/'support_variation.csv').query(
        "scope=='original' and stage=='full' and policy=='cv_oa' and dataset!='self24ports'")[
        ['dataset','delta_oa_sd','delta_ba_sd','ba_positive','ba_negative']]
    port_effect = primary[primary.dataset=='self24ports'].sort_values('delta_ba')
    fig, axes = plt.subplots(1,2,figsize=(13,8),gridspec_kw={'width_ratios':[1.6,1]})
    colors = np.where(port_effect.delta_ba>=0,'#287b65','#bf5555')
    axes[0].barh(port_effect.task,port_effect.delta_ba,color=colors)
    axes[0].axvline(0,color='#777777',lw=.8)
    axes[0].set_xlabel('Full-model BA change (percentage points)')
    axes[0].set_title('Frozen support-CV rule: all 23 ports')
    order = ['self24ports','f0921','oss2','oss1']
    ds = main_rows.set_index('dataset').loc[order]
    xx = np.arange(len(order))
    axes[1].bar(xx-.18,ds.delta_oa,width=.36,label='OA',color='#287b65')
    axes[1].bar(xx+.18,ds.delta_ba,width=.36,label='BA',color='#bb7840')
    axes[1].axhline(0,color='#777777',lw=.8)
    axes[1].set_xticks(xx,[NAMES[k] for k in order],rotation=25,ha='right')
    axes[1].set_ylabel('Change (percentage points)')
    axes[1].set_title('Matched baseline within each dataset')
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/'coverage_results.png',dpi=170)
    plt.close(fig)
    body = ['# 冻结候选扩大覆盖：完整模型接法',
        '', '日期：2026-10-12。四接法、CV规则、支持抽样和超参数冻结后，完成25个完整模型任务与1个OSS1紧凑任务；没有新增神经训练、没有根据新增查询结果调参或换规则。',
        '', '## 先说结论',
        '', '- 用户提出先扩大覆盖是必要的：旧三港主CV规则平均BA变化为-0.1161个百分点，新增二十港为+0.0089，全部二十三港为-0.0074。旧诊断港不能代表全部任务；扩大之后整体仍基本持平，不能称为明显突破。',
        '- OSS2主CV规则OA/BA分别+0.2934/+0.2253个百分点，存在局部净收益；但五组支持中BA为两正、一负、两持平，当前仍是同一批历史数据的扩大分析，不是独立确认。',
        '- F_0921主CV规则OA约持平、BA-0.5019；固定resolve接法OA/BA却为+0.1086/+0.0324。这区分了候选接法与选择规则的表现，不能据此事后改用F专属最优规则。',
        '- OSS1仅是紧凑版参考。主CV规则BA+0.0207、OA-0.1556；固定retain_full的BA+0.8450伴随OA-1.2296，是类别收益与总体正确率的取舍，不是无条件获益。',
        '- 暂不替换论文主模型，也不一票否决候选。当前支持的是明确范围的小收益和选择迁移问题；继续开发必须针对可利用的收益空间，而不是重新堆模块。',
        '', '## 主结果：既有cv_oa规则', '', table(main_rows),
        '', 'OA/BA及增量均为百分点单位；macro Brier变化为原始风险单位，负数表示改善。23港先逐港平均五组支持，再等港平均；外部数据分别统计。OSS1不是同版分频完整模型，不混入23港或外部总体均值。',
        '', '## 原三港与新增二十港', '', table(parts[parts.policy=='cv_oa']),
        '', '该分组在新结果读取前确定。它回答原三个诊断港能否代表其余港口，不用于按目标成绩重新选择方法。',
        '', '## 所有冻结候选', '', table(displays),
        '', 'cv_nll保留为预定诊断，不因它在某个目标上较好而替换cv_oa。固定接法的正结果同样保留，但不是新部署选择规则。',
        '', '## 逐港与支持波动', '', table(spread[spread.policy=='cv_oa']),
        '', '逐任务表：', '', table(primary[['dataset','task','oa','ba','delta_oa','delta_ba']]),
        '', '支持抽样波动见support_variation.csv；每组选择见choices.csv，支持组成与该组收益见support_outcomes.csv。五组支持不是五个独立训练种子；不能把所有小负值事后解释成随机。',
        '', '## 公共查询复核', '', table(common),
        '', '以上固定为各任务五组查询交集，不与原查询绝对分数混算。两种口径均为历史外层行排除，不称严格未见船验证。',
        '', '## 收益在哪一步保留或消失', '', table(stages),
        '', '上述是同一冻结策略的阶段输出，不是互不相关的因果效应分解。F的先验BA+0.4780、专家输出-0.6170、最终融合-0.5019，说明只看先验改善不足以决定部署价值。23港专家BA+0.0275而融合-0.0074，同时三个阶段macro Brier均改善；应分别判断概率证据、专家识别和系统结果。OSS2三个阶段OA/BA与macro Brier均改善。',
        '', '外部数据五组支持的配对增量标准差与正负次数：', '', table(variation),
        '', '正负次数未达到五的部分为持平。标准差反映现有五组支持的敏感性，不是独立数据集或独立训练种子的置信区间；不能据此直接把退化归为随机。',
        '', '逐组支持与选择结果：', '', table(joined[joined.dataset!='self24ports'][
            ['dataset','draw','classes_support','arm','delta_oa','delta_ba']]),
        '', 'OSS2的五组BA变化为0、0、+0.0757、-0.0637、+1.1144个百分点，平均正收益主要来自第4组（编号从0开始）。F的第3、4组均选择retain_full，BA分别-0.6857、-1.8563；其中第3组支持覆盖八类，说明“支持缺类”不足以单独解释该现象。这些定位的是抽样与接法的作用范围，尚未证明某个支持组成统计量能提前预测收益。',
        '', '## 类别边界', '', table(class_effects),
        '', '逐类值先在任务内平均，再在含该类的任务间等权；ports_with_class给出覆盖数。不是所有港口每类都存在，也不能将此表类均值再平均以替代原等港BA。',
        '', '## 实现与验证',
        '', '- 旧四任务同支持候选输出、内部OOF预测及分折重放；各任务original的基础、专家及融合首选类别核对历史参照。',
        '- 初次使用ClearSAR sklearn1.8.0时，同seed分折与原1.7.2不同；分折统一采用1.7.2。F的数值重放另用ClearSAR，并读取1.7.2预生成且与支持ID/标签/分组核对的分折，不修改选择规则。历史PCA反序列化版本警告保留，实际端点和旧结果重放是本次兼容性依据。',
        '- OSS2按旧接入整批PCA布局、F按既有2048分块计算；外部数据旧概率有约1e-7量级数值差，单列最大误差，不冒称逐值精确。旧候选及OOF首选类别、分折和CV选择必须一致，概率容差3e-7另加相对1e-7。此次重放没有唯一定位浮点差异来源，不把它讲成研究机制。',
        '- 每个任务的选择与全候选预测先保存；26任务全部冻结之后才统一读取本轮查询指标。旧开发任务结果并非未知。',
        '- 各类救回减误伤严格还原BA差；已知支持同船同折，未知身份按样本分组；缺类支持不改变八类输出空间。',
        '- 成本记录于各任务FROZEN.json，仅本次读出/CV及缓存写出，不是端到端图像推断成本。',
        '', '## 范围与下一步',
        '', '本报告扩大了已有方案的证据范围，不自动证明因果因素或所有任务稳定获益。依据实际净收益、支持波动和类别代价判断适用范围，不要求每港每类全胜；若发现新的适用条件，它仍是探索性条件，不能回头据此挑选本轮主结果。',
        '', '尚未改变训练配方、骨干、输入处理或论文主模型。预算修正路线的覆盖情况见同级budget目录，不能用本报告25任务完成数代替那条路线的完成数。',
        '', '![覆盖结果](E:/临时会话/frozen_candidate_coverage_20261012/readout/coverage_results.png)', '']
    (OUT/'FINDINGS_ZH.md').write_text('\n'.join(body),encoding='utf-8')


if __name__=='__main__':
    main()
