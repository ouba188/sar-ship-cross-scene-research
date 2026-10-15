"""Summarize the frozen interventions without selecting a deployment arm."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
summary = pd.read_csv(ROOT/'summary.csv')
paired = pd.read_csv(ROOT/'paired.csv')
retrieval = pd.read_csv(ROOT/'retrieval_summary.csv')
operators = pd.read_csv(ROOT/'operator_diagnostics.csv')
names = dict(raw='原模型', initial='完整支持修正', relation_only='只改查询—案例关系',
    coefficient_only='只改校正系数', strength_match='统一强度匹配',
    class_strength_match='逐类强度匹配', simple='旧版固定缩放')


def table(frame, columns):
    lines = ['| '+' | '.join(columns.values())+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in frame.iterrows():
        values = []
        for key in columns:
            value = row[key]
            if key == 'arm':
                value = names[value]
            elif isinstance(value, (float, np.floating)):
                value = f'{value:.6f}'
            values.append(str(value))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


fusion = summary[summary.layer=='fusion']
focus = fusion[(fusion.dataset=='f0921') & (fusion.budget==16)]
initial = fusion[fusion.arm=='initial']
relation = retrieval[retrieval.arm=='initial'].merge(retrieval[retrieval.arm=='raw'],
    on=['dataset', 'budget'], suffixes=('_initial', '_raw'), validate='one_to_one')
relation['map_gain_pp'] = 100*(relation.macro_map_at_r_initial-relation.macro_map_at_r_raw)
relation['p1_gain_pp'] = 100*(relation.macro_p1_initial-relation.macro_p1_raw)
relation.to_csv(ROOT/'retrieval_paired_summary.csv', index=False)

f = paired[(paired.dataset=='f0921') & (paired.budget==16) & (paired.layer=='fusion')]
advantage = f[f.arm=='initial'].merge(f[f.arm=='class_strength_match'], on='draw', suffixes=('_initial', '_scale'))
extra_ba = advantage.ba_initial-advantage.ba_scale
extra_oa = advantage.oa_initial-advantage.oa_scale
full_table = table(initial, dict(dataset='数据', budget='B', oa='OA%', ba='BA%', delta_oa='ΔOA(pp)',
    delta_ba='ΔBA(pp)', positive_ba_draws='BA正收益抽样数/5', delta_macro_brier='Δ类别宏Brier'))
focus_table = table(focus, dict(arm='对照', delta_oa='ΔOA(pp)', delta_ba='ΔBA(pp)',
    delta_ba_sd='ΔBA抽样SD', positive_ba_draws='BA正收益抽样数/5', delta_macro_brier='Δ类别宏Brier'))
retrieval_table = table(relation, dict(dataset='数据', budget='B', macro_map_at_r_raw='原宏MAP@R',
    macro_map_at_r_initial='修正宏MAP@R', map_gain_pp='Δ宏MAP@R(pp)', p1_gain_pp='Δ宏P@1(pp)',
    coverage_initial='同类支持覆盖率'))

text = f'''# 支持修正为什么有效：本轮机制诊断

## 先说结论

**F_0921 少样本支持修正的正信号保留，但它不是单纯“把邻居找得更准”，也不能全部归功于新增类别信息。** 当前证据更符合：源域同类信息监督改变支持表示，同时改变校正器的求解与作用强度；配套使用可有净收益。逐类强度匹配已能复现一部分收益，完整关系修正超出该替代方案的增益仍有明显抽样依赖。

本轮没有新增模块、训练或调参。两数据集×三预算×五抽样，共30预测单元、四种新计算干预；同时复用原模型、完整修正、旧缩放。主模型未替换。侧线输入因素研究未修改。

## 1. 实际改了什么

这个版本的训练目标是**同类其他船的均值**，不是同船历史教师；函数名 image_history 是历史命名，不能据此写成“目标船历史知识注入”。预测支持修正时不输入目标船类别，已有支持标签用于后续岭回归求解。

```mermaid
flowchart LR
    A[源域同类异船案例] --> B[拟合线性残差映射]
    C[目标港已标注支持案例] --> D[选取第二路576维坐标]
    B --> E[中心化变换并恢复总体尺度]
    D --> E
    E --> F[新的查询与支持案例关系]
    E --> G[新的支持自关系]
    G --> H[重新求解校正系数]
    F --> I[专家类别分数]
    H --> I
    J[固定专家先验] --> I
    I --> K[与固定基础分支等权概率融合]
```

R50坐标、查询表示、两路先验与基础分支全部不动。改的是第二路保留64维和原坐标补充512维，共576维支持坐标。

现有代码等价于以下变换（仅写这576维）：

`C = mean(R) + a * (R - mean(R)) @ (I + dose * W)`

其中 a 恢复原支持集合的中心化总能量。残差预测的常量/均值项在中心恢复步骤被消去。这个恒等式在30单元全部成立；中心最大误差 {operators.centroid_error.max():.3g}，能量比接近1。

因此它是**依赖支持集合、受类别监督的方向重组**，不是简单抬高亮度/特征总幅度，也不是已经分离出的纯因果因素。剂量、归一化和支持构成会共同影响其作用。

## 2. 哪条计算路径带来收益

设 R/C 是原/修正支持，Q 是查询。原得分为 `P + Q R^T a0`，修正为 `P + Q C^T a1`。

- 只改关系：`P + Q C^T a0`。
- 只改系数：`P + Q R^T a1`。
- 统一/逐类强度匹配：将原读出投影到完整修正读出，分别只允许一个系数或每类一个系数。不使用查询标签，不搜索超参数。

以下均为 **F_0921 B16，最终融合，相对原模型**。Brier变化负数表示改善。

{focus_table}

**第一条可确认的计算机制：在这个设置中，关系变化与重新求解不能拆开直接用。** 两个交叉干预平均退化，而联合修正平均正收益。交叉干预不是新的对称核模型，结论是配套计算的重要性，不是两个独立物理因素的证明。

**第二条重要边界：分数强度也解释了部分收益。** 完整修正相对逐类强度匹配平均只多 OA {extra_oa.mean():+.6f}、BA {extra_ba.mean():+.6f} 个百分点；额外BA收益仅 {int((extra_ba>0).sum())}/5 组为正。逐类强度匹配的类别宏Brier改善还更大。不能把 .868 和 .656 的比值当作因果贡献比例，也不能宣布复杂修正已稳定优于合理缩放。

两个投影匹配对照仍依赖完整修正读出，只是机制对照，**不是无需修正器的低成本部署方法**。它们也没有穷尽所有校准方法。

## 3. 案例关系是否真的改善

固定最多4096个不依赖标签抽取的查询，按实际点积关系检索支持案例。排除可识别的同MMSI；仅在仍有同类候选的查询上计算MAP@R，单独报告覆盖率。F用4096查询，OSS2用全部3440共同查询，均五抽样均值。

{retrieval_table}

F_B16 的宏MAP@R与宏P@1都改善，说明不只有最终决策的偶然翻转。但 OSS2_B16 的关系质量和系统BA同时退化；F_B64、OSS2_B64 的宏MAP@R改善却未兑现为总体分类收益。**“学到了更有用的案例关系”和“当前读出带来净收益”确实必须分开评价。**

这是目标支持案例检索，不冒充严格跨港检索。MAP@R的候选数和同类数量随预算改变，不跨B直接比较绝对值判断表征优劣。本轮不是因果分离证明。

## 4. 保留所有作用范围

完整修正对照原模型，最终融合结果：

{full_table}

F_B16 正信号不能外推成任何数据、预算下都适用。OSS2_B128虽平均略正，但仅1/5抽样BA正收益；不把它与F_B16的4/5混为相同可靠程度。B变化时 dose=1/.5/.25、岭强度=B/8 也随之改变，因此不能把跨B差异唯一归因于标注数量。

![固定计算干预](mechanism_budget.png)

图为配对ΔBA均值，误差线是五次支持抽样的标准差，不是置信区间，更不是五次独立训练。

## 5. 哪些类别、哪些抽样在起作用

F_B16 的平均BA正贡献主要是散货船 +0.807095、渔船 +0.731804、杂货船 +0.569629、海工补给船 +0.355244 个百分点；原油轮 -0.652494、成品油/化学品轮 -0.517394、拖轮 -0.369342 等损失抵消了相当部分。渔船召回率平均 +5.854430 个百分点，五组都改善。

**特别有解释价值的是：五组支持中渔船数量都为零。** 这部分识别改善不可能由“检索到了更好的同类目标支持案例”产生。检索MAP@R又只统计有同类候选的查询，因此其改善不能直接解释缺席类别的改善。

针对这条观察追加了既有分数分解，未新增模型：固定每个查询在原专家中的最强错误竞争类，比较正确类与该竞争类的差值。F_B16渔船的支持校正平均把原先验的这一差值压低 **0.701419**，完整修正后压低 **0.397604**，即少压低 **0.303816**；五组该差值均向好。这支持“减轻不完整支持集对已有类别证据的不利作用”这一计算解释，而不是“获得了新的同类目标案例”。

作为参照，OSS2_B16渔船每组有1至3个支持，原校正对同一差值的平均贡献为 +0.144041，修正后为 -0.087535；该类最终融合召回平均下降9.107143个百分点。相同源域映射并不保证在另一组目标案例上保留有用方向。

这仍是**结果后提出的、固定竞争类的专家分数描述**，不是独立因果验证。不能用均值解释每个融合决定；平均差值变好不保证top-1变好，因为其他竞争类别、样本分布和基础分支也起作用。分解见 prior_margin.csv 与 PRIOR_MARGIN_NOTE.json。

另外，draw 2贡献了五抽样平均BA净增益的69.156336%。只为描述集中程度而去掉它，剩余四组BA仍 +0.334817，但OA为 -0.147154；正式结果始终保留全部五组。不能把跨抽样重复出现的同一查询当成新增独立样本。

独立诊断见 [类别与抽样分析](diagnostics/FINDINGS.md)；逐类净救回、误伤、支持覆盖以及 initial/simple 比较均保留。

## 6. 下一步设计如何收紧

1. 保留已有修正器作为范围明确的候选，不默认全任务启用；本轮不按查询结果挑最优组合部署。
2. 下一个需要解决的具体问题是：**怎样从训练/支持侧识别“方向重组值得用”的情形，而不把已经能由强度调整得到的收益重复包装成新信息。**
3. 开发前先在源留船任务上，将完整修正与本轮强度对照比较，定位可重复的额外净收益；再判断支持内可观测量能否预测这种收益。只有具备收益空间且可利用，才开发选择或约束模块。本轮没有完成这一步，也不宣称已有选择器。
4. 侧线继续研究第二路输入证据为何互补；本轮解释支持校正的计算机制。两条证据链可以衔接，但数值收益不能相加，也不能互相替代。

## 7. 验证与记录

- 全30单元 raw/initial 分数重放最大误差0；基础分支及原预测只读复用。
- 四个针对性测试通过：计算路径与显式核乘法一致、无变换时全部对照回到原模型、同船排除、公开数据身份字段。
- 首次检索统计漏识别 OSS2 的 mmsi_or_ship_id 字段；已修复并重算检索，分类预测与指标不变。最终两数据集查询/支持身份均可识别。
- 预测冻结耗时74.59秒，CPU2线程，无GPU、新训练或外部数据下载；不含评价及报告耗时。
- 固定softmax温度1，概率风险用于一致口径比较，不宣称已校准。历史开发数据复用、行级支持/查询隔离和原预训练暴露边界沿用。
- 五次支持抽样的配对均值/SD描述抽样敏感性，不是训练方差、独立泛化确认或显著性保证。未新增训练，不为凑三种子重复确定性计算。
- 协议：[PROTOCOL_ZH.md](PROTOCOL_ZH.md)；全指标：[summary.csv](summary.csv)；逐抽样：[paired.csv](paired.csv)；检索：[retrieval.csv](retrieval.csv)；计算路径边际：[margin_paths.csv](margin_paths.csv)。预测缓存路径见 PREDICTIONS_FROZEN.json，权重另存于 weights/。
'''
(ROOT/'FINDINGS_ZH.md').write_text(text, encoding='utf-8')

arms = ['initial', 'relation_only', 'coefficient_only', 'class_strength_match']
labels = ['Full correction', 'Relation only', 'Coefficient only', 'Class-wise strength']
colors = ['#228833', '#4477AA', '#CC6677', '#AA4499']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, ds, title in zip(axes, ['f0921', 'oss2'], ['F_0921', 'OpenSARShip 2.0']):
    for j, (arm, label, color) in enumerate(zip(arms, labels, colors)):
        data = fusion[(fusion.dataset==ds) & (fusion.arm==arm)].set_index('budget').loc[[16,64,128]]
        ax.bar(np.arange(3)+(j-1.5)*.18, data.delta_ba, width=.17, color=color,
               yerr=data.delta_ba_sd, capsize=2, error_kw={'linewidth': .65}, label=label)
    ax.axhline(0, color='#222222', linewidth=.7)
    ax.set_xticks(np.arange(3), ['16', '64', '128'])
    ax.set_xlabel('Labeled support budget')
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Fusion BA change (percentage points)')
fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='lower center', frameon=False, fontsize=9)
fig.suptitle('Fixed support-map interventions; mean and SD of five paired draws', fontsize=11)
fig.tight_layout(rect=(0, .09, 1, .94))
fig.savefig(ROOT/'mechanism_budget.png', dpi=160)
plt.close(fig)
print('REPORT COMPLETE', ROOT/'FINDINGS_ZH.md')
