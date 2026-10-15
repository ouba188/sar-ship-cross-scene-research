# 保留原主路的条件引导案例关系修正

## 结论

条件版目标BA38.261%，相对基线38.304%下降0.043个百分点，相对普通上下文38.019%高0.242个百分点。原来替换整个判别表示的大幅退化没有再出现，但没有取得跨港净收益，不加入主模型。
源域留出船舶任务BA由40.425%升至40.960%（+0.535个百分点）。该任务64支持来自多个源港混合人群，而目标评价每港64支持，任务组成不同；不能将两者差别仅归因于新港条件。

## 目标港结果

| arm | OA | BA |
| --- | --- | --- |
| baseline | 37.195 | 38.304 |
| condition_context | 37.201 | 38.261 |
| ordinary_context | 37.128 | 38.019 |

| port | arm | OA | BA |
| --- | --- | --- | --- |
| Los Angeles | baseline | 40.404 | 38.676 |
| Los Angeles | condition_context | 40.546 | 38.761 |
| Los Angeles | ordinary_context | 40.535 | 38.086 |
| Port Klang | baseline | 33.295 | 32.434 |
| Port Klang | condition_context | 33.235 | 32.414 |
| Port Klang | ordinary_context | 33.115 | 32.282 |
| Santos | baseline | 37.886 | 43.803 |
| Santos | condition_context | 37.821 | 43.608 |
| Santos | ordinary_context | 37.735 | 43.688 |

| arm | coverage | recall | task_groups |
| --- | --- | --- | --- |
| baseline | absent | 33.503 | 4 |
| baseline | multiple | 36.869 | 9 |
| baseline | singleton | 38.375 | 5 |
| condition_context | absent | 33.832 | 4 |
| condition_context | multiple | 36.745 | 9 |
| condition_context | singleton | 38.557 | 5 |
| ordinary_context | absent | 33.832 | 4 |
| ordinary_context | multiple | 36.502 | 9 |
| ordinary_context | singleton | 38.557 | 5 |

## 源域身份留出诊断

| arm | OA | BA |
| --- | --- | --- |
| baseline | 41.897 | 40.425 |
| condition_context | 42.104 | 40.960 |
| ordinary_context | 41.945 | 40.583 |

留出的302艘源船不参与条件编码器或修正网络训练。每次64艘作支持，其余身份查询；可共享港口及大景。这里是独立身份诊断，不是源训练episode查询表现。

## 本轮管线

- 原512维特征及其源分类分数完整保留。条件/普通上下文向量均32维、冻结；输入同样的零初始化线性层，经0.1*tanh生成512维增益，调节原标准化特征后形成支持核。
- ordinary_context来自上轮同结构但无关系损失网络；condition_context来自同船/同景关系监督网络。两者残差参数均16896、相同源episodes、120步Adam、学习率0.01。
- 源域模拟64支持任务，在支持集之外的船舶上监督最终残差分类结果，不直接奖励同船向量相似。训练查询参与梯度，因此不把它们称为独立验证。
- 源任务分类器排除整个模拟目标港及其所有船舶身份；已有冻结条件编码器训练曾使用这些源港身份，因此这不是对整个系统严格留港的验证。
- 目标港只用原64标签拟合相同的残差分类头，不额外更新神经网络或按目标港挑模型。原先验固定，支持未覆盖类别的分数严格不变，但仍可能被其他类别竞争压过。
- 只检验条件分支如何使用；没有让128维稳定分支替代主路，没有恢复频率实验。条件向量仍可能含船型信息，不能称纯环境因子。

## 实际影响

| arm | gain_rms | relative_update_rms | rescue | harm | mean_true_margin_change | max_abs_score_change | top_support_switch_pct | top_support_class_match_before | top_support_class_match_after | absent_class_score_max_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| condition_context | 0.074 | 0.077 | 3.889 | 4.667 | -0.000 | 0.043 | 6.538 | 35.436 | 35.358 | 0.000 |
| ordinary_context | 0.074 | 0.076 | 3.778 | 6.333 | -0.001 | 0.050 | 7.677 | 35.436 | 35.283 | 0.000 |

rescue/harm是每组查询计数的平均值，查询组重复，不能跨组相加当作独立船数。最近支持案例仅为诊断，真实预测使用全部64支持的带符号残差贡献。

## 边界

- 本轮两个源港可构成64支持任务，共6个固定训练episodes；修正网络源任务覆盖有限。没有用目标查询选超参数、训练轮次或版本。
- Jebel Ali每个训练任务查询22艘，Singapore为14艘；6个任务的查询类别均被对应支持集覆盖。本轮不是通过缺类正例训练学会保护缺类，而是通过固定先验保证缺类原分数不变。
- 冻结R18小试、单次神经初始化、原3港9次支持抽样；不是完整双路模型或独立外部确认。继承原预训练曝光与目标支持/查询共景边界。

## 条件向量是否真的需要逐船变化

| arm | mode | OA | BA |
| --- | --- | --- | --- |
| condition_context | dynamic | 37.201 | 38.261 |
| condition_context | source_average | 37.203 | 38.303 |
| ordinary_context | dynamic | 37.128 | 38.019 |
| ordinary_context | source_average | 37.178 | 38.288 |

| arm | OA | BA |
| --- | --- | --- |
| baseline | 41.897 | 40.425 |
| condition_context__dynamic | 42.104 | 40.960 |
| condition_context__source_average | 42.002 | 40.725 |
| ordinary_context__dynamic | 41.945 | 40.583 |
| ordinary_context__source_average | 41.897 | 40.425 |

固定已学权重，只将逐船增益替换为源训练1744张图上的平均增益，不重新训练。条件版在目标港由动态38.261%变为固定平均38.303%，后者基本等于原基线；源留出任务则由动态40.960%变为固定平均40.725%。动态条件使用在源内有信号，但本轮未显示跨港额外收益。

| arm | source_dynamic_gain_rms | source_average_gain_rms | sample_specific_gain_rms |
| --- | --- | --- | --- |
| ordinary_context | 0.074 | 0.011 | 0.073 |
| condition_context | 0.075 | 0.011 | 0.074 |

平均增益RMS约1.12%，动态约7.49%。此干预同时减小更新幅度，不是等幅度静态模型对照，因此不能单独证明样本自适应一定无效，也不能据此选择目标成绩最好的变体部署。

## 失效定位

- 原先验严格不变；目标支持缺类分数实际最大变化为0。缺类召回仍从33.503%变为33.832%，来自其他类别竞争变化，而非给缺类新增标签或修改其源分数。
- 条件版实际特征更新RMS约7.70%，约6.54%的查询更换了点积最高的支持案例。不是模块未接入或没有产生变化。最近案例诊断不等价于完整带符号残差分类。
- 每组平均纠正3.89张、误改4.67张；目标平均真实类间隔变化-0.000299。更新产生了正负两方面作用，尚未形成净收益。重复抽样查询不独立，不把这些计数相加当独立样本。
- 两项针对性测试通过；9组原基线、18组动态版本回放预测均一致。未发现可解释退化的接入错误，但这些检查不能证明没有任何实现问题。
- 剩余假设是条件校正规则的跨港可迁移性、源任务覆盖与更新幅度之间的关系。当前6个源任务只覆盖2港是实际限制，但尚未证明它是唯一根因。

## 后续取舍

保留本轮的原判别主路保护和条件关系学习作为候选，不宣称域不变因子已经被纯净分离或跨港提点成功。若继续，优先扩大源域模拟任务的港口/条件覆盖，固定其他设计后观察收益能否迁移；本轮没有追加新训练或超参数搜索。
