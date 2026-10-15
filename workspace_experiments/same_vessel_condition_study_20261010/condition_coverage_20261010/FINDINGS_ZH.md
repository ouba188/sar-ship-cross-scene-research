# 条件引导修正：源训练港口覆盖对照

## 结论

在相同32支持/16查询、120步预算下，源任务从2港扩到9港没有提高目标成绩。条件版BA38.274% -> 38.095%（-0.179个百分点），普通上下文38.414% -> 38.153%（-0.261个百分点）。本轮不能将先前未提点简单解释为覆盖港口太少。
原基线38.304%。2港普通上下文有+0.110个百分点BA的小正结果，完整保留；但它不是关系监督条件分支带来的增益，不能用来证明因子分离有效。单次神经初始化及重叠查询抽样也不足以确认稳定提升。

## 目标结果

| arm | OA | BA |
| --- | --- | --- |
| baseline | 37.195 | 38.304 |
| broad32_condition_context | 37.119 | 38.095 |
| broad32_ordinary_context | 37.110 | 38.153 |
| narrow32_condition_context | 37.198 | 38.274 |
| narrow32_ordinary_context | 37.258 | 38.414 |

| context | delta_OA | delta_BA | positive_tasks | negative_tasks |
| --- | --- | --- | --- | --- |
| condition_context | -0.079 | -0.179 | 4 | 5 |
| ordinary_context | -0.148 | -0.261 | 2 | 7 |

主比较为相同32支持/16查询、相同120步训练下的broad32减narrow32。真实目标评估仍为原3港9次64标签支持，不改变目标标注预算。

## 分港与缺类

| port | arm | OA | BA |
| --- | --- | --- | --- |
| Los Angeles | baseline | 40.404 | 38.676 |
| Los Angeles | broad32_condition_context | 40.328 | 38.252 |
| Los Angeles | broad32_ordinary_context | 40.399 | 38.612 |
| Los Angeles | narrow32_condition_context | 40.693 | 38.938 |
| Los Angeles | narrow32_ordinary_context | 40.705 | 39.256 |
| Port Klang | baseline | 33.295 | 32.434 |
| Port Klang | broad32_condition_context | 33.185 | 32.354 |
| Port Klang | broad32_ordinary_context | 33.175 | 32.272 |
| Port Klang | narrow32_condition_context | 33.165 | 32.415 |
| Port Klang | narrow32_ordinary_context | 33.185 | 32.409 |
| Santos | baseline | 37.886 | 43.803 |
| Santos | broad32_condition_context | 37.843 | 43.679 |
| Santos | broad32_ordinary_context | 37.757 | 43.575 |
| Santos | narrow32_condition_context | 37.735 | 43.470 |
| Santos | narrow32_ordinary_context | 37.886 | 43.577 |

| arm | coverage | recall | task_groups |
| --- | --- | --- | --- |
| baseline | absent | 33.503 | 4 |
| baseline | multiple | 36.869 | 9 |
| baseline | singleton | 38.375 | 5 |
| broad32_condition_context | absent | 33.503 | 4 |
| broad32_condition_context | multiple | 36.619 | 9 |
| broad32_condition_context | singleton | 38.642 | 5 |
| broad32_ordinary_context | absent | 33.503 | 4 |
| broad32_ordinary_context | multiple | 36.679 | 9 |
| broad32_ordinary_context | singleton | 38.375 | 5 |
| narrow32_condition_context | absent | 33.832 | 4 |
| narrow32_condition_context | multiple | 36.731 | 9 |
| narrow32_condition_context | singleton | 38.909 | 5 |
| narrow32_ordinary_context | absent | 33.832 | 4 |
| narrow32_ordinary_context | multiple | 36.887 | 9 |
| narrow32_ordinary_context | singleton | 38.909 | 5 |

缺类召回仅4组任务；各支持抽样的查询人群有重叠。

## 源域身份留出

| arm | OA | BA |
| --- | --- | --- |
| baseline | 41.897 | 40.425 |
| broad32_condition_context | 41.846 | 40.225 |
| broad32_ordinary_context | 41.790 | 40.402 |
| narrow32_condition_context | 42.104 | 40.870 |
| narrow32_ordinary_context | 41.946 | 40.762 |

仍为原302艘留出源船，64支持来自混合源港，与逐目标港64支持设定不同。没有用这组诊断选轮次或超参数。

## 更新是否生效

| arm | gain_rms | relative_update_rms | rescue | harm | mean_true_margin_change | max_abs_score_change | top_support_switch_pct | top_support_class_match_before | top_support_class_match_after | absent_class_score_max_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad32_condition_context | 0.063 | 0.073 | 3.889 | 5.444 | 0.000 | 0.040 | 6.756 | 35.436 | 35.563 | 0.000 |
| broad32_ordinary_context | 0.063 | 0.071 | 4.333 | 6.333 | 0.000 | 0.039 | 7.799 | 35.436 | 35.622 | 0.000 |
| narrow32_condition_context | 0.072 | 0.074 | 4.222 | 6.000 | 0.000 | 0.041 | 7.300 | 35.436 | 35.259 | 0.000 |
| narrow32_ordinary_context | 0.071 | 0.075 | 5.333 | 6.111 | 0.001 | 0.042 | 8.947 | 35.436 | 35.446 | 0.000 |

所有模型保留原源分类分数，仅调整原512维案例关系；支持缺类分数严格不变，但其他类别竞争仍可改变缺类召回。

## 控制范围

- narrow32仅Jebel Ali和Singapore；broad32加入所有至少48艘且参考源集仍有8类的源港。窄覆盖任务直接取宽覆盖池中的同一配对，无独立重新抽样。
- 每任务32支持船、16独立查询船，各船只取一张。两组均120步、Adam0.01、16896个参数、相同±10%限制；宽覆盖不增加梯度步数。
- 普通上下文/条件上下文编码器、归一化、原源分类器与302艘诊断身份全部冻结。全部使用原1744源图/605船，没有新增人工标签或真实目标训练数据。
- 各港按轮次交错，宽覆盖港训练次数尽可能均衡；重复源查询参与训练，不是独立确认。参考分类器排除整个模拟目标港及其船舶，冻结上下文网络的旧源训练曝光未撤销。
- 比上轮64支持训练减少源支持数以容纳更多港口。因此上轮结果只作历史参考，不能将其与broad32的差值全部归因于港口覆盖。
- 不按目标成绩挑港口、权重或版本；CPU单线程，无主线程或骨干改动。冻结R18开发小试，不是完整主模型或因果鉴定。

## 为什么扩覆盖仍然没有迁移收益

| context | curriculum | group | mse | OA | BA |
| --- | --- | --- | --- | --- | --- |
| condition_context | broad32 | added7 | 0.313 | 58.036 | 58.796 |
| condition_context | broad32 | old2 | 0.396 | 36.458 | 41.047 |
| condition_context | identity | added7 | 0.318 | 57.143 | 58.153 |
| condition_context | identity | old2 | 0.400 | 35.417 | 39.856 |
| condition_context | narrow32 | added7 | 0.317 | 57.738 | 58.411 |
| condition_context | narrow32 | old2 | 0.393 | 36.458 | 41.047 |
| ordinary_context | broad32 | added7 | 0.312 | 57.738 | 58.682 |
| ordinary_context | broad32 | old2 | 0.396 | 36.458 | 41.047 |
| ordinary_context | identity | added7 | 0.318 | 57.143 | 58.153 |
| ordinary_context | identity | old2 | 0.400 | 35.417 | 39.856 |
| ordinary_context | narrow32 | added7 | 0.318 | 57.440 | 58.298 |
| ordinary_context | narrow32 | old2 | 0.392 | 36.458 | 41.047 |

这是对同一批固定源训练任务的回放，不是独立验证。条件版在新增7港的MSE由未修正0.317587降至0.312667，2港训练模型在这些任务上为0.317471。说明扩覆盖确实学到了新增源任务，不是完全没有更新或训练没有作用。
但原2港的MSE由窄覆盖0.392529升至宽覆盖0.396246，仍好于未修正0.400227。扩覆盖产生了源任务间的取舍；源身份留出BA也从窄覆盖40.870%降至宽覆盖40.225%，并非只在目标3港出现下降。

| context | mean_cosine | min_cosine | negative_pairs | total_pairs |
| --- | --- | --- | --- | --- |
| condition_context | 0.010 | -0.086 | 15 | 36 |
| ordinary_context | 0.018 | -0.040 | 8 | 36 |

在零修正初始化处，先求各港3个任务的平均梯度，再比较9港间36对方向：条件版15对余弦为负，普通上下文8对为负，均值分别0.0097和0.0181。这是局部优化方向冲突/差异的线索，不是港口物理机制鉴定，也不能唯一证明最终退化的原因。Adam、非线性与后续梯度变化没有被这个静态诊断完整描述。
固定120步时，窄覆盖每港60步，宽覆盖每港13或14步。因此已检验的是固定总计算预算下扩覆盖，不是每港保持同等训练次数的比较；训练不足与任务冲突尚未完全分离，本轮不追加步数搜索。

## 实现与机制边界

- 4项检查通过，包括32支持训练切片、身份排除、零修正恒等及固定先验；9组原基线预测回放一致。
- 宽覆盖条件版实际特征更新RMS约7.29%，6.76%的查询更换了点积最高的支持案例，说明模块确实介入。每组平均纠正3.89张、误改5.44张，仍未产生净收益。
- 所有目标支持缺类分数的最大变化为0。新增27个源任务只有1个存在查询类别未被支持覆盖，不能宣称已充分训练缺类恢复。
- 本轮没有证明条件因子无价值，也没有发现足以解释结果的接入错误。更值得检验的是：编码出的条件差异，是否对应可复用的判别修正方向，而非只会区分场景。
- 不再把增加源港数量当成现成修复，不将本轮条件模块并入主模型。后续若继续，应针对条件信息与正确修正方向的对应关系设计实验，而不是继续无差别扩港或加大改动幅度。
