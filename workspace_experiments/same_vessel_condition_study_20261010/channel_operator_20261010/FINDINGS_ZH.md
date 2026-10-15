# 内部通道软抑制：首轮算子级实验

## 结论先行

真实内部门控已实现并完成试验。定向可学习门控的三港等权 OA 改变 +0.398 个百分点，BA 改变 -0.088 个百分点。优于本轮三个随机通道对照的三港平均指标，但尚未超过未门控基线的平均 BA，不能作为成熟提点模块。

根因测量表明：选中通道的抑制确实降低了内部共同场景变化；分类头重新拟合抵消了大部分分数层降幅。补上固定场景分数约束后，平均 BA 仍未改善。因此，分类头补偿是一个已定位的中间环节，但不是退化的全部解释；当前通道统计尚未区分有害依赖和有用判别成分。

Santos 的得失尤其明确：散货船净增9个正确样本，但成品油/化学品船净减4个、原油船减1个、拖轮减2个、海工补给船减1个。不能用多数类收益掩盖其他类的损失。

本轮不是末端特征减法。门控位于 SAR R18 最后一个残差块之前，抑制后的响应继续经过真实卷积、BN、ReLU 和残差相加。骨干权重固定，门控与源域分类头交替优化。

## 范围

- 训练 939 个源港样本，三港共同留出，评估 5654 个样本；不是三个独立 LOPO，也不是主模型的64标签实验。
- 源域排除三个目标港、目标船舶身份及场景；目标标签只在训练完成后计算指标。历史神经预训练曝光仍然存在。
- 512个内部通道中选64个。选择依据为源港多船型共同场景变化与类别均值差异的相对关系，并不把它们认定为纯噪声神经元。
- 三次随机对照只是随机通道集合，不是三个独立神经网络训练种子；固定三轮，不按目标结果选择版本。

## 三港等权结果

| arm | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- |
| baseline | 40.752 | 35.771 | 0.000 | 0.000 |
| targeted_fixed | 40.965 | 35.706 | 0.213 | -0.066 |
| targeted_learned | 41.150 | 35.683 | 0.398 | -0.088 |
| random_learned_0 | 40.279 | 35.540 | -0.472 | -0.231 |
| random_learned_1 | 40.344 | 35.204 | -0.408 | -0.568 |
| random_learned_2 | 40.739 | 35.584 | -0.013 | -0.187 |

OA、BA以百分数计，delta以百分点计。三港等权均值不是把所有查询合并的OA。

## 逐港结果

| arm | port | n | OA | BA | delta_BA | rescue | harm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | Santos | 1653 | 51.845 | 45.064 | 0.000 | 0 | 0 |
| baseline | Port Klang | 3425 | 33.431 | 27.563 | 0.000 | 0 | 0 |
| baseline | Los Angeles | 576 | 36.979 | 34.686 | 0.000 | 0 | 0 |
| targeted_fixed | Santos | 1653 | 51.906 | 44.510 | -0.554 | 17 | 16 |
| targeted_fixed | Port Klang | 3425 | 33.489 | 27.657 | 0.094 | 38 | 36 |
| targeted_fixed | Los Angeles | 576 | 37.500 | 34.950 | 0.264 | 7 | 4 |
| targeted_learned | Santos | 1653 | 51.966 | 44.135 | -0.929 | 25 | 23 |
| targeted_learned | Port Klang | 3425 | 33.810 | 27.748 | 0.185 | 49 | 36 |
| targeted_learned | Los Angeles | 576 | 37.674 | 35.166 | 0.479 | 10 | 6 |
| random_learned_0 | Santos | 1653 | 51.180 | 44.783 | -0.281 | 19 | 30 |
| random_learned_0 | Port Klang | 3425 | 33.547 | 27.800 | 0.236 | 54 | 50 |
| random_learned_0 | Los Angeles | 576 | 36.111 | 34.038 | -0.649 | 6 | 11 |
| random_learned_1 | Santos | 1653 | 52.269 | 44.226 | -0.839 | 31 | 24 |
| random_learned_1 | Port Klang | 3425 | 33.693 | 27.913 | 0.350 | 40 | 31 |
| random_learned_1 | Los Angeles | 576 | 35.069 | 33.472 | -1.214 | 7 | 18 |
| random_learned_2 | Santos | 1653 | 50.938 | 44.437 | -0.627 | 18 | 33 |
| random_learned_2 | Port Klang | 3425 | 33.431 | 27.686 | 0.123 | 44 | 44 |
| random_learned_2 | Los Angeles | 576 | 37.847 | 34.631 | -0.056 | 12 | 7 |

## 抑制是否真正发生

| arm | channels | gate_mean | gate_min | at_floor | at_ceiling | mean_scene_component | mean_class_variance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_fixed | 64 | 0.750 | 0.750 | 0 | 0 | 0.338 | 0.041 |
| targeted_learned | 64 | 0.611 | 0.544 | 0 | 0 | 0.338 | 0.041 |
| random_learned_0 | 64 | 0.595 | 0.500 | 2 | 0 | 0.089 | 0.122 |
| random_learned_1 | 64 | 0.617 | 0.500 | 1 | 0 | 0.094 | 0.149 |
| random_learned_2 | 64 | 0.611 | 0.531 | 0 | 0 | 0.099 | 0.154 |

| arm | source_scene_score_energy_ratio | source_centroid_score_energy_ratio | source_BA | source_score_relative_change |
| --- | --- | --- | --- | --- |
| baseline | 1.000 | 1.000 | 59.317 | 0.000 |
| targeted_fixed | 0.989 | 0.985 | 59.154 | 0.022 |
| targeted_learned | 0.988 | 0.986 | 58.578 | 0.026 |
| random_learned_0 | 1.000 | 1.002 | 59.188 | 0.029 |
| random_learned_1 | 1.003 | 0.980 | 58.930 | 0.026 |
| random_learned_2 | 0.973 | 0.984 | 59.231 | 0.025 |

场景能量为源域不同类别场景残差平均后的分数能量，不是已识别的纯噪声方差；类别均值能量也不等于全部船型语义。两个指标下降或保留不能单独证明跨港泛化。

## 定向可学习版本逐类得失

| port | class_id | n | baseline_recall | recall | delta_recall | net_correct |
| --- | --- | --- | --- | --- | --- | --- |
| Santos | 0 | 898 | 67.261 | 68.263 | 1.002 | 9 |
| Santos | 1 | 10 | 90.000 | 90.000 | 0.000 | 0 |
| Santos | 2 | 88 | 9.091 | 9.091 | 0.000 | 0 |
| Santos | 3 | 191 | 50.785 | 48.691 | -2.094 | -4 |
| Santos | 4 | 197 | 45.685 | 46.193 | 0.508 | 1 |
| Santos | 5 | 56 | 62.500 | 60.714 | -1.786 | -1 |
| Santos | 6 | 188 | 3.191 | 2.128 | -1.064 | -2 |
| Santos | 7 | 25 | 32.000 | 28.000 | -4.000 | -1 |
| Port Klang | 0 | 882 | 69.274 | 69.841 | 0.567 | 5 |
| Port Klang | 1 | 57 | 19.298 | 19.298 | 0.000 | 0 |
| Port Klang | 2 | 394 | 36.802 | 38.071 | 1.269 | 5 |
| Port Klang | 3 | 895 | 14.525 | 14.749 | 0.223 | 2 |
| Port Klang | 4 | 651 | 19.969 | 19.969 | 0.000 | 0 |
| Port Klang | 5 | 273 | 33.700 | 34.432 | 0.733 | 2 |
| Port Klang | 6 | 197 | 4.569 | 4.569 | 0.000 | 0 |
| Port Klang | 7 | 76 | 22.368 | 21.053 | -1.316 | -1 |
| Los Angeles | 0 | 89 | 13.483 | 16.854 | 3.371 | 3 |
| Los Angeles | 1 | 19 | 94.737 | 94.737 | 0.000 | 0 |
| Los Angeles | 2 | 33 | 12.121 | 12.121 | 0.000 | 0 |
| Los Angeles | 3 | 97 | 27.835 | 27.835 | 0.000 | 0 |
| Los Angeles | 4 | 65 | 60.000 | 60.000 | 0.000 | 0 |
| Los Angeles | 5 | 215 | 50.233 | 50.698 | 0.465 | 1 |
| Los Angeles | 6 | 41 | 7.317 | 7.317 | 0.000 | 0 |
| Los Angeles | 7 | 17 | 11.765 | 11.765 | 0.000 | 0 |

## 可解释性与下一步判断

- pooled_channel_trace.csv 给出先前候选方向回溯到原始池化通道的线性载荷；internal_channel_ranking.csv 对应本轮真实内部层，二者层位不同，不混为同一神经元。
- 先确认 REPLAY.json 的原始输出复现与训练梯度；再判断定向是否优于相同训练的随机抑制，不能只与不改动版本比较。
- 如果门控改变而原有场景分数能量没有下降，说明通道选择与最终决策敏感性之间仍存在缺口；不能归因为门控没有接入。
- 如果随机抑制也改善，当前增益可能含普通正则化效应；若定向不能超过随机，不把本轮包装成专门消除港口噪声。
- 仅做最后一个残差块的门控与分类头协同，尚未微调骨干卷积，也没有证明其他层位、条件门控或更大样本训练无效。

所有分组得失保存在 class_attribution.csv 和 targeted_changed_cases.csv。不按目标港逐一挑选获胜方法。

## 分层根因定位

| arm | input_shared_energy_ratio | output_shared_energy_ratio | original_head_scene_energy_ratio | refitted_head_scene_energy_ratio |
| --- | --- | --- | --- | --- |
| baseline | 1.000 | 1.000 | 1.000 | 1.000 |
| targeted_fixed | 0.822 | 0.830 | 0.907 | 0.989 |
| targeted_learned | 0.746 | 0.783 | 0.876 | 0.988 |
| random_learned_0 | 0.926 | 0.962 | 0.956 | 1.000 |
| random_learned_1 | 0.928 | 0.965 | 0.963 | 1.003 |
| random_learned_2 | 0.922 | 0.943 | 0.855 | 0.973 |

所有比值均相对未门控版本，指标尺度固定。输入与输出表示只计算跨类别共同协变的正对角部分。

同一套门控特征在原分类头与重新拟合分类头下的对比，直接定位了头部重新加权对场景分数响应的影响。它不说明分类头有意学习港口，也不把这些协变分量认定为纯噪声。

## 针对性协调对照

发现分类头重新加权抵消抑制后，追加固定两因素对照：有/无内部门控 × 普通/场景约束分类头。门控沿用原权重，不重新训练；约束沿用前轮 rank=8、tau=0.1，不搜索目标港参数。

| arm | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- |
| baseline__ordinary | 40.752 | 35.771 | 0.000 | 0.000 |
| baseline__scene | 40.634 | 35.162 | -0.118 | -0.609 |
| targeted_learned__ordinary | 41.150 | 35.683 | 0.398 | -0.088 |
| targeted_learned__scene | 40.666 | 35.111 | -0.086 | -0.661 |

| arm | port | OA | BA | delta_BA | rescue | harm |
| --- | --- | --- | --- | --- | --- | --- |
| baseline__ordinary | Santos | 51.845 | 45.064 | 0.000 | 0 | 0 |
| baseline__ordinary | Port Klang | 33.431 | 27.563 | 0.000 | 0 | 0 |
| baseline__ordinary | Los Angeles | 36.979 | 34.686 | 0.000 | 0 | 0 |
| baseline__scene | Santos | 52.390 | 43.581 | -1.484 | 36 | 27 |
| baseline__scene | Port Klang | 33.401 | 27.581 | 0.018 | 49 | 50 |
| baseline__scene | Los Angeles | 36.111 | 34.325 | -0.361 | 6 | 11 |
| targeted_learned__ordinary | Santos | 51.966 | 44.135 | -0.929 | 25 | 23 |
| targeted_learned__ordinary | Port Klang | 33.810 | 27.748 | 0.185 | 49 | 36 |
| targeted_learned__ordinary | Los Angeles | 37.674 | 35.166 | 0.479 | 10 | 6 |
| targeted_learned__scene | Santos | 52.571 | 43.133 | -1.931 | 48 | 36 |
| targeted_learned__scene | Port Klang | 33.489 | 27.596 | 0.033 | 68 | 66 |
| targeted_learned__scene | Los Angeles | 35.938 | 34.603 | -0.083 | 7 | 13 |

| arm | source_scene_energy | source_BA |
| --- | --- | --- |
| baseline__ordinary | 0.111 | 59.317 |
| baseline__scene | 0.059 | 59.120 |
| targeted_learned__ordinary | 0.110 | 58.578 |
| targeted_learned__scene | 0.058 | 58.629 |

这是针对已观察问题的研发纠错，不是独立确认。必须与仅约束分类头比较，不能把头部自身收益归给门控。
