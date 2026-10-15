# 少量目标标签监督内部响应适配

## 本轮回答的问题

不是继续删去场景敏感通道，而是允许同一适配程序用目标支持标签学习如何增强、减弱或偏移内部响应。必须分别判断：目标标签本身是否有用；内部适配是否超过只适配分类头；源域场景信息是否超过普通内部适配。

## 实验设置

- Santos、Port Klang、Los Angeles三个目标港，各3个固定支持抽样，每次64艘船各取一片；同一组标签被分类头与适配器共用，不增加预算。
- 所有版本共用同一支持/查询划分。查询排除支持船舶的所有重复观测；支持与查询可能来自同一大景，因此不是目标域跨大景独立验证。
- 骨干固定，内部适配器位于R18最后一个残差块之前。普通/场景引导适配均1024个参数，均从不改变网络的状态初始化，训练24步。
- 场景引导只改变全类别共享的参数调整先验，先验强度均值与普通版本相同。没有船型专用门控，也不按目标港结果选择配置；各目标港通过相同程序从自己的64个支持标签学习适配参数。
- 识别头保留源域八类先验；残差头使用同一支持集的类别平衡权重。支持集中未出现类不额外拟合负残差，其分数保留当前表征下的源域先验。
- 支持留一损失、源域分数保持和初始状态约束一起训练；留一计算固定完整支持集的权重与出现类别掩码，不逐次重估，且用于优化，所以不是独立验证成绩。
- 使用相同939个源域标签的算子级小试，不是当前完整双路模型，也不是所有港口或外部数据集成绩。旧预训练曝光及开发集重复使用的边界保留。

## 总体结果

| arm | OA | BA | delta_OA_vs_source | delta_BA_vs_source | delta_OA_vs_head | delta_BA_vs_head | delta_BA_vs_ordinary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| source_only | 40.309 | 35.348 | 0.000 | 0.000 | 3.114 | -2.956 | -2.729 |
| head_only | 37.195 | 38.304 | -3.114 | 2.956 | 0.000 | 0.000 | 0.227 |
| ordinary_adapter | 38.643 | 38.077 | -1.666 | 2.729 | 1.448 | -0.227 | 0.000 |
| scene_guided_adapter | 38.640 | 38.112 | -1.669 | 2.764 | 1.445 | -0.192 | 0.035 |

百分数与百分点。先对每港3次支持抽样取平均，再对3港等权；这些抽样共享查询人群，不是9个独立数据集，也不是3个神经预训练种子。

## 逐港结果

| port | arm | OA | BA | delta_BA_vs_head | delta_BA_vs_ordinary |
| --- | --- | --- | --- | --- | --- |
| Santos | source_only | 51.655 | 44.617 | 0.814 | 3.518 |
| Santos | head_only | 37.886 | 43.803 | 0.000 | 2.705 |
| Santos | ordinary_adapter | 36.208 | 41.099 | -2.705 | 0.000 |
| Santos | scene_guided_adapter | 36.122 | 41.035 | -2.768 | -0.064 |
| Port Klang | source_only | 33.574 | 27.646 | -4.787 | -7.743 |
| Port Klang | head_only | 33.295 | 32.434 | 0.000 | -2.956 |
| Port Klang | ordinary_adapter | 37.433 | 35.389 | 2.956 | 0.000 |
| Port Klang | scene_guided_adapter | 37.363 | 35.388 | 2.954 | -0.001 |
| Los Angeles | source_only | 35.699 | 33.780 | -4.896 | -3.962 |
| Los Angeles | head_only | 40.404 | 38.676 | 0.000 | 0.934 |
| Los Angeles | ordinary_adapter | 42.287 | 37.742 | -0.934 | 0.000 |
| Los Angeles | scene_guided_adapter | 42.435 | 37.913 | -0.763 | 0.171 |

## 抽样稳定性

| arm | tasks | positive_vs_head | negative_vs_head | positive_vs_ordinary | negative_vs_ordinary |
| --- | --- | --- | --- | --- | --- |
| source_only | 9 | 2 | 7 | 3 | 6 |
| head_only | 9 | 0 | 0 | 6 | 3 |
| ordinary_adapter | 9 | 3 | 6 | 0 | 0 |
| scene_guided_adapter | 9 | 3 | 6 | 5 | 3 |

| port | draw | support | query | support_classes | query_classes | removed_same_vessel |
| --- | --- | --- | --- | --- | --- | --- |
| Santos | 0 | 64 | 1554 | 6 | 8 | 35 |
| Santos | 1 | 64 | 1551 | 7 | 8 | 38 |
| Santos | 2 | 64 | 1549 | 8 | 8 | 40 |
| Port Klang | 0 | 64 | 3322 | 7 | 8 | 39 |
| Port Klang | 1 | 64 | 3330 | 7 | 8 | 31 |
| Port Klang | 2 | 64 | 3326 | 8 | 8 | 35 |
| Los Angeles | 0 | 64 | 434 | 8 | 8 | 78 |
| Los Angeles | 1 | 64 | 472 | 8 | 8 | 40 |
| Los Angeles | 2 | 64 | 467 | 8 | 8 | 45 |

## 场景信息是否真的改变了适配

| arm | parameter_rms | guided_top64_update_fraction | support_LOO_BA |
| --- | --- | --- | --- |
| ordinary_adapter | 0.155 | 0.123 | 48.395 |
| scene_guided_adapter | 0.156 | 0.141 | 48.395 |

| task | relative_parameter_difference | parameter_cosine | prediction_disagreements |
| --- | --- | --- | --- |
| Los_Angeles_draw0_models | 0.129 | 0.992 | 1 |
| Los_Angeles_draw1_models | 0.148 | 0.989 | 1 |
| Los_Angeles_draw2_models | 0.148 | 0.989 | 0 |
| Port_Klang_draw0_models | 0.134 | 0.991 | 15 |
| Port_Klang_draw1_models | 0.129 | 0.992 | 16 |
| Port_Klang_draw2_models | 0.123 | 0.993 | 10 |
| Santos_draw0_models | 0.153 | 0.989 | 9 |
| Santos_draw1_models | 0.137 | 0.991 | 4 |
| Santos_draw2_models | 0.140 | 0.990 | 3 |

先验修改了更新位置，不等于额外提高了查询准确率。若与普通版本参数或预测接近，应报告引导作用有限，不能把普通适配的全部收益归给场景机制。

## 训练是否在工作

| task | arm | initial_loss | target_loo_loss | relative_training_loss_drop | source_preservation_loss | parameter_rms |
| --- | --- | --- | --- | --- | --- | --- |
| Santos_draw0 | ordinary_adapter | 0.441 | 0.350 | 0.207 | 0.023 | 0.136 |
| Santos_draw0 | scene_guided_adapter | 0.441 | 0.350 | 0.207 | 0.023 | 0.138 |
| Santos_draw1 | ordinary_adapter | 0.358 | 0.270 | 0.247 | 0.020 | 0.150 |
| Santos_draw1 | scene_guided_adapter | 0.358 | 0.270 | 0.247 | 0.020 | 0.151 |
| Santos_draw2 | ordinary_adapter | 0.374 | 0.284 | 0.240 | 0.021 | 0.151 |
| Santos_draw2 | scene_guided_adapter | 0.374 | 0.284 | 0.240 | 0.021 | 0.153 |
| Port_Klang_draw0 | ordinary_adapter | 0.410 | 0.318 | 0.225 | 0.018 | 0.163 |
| Port_Klang_draw0 | scene_guided_adapter | 0.410 | 0.318 | 0.224 | 0.018 | 0.164 |
| Port_Klang_draw1 | ordinary_adapter | 0.429 | 0.306 | 0.287 | 0.041 | 0.156 |
| Port_Klang_draw1 | scene_guided_adapter | 0.429 | 0.306 | 0.286 | 0.041 | 0.156 |
| Port_Klang_draw2 | ordinary_adapter | 0.455 | 0.335 | 0.264 | 0.020 | 0.159 |
| Port_Klang_draw2 | scene_guided_adapter | 0.455 | 0.335 | 0.264 | 0.020 | 0.161 |
| Los_Angeles_draw0 | ordinary_adapter | 0.425 | 0.336 | 0.209 | 0.016 | 0.158 |
| Los_Angeles_draw0 | scene_guided_adapter | 0.425 | 0.336 | 0.209 | 0.016 | 0.159 |
| Los_Angeles_draw1 | ordinary_adapter | 0.400 | 0.322 | 0.194 | 0.019 | 0.161 |
| Los_Angeles_draw1 | scene_guided_adapter | 0.400 | 0.322 | 0.194 | 0.019 | 0.163 |
| Los_Angeles_draw2 | ordinary_adapter | 0.408 | 0.327 | 0.199 | 0.012 | 0.159 |
| Los_Angeles_draw2 | scene_guided_adapter | 0.408 | 0.327 | 0.199 | 0.013 | 0.161 |

训练损失下降而查询下降时，应区分少样本过拟合、表示/读出目标不一致和引导先验不可靠，不凭单个负结果判定目标适配不可能。

## 已定位并修正的头部问题

首版采用不加权残差头，且把支持集缺类也作为负证据更新，与类别平衡训练目标和闭集源域先验不协调。在完成Santos三个支持抽样后中止，原结果保留于 target_response_adapter_20261010；Port Klang首个任务不完整，不计入结果。

修正版独立输出，统一对所有对照应用类别平衡与缺类先验保留；支持集、标签预算、训练步数和其他超参数不变。加权留一公式已与显式逐样本删除拟合核对，缺类残差系数严格为零。

本轮完整完成 9 个任务、18 次内部适配。CPU单线程耗时 324.0 秒，未修改主线程模型或使用其GPU。

## 同特征同标签的头部纠错

| port | draw | OA_legacy | OA_balanced | BA_legacy | BA_balanced |
| --- | --- | --- | --- | --- | --- |
| Santos | 0 | 58.687 | 33.462 | 31.910 | 43.185 |
| Santos | 1 | 55.835 | 38.169 | 28.537 | 42.392 |
| Santos | 2 | 58.554 | 42.027 | 32.100 | 45.833 |

## 标签覆盖与误伤

| arm | coverage_group | mean_recall | delta_vs_head | task_groups |
| --- | --- | --- | --- | --- |
| head_only | absent | 33.503 | 0.000 | 4 |
| head_only | multiple | 36.869 | 0.000 | 9 |
| head_only | singleton | 38.375 | 0.000 | 5 |
| ordinary_adapter | absent | 21.723 | -11.780 | 4 |
| ordinary_adapter | multiple | 37.945 | 1.076 | 9 |
| ordinary_adapter | singleton | 38.527 | 0.152 | 5 |
| scene_guided_adapter | absent | 21.723 | -11.780 | 4 |
| scene_guided_adapter | multiple | 37.965 | 1.096 | 9 |
| scene_guided_adapter | singleton | 38.527 | 0.152 | 5 |
| source_only | absent | 37.479 | 3.976 | 4 |
| source_only | multiple | 33.165 | -3.704 | 9 |
| source_only | singleton | 41.560 | 3.185 | 5 |

先在各任务的标签覆盖组内平均类别召回，再平均可用任务组；不是总体BA的直接贡献，也不是独立样本估计。支持缺类组仅4个任务组。

支持未覆盖类的平均召回由33.503降至21.723，下降11.780个百分点。虽然缺类残差系数保持为零，内部适配仍改变其源分类头的输入；冻结权重不等于冻结判别函数。其他类别分数改变也会影响竞争结果。

## 逐样本分数分解

| port | draw | kind | instance_id | true_class | fixed_rival | old_margin | new_margin | source_prior_term | feature_with_old_head_term | support_head_refit_term |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Santos | 0 | rescue | pair_0067789 | 0 | 3 | -0.006 | 0.034 | -0.113 | 0.048 | 0.106 |
| Santos | 0 | rescue | returned_20260915_0002846 | 0 | 5 | -0.187 | 0.380 | 0.483 | 0.008 | 0.076 |
| Santos | 0 | rescue | returned_20260915_0003147 | 0 | 5 | -0.052 | 0.668 | 0.653 | 0.018 | 0.049 |
| Santos | 0 | rescue | returned_20260915_0002980 | 0 | 4 | -0.126 | 0.016 | 0.027 | 0.108 | 0.006 |
| Santos | 0 | harm | returned_20260915_0003104 | 0 | 4 | 0.065 | -0.035 | -0.198 | 0.084 | 0.015 |
| Santos | 0 | harm | returned_local_knowledge_0014716 | 4 | 1 | 0.650 | -0.077 | -0.717 | -0.054 | 0.045 |
| Santos | 0 | harm | returned_20260915_0003169 | 0 | 6 | 0.129 | -0.142 | -0.410 | 0.067 | 0.073 |
| Santos | 0 | harm | returned_local_knowledge_0014387 | 5 | 4 | 0.201 | -0.379 | -0.634 | 0.075 | -0.021 |
| Port Klang | 0 | rescue | pair_0052380 | 0 | 2 | -0.007 | 0.005 | 0.005 | 0.041 | -0.033 |
| Port Klang | 0 | rescue | returned_local_knowledge_0009910 | 4 | 3 | -0.006 | 0.003 | -0.007 | -0.024 | 0.040 |
| Port Klang | 0 | rescue | pair_0052382 | 0 | 3 | -0.010 | 0.079 | 0.143 | 0.013 | -0.066 |
| Port Klang | 0 | rescue | returned_20260915_0001738 | 0 | 5 | -0.035 | 0.075 | 0.006 | 0.074 | 0.030 |
| Port Klang | 0 | harm | returned_local_knowledge_0008568 | 4 | 6 | 0.044 | -0.351 | -0.373 | 0.003 | -0.026 |
| Port Klang | 0 | harm | returned_local_knowledge_0008000 | 4 | 0 | 0.056 | -0.006 | -0.085 | -0.071 | 0.094 |
| Port Klang | 0 | harm | pair_0052388 | 0 | 2 | 0.003 | -0.053 | -0.142 | 0.166 | -0.081 |
| Port Klang | 0 | harm | returned_20260915_0001586 | 0 | 4 | 0.026 | -0.010 | 0.002 | 0.025 | -0.063 |
| Los Angeles | 0 | rescue | returned_local_knowledge_0005572 | 5 | 0 | -0.007 | 0.166 | 0.318 | 0.008 | -0.153 |
| Los Angeles | 0 | rescue | returned_local_knowledge_0005650 | 5 | 6 | -0.128 | 0.099 | 0.571 | 0.181 | -0.524 |
| Los Angeles | 0 | rescue | returned_20260915_0000931 | 0 | 3 | -0.030 | 0.183 | 0.152 | -0.014 | 0.076 |
| Los Angeles | 0 | rescue | returned_local_knowledge_0005479 | 5 | 0 | -0.033 | 0.308 | 0.467 | 0.031 | -0.157 |
| Los Angeles | 0 | harm | returned_local_knowledge_0005398 | 3 | 0 | 0.118 | -0.022 | -0.091 | 0.013 | -0.061 |
| Los Angeles | 0 | harm | returned_local_knowledge_0005465 | 3 | 0 | 0.057 | -0.058 | -0.054 | -0.037 | -0.024 |
| Los Angeles | 0 | harm | returned_local_knowledge_0005576 | 5 | 4 | 0.157 | -0.015 | -0.216 | 0.019 | 0.024 |
| Los Angeles | 0 | harm | returned_local_knowledge_0005524 | 4 | 5 | 0.067 | -0.031 | -0.066 | -0.036 | 0.004 |

上述24例是每港第0次抽样中各取前4个纠错与前4个误伤，不代表总体贡献比例。固定竞争类别后，分数间隔变化精确拆为：源先验变化、旧残差头下的特征变化、支持头重拟合。各例预测已回放核对。这是依赖分解顺序的计算归因，不是散射或杂波的物理因果证明。

## 先验分数与支持关系交叉对照

复用全部9个任务的两种已训练适配器，不重新训练网络、不增加标签。第一字母为源分类头先验所用表征，第二字母为支持关系所用表征；F为原冻结表征，A为适配后表征。每格使用相同64个标签重新拟合平衡残差头。FF为仅适配头，AA为完整内部适配。

| adapter | cell | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- | --- |
| ordinary_adapter | AA | 38.643 | 38.077 | 1.448 | -0.227 |
| ordinary_adapter | AF | 38.856 | 38.157 | 1.661 | -0.147 |
| ordinary_adapter | FA | 37.198 | 38.175 | 0.003 | -0.129 |
| ordinary_adapter | FF | 37.195 | 38.304 | 0.000 | 0.000 |
| scene_guided_adapter | AA | 38.640 | 38.112 | 1.445 | -0.192 |
| scene_guided_adapter | AF | 38.872 | 38.174 | 1.677 | -0.130 |
| scene_guided_adapter | FA | 37.235 | 38.234 | 0.040 | -0.070 |
| scene_guided_adapter | FF | 37.195 | 38.304 | 0.000 | 0.000 |

| port | adapter | cell | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- | --- | --- |
| Los Angeles | ordinary_adapter | AA | 42.287 | 37.742 | 1.883 | -0.934 |
| Los Angeles | ordinary_adapter | AF | 41.989 | 37.893 | 1.584 | -0.783 |
| Los Angeles | ordinary_adapter | FA | 40.768 | 38.426 | 0.364 | -0.250 |
| Los Angeles | ordinary_adapter | FF | 40.404 | 38.676 | 0.000 | 0.000 |
| Los Angeles | scene_guided_adapter | AA | 42.435 | 37.913 | 2.031 | -0.763 |
| Los Angeles | scene_guided_adapter | AF | 42.060 | 37.975 | 1.656 | -0.701 |
| Los Angeles | scene_guided_adapter | FA | 40.910 | 38.603 | 0.506 | -0.072 |
| Los Angeles | scene_guided_adapter | FF | 40.404 | 38.676 | 0.000 | 0.000 |
| Port Klang | ordinary_adapter | AA | 37.433 | 35.389 | 4.138 | 2.956 |
| Port Klang | ordinary_adapter | AF | 38.134 | 35.443 | 4.840 | 3.010 |
| Port Klang | ordinary_adapter | FA | 33.135 | 32.290 | -0.160 | -0.144 |
| Port Klang | ordinary_adapter | FF | 33.295 | 32.434 | 0.000 | 0.000 |
| Port Klang | scene_guided_adapter | AA | 37.363 | 35.388 | 4.068 | 2.954 |
| Port Klang | scene_guided_adapter | AF | 38.155 | 35.483 | 4.860 | 3.049 |
| Port Klang | scene_guided_adapter | FA | 33.125 | 32.295 | -0.170 | -0.138 |
| Port Klang | scene_guided_adapter | FF | 33.295 | 32.434 | 0.000 | 0.000 |
| Santos | ordinary_adapter | AA | 36.208 | 41.099 | -1.678 | -2.705 |
| Santos | ordinary_adapter | AF | 36.444 | 41.135 | -1.442 | -2.668 |
| Santos | ordinary_adapter | FA | 37.693 | 43.809 | -0.194 | 0.006 |
| Santos | ordinary_adapter | FF | 37.886 | 43.803 | 0.000 | 0.000 |
| Santos | scene_guided_adapter | AA | 36.122 | 41.035 | -1.764 | -2.768 |
| Santos | scene_guided_adapter | AF | 36.401 | 41.065 | -1.485 | -2.738 |
| Santos | scene_guided_adapter | FA | 37.671 | 43.803 | -0.215 | -0.000 |
| Santos | scene_guided_adapter | FF | 37.886 | 43.803 | 0.000 | 0.000 |

| adapter | cell | coverage_group | recall |
| --- | --- | --- | --- |
| ordinary_adapter | AA | absent | 21.723 |
| ordinary_adapter | AA | multiple | 37.945 |
| ordinary_adapter | AA | singleton | 38.527 |
| ordinary_adapter | AF | absent | 21.723 |
| ordinary_adapter | AF | multiple | 38.157 |
| ordinary_adapter | AF | singleton | 37.160 |
| ordinary_adapter | FA | absent | 33.170 |
| ordinary_adapter | FA | multiple | 36.926 |
| ordinary_adapter | FA | singleton | 37.275 |
| ordinary_adapter | FF | absent | 33.503 |
| ordinary_adapter | FF | multiple | 36.869 |
| ordinary_adapter | FF | singleton | 38.375 |
| scene_guided_adapter | AA | absent | 21.723 |
| scene_guided_adapter | AA | multiple | 37.965 |
| scene_guided_adapter | AA | singleton | 38.527 |
| scene_guided_adapter | AF | absent | 21.723 |
| scene_guided_adapter | AF | multiple | 38.168 |
| scene_guided_adapter | AF | singleton | 37.160 |
| scene_guided_adapter | FA | absent | 33.170 |
| scene_guided_adapter | FA | multiple | 36.985 |
| scene_guided_adapter | FA | singleton | 37.275 |
| scene_guided_adapter | FF | absent | 33.503 |
| scene_guided_adapter | FF | multiple | 36.869 |
| scene_guided_adapter | FF | singleton | 38.375 |

两种适配器在所有任务的FF/AA对角线共36次回放均与原预测一致。FA/AF是对已按AA目标训练模型的诊断交换，不是针对各结构重新训练的公平最终模型对比，也不据此按港挑选部署版本。

场景引导版本在巴生港：AA相对FF的BA提高2.954点，AF提高3.049点，FA下降0.138点。桑托斯港：AA下降2.768点，AF下降2.738点，FA基本不变。洛杉矶：AA下降0.763点，AF下降0.701点，FA下降0.072点。普通适配也呈相同方向。

在这批固定模型中，主要收益与损失随适配后的先验分数路径保留，而不是随支持关系几何变化保留；不代表已证明先验路径单独训练必然最优。

支持缺类组在FF、FA、AF、AA的召回分别为33.503、33.170、21.723、21.723。保留原先验基本恢复缺类表现，但同时丢失巴生港的主要增益，说明不能仅靠冻结该路径同时解决所有问题。

## 本轮收口与下一步

已确认：内部适配确实改变决策；巴生港三次抽样均有BA正收益；其他两港没有同样受益。总体场景引导适配相对仅适配头OA提高1.445点、BA下降0.192点；场景引导相对普通适配的BA仅提高0.035点，尚无有意义的额外收益证据。

已定位：收益/损失主要沿源分类头在新表征上的分数响应传递；支持缺类的原有判别未被有效保护。尚未定位：具体散射因素、真正非语义噪声，以及源域场景敏感排序为何不能有效指导目标更新。标签覆盖分析是事后诊断，不用于制定类别专用规则。

建议下轮只检验一个方向：由源港模拟64标签支持任务，学习全类别共享的更新约束，使少量目标标签可校正分数响应，又能保护支持未覆盖的判别关系。与普通适配保持相同预算和容量；不按目标港结果选择结构或强度。该方案尚未实施，不能写成已有提升。
