# 保留通道的观测条件响应校正

## 结论先行

主检验版本为 condition_gain__refit。三港等权平均 OA 改变 -2.589 个百分点，BA 改变 -1.956 个百分点。不能依据表中目标港结果逐港挑选不同版本。

本轮未获得正增益，不并入主方案。不是实现没有接入：原始预测复现、条件缺失回退和空间排列保持均通过检查。核心限制是当前同船拟合得到的统一条件响应关系跨港迁移弱；重拟合分类头与源域强度校准均未使其超过基线。这里仅否定本轮固定条件函数，不否定场景因素存在，也不要求改用船型专用规则。

本轮没有按船型、类别混淆或目标港设置规则。候选输入为入射角、升降轨、卫星平台；同一套系数用于全部船型与港口。平台在同船同港重复组内没有变化，无法单独识别平台效应，其系数实际为零。有效校正来自入射角及升降轨项。船型标签仅用于训练识别头和评价。

## 方法

从排除三个目标港及其船舶身份后的源域中，使用 7460 次重复观测、2527 艘船、13 个港。每个同船同港组内去均值，拟合观测条件差异与通道对数平均响应差异的回归。组内固定的船舶结构差异不进入回归，但随时间变化的装载、背景等未测因素仍可能混杂，不能直接解释为因果效应。

公式：log(mu_ic) = vessel_port_offset_c + phi(condition_i) B_c + residual_ic。

推理：gain_ic = clip(exp(-0.5 * (phi(condition_i)-source_reference) B_c), 0.5, 2)。

h_prime_icuv = gain_ic * h_icuv。所有通道保留，既可减弱也可增强，通道内相对空间排列不变。它校正的是条件相关幅度，不是从每个神经元中分离出了纯噪声。缺失条件时 gain=1。随后继续经过原始非线性残差块。

## 对照与范围

- baseline：不处理；固定头与重新拟合头都精确复现前轮原始查询预测。
- sample_mean_shrink：不区分条件来源，统一把样本通道均值往源域均值收缩；同样只处理条件已知样本。
- condition_gain：本轮条件相关响应校正，统一固定强度0.5。
- permuted_channel_gain：固定一次通道对应关系置乱；每个样本的校正幅度集合与正确版本完全相同，检查对应关系是否重要。不是新增跨域数据集置乱任务。
- 每个表示同时比较原分类头和源域重新拟合分类头，不增加目标标注或目标条件统计。
- 分类头沿用相同939个源域标签；响应估计额外利用重复船舶身份和观测元数据。这是新增信息，不能称为完全同信息量的架构比较。
- 三港共同留出；历史预训练曝光保留。不是完整主模型的64标签结果或独立确认集。

## 源域关系是否可迁移

| split | fold | train | test | prediction_R2 | half_correction_variation_reduction | positive_channels |
| --- | --- | --- | --- | --- | --- | --- |
| identity | 0 | 4871 | 2589 | 0.008 | 0.013 | 295 |
| identity | 1 | 5080 | 2380 | 0.013 | 0.014 | 349 |
| identity | 2 | 4969 | 2491 | 0.016 | 0.014 | 351 |
| port | 0 | 1313 | 6088 | -0.007 | 0.001 | 186 |
| port | 1 | 6271 | 1150 | -0.042 | -0.003 | 147 |
| port | 2 | 7230 | 222 | -0.006 | 0.004 | 243 |

prediction_R2 相对“不预测任何同船变化”；variation_reduction 测量固定半强度校正后同船变化的减少。正负结果均保留。船舶留出与港口留出回答不同问题，不能用前者代替后者。R2不是分类准确率，也不是已解释的全部物理噪声比例。

## 三港等权指标

| arm | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- |
| baseline__fixed | 40.752 | 35.771 | 0.000 | 0.000 |
| baseline__refit | 40.752 | 35.771 | 0.000 | 0.000 |
| sample_mean_shrink__fixed | 39.673 | 32.448 | -1.078 | -3.323 |
| sample_mean_shrink__refit | 37.189 | 33.470 | -3.563 | -2.302 |
| condition_gain__fixed | 38.621 | 33.030 | -2.131 | -2.742 |
| condition_gain__refit | 38.162 | 33.815 | -2.589 | -1.956 |
| permuted_channel_gain__fixed | 38.641 | 35.095 | -2.111 | -0.677 |
| permuted_channel_gain__refit | 38.077 | 34.395 | -2.674 | -1.376 |

## 逐港结果

| arm | port | n | OA | BA | delta_OA | delta_BA | rescue | harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline__fixed | Santos | 1653 | 51.845 | 45.064 | 0.000 | 0.000 | 0 | 0 |
| baseline__fixed | Port Klang | 3425 | 33.431 | 27.563 | 0.000 | 0.000 | 0 | 0 |
| baseline__fixed | Los Angeles | 576 | 36.979 | 34.686 | 0.000 | 0.000 | 0 | 0 |
| baseline__refit | Santos | 1653 | 51.845 | 45.064 | 0.000 | 0.000 | 0 | 0 |
| baseline__refit | Port Klang | 3425 | 33.431 | 27.563 | 0.000 | 0.000 | 0 | 0 |
| baseline__refit | Los Angeles | 576 | 36.979 | 34.686 | 0.000 | 0.000 | 0 | 0 |
| sample_mean_shrink__fixed | Santos | 1653 | 52.269 | 40.080 | 0.423 | -4.984 | 108 | 101 |
| sample_mean_shrink__fixed | Port Klang | 3425 | 32.029 | 25.081 | -1.401 | -2.482 | 97 | 145 |
| sample_mean_shrink__fixed | Los Angeles | 576 | 34.722 | 32.182 | -2.257 | -2.504 | 21 | 34 |
| sample_mean_shrink__refit | Santos | 1653 | 46.763 | 40.326 | -5.082 | -4.738 | 100 | 184 |
| sample_mean_shrink__refit | Port Klang | 3425 | 30.949 | 26.907 | -2.482 | -0.656 | 143 | 228 |
| sample_mean_shrink__refit | Los Angeles | 576 | 33.854 | 33.176 | -3.125 | -1.510 | 23 | 41 |
| condition_gain__fixed | Santos | 1653 | 50.212 | 40.294 | -1.633 | -4.770 | 96 | 123 |
| condition_gain__fixed | Port Klang | 3425 | 31.796 | 26.339 | -1.635 | -1.224 | 59 | 115 |
| condition_gain__fixed | Los Angeles | 576 | 33.854 | 32.455 | -3.125 | -2.231 | 7 | 25 |
| condition_gain__refit | Santos | 1653 | 49.970 | 43.015 | -1.875 | -2.049 | 105 | 136 |
| condition_gain__refit | Port Klang | 3425 | 31.358 | 26.114 | -2.073 | -1.449 | 106 | 177 |
| condition_gain__refit | Los Angeles | 576 | 33.160 | 32.316 | -3.819 | -2.370 | 12 | 34 |
| permuted_channel_gain__fixed | Santos | 1653 | 45.917 | 44.685 | -5.929 | -0.379 | 41 | 139 |
| permuted_channel_gain__fixed | Port Klang | 3425 | 33.547 | 27.112 | 0.117 | -0.451 | 61 | 57 |
| permuted_channel_gain__fixed | Los Angeles | 576 | 36.458 | 33.487 | -0.521 | -1.199 | 10 | 13 |
| permuted_channel_gain__refit | Santos | 1653 | 45.856 | 42.194 | -5.989 | -2.871 | 56 | 155 |
| permuted_channel_gain__refit | Port Klang | 3425 | 32.438 | 26.582 | -0.993 | -0.982 | 107 | 141 |
| permuted_channel_gain__refit | Los Angeles | 576 | 35.938 | 34.411 | -1.042 | -0.276 | 19 | 25 |

## 校正覆盖与强度

| port | n | available | mean_abs_log_gain | gain_q01 | gain_q99 | capped_fraction |
| --- | --- | --- | --- | --- | --- | --- |
| source | 939 | 378 | 0.064 | 0.637 | 1.548 | 0.004 |
| Santos | 1653 | 1162 | 0.131 | 0.607 | 1.795 | 0.007 |
| Port Klang | 3425 | 2021 | 0.106 | 0.572 | 1.670 | 0.006 |
| Los Angeles | 576 | 292 | 0.085 | 0.608 | 1.646 | 0.005 |

## 决策敏感性

| arm | source_scene_score_energy | source_BA |
| --- | --- | --- |
| baseline__fixed | 0.111 | 59.317 |
| baseline__refit | 0.111 | 59.317 |
| sample_mean_shrink__fixed | 0.088 | 55.598 |
| sample_mean_shrink__refit | 0.119 | 58.446 |
| condition_gain__fixed | 0.117 | 57.644 |
| condition_gain__refit | 0.110 | 59.439 |
| permuted_channel_gain__fixed | 0.113 | 58.329 |
| permuted_channel_gain__refit | 0.108 | 61.028 |

## 如何判读失败

先看接入、预测复现和校正强度，再区分两种情况：一是观测条件与响应的关系本身在换港后失效；二是关系能迁移，但被校正的变化并非全是分类有害因素。若只在同港不同船上可预测，不应宣称已找到跨港不变量。

原分类头与重拟合头的差异用于定位训练协调问题；样本均值收缩和通道错配对照用于判断收益是否来自普遍的幅度扰动。逐类结果保存在 by_class.csv，仅用于诊断，不用于制定船型专用规则。

总耗时 91.4 秒，CPU单线程，未使用或中断主线程GPU任务。

## 条件缺失与训练协调

| arm | metadata_known | n | predictions_changed | rescue | harm |
| --- | --- | --- | --- | --- | --- |
| condition_gain__fixed | False | 2179 | 0 | 0 | 0 |
| condition_gain__fixed | True | 3475 | 787 | 162 | 263 |
| condition_gain__refit | False | 2179 | 195 | 60 | 46 |
| condition_gain__refit | True | 3475 | 1028 | 163 | 301 |

固定原分类头时，缺失条件的样本预测完全不变。重新拟合头后，这些样本也可能变化，这是共享分类头更新的影响，不是给缺失条件填了虚构校正值。

## 源港校准与负结果定位

原港口交叉验证三个验证折样本量为6088、1150、222。为检查是否仅由划分失衡造成，追加按港口样本量贪心平衡的三折；原结果完整保留。没有用目标港结果选择划分。

| fold | train | test | prediction_R2 | half_reduction |
| --- | --- | --- | --- | --- |
| 0 | 4139 | 3257 | 0.004 | 0.003 |
| 1 | 5304 | 2097 | -0.007 | 0.001 |
| 2 | 5252 | 2106 | -0.040 | -0.006 |

| port | n | vessels | prediction_R2 |
| --- | --- | --- | --- |
| Ningbo-Zhoushan | 3257 | 1111 | 0.004 |
| Busan | 421 | 152 | -0.008 |
| Jebel Ali | 1304 | 401 | -0.002 |
| Newcastle | 10 | 5 | -0.012 |
| Port Said | 251 | 85 | 0.000 |
| Qingdao | 102 | 46 | -0.000 |
| Richards Bay | 9 | 4 | -0.020 |
| Callao | 74 | 24 | -0.072 |
| Fujairah | 268 | 83 | -0.033 |
| New York | 210 | 54 | -0.064 |
| Shanghai | 565 | 228 | 0.001 |
| Singapore | 952 | 359 | -0.019 |
| Sydney Botany | 37 | 17 | -0.045 |

平衡后源港OOF整体 R2=-0.02306。用这些OOF预测拟合一个全船型、全港共享的响应强度系数 k=0.17292，不做参数搜索，不按类别、通道或目标港设置不同系数。此校准使用了OOF数据，校准后的OOF拟合误差不作为独立验证。

| head | OA | BA | delta_OA | delta_BA |
| --- | --- | --- | --- | --- |
| fixed | 40.613 | 34.994 | -0.138 | -0.777 |
| refit | 40.287 | 34.717 | -0.465 | -1.054 |

| head | port | OA | BA | delta_BA | rescue | harm |
| --- | --- | --- | --- | --- | --- | --- |
| fixed | Santos | 52.329 | 43.768 | -1.296 | 26 | 18 |
| fixed | Port Klang | 33.226 | 27.306 | -0.258 | 13 | 20 |
| fixed | Los Angeles | 36.285 | 33.910 | -0.777 | 3 | 7 |
| refit | Santos | 52.511 | 43.599 | -1.466 | 31 | 20 |
| refit | Port Klang | 32.759 | 26.959 | -0.604 | 24 | 47 |
| refit | Los Angeles | 35.590 | 33.595 | -1.091 | 2 | 10 |

校准减轻损失，但未带来超过原模型的平均BA。跨港预测弱也出现在Busan、Fujairah、New York、Singapore等样本相对较多的源港，不能全部归于少数小港权重。平台组内方差与平台系数均为零，不把平台列的存在写成已学得平台校正机制。

可支持的结论是：当前全局入射角/轨道到通道幅度的固定回归式不足以可靠描述跨港响应变化。未测观测条件、船体朝向与场景交互、回归形式不足等仍是待区分解释；不能据此直接归因于某种具体物理噪声。
