# 任务训练后 SSL-R18 单独读出：固定 CPU 对照

日期：2026-10-10。脚本：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r18_readout.py`。

本控制只使用已有合法 source/target 特征，拟合源端分类头和原五组目标支持残差。没有新的神经训练、特征提取、GPU 运算或超参数选择。原任务训练、主报告、记忆、cases 与 crossover 脚本不在写入范围内。实际完成范围以本目录 `RUN_STATUS.json`、最终 `COMPLETE.json` 为准。

## 四个预先固定的臂

| mode | 输入与读出 |
| --- | --- |
| `r50_frozen_c4` | 已缓存冻结 C4-R50 2048维，L2单位化后乘 sqrt32，加常数1 |
| `r18_task_trained` | 各任务原有训练后 SSL-R18 512维，L2单位化后乘 sqrt32，加常数1 |
| `concat_shared_head` | 拼接两块各自单位化特征，统一乘 sqrt16，加一个常数1，共享一个分类头 |
| `equal_single_softmax` | 前两个单路经64支持适配后的 logits 分别做 softmax，概率等权平均；没有新的拟合 |

三个源头分别在同一组合法源行及其类别标签上拟合 balanced ridge，alpha固定1，偏置不惩罚。目标保持同5组64标签，balanced residual ridge的lambda固定8；目标支持权重只按该draw实际出现的类别计算，支持缺类仍可由源先验提供该类分数。

`current_dual_reference` 直接来自 `backbone_ablation_20261010/targets/{dataset}/{task}/predictions.npz` 的 `dual_refit_late`。它是完整旧双路的固定参考，不是为四个臂新训练、也不是相同紧凑读出的架构干预。

## 等价算子与内存口径

令单位化特征为 U、V。源求解改在单位坐标进行：原缩放特征上的系数L2惩罚alpha1，等价于U/V单位坐标系数的惩罚1/32；联合臂为1/16。偏置惩罚始终为0。一次分块累计 `[U,V,1]` 的加权Gram/RHS，取相应子矩阵后分别解三个头，联合头不是两个单路头的平均。

目标支持核严格为：

```text
R50:   32 * U_query @ U_support.T + 1
R18:   32 * V_query @ V_support.T + 1
Concat:16 * (U_query @ U_support.T + V_query @ V_support.T) + 1
```

复用 `unified_head_core.coefficients` 计算平衡残差。支持阶段的常数坐标与原 `aligned_residual_ops.residual_scores` 一样参与残差正则化，不把源头的无惩罚偏置规则误用到支持修正。

单位化使用原式 float32 `F.normalize(...,eps=1e-12)`，随后的尺度、源统计量和支持求解使用float64。这与显式缩放特征的数学算子等价，但不声称与旧float32读出逐位相同。小型对照中：源头显式特征与结构化求解最大差约7.22e-16，残差与原函数差约2.89e-15；与原 `fit_source_prior/geometry` 的float32缩放对照先验差为2.72e-8。见 `SELF_TEST.json`。

另有用户指定的重复分支小矩阵检验：B=A，两份单位特征分别乘sqrt16并仅加一次常数，检查联合核、alpha1源先验和lambda8支持岭分类均还原A单路，且分类预测完全相同。它只检验实现的归一化/正则化/常数坐标是否正确，不把增加特征维度本身当作互补证据；没有跑真实25任务的重复网络。

每批512行，只在块内转float64；不生成F全量float64特征。F的R50/R18特征和输出scores/priors以mmap流式读写。源Gram为固定2561x2561，约50MiB；OSS2源R18压缩缓存只展开已有36996x512 float32子集。源头拟合、目标核和NumPy/SciPy/Torch均限制两线程，无模型前向。

## 合法索引与谱系

- LOPO源R18的 `source_features/{port}/features.npy` 已按同目录 `rows.npy` 排列，禁止再次用全局行号索引。目标 `matched_joint/joint_extra.npy` 按该任务目标行排列。
- F源 `joint_residual_pilot_20261008/joint/source_extra.npy` 是全38091行矩阵，必须先取该任务合法29717条source rows；目标为123489行。
- OSS2源来自 `compact_fusion_20261010/cache/oss2_source_extra.npz`，已是36996条合法源行子集；目标3745行。
- R50源为 `c4_base_trial_20261009/pooled_c4.npy`，外部目标为 `cross_dataset_c4_20261009/{dataset}/pooled_c4.npy`；外部sample_ids与目标设计严格对齐。
- 每个任务的 `INPUTS.json` 保存全部实际路径、矩阵形状、允许源行、任务训练后layer4谱系以及既有双路参考。原ID、support、support_labels、eval_masks均保留并检查，查询标签只用于固定预测落盘后的评价。

R18是在原双路任务上下文中训练的，已经见过内层源标签；本轮没有新增内层独立验证，也不把它包装成独立优化到最佳的R18单骨干。不同预训练、输入和训练经历未被隔离；F历史无标签SSL暴露仍适用。

**R50控制口径不能混同：**本轮R50复用历史 `pooled_c4.npy`；主线程任务训练的frozen control由当前bf16网络从像素重新提取。既有源初始化探针曾测到约0.004的relative geometry差异，本轮不假定两者逐位相同，不把该探针扩大为全源/全目标差异估计，也不直接归因为舍入。同缓存三种特征读出及其等权融合在本轮内部可比较，但若数字不同于task脚本的frozen control，不强制改预测，也不称完全同一控制。

**输入可观测性限制：**主代理本轮代码/权重核查确认，当前及历史R50提取器的 `conv1=meanRGB repeat2*1.5` 使VV/VH权重相等，prefix冻结。正式凭据 [OBSERVABILITY.json](E:/Hermes/paper_tracking/sensitivity/strong_single_polarization_20261010/OBSERVABILITY.json) 记录float64旧反对称响应max为1.0246394606700237e-14、新响应RMS为1.7404581770394036、共同模响应差max为6.375879291198316e-7，总参数均为23521288；此前3.6e-7仅为float32初探，以落盘审计为准。在编码器输入坐标上只保留两通道之和，R18则输入VH。这支持输入信息互补这一候选解释，不证明其性能贡献或双骨干必需性。本轮没有重测前向探针；性能归因等待主线同参数量、同前向次数的差分保留单路控制（源目标重新提取并同读出refit）。本轮不扩任务，原预测和指标保持不变。

主线在读取本轮全量最终指标前预先固定最终explain的已测单路集合，纳入R18-only。对接时**只取 `r18_task_trained`**；`concat_shared_head` 与 `equal_single_softmax` 都需要两个骨干，不进入该单路集合。本轮没有修改旧 `strong_single_multidata_explain.py`。

## 统计与后续诊断接口

每个task/draw输出OA、BA、逐类召回以及配对rescue/harm。BA仅平均当前查询子集实际出现的类别；全数据集先平均任务内五draw，再等权平均任务。23港不是按样本量池化，五draw也不是神经种子。

固定配对为：两个combined臂分别对每个单路；四个臂各对currentdual；R18单路对R50单路。rescue始终表示candidate正确而reference错误，harm为相反情形。输出不只保留正向救回，不按目标结果挑主方案。

每任务 `targets/{dataset}/{task}/`：

- `predictions.npz`：`predictions` `[5,4,N]`，`modes`，`margins` `[5,4,N]`，原 `source_rows/rows/ids/support/support_labels/eval_masks`，以及单独的 `current_dual_predictions` `[5,N]`。
- `score_units`：前三臂为 `logits`，等权融合为 `softmax_probability`。margin同样继承各臂原分数单位，不跨单位混合置信档。
- `scores.npy`：float64 `[5,4,N,8]`，与四个modes顺序完全对应；可mmap，预测直接由这些原始分数argmax得到。
- `priors.npy`：float64 `[3,N,8]`，对应三个独立源头。
- `source_heads.npz`：各头的单位坐标系数，末行为源偏置，附合法source_rows、源类别数与缩放记录。
- `support_coefficients.npz`：每个draw/三种核的64x8残差系数，便于后续重建先验/支持关系贡献。
- `PREDICTED.json`：固定参数、时长、设备、精度与“预测先于查询评价”记录。

根目录：`summary.csv`、`task_means.csv`、`metrics_by_task_draw.csv`、`class_recall.csv`、`rescue_harm.csv`、`class_rescue_harm.csv`、`task_effects.csv`、`effect_summary.csv`、自动 `FINDINGS.md`。失败写为独立 `FAILURE_时间戳.json`，不删除失败或修改参数争取正结果。

## 调用

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' -B 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r18_readout.py' --stage self-test
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' -B 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r18_readout.py' --stage check --tasks Mombasa f0921
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' -B 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r18_readout.py' --stage run
```

`--stage run` 默认全部25任务，已成功落盘且协议一致的预测直接复用；`--stage report` 仅重新汇总既有预测。`--tasks`可限定任务名或`dataset/task`；报告明确实际覆盖，不把部分运行写成25任务已完成。运行环境名称含CUDA，但该脚本始终只在CPU运算。
