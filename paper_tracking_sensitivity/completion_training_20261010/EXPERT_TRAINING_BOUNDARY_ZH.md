# 最新专家训练消融的具体边界

本轮核对日期：2026-10-10。此文件记录代码和既有产物能证明什么，不把“可以实现”写成“已经重训”。

## 本轮能直接完成的部分

25 个任务的分频 `unsplit / fixed / selective` 均已有实际源端训练检查点。每臂固定 6 轮、同一允许源队列、同一 64 支持模拟任务、末轮检查点；本轮重新将它们接入当前基础头、等权双头融合及统一头。它们不是随机初始化分支，也不是对同一检查点删坐标伪装成架构重训。

其中 `selective_uniform` 只是已训练 selective 检查点的推理期等权消融，不是第四种独立训练架构。分频训练损失原本针对源模拟任务的基础头支持残差输出；本轮变更读出不意味着它曾经针对双头融合或统一头的最终损失联合训练。

## 专家仍缺哪一项

历史神经专家训练由 `train_joint_residual.py` 和 `joint_residual_ops.py` 执行：R18 layer4 接收梯度，保留 PCA 核与补空间核进入支持残差求解，然后用源端查询交叉熵训练。历史 aligned-only 和 joint 是真实独立训练臂。

当前 24 港专家在该训练结果之后又加入三种算子：

1. 在 R50 保留坐标和补空间中衰减候选漂移坐标，幅度固定 0.5。
2. 源分类先验显式使用候选稳定坐标，支持核加 `0.25 * stable_kernel`。
3. 源分类先验加入入射角特征，支持核加 `0.125 * (u @ u_support.T)`。

这三项见 `run_protected_drift_activation.py`、`direct_incidence_conditioning.py` 和 `unified_head_expert.py`。当前源先验还在冻结的训练后 R18 特征上重新拟合。历史 `ComplementExpertGeometry` 中没有这些最新项，也没有重拟合后的完整最新源先验。因此，虽然 R50 的稳定、漂移和角度本身不接收神经梯度，它们改变残差求解矩阵与目标残差后，仍可能改变传回 R18 layer4 的梯度，不能声称旧梯度算子与新部署算子完全一致。

具体缺项是：**在相同源模拟任务和相同初始化上，对照旧梯度算子与最新完整梯度算子训练 R18 layer4，然后重新提取特征并用同一目标 64 标签读出。** 不是缺少最新模型的前向公式，也不是所有历史训练四臂都不存在。

两个外部任务的当前冻结主专家仍使用原 retained/complement 配方。单独开展的外部稳定/角度实验，尤其恢复后的 F_0921 入射角版本，没有替换这个主专家；不能将其新结果混入本轮选优。

## 最小可执行补法及本轮状态

最小有意义的下一组是“旧算子继续训练”和“新算子继续训练”两个同源配对臂：同一当前 layer4 初始化、相同六轮支持/查询顺序、学习率、损失和最终轮取值，冻结 Qs/Qd/归一化与源先验，保留原冻结模型作为参考。这样能区别算子变化与仅仅多训练六轮的作用。所有评估仍使用原五组 64 标签，不能用查询真值选择港口、轮数或权重。

这还涉及将最新源先验、R50 漂移处理、稳定/角度核正确接入可微求解，以及训练后目标特征提取。若直接只重训新算子、不配对旧算子继续训练，就无法单独归因于训练/部署一致性。

**本支线没有申请或使用 GPU，没有启动上述专家继续训练，也没有将该项标为已完成。** 用户要求优先一次补齐的 25 任务分频训练架构当前读出已另行执行，其完成状态以 `COMPLETE.json` 为准。完整单骨干训练和端到端成本由其他支线负责，本文件不替其报完成。

## 核对入口

- 源训练脚本：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/train_joint_residual.py`
- 历史可微算子：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/joint_residual_ops.py`
- 最新专家前向与核：`E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_expert.py`
- 当前共享核心固定表示消融：`E:/Hermes/paper_tracking/sensitivity/external_knowledge_20261010/FINDINGS_ZH.md`
- 历史真实训练四臂的当前双头融合读出：`E:/Hermes/paper_tracking/sensitivity/backbone_ablation_20261010/complement_and_historical/FINDINGS_ZH.md`

以上各实验回答不同问题，保留各自的实现、时间顺序和比较边界，不拼接成一套未实际执行的完整重训练表。
