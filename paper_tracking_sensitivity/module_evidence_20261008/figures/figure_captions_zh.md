# 模块架构与流程图图注

日期：2026-10-08。对应当前联合版本，不呈现性能比较或泛化结论。图中文字采用英文，便于论文排版。每张图均提供同名 PDF、SVG 和 300 dpi PNG；合并版为 `module_pipeline_diagrams.pdf`，按下列顺序排列。

## 图 1：整体流程

文件：`01_overall_pipeline.png`

模型分为离线源域拟合与表征训练、目标域共享 64 标签适配、查询推理三个阶段。冻结的配对 VV/VH ResNet-50 输出 2048 维特征；VH 灰度图复制为三通道后输入 SimCLR 预训练的 ResNet-18，输出 512 维主干特征。M3 仅在源域训练阶段更新 R18 的 layer4；目标域适配与推理期间两个编码器均固定。M2 提供联合专家特征，M1 对基础分支和专家分支使用同一组 64 个标签拟合残差校正，M4 再以固定源域门控选择至多 5% 的查询接管。源知识指源域拟合的 PCA、归一化统计、分类头及门控，不是 AIS 或港口设施输入；身份标识只用于划分与排重。

R18 最初随机初始化，经历史 F0921 无标签池的 SimCLR/NT-Xent 预训练获得。预训练投影 MLP 为 512→256→128，部署丢弃该投影头，故图中未将其画作当前专家头。此无标签历史暴露不等同于“编码器从未接触目标域”。

## 图 2：M1 源先验锚定的双分支适配

文件：`02_m1_source_prior_adaptation.png`

源端在身份排重后拟合 PCA、标准化统计及类别平衡的 ridge-1 分类头；用于源端 episode 或 OOF 的拟合还排除留出港口。基础分支使用 R50 PCA128，专家源先验使用标准化的 R50 PCA128 与 R18 PCA64 拼接特征。联合版本的专家残差求解接收 M2 的 2753 维特征，而非仅使用旧版 193 维特征。基础与专家共享完全相同的 64 个支持标签，构造正类 +1、其他类 -1 的目标，并以支持集中出现的类别数及类别频数计算权重。两分支均执行 ridge-8 的 64×64 对偶求解，查询分数等于源先验分数加残差校正；偏置作为一个坐标共同受惩罚。目标查询标签不参与求解。

## 图 3：M2 保留主成分与正交补空间

文件：`03_m2_retained_complement_geometry.png`

对 R50 与 R18 特征分别减去各自源域 PCA 均值，按源域固定子空间计算保留分量与正交残差。保留分支拼接 PCA128 与 PCA64，按实现进行源端标准化，再除以 √192；补空间分支保留 2048+512 个原坐标中的残差坐标，使用源端补空间均值和尺度归一化，再除以 √2560。两分支分别乘 √0.5，最后仅追加一个偏置，得到 2753 维联合映射，对应核为 0.5K_p+0.5K_d+1。2560 是补空间的表示坐标数，并不声称残差具有 2560 个独立自由度。补空间来自显式正交投影，不是可学习注意力；源先验头仍只读取保留特征。

## 图 4：M3 穿过实际部署求解器的源域训练

文件：`04_m3_deployment_aligned_training.png`

每个源港口 episode 使用留出港口内的 64 个支持样本及不相交查询样本，查询按 128 大小分批（末批可更小）。其源先验、PCA 和统计由其他港口且经过身份排重的样本拟合。前向路径为 R18 layer4、M2 联合特征、固定参数先验头、实际 ridge-8 支持集求解、源类别平衡的查询交叉熵。红色虚线表示梯度经支持与查询特征、先验分数及求解过程回传，仅更新 layer4。PCA、归一化统计及先验头参数固定，但专家先验分数随编码器更新而变化；没有训练后重新拟合分类头。训练使用 20 个源港口 episode、六个 epoch、最终 checkpoint，冻结 BN 运行统计；不使用目标标签训练编码器，也没有额外的 5% 接管交叉熵。

## 图 5：M4 源端 OOF 可靠性学习与批量接管

文件：`05_m4_oof_reliability_handoff.png`

源端从港口外 OOF 的基础与专家分数提取四维输入，顺序为专家 margin、基础 margin、专家熵、基础熵。所有源样本先按港口等权、港口内已出现类别等权计算权重及标准化统计，再在分歧样本中以恢复正确性（delta=+1）和造成错误（delta=-1）训练逻辑回归，排除 delta=0。实现权重为 1/(港口数×该港口出现类别数×该港口该类样本数)，LR 拟合再乘源样本总数；无额外的二元类别平衡选项。

联合版本继承已有门控参数，不重新用联合模型训练门控；目标端输入实际适配后的基础与接收专家分数，四维置信度均由完整八类分数计算。候选必须不是 64 个支持样本、两分支预测不同、专家预测类别在支持集中出现且门控分数大于零。按门控分数降序、行 ID 升序稳定排序，选择至多 floor(0.05×N_query) 个候选；候选不足时不会补齐。入选输出专家预测，其余输出基础预测。专家特征及两路分数在选择前已经计算，预算约束的是接管数，不代表计算节省。源 OOF 仅针对固定编码器条件下的头部拟合，不能称为编码器级完全留出验证。

## 实现依据

- `../../joint_residual_pilot_20261008/JOINT_FINDINGS.md`：当前联合版本、冻结范围、训练方案与使用边界。
- `../../oof_reliability_check/joint_residual_ops.py`：`ComplementExpertGeometry.forward` 的半权重拼接、补空间归一化和单偏置。
- `../../oof_reliability_check/aligned_residual_ops.py`：`FixedExpertGeometry.forward` 的动态先验；`residual_scores` 的实际可微 ridge 求解。
- `../../oof_reliability_check/residual_expert.py`：`source_geometry`、`fit_correction` 的标准化、正负一目标和支持类别权重。
- `../../oof_reliability_check/run_opensarship2_transfer.py`：`fit_source`、`target_geometry`、`freeze_draw` 的源头拟合及双分支适配。
- `../../oof_reliability_check/prepare_aligned_residual.py`、`prepare_joint_residual.py`、`train_joint_residual.py`：源端留出 episode、冻结几何与联合训练前向/反向。
- `../../oof_reliability_check/run_oof_reliability.py`：`fit_reliability` 的置信度顺序、过滤时机和加权 LR。
- `../../oof_reliability_check/frozen_modules/report_bdh_pixel_handoff.py`：`source_weights` 的港口和类别等权。
- `../../oof_reliability_check/receiver_consistent_handoff.py`：`gate_scores` 读取接收专家实际分数。
- `../../oof_reliability_check/projection_residual_expert.py`：`freeze_kernel_draw` 的支持排除、已见类限制、正分数门槛及预算。
- `../../oof_reliability_check/budgeted_residual_handoff.py`：`ranked_mask` 的稳定排序。
- `E:/临时会话/visual_reliable_baseline/e206_ssl_pretrain.py`：SimCLR/NT-Xent、随机初始化 R18、预训练投影头及历史 F0921 无标签池。
