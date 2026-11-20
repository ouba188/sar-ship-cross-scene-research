# 复用来源与研究边界

本包是本地 SAR 跨港研究管线的独立整理与扩展，当前状态为代码实现及小规模正确性验证。尚未以这套完整配置完成新的正式 LOPO 对照，因此不能声称新增模块已经涨点、已经完成因果解耦，或复现了下述论文的结果。

## ZoomViT：本地源码实际提供了什么

核对的本地快照为 `E:/Docms/ZoomViT-main/ZoomViT-main`，重点文件为 `README.md`、`models_v2.py`、`engine.py`、`losses.py`。这里记录本地已检查版本的事实，不将其表述为上游仓库永久状态。

`README.md` 的 Training 节写明：

> Stay tuned! The training framework for **Zoomer** will be released soon.

因此此快照不能提供 Zoomer 训练目标生成、类别先验蒸馏与整套训练阶段的完整复现。README 同时声明 Apache 2.0；若后续直接复制上游实现，仍需保留相应许可证与版权说明。本次包没有直接 vendoring 这些上游源码。

`models_v2.py::Zoomer` 使用 ResNet-18 的浅层特征，池化到 28×28，经 64→1 卷积、28²→14² 线性投影与 sigmoid，产生单图的一张 14×14 重要性图。`Importance_Score` 从硬编码的 `Zoomer_70.pth` 加载模型并冻结参数，再按 patch 大小调整重要性图。该接口不接收查询类别标签，也没有当前 SAR 方案所需的每候选类别 × 双极化 × 区域 × 干预效应张量。

`engine.py` 是分类器训练循环；`losses.py::DistillationLoss` 提供普通分类输出蒸馏，包括 soft/hard teacher logits。它们不能替代 README 所说明尚未发布的 Zoomer 训练框架，也不是本包的区域有符号效应监督实现。

可复用的准确范围是：轻量单图预测器提供可部署的重要性提示，按提示调整局部 patch 粒度，并在下游分类器训练时协调先验模块与主干。原论文方法阅读记录还支持比较相同分辨率/预算以及区分先验训练与下游适配阶段。相关论文题名为 *Vision transformers need zoomer: Efficient ViT with categorical attention prior-guided zoom adapter*，DOI 为 `10.1016/j.neunet.2026.109514`；论文阅读记录见原研究目录 `multiscale_prototype_design_20261120/DESIGN_ZH.md`。

本包作出的任务特定改变包括：

- 输入为保持原始像素尺度的 VV/VH SAR chip，先验来自本折源教师的 M1 弱响应及原始区域干预测量。
- 学生预测全部候选类别的注意力和有符号效应；推理无需查询真标签或教师。
- 先验条件化父区域/子区域选择，选后的局部 token 保留到真实原型对应，不提前平均成单个类别特征。
- 参考来自源端真实船只，带来源与身份约束，匹配残差回写当前模型的 Layer3。

这些是本任务的实现选择，既不是直接使用上游 Zoomer 权重，也不等于作者训练流程复现。是否优于仅内容选择、均匀先验或静态细区域，需要在本数据上独立比较。

## 多尺度自适应参考

MSViT（*MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers*，ICCV Workshops，arXiv `2307.02321`）及其本地阅读记录 `paper_draft_20261108/literature_msvit_20261118/READING_ZH.md` 用于理解内容条件的粗细候选、预算约束和公平比较。

本包采用保留空间归属的父/子区域选择，没有复制其 Gumbel 训练策略、GBaS 或稀疏打包执行。当前 `scale_cost` 是期望额外活跃 token 的代理目标，填充张量中的候选槽位数保持固定，不能将其称作实际 FLOPs 或吞吐量收益。需要以同输入、同训练预算、相同原型数比较 fine/coarse/soft/hard，并另测部署成本。

## 本地历史实现的继承与改动

下列路径以 `E:/临时会话` 为原工作区根目录，仅说明实现来源；运行本包无需这些历史目录：

| 来源 | 本包继承的部分 | 本次明确改变 |
|---|---|---|
| `port_coverage_pilot_20261116/pcea_intervention_transfer_20261118/unified_model.py` 及其 backbone/M1/M2 依赖 | ResNet-50 分类、M1 关系反馈、M2 区域读写、guided classifier 的位置与接口 | 拆成 early/middle/finish；设备可选；移除自动历史权重加载；冻结 BN running stats |
| `sar_m123_current_20261031/sar_m123/input.py`、`tokens.py`；`paper_draft_20261108/m3_native_readout_20261118/`、`m3_content_scale_20261118/`、`m3_late_axis_evidence_20261118/` | 原始几何积分、区域 token、粗细候选、局部匹配与回写的已有实现经验 | 统一源/查询接口，补第三层参与、候选条件与独立模块配置 |
| 同研究目录 PCEA/CCREL | 低层/区域证据用于 Layer3 内部交换 | 源统计显式作为 bank 内容传入，移除可串 bank 的共享校准状态 |
| 同研究目录 `objective.py` | 原始尺度高/中频衰减与强度变化对照 | 明确全候选有符号 margin 效应、正确真类注意力弱监督、在线冻结教师 |
| `multiscale_prototype_design_20261120/ARCHITECTURE_V2_ZH.md` 与 `DESIGN_ZH.md` | 数据约束、监督语义、候选/匹配/回写逻辑 | 落成不依赖历史脚本的可运行包与 CLI |

历史负结果、全港口开发结果与原始测量应保留，不能因新包实现而被替换成“已成功”。旧的正增益缓存与本包有符号效应也不能不经语义转换直接等同。当前包拒绝把没有源折训练来源记录的普通研究 checkpoint 当作干净 source-only 初始化/教师；ImageNet 是独立的显式骨干初始化入口。

## 尚需实验回答的问题

1. 源教师弱注意力与有符号效应是否在固定原型身份/预算下改善局部选择和完整分类；需要分离内容 selector、注意力、效应三者贡献。亮度对照匹配的是强度变化总和，不是能量；当前 loss 不消除亮度效应，因此不能声称已完成物理因果识别。
2. 第三层是否在语义和局部对应之间提供有用补充；比较时保持质量归一化、原型数和优化设置，避免把更多层/参数等同于独立证据。
3. Layer1 通道门控与合法 donor 交换是否改善跨港稳健性；剩余分量可能仍有类别信息，门控名称不构成已识别环境因素的证据。
4. coverage 原型相对同预算 random 的变化来自更好的覆盖、源港组成，还是重复观察/任务难度；需要按船、港、类别及极化条件追踪实际成员和预测翻转。

这些问题适合在实现通过后用有限配对对照验证，不要求先证明涨点再训练。完整收益、适用范围和错误案例的最终判断以实际目标折结果及原始 chip 回看为准。
