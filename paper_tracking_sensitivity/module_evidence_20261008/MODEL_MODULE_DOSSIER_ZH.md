# 跨港预算化专家交接：联合版本的模块、文献、流程与实验依据

更新日期：2026-10-08。用途：论文讨论底稿与方法/实验章节的事实底表。首次整理核对已有代码、结果并制作流程图；随后按用户要求启动并逐项补齐缺失实验。

**同日补测已完成六类核心任务。** 本文8.1/8.4节保留原主线数值，8.2节已回填新的四臂匹配结果。完整过程、正负结果和限制见[补测结论](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/COMPLETION_FINDINGS_ZH.md)与[全数值汇总](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/COMPLETED_RESULTS.md)。历史工件不覆盖，新旧参考明确区分。

**新增预算扫描已完成。** 固定联合模型、相同64标签和5次支持集抽样，在六个数据集上比较六档门控上限及全量专家，共34任务、1190条结果，原5%精确复现。见[完整预算报告](E:/Hermes/paper_tracking/sensitivity/handoff_budget_sweep_20261008/BUDGET_FINDINGS_ZH.md)。24港、F_0921和OSS2放宽上限后OA/BA同时改善；OSS1存在OA/BA权衡，PACS大幅放宽退化，Office-Home连5%也低于Base64。因此下文5%是固定低预算工作点，不是普遍最优预算；预算放宽的收益也不能算作M2/M3新增模块收益。

## 1. 先给结论

当前联合版本由四个功能模块组成：**M1 源先验锚定的双分支少样本适配、M2 保留空间与补空间联合建模、M3 部署求解器对齐的专家表征训练、M4 接收者一致的预算化交接**。

其中，M1/M4 构成已有主线，M2/M3 是近期加入的联合改动。四个功能模块不等于四项已经成立的原创算法。最适合集中论证的新增机制是：**让专家保留源域降维未保留的信息，并通过部署时真正使用的少样本残差求解器学习如何利用这些信息，最终进入有限预算的交接系统。**

联合版本已完成自建24港中的23个可评估目标港、F_0921、OpenSARShip 1.0/2.0、PACS和Office-Home的匹配2×2结果。23港相对匹配参考的最终OA/BA增益为 **+0.307381/+0.395702个百分点**；F_0921为 **+0.124286/+0.325071**；OSS2为 **+0.755230/+0.618145**；OSS1为 **+0.233463/−0.572066**；PACS为 **−0.528798/−0.451192**；Office-Home为 **−1.155038/−1.127123**。三个SAR任务两项指标同时改善，但已有证据直接排除了“联合模块跨数据集普遍提升”的表述。

原主线在自建 24 港中的 23 个合格目标港、F_0921、OpenSARShip 1.0/2.0、PACS 和 Office-Home 上有结果。新增 M2 已完成这些任务的补测。PACS/Office-Home 原监督 R18 权重未保存，新四臂试验必须重训源域编码器，单独标为 retrained_rgb_reference，不能与旧特征基线混算增益。

## 2. 文件导航

- 五页流程图 PDF：[module_pipeline_diagrams.pdf](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/module_pipeline_diagrams.pdf)。每页另有可编辑 SVG、矢量 PDF、300 dpi PNG。
- 全部数值与协议：[MODULE_DATASET_EVIDENCE_ZH.md](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/MODULE_DATASET_EVIDENCE_ZH.md)。包括每个通用目标域和各类未运行项。
- 机器可读证据：[module_dataset_evidence.json](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/module_dataset_evidence.json)。逐行保留原始结果路径与字段选择器。
- 原主线数值：[main_dataset_results.csv](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/main_dataset_results.csv)。自建逐港数值：[self23ports_detail.csv](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/self23ports_detail.csv)。
- 联合对照：[f0921_matched_joint_2x2.csv](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/f0921_matched_joint_2x2.csv)。模块覆盖：[module_coverage.csv](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/tables/module_coverage.csv)。
- 当前六类四臂对照：[core_matched_results.csv](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/core_matched_results.csv)。当前模块覆盖：[module_coverage_completed.csv](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/module_coverage_completed.csv)。前一行旧覆盖表是补测前快照，不再用于判断当前缺项。
- 参考文献：[references.bib](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/references.bib)。下文 [R1] 等编号对应该文件注释中的编号。

## 3. 总体管线与阶段边界

![Overall pipeline](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/01_overall_pipeline.png)

### 3.1 三个阶段

1. **源域离线阶段**：源港图像、类别、港口划分进入双编码器；拟合源域 PCA、标准化、基础/专家分类先验与 source-head-OOF 可靠性交接器。在源港支持/查询情景中，经 M2 和真实 M1 求解器训练 M3，仅更新专家 R18 的 layer4。
2. **目标域适配阶段**：抽取 64 个目标标注样本；基础分支和专家分支共享这同一组标签。编码器、源统计量和交接器固定，通过两次残差岭回归拟合目标修正项。目标标签不用于本轮 M3 编码器训练。
3. **查询阶段**：查询样本同时得到适配后的基础预测与专家预测。M4 在满足资格的样本中排序，最多交接查询集的 5%；被选中者采用专家预测，其他采用已适配的基础预测。

这里有两个独立预算：64 是标注预算，5% 是专家交接比例。基础模型可在全部查询上受到 64 标签适配的影响；5% 不限制它相对原源模型改变多少预测。当前实现先计算双分支，再选择交接，尚不能声称节省 95% 的专家计算。

### 3.2 输入、编码器和维数

| 部件 | 当前 SAR 实现 | 输出与训练状态 |
| --- | --- | --- |
| 基础编码器 | ImageNet 预训练 ResNet-50，适配 VV/VH 双通道输入 | 2048 维，全程冻结 |
| 专家补充编码器 | SimCLR/NT-Xent 预训练 ResNet-18；VH 灰度重复成三通道 | 512 维；M3 只更新 layer4 |
| 原基础表征 | 源域 R50 PCA128 与标准化 | 128 维，适配时加一个偏置为 129 维 |
| 原专家表征 | R50 PCA128 与 R18 PCA64 联合后标准化 | 192 维，原适配特征加偏置为 193 维 |
| M2 联合专家表征 | 保留 192 维 + 两编码器补空间的 2560 个坐标 + 一个偏置 | 2753 维，不等于 2753 个独立信息方向 |

ResNet 架构依据 [R1]；本地 `e206_ssl_pretrain.py` 使用的是 SimCLR 式双增强 NT-Xent，依据 [R2]，不是 BYOL/DINO。其预训练投影 MLP 为 512→256→128；当前部署只使用 512 维骨干输出，不使用该投影头。原预训练语料包含历史 F_0921 无标签池，故 F_0921 不能写成编码器从未接触的盲外部数据。

这里使用的“跨港知识”是源港的特征统计、类别边界和基础/专家纠错经验。当前模块不输入港口设施、泊位功能、地理邻接图或 AIS 轨迹。AIS 参与标签/数据构建不等于模型使用了轨迹知识。

## 4. M1：源先验锚定的双分支少样本适配

![M1](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/02_m1_source_prior_adaptation.png)

### 4.1 要解决什么，为什么这样做

源港训练的分类边界迁移到目标港后会偏移，而 64 个标签不足以重新训练完整网络。M1 保留源模型得分作为起点，只学习目标域需要的修正；同时让基础分支也使用同样的 64 标签，避免把“专家多用了标签”误写成专家交接机制的优势。

两个分支适配相互独立，但支持集相同。专家可以使用更多特征，不能多用目标标签。

### 4.2 逐步计算

1. 在源域以类别平衡的 RidgeClassifier 拟合基础和专家先验，源头正则强度为 1，正常拟合截距。
2. 固定源 PCA、均值/尺度及源分类头参数。目标支持集记为 S，|S|=64；目标查询为 Q，与 S 分开。
3. 为分支 b 构造特征映射 phi_b，包含一个显式偏置列；基础使用 129 维，联合专家使用 2753 维。
4. 将支持标签编码为每类 +1/-1 目标，减去对应源头得分，得到要拟合的残差。
5. 用类别平衡权重求解正则强度为 8 的残差岭回归。
6. 查询得分等于源得分加残差修正；类别取 argmax。

设 C 为固定输出类别数、C_S 为支持集实际观察到的类别数，n_c 为支持集中类别 c 的样本数：

\[
t_i=2\operatorname{onehot}(y_i)-1,\qquad
w_i=\frac{64}{C_S n_{y_i}}.
\]

\[
\Delta W_b=\arg\min_W\sum_{i\in S}w_i\|t_i-s_b^{src}(x_i)-\phi_b(x_i)W\|_2^2+8\|W\|_F^2,
\qquad s_b^{adapt}(x)=s_b^{src}(x)+\phi_b(x)\Delta W_b.
\]

令 A=diag(sqrt(w))Phi_S，则实际求解为：

\[
\Delta W_b=A^T(AA^T+8I_{64})^{-1}\operatorname{diag}(\sqrt w)(T-S_b^{src}).
\]

代码使用线性方程求解，不显式求逆。求解矩阵仅 64×64；偏置列只有一个，并受同一正则项约束。源分类头参数固定不等于专家源得分固定：M3 改变输入特征时，源得分也随之改变。

### 4.3 参考文献与实际区别

- [R3] Hoerl 与 Kennard：岭回归的数学基础。
- [R4] Bertinetto 等，R2D2：可微闭式少样本求解器及对偶小矩阵求解，是最直接的方法前序。
- [R5] Long 等，Residual Transfer Networks：源/目标分类器残差关系的相关工作。当前方法没有复现其完整无监督域适配损失，因此不能直接称为 RTN 模块。

本项目的具体选择是“固定源先验 + 类平衡目标残差 + 同组 64 标签的双分支公平适配”。残差、岭回归本身不是新发明。

### 4.4 实际效果

原主线 Base64 相对 Source 的 BA 增益：23 港均值 +5.271075、F_0921 +4.587546、OSS1 +10.277297、OSS2 +9.837550、PACS +10.889052、Office-Home +0.732512 个百分点。

这说明使用少量目标标签进行整体适配有实际价值。但 Source→Base64 同时改变了目标标签访问和适配过程，**不是“源锚定”相对其他少样本头的独立消融**，不能把全部收益归因于源锚定这一细节。

代码入口：[residual_expert.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/residual_expert.py)、[aligned_residual_ops.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/aligned_residual_ops.py)。

## 5. M2：保留空间与补空间联合建模

![M2](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/03_m2_retained_complement_geometry.png)

### 5.1 要解决什么，为什么这样做

源域 PCA 优先保留源域方差较大的方向，但这些方向不必覆盖目标域全部判别证据。原专家只读取 192 维保留空间，即使原始编码器输出还含有有用方向，也无法用 64 标签对这些方向作出修正。

M2 不重新引入外部知识，而是让目标残差头同时读取原保留空间和原来未读取的补空间。设计目标是减少降维这一环节的信息丢弃，不是保证修复原始裁片、标签或骨干已经丢失的信息。

### 5.2 逐步计算

1. R50 输出 h_B∈R^2048，R18 输出 h_E∈R^512。
2. 使用源域固定 PCA，分别保留 128/64 维，构成标准化保留表征 z∈R^192。
3. 对每个编码器，计算其中心化输出减去保留空间重建，获得正交补空间残差。
4. 用源域残差统计进行逐坐标中心化和尺度归一化，拼接为 d∈R^2560。
5. 对两个块分别按维数归一化，并给予固定 1/2 的核权重；仅添加一个共享偏置。
6. 原源专家头仍使用保留表征给出先验，目标残差头使用联合特征修正。

若 P 的行为 PCA 正交基：

\[
r(h)=(h-\mu)-((h-\mu)P^T)P,
\qquad \phi_E=[z/\sqrt{2\cdot192},\ d/\sqrt{2\cdot2560},\ 1].
\]

对应核为：

\[
K_E=\tfrac12 K_{retained}+\tfrac12 K_{complement}+\mathbf1\mathbf1^T.
\]

两个权重为固定值，不是注意力权重，也不是训练得到的多核选择器。2560 是补空间在原坐标系里的表示长度；其线性秩不超过 (2048−128)+(512−64)=2368。不能把增加的坐标数等同于新增独立信息量。

### 5.3 文献定位

- [R6] Jolliffe：PCA 投影与重建的数学依据。
- [R8] SimpleMKL：核组合的相关研究；本实现只采用固定加权和，不实现其学习核权重的优化。
- [R7] Domain Separation Networks：分离共享/私有表征的相关工作，适合在相关工作中比较。本方法不是 DSN，没有其私有编码器、重建器或完整域对抗训练；也不能将 PCA 保留/补空间直接命名为已证实的域不变/域特有语义。

具体结构来自本项目对原专家几何的改造，不能把上述任一篇说成整套模块的直接来源。

### 5.4 实际效果与边界

历史独立 M2 完整系统相对各自原主线的 ΔOA/ΔBA：F_0921 **+0.195260/−0.101876**，OSS1 **+0.217899/−0.359902**，OSS2 **+0.059766/+0.033502** 个百分点。

结论：补空间有时提高 OA，但单独使用不能保证 BA；F_0921/OSS1 上出现 OA 与 BA 的取舍。该结果支持保留这条信息路径供 M3 利用，不支持“补空间普遍提升所有类别”。源监督压缩为 8 维的旧候选没有纳入联合版本；半权保留空间对照属于已有诊断，不另算创新模块。

代码入口：[projection_residual_expert.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/projection_residual_expert.py)、[joint_residual_ops.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/joint_residual_ops.py)。

## 6. M3：部署求解器对齐的专家表征训练

![M3](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/04_m3_deployment_aligned_training.png)

图中20个源港情景是初始F_0921配置，不是所有任务的固定数量：本轮逐港补测每个外层任务为22个源港情景，通用域每个外层任务为3个源域情景；六轮tail训练不变。

### 6.1 要解决什么，为什么这样做

普通分类交叉熵训练出“对当前训练头有用”的特征，不保证这些特征经过源先验、标准化、64 标签残差岭回归后仍最有用。加入 M2 后可读取的信息更多，但特征是否方便少样本求解器利用仍需训练。

M3 将部署时的真实求解器置入训练图，让查询损失反传到专家表征。重点不是再造一个更大的头，而是改变专家学到什么，使其适合最终采用的适配规则。

### 6.2 源港情景训练的实际步骤

1. 构造源域留出情景（F_0921 初始联合试验为20个；后续数量依目标排除后的可用源港/源域确定）。每个情景从一个留出源港/域抽取支持64和查询批次128；源先验拟合排除该港以及 SAR 既有同 MMSI/产品重叠。
2. 固定 R50、R18 前缀、源 PCA/标准化/源分类头参数，固定 BatchNorm 运行统计。
3. 支持和查询同时经过可训练的 R18 layer4，得到新的 512 维特征。
4. 经 M2 构造保留/补空间联合表征；使用 M1 的实际强度 8、实际类别平衡和显式偏置求解残差头。
5. 对情景查询计算类别平衡交叉熵。
6. 梯度经过查询特征、支持特征、源先验得分与线性方程求解器，回传更新 layer4。
7. 训练完成后冻结编码器；部署时只用真正目标域的 64 标签拟合两个残差头。

\[
\mathcal L(\theta)=\operatorname{BalancedCE}_{Q_p}
\left[s^{src}_{E,\theta}(x)+\phi_{E,\theta}(x)\Delta W_E(S_p;\theta),y\right].
\]

所有新增 tail 训练均为6 epochs；F_0921 初始联合试验每轮28402个查询行，其他任务按各自合法源行生成情景，不能照抄该行数。AdamW 学习率1e−5、weight decay 1e−4，梯度裁剪5，训练 seed 20261008。使用末轮模型，没有用目标查询成绩挑选最佳 epoch。aligned-only 与 joint 使用匹配初始化、顺序和日程。

**对齐范围需准确表述**：这里对齐的是“表征→少样本残差求解器→专家查询分类”。没有对硬 top-k 或最终交接收益反传，没有额外 gate-CE，也不是端到端优化最终预算准确率。源头按情景排除与完整编码器嵌套 OOF 不是同一个概念。

### 6.3 文献定位

- [R4] R2D2：通过可微岭回归训练少样本特征，是最接近的基础。
- [R9] MetaOptNet：通过可微凸优化训练适于线性分类器的嵌入。本实现为源先验残差岭回归，不是其 SVM/QP 实现。
- [R12] AdamW：优化器依据，仅为实现组件，不作为创新点。

因此，不能宣称“首次可微少样本求解器”。可以讨论的本项目特征是：跨港源情景、固定源先验、部署一致的残差几何、共享 64 标签约束，以及与 M2 的联合训练。

### 6.4 实际效果

F_0921 同轮 aligned-only 相对参考：最终 OA **−0.307879**、BA **+0.328281** 个百分点。联合 M2 后，相对 aligned-only，OA **+0.432165**、BA **−0.003211**。这意味着本轮联合主要收回了对齐训练的 OA 损失，同时保留了几乎相同的 BA，而不是在两个指标上都超过各单模块的最大值。

专家独立预测全部查询时，参考 Expert BA=25.982723、aligned-only=27.041201、joint=27.147474；最终系统 BA 则分别为 25.106684、25.434966、25.431755。专家自身改善不等于全部转化为有限交接系统的改善。

新增公开SAR验证：OSS2的aligned-only为32.882369/30.318171，联合为33.137734/30.556508，联合超过匹配参考且超过单加M3；OSS1联合仍有BA下降。新增RGB验证：PACS/Office-Home联合专家全查询BA分别下降6.074592/9.931260个百分点，最终系统BA下降0.451192/1.127123个百分点。因此通用域负结果不是仅有门控未利用专家改善，专家本身也退化；具体训练失效原因尚未被这些消融单独证明。

主数据集补测：23港M3单独为56.756475/50.577416，联合为56.961200/50.777800。联合相对M3再增加0.204726 OA / 0.200384 BA个百分点；相对参考，15港BA提高、8港下降。该结果支持主数据集上的联合平均收益，不是无损或逐港一致性保证。

代码入口：[prepare_joint_residual.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/prepare_joint_residual.py)、[train_joint_residual.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/train_joint_residual.py)、[evaluate_joint_residual.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/evaluate_joint_residual.py)。

## 7. M4：接收者一致的预算化交接

![M4](E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/figures/05_m4_oof_reliability_handoff.png)

### 7.1 要解决什么，为什么这样做

“基础模型不确定”不等于“专家会答对”，全部替换也可能损害原本正确的样本。M4 根据两个模型的相对可靠性选择交接对象，在预算内让专家替换基础预测。

“接收者一致”指部署时资格和路由特征以真正将接收样本的适配后专家、以及真正的 Base64 为准，而非继续按旧源基础模型判断分歧。当前仍使用既有冻结的源 OOF 逻辑回归门控，并未完成门控针对新专家的端到端联合学习。

### 7.2 训练管线

1. 按源港得到分类头的 OOF 基础/专家得分。
2. 构造四个门控特征：专家 margin、基础 margin、专家 entropy、基础 entropy。
3. 根据源标签计算 delta=I(专家正确)−I(基础正确)。
4. 仅保留 delta≠0 的信息性分歧：专家纠正基础错误为正类，专家损害基础正确为负类；两者都错的零效用样本不进入该 LR 训练。
5. 使用等港、港内等类的权重拟合 C=1 的逻辑回归。权重先在源样本上计算，再筛选信息性分歧。
6. 固定门控参数。函数中另拟合的岭效用回归器不是本轮实际路由器，实际调用的是 `model['lr']`。

### 7.3 目标部署管线

1. 移除 64 个支持行，定义查询数 N_Q。
2. 用适配后的基础/专家输出构造门控特征。
3. 候选须同时满足：两者预测不同；专家预测类别已在 64 支持标签中出现；门控分数大于 0。
4. 按分数降序、行号升序作稳定排序。
5. 最多选 floor(0.05 N_Q) 行交接。候选不足时不强行补齐。
6. 被选中行用专家预测，其余用基础预测。查询标签仅用于最终评价。

\[
\widehat y(x)=\begin{cases}\widehat y_E(x),&x\in H,\\\widehat y_B(x),&x\notin H,\end{cases}
\qquad |H|\leq\lfloor0.05N_Q\rfloor.
\]

LR 在信息性分歧上学习，不是对所有样本校准过的净增益概率，不能赋予其未证明的风险保证。Office-Home 的 64 标签未覆盖所有类时，未见类仍在完整查询评价中，只是专家预测该类时不能进入交接候选。

### 7.4 文献定位与效果

- [R10] Learning to Defer：分类器与专家间的分工依据。本文没有采用该文一致性替代损失，不能继承其理论保证。
- [R11] Adaptive Neural Networks：预算约束的选择性推理相关工作。其计算节省与当前“先算专家、后决定是否替换”的预算含义不同。

原主线 Full5−Base64 的 BA 增益：23 港 +1.842028、F_0921 +0.203362、OSS1 +0.820242、OSS2 +0.643091、PACS 等域 +1.631937、Office-Home 等域 +0.678186 个百分点。23 港中 21 港 BA 为正。

这些是整个专家交接阶段的增量，不能单独归因于 LR 门控优于其他路由器。也存在负域：PACS/photo 为 −3.213599 BA，Office-Home/Art 为 −0.090568 BA。当前不具备跨域无损交接保证。

代码入口：[run_oof_reliability.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_oof_reliability.py)、[receiver_consistent_handoff.py](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/receiver_consistent_handoff.py)。

## 8. 各数据集数值

所有 OA/BA 单位为 %，增量为百分点。下面的 6 位小数用于工件对账，不代表统计精度；论文展示通常保留 2 位，并配套重复抽样说明。OA 是总正确比例；BA 是实际评价类别召回率的等权平均。二者不可互代。

### 8.1 原主线：Source → M1 Base64 → M1+M4 Full5

| 数据集 | Source OA/BA | Base64 OA/BA | 原主线 Full5 OA/BA | M1阶段 ΔOA/ΔBA | M4阶段 ΔOA/ΔBA |
| --- | --- | --- | --- | --- | --- |
| 自建24港：23合格港等权 | 44.094334/43.268995 | 54.794330/48.540070 | 56.653819/50.382098 | +10.699996/+5.271075 | +1.859489/+1.842028 |
| F_0921 | 24.207575/20.314242 | 28.937087/24.901788 | 29.633543/25.105149 | +4.729512/+4.587546 | +0.696455/+0.203362 |
| OpenSARShip 1.0 | 38.163424/21.233037 | 40.513619/31.510334 | 43.066148/32.330577 | +2.350195/+10.277297 | +2.552529/+0.820242 |
| OpenSARShip 2.0 | 18.565607/19.457723 | 30.638413/29.295273 | 32.382505/29.938364 | +12.072806/+9.837550 | +1.744091/+0.643091 |
| PACS：四域等权 | 67.529691/67.348009 | 76.024350/78.237061 | 77.969497/79.868998 | +8.494658/+10.889052 | +1.945148/+1.631937 |
| Office-Home：四域等权 | 65.520207/64.120288 | 66.558133/64.852800 | 67.285482/65.530986 | +1.037926/+0.732512 | +0.727349/+0.678186 |

这些是阶段差值，不是全部模块逐一独立消融。尤其 PACS/Office-Home 使用各自 RGB 编码器和域划分，不是将 SAR 训练好的联合检查点直接迁移过去。

### 8.2 新模块覆盖与完整系统成绩

| 数据集 | 仅加M2 OA/BA | 仅加M3 OA/BA | 联合M2+M3 OA/BA |
| --- | --- | --- | --- |
| 自建24港：23合格港匹配参考 | 56.679312/50.258322 | 56.756475/50.577416 | 56.961200/50.777800 |
| F_0921：最新匹配轮次 | 29.829289/25.003473 | 29.326474/25.434966 | **29.758639/25.431755** |
| OpenSARShip 1.0 | 43.284047/31.970674 | 43.003891/31.848632 | 43.299611/31.758511 |
| OpenSARShip 2.0 | 32.442271/29.971866 | 32.882369/30.318171 | 33.137734/30.556508 |
| PACS：重训RGB匹配参考 | 77.813689/79.636341 | 77.349439/79.266995 | 77.280367/79.184320 |
| Office-Home：重训RGB匹配参考 | 67.096863/65.372944 | 66.252547/64.557644 | 65.973213/64.265466 |
| FMoW | 当前协议未找到结果 | 当前协议未找到结果 | 当前协议未找到结果 |
| Camelyon17 | 当前协议未找到结果 | 当前协议未找到结果 | 当前协议未找到结果 |

本表每个数是加入该模块后的**完整部署系统**，不是该模块自身的“准确率”。“未运行”不能填0，也不能用旧主线或表征诊断实验代替。PACS新四臂参考为77.809164/79.635512，Office-Home为67.128250/65.392589，不能减去8.1节的旧参考。原缓存上的M2另为：PACS77.992571/79.882573，Office-Home67.310114/65.564288。24港四臂采用相同float16前缀的初始tail参考，其最终OA/BA恰与旧参考相同（56.653819/50.382098），专家全查询数值略有差别，工件仍分开保留。

### 8.3 F_0921 最新匹配 2×2

固定基础分支、同组支持标签、M4 规则以及 6171/123425 次交接，仅改变 M2 和 M3。

| 分支 | M2 | M3 | 最终OA | 最终BA | 专家全查询OA | 专家全查询BA |
| --- | --- | --- | --- | --- | --- | --- |
| 本轮参考 | 否 | 否 | 29.634353 | 25.106684 | 31.055945 | 25.982723 |
| 补空间单独 | 是 | 否 | 29.829289 | 25.003473 | 32.808102 | 25.834198 |
| 对齐训练单独 | 否 | 是 | 29.326474 | 25.434966 | 29.806441 | 27.041201 |
| 联合 | 是 | 是 | 29.758639 | 25.431755 | 31.874580 | 27.147474 |

联合相对参考：+0.124286 OA / +0.325071 BA，两个指标均在 3/5 次抽样为正。联合相对 aligned-only：+0.432165 OA（5/5 为正）/−0.003211 BA。联合相对 complement-only：−0.070650 OA/+0.428282 BA（BA 5/5 为正）。

描述性二因素交互项 joint−aligned−complement+reference 为 +0.237229 OA/+0.100001 BA；这里只作加性关系描述，没有完成交互显著性推断。不能把它写成证明了统计显著协同。

生产参考 29.633543/25.105149 与本轮参考 29.634353/25.106684 略有差别，全部比较使用各自匹配参考。历史独立 M3 的 BA 为 25.435555，也不替换本轮 25.434966。

### 8.4 通用跨域逐域成绩：只属于已有框架

| 数据集/目标域 | Base64 OA/BA | Full5 OA/BA | ΔOA/ΔBA |
| --- | --- | --- | --- |
| PACS / art_painting | 78.608871/78.395563 | 80.655242/80.612386 | +2.046371/+2.216823 |
| PACS / cartoon | 70.789474/74.436160 | 74.745614/77.864359 | +3.956140/+3.428199 |
| PACS / photo | 97.721046/97.199079 | 95.105853/93.985480 | −2.615193/−3.213599 |
| PACS / sketch | 56.978008/62.917442 | 61.371281/67.013767 | +4.393273/+4.096325 |
| Office-Home / Art | 66.559458/62.954531 | 66.347863/62.863962 | −0.211595/−0.090568 |
| Office-Home / Clipart | 49.476866/49.630514 | 51.364799/51.342740 | +1.887933/+1.712226 |
| Office-Home / Product | 74.240000/72.438926 | 74.806857/72.893490 | +0.566857/+0.454565 |
| Office-Home / Real World | 75.956208/74.387229 | 76.622409/75.023750 | +0.666201/+0.636521 |

### 8.5 样本、标签预算与可比范围

| 数据集 | 当前目标样本/每轮查询数 | 类别 | 64标签如何计数 |
| --- | --- | --- | --- |
| 自建24港的23合格港 | 38027/36555 | 8 | 每目标港64；23任务合计1472/轮 |
| F_0921 | 123489/123425 | 8 | 每轮全数据集共享64 |
| OSS1兼容子集 | 1349/1285 | 8 | 每轮全数据集共享64 |
| OSS2兼容子集 | 3745/3681 | 8 | 每轮全数据集共享64 |
| PACS四域任务合计 | 9991/9735 | 7 | 每目标域64；四任务合计256/轮 |
| Office-Home四域任务合计 | 15588/15332 | 65 | 每目标域64；四任务合计256/轮 |

自建原库 38091 行、24 港；主线当前纳入 23 个合格目标任务，不应写成 24 个港均完成当前联合评估。未纳入的是 Melbourne，恰有 64 行；取完支持后没有剩余查询，代码按 `len(q) <= ANNOT` 跳过，并非按成绩剔除。见[港口样本记录](E:/Hermes/paper_tracking/sensitivity/port_context_audit_20261007/query_source_coverage.csv:10)和[任务筛选](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_oof_reliability.py:291)。

主线先平均港内五次抽样，再等权平均港口；OA 也不是跨港按样本数汇总。PACS/Office-Home 同样采用等目标域均值。五次支持抽样使用 seed 0、1、2、3、4，均匀无放回抽取 64 行，无类别配额；这些不是五次独立神经网络训练 seed。同一任务的各比较分支共享支持集，不同数据集/目标域只共享抽样规则，不共享样本。Office-Home 支持覆盖 34–46 类，完整查询的 65 类仍保留。

OpenSARShip 表是统一细标签后的兼容子集，OSS1 为兼容 GRDH 子集，不是公开数据全集原生协议。不能将这里的 OA 直接和其他论文不同类数/切分的 OA 比高低。公开数据需分别引用 [R13]/[R14]；PACS/Office-Home 分别引用 [R15]/[R16]。

PACS/Office-Home 使用冻结 ImageNet R50 与按外层源域监督微调的 R18，输入是同一 RGB 图像。源内部的 head-OOF 不等于 encoder-OOF，不能写成严格嵌套表征留出。SAR 支持/查询为行级划分，非同船完全隔离；这些数据都有历史开发使用。

### 8.6 可选数据与未覆盖部分

FUSAR 原主线：Source 46.965265/22.463793；Base64 51.608775/28.446428；Full5 52.650823/28.894663。实际存在 7 个目标类、1158 行、查询 1094；单图复制为两通道，不构成双极化实验。仅作为独立附表，尚无 M2/M3 联合结果。

FMoW 查到旧的 8 类、五地区、20% 预算实验，不能填进当前 shared64/5% 表；Camelyon17 确认存在数据与特征，当前协议结果在本次有界检索中未找到。具体范围写入证据 JSON。这里不声称历史上从未跑过这些数据集。

## 9. 论文怎么组织这些模块

建议按“问题→机制→证据”写，而不是把四个名字均称为全新模块：

1. **任务与主线**：跨港分布偏移下，64 目标标注与 5% 专家交接预算同时存在；M1 提供同等标签条件的可适配基座，M4 执行有限替换。
2. **信息利用机制**：M2 将源 PCA 未保留的编码器证据送入专家残差适配，而非再堆一个注意力名称。
3. **训练部署对齐机制**：M3 经实际残差求解器优化表征，使补空间信息可被少量标签使用；这是联合版本主要值得论证的部分。
4. **系统结果**：原主线跨数据集结果支持框架可迁移；23港均值、F_0921与OSS2的2×2支持联合改造在这些开发任务上的收益；OSS1的指标取舍以及PACS/Office-Home负结果限定适用范围。不能将框架广泛可运行等同于新增模块普遍有效。

目前可写：“联合改造在23港等权均值、F_0921与OSS2开发任务上同时提高相对匹配参考的OA和BA，但不具有跨数据集一致收益。”

目前不可写：“四个创新模块在所有数据集持续提点”“首次提出可微少样本求解”“5%计算完成专家推理”“引入港口设施/AIS轨迹知识”“取得盲外部验证或无损保证”。

四模块架构可以作为方法章节的组织骨架；本轮已补齐六类核心任务的联合证据，得到明确的正负结果。是否达到某期刊创新标准不能由模块数或这些参考文献的发表级别推出，本资料也不把本次有限文献核验当成穷尽式新颖性检索。

## 10. 模块对应参考文献

| 部分 | 直接基础/实现依据 | 相关工作而非直接实现 | 引用时应说明的差别 |
| --- | --- | --- | --- |
| 编码器与预训练 | [R1] ResNet；[R2] SimCLR | 无需额外贴注意力文献 | 自建SAR预训练配置不等于原文配置 |
| M1 | [R3] Ridge；[R4] R2D2 | [R5] RTN | 本文拟合源得分残差，两分支共享64标签 |
| M2 | [R6] PCA | [R7] DSN；[R8] SimpleMKL | 固定PCA补空间、固定核权重，不是DSN或可学习MKL |
| M3 | [R4] 可微岭回归；[R12] AdamW | [R9] MetaOptNet | 实际残差部署头进入训练；不对硬预算门控反传 |
| M4 | 自有source-head-OOF LR规则 | [R10] L2D；[R11]预算推理 | 未复现其理论一致性损失或计算节省机制 |
| 数据集 | [R13] OSS1；[R14] OSS2；[R15] PACS；[R16] Office-Home | 自建24港/F_0921引用自身数据构建记录 | 不编造自建数据的外部发表文献 |

### 参考文献表

[R1] K. He, X. Zhang, S. Ren, J. Sun. **Deep Residual Learning for Image Recognition.** CVPR, 2016, pp. 770–778. arXiv:1512.03385.

[R2] T. Chen, S. Kornblith, M. Norouzi, G. Hinton. **A Simple Framework for Contrastive Learning of Visual Representations.** ICML, PMLR 119, 2020, pp. 1597–1607.

[R3] A. E. Hoerl, R. W. Kennard. **Ridge Regression: Biased Estimation for Nonorthogonal Problems.** Technometrics, 12(1), 1970, pp. 55–67. DOI:10.1080/00401706.1970.10488634. 出版网页的2012是在线数字化日期，非论文年份。

[R4] L. Bertinetto, J. F. Henriques, P. H. S. Torr, A. Vedaldi. **Meta-learning with differentiable closed-form solvers.** ICLR, 2019. arXiv:1805.08136.

[R5] M. Long, H. Zhu, J. Wang, M. I. Jordan. **Unsupervised Domain Adaptation with Residual Transfer Networks.** Advances in Neural Information Processing Systems, vol. 29, 2016.

[R6] I. T. Jolliffe. **Principal Component Analysis.** 2nd ed., Springer, 2002. DOI:10.1007/b98835.

[R7] K. Bousmalis, G. Trigeorgis, N. Silberman, D. Krishnan, D. Erhan. **Domain Separation Networks.** Advances in Neural Information Processing Systems, vol. 29, 2016. arXiv:1608.06019.

[R8] A. Rakotomamonjy, F. R. Bach, S. Canu, Y. Grandvalet. **SimpleMKL.** Journal of Machine Learning Research, 9(83), 2008, pp. 2491–2521.

[R9] K. Lee, S. Maji, A. Ravichandran, S. Soatto. **Meta-Learning With Differentiable Convex Optimization.** CVPR, 2019, pp. 10657–10665.

[R10] H. Mozannar, D. Sontag. **Consistent Estimators for Learning to Defer to an Expert.** ICML, PMLR 119, 2020, pp. 7076–7087.

[R11] T. Bolukbasi, J. Wang, O. Dekel, V. Saligrama. **Adaptive Neural Networks for Efficient Inference.** ICML, PMLR 70, 2017, pp. 527–536.

[R12] I. Loshchilov, F. Hutter. **Decoupled Weight Decay Regularization.** ICLR, 2019. arXiv:1711.05101.

[R13] L. Huang, B. Liu, B. Li, W. Guo, W. Yu, Z. Zhang, W. Yu. **OpenSARShip: A Dataset Dedicated to Sentinel-1 Ship Interpretation.** IEEE JSTARS, 11(1), 2018, pp. 195–208. DOI:10.1109/JSTARS.2017.2755672. 完整作者名见BibTeX，期刊卷期年份为2018。

[R14] B. Li, B. Liu, L. Huang, W. Guo, Z. Zhang, W. Yu. **OpenSARShip 2.0: A large-volume dataset for deeper interpretation of ship targets in Sentinel-1 imagery.** 2017 SAR in Big Data Era: Models, Methods and Applications (BIGSARDATA), 2017, pp. 1–5. 不是IGARSS条目。

[R15] D. Li, Y. Yang, Y.-Z. Song, T. M. Hospedales. **Deeper, Broader and Artier Domain Generalization.** ICCV, 2017, pp. 5542–5550（CVF开放版本）。

[R16] H. Venkateswara, J. Eusebio, S. Chakraborty, S. Panchanathan. **Deep Hashing Network for Unsupervised Domain Adaptation.** CVPR, 2017, pp. 5018–5027. 这里引用其发布的Office-Home数据，不代表实现了其哈希方法。

书目信息通过论文原文、正式会议/期刊页面、作者机构和数据集官方页面核对。BibTeX 中保留核验入口。相关工作用于说明技术关系，不作为本项目实验结果来源；所有实验数字只来自本地工件。
