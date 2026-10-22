# 有符号空间分布读出的有界文献比较

状态：2026-10-23，已完成并按用户回传的旧 split matched-controls 结果更新。核心比较仍限三篇已实际读取方法正文的原始论文：WILDCAT、B-CNN、OpenMax；它们是方法近邻，不是近年论文清单。不扩检、不重复展开 CSRA/SRM/GSoP，不训练、不修改父代码、不 Git。当前优先建议改为：把 score-only 校准纳入强基线，再检验局部分布的额外价值；源参考标定降为待证实假设。

## 先确定实际问题

最新用户回传：旧 split 60 条训练轨迹完成，signed + J 三种子净 +11/+13/+5，平均 OA 约81.68，但 BA 约79.92，低于固定 layer3 A 的80.13；score-only + J 净 +85/+79/+79，平均 OA约82.93、BA约80.58，均明显优于 signed。新 split 的 score-only OA也更好；未据此补写未提供的新 split BA。上述数值为用户提供的主线实证，本支线未重跑或另选模型。

因此，局部矩相对未校准 A 的部分正值不足以证明它超越了已有均值分数中的可利用信息。当前最强的解释边界是：**已测试的固定 W 局部矩读出没有证明比 score-only 决策校准更强，也未证明在校准基线上有条件增量。** 这不等于证明所有局部分布都无信息；也不能将 score-only 的收益算作空间池化贡献。

当前 source_utility_diagnosis 已显示：66 个 selected heads 的源验证 balanced NLL 均改善，其中47个源验证 OA 下降。损失并非只在目标域才发生；优化/选择目标的纠正优先于扩大结构。固定头删除路径的正结果也不能代替对应结构重训结果。

八类范数诊断进一步显示，源上 crude-chemical 的平均 std 差 +0.448504，可按对称乘积分解为范数项 -0.083012、单位方向响应项 +0.531516。这排除了“只是 W 更长”的笼统解释，但既没有证明各类响应可直接比较，也没有证明共享效用映射过强。类间响应尺度不同，不等于类别条件下的效用曲线必然不同。

本轮实际复看既有两张原窗面板：pair_0031218 的 VV64/VH72 为紧凑团状亮结构，conditional_signed 将散货对固定错类分差 -0.107590 改为 +0.110199；pair_0031199 的 VV64/VH88 为较长亮结构并有框外延伸，同一臂将 +0.156864 改为 -0.080524。二者支持“有符号统计能改变竞争、但方向不保证有利”的问题设定，不支持按外观命名散射机理。没有重新选图或把两个案例当总体比例。

## 当前公式究竟保留什么

设每个极化、每个 C4 rotation 的有效空间权重为 a_p，和为1。当前三类原窗使用均匀权重；八类对应实际有效 footprint。令

`r_cp = (W_c-Wbar) · (F_p-mean_a(F))`

`sigma_c = sqrt(sum_p a_p r_cp^2)`

`kappa_c = sum_p a_p r_cp |r_cp| / max(sigma_c, epsilon)`。

sigma 是有效类别方向上的投影标准差，等于 `sqrt(w_c^T Cov_a(F) w_c)`；它是二阶统计的特定投影，不是完整协方差。kappa 是正负半轴能量不平衡，经 sigma 归一后保持分数单位，不是传统三阶偏度，也不是 signed-square-root covariance。

在 sigma 非零且未触发数值下限时，定义 `E+ = sum a_p max(r_cp,0)^2`、`E- = sum a_p max(-r_cp,0)^2`，则 `sigma^2=E++E-`、`sigma*kappa=E+-E-`。因此单个 pol/rotation 的 sigma/kappa 等价于两个半轴二阶能量的重参数化；不能仅靠改名声称新统计。它的有效差别在于 r 反号时 sigma 不变而 kappa 反号，补回 unsigned dispersion 丢失的一种不对称性。最终先非线性池化再 pol 求和/C4 平均，不能从最终聚合 sigma/kappa 精确反推每个 rotation 的 E+/E-。

当前 shared gate 的五个输入是类中心化 A、sigma、类中心化 sigma、kappa、类中心化 kappa。实际代码为 GELU，而不是早期口头 ReLU。signed-only 用共享5→16→1网络产生 tanh 系数乘原 kappa，零末层回退 A；联合臂输出两个系数乘 sigma/kappa。共享参数、同步类别置换等变、源 OOF 的类别语义轴、保留冻结 A，是明确结构约束；小 MLP、门控、类中心化本身不是独立贡献。全局跨样本/类别标准化并没有逐类对齐响应分布。

## 三篇相关主文

### 1. WILDCAT：最接近类别空间证据聚合

Durand et al., CVPR 2017, *WILDCAT: Weakly Supervised Learning of Deep ConvNets for Image Classification, Pointwise Localization and Segmentation*。实际读取 Section 3.1–3.4，PDF第3–5页，特别 Eq.(1)–(3)。

每类先学习 M 个局部响应图，再按类平均为 z_c(p)。空间读出是

`s_c = mean(top-k+ z_c) + alpha * mean(bottom-k- z_c)`。

这是类别相关的两端响应聚合；底部实例是相对低分区域，不要求它们的数值一定小于0。论文继承负证据思想并允许两端权重不同。M个图均值在函数上可折叠成单个线性卷积，作者明确承认，价值在训练组织而非函数凭空新增。

它依赖图像级弱监督、可学习的类别图和骨干微调，目标含多标签识别及定位；不是冻结强 GAP 分类器的源 OOF 残差修正。与我们同类之处是“先投影成类别响应，再非线性聚合空间分布”；不同处是排序极值/截尾均值，对比中心响应的半轴能量及任务效用门控。故不能主张首次利用正负空间证据，但可以检验保留强 A、只读其遗漏分布的增量价值。

### 2. B-CNN：signed square root 不是当前 signed energy

Lin, RoyChowdhury and Maji, ICCV 2015, *Bilinear CNN Models for Fine-grained Visual Recognition*。实际读取 Section 2–2.1、3.1，尤其 PDF第3页的池化及归一化。

两路局部特征 A_p、B_p 形成 `X=sum_p A_p B_p^T`，向量化后作 `y=sign(x)*sqrt(abs(x))`，再 L2 归一化，送分类器并可端到端微调。X一般是未中心化的交叉二阶矩，不应一概改写成中心协方差。

signed square root 是对已经池化的双线性坐标逐元素变换；“sign”来自该坐标的值，而不是分别统计局部类别证据的正峰/负谷。若两路都是 ReLU 特征，外积坐标甚至全部非负。对同一路中心特征整体反号，协方差不变，后续矩阵或逐元素平方根也不能恢复丢失的反号信息；我们的 kappa 则会变号。

所以两者共享非一阶、无序空间统计的背景，但不是同一公式或只差名字。B-CNN保留广泛通道交互；我们只沿冻结判别方向做低维统计，再修正原分类分数。区别是压缩方式与任务接入，不是“比二阶更高阶”这一笼统说法。

### 3. OpenMax：最接近类参考标定，但不是空间池化

Bendale and Boult, CVPR 2016, *Towards Open Set Deep Networks*。实际读取 Section 2.1–2.3、Algorithm 1–2，PDF第3–5页。

对每个真实类别 c，用正确分类的训练样本 activation vector 建均值 `mu_c`，对距离 `d_c(x)=||v(x)-mu_c||` 的最大尾部拟合类专属 Weibull 模型。推理按其尾部模型及排名修正高分类型：`vhat_c=v_c*omega_c`，把移除部分 `sum_c v_c*(1-omega_c)` 分配给未知类，再归一化。关键是各类 activation 分布不统一，因此用源类参考把距离映射到可比较的异常程度。

这是类条件参考标定的明确先例；不是局部空间统计、不是闭集类别修正、也没有我们的 sigma/kappa。它还只用正确训练样本、含尾部模型及未知类拒识假设，不能原样搬入本任务，更不能把参考分位数称已校准的分类正确概率。

对我们的启发仅是：相同原始数值在不同类参考分布中可能含义不同。是否能改善 closed-set shared gate，需要任务证据，不能由 OpenMax 的开放集结果推导。

## 源参考标定与二阶池化的可检验差别

普通二阶池化在一张图内部沿空间位置统计；源参考标定则在一批源图之间，判断该图某个类别方向的统计量相对这个方向的常态有多异常。它们沿不同样本轴操作，不能混称同一种“分布池化”。

概念上可记 `G_(reader,c,m)(t)=P_source[m_c(X)<=t]`，其中 m 为 sigma 或 kappa。这里的 c 首先表示**被投影的类别方向**：同一参考源池中的每张图都可对每个方向贡献统计，不要求未知查询的真实标签。它不同于 OpenMax 使用真实 c 类的正样本参考。若改用 `P(m_c|Y=c)`，那是另一项类条件建模选择，不能偷换。

逐方向分位数映射会引入跨源样本的信息和类专属参考，其作用是让共享 gate 接收“相对该方向常态的位置”，而非强行把 raw sigma 当同一量纲的效用。固定参考下单调映射并不创造图内新信息；它改变的是共享函数的参数化、归纳偏置和源统计输入。有限样本经验CDF有台阶和尾部截断，也并非严格无损可逆。

可检验命题不是“归一化一定更好”，而是：**在相近原 A 竞争状态下，原始响应尺度不同造成的共享效用映射不一致，能否由源参考坐标减轻，而无需学习独立类偏置。** 若问题只是均值/尺度，逐类标准化就可能足够；若相对分位相同的样本仍有相反效用，单纯参考标定也不够。现有范数表并没有完成这些检验。

## 收紧后的下一步问题

唯一优先问题是：**局部分布是否包含强 score-only 校准尚未利用的信息？** 记原 A 为冻结均值读出，实际完整 score-only 映射为 C，强基线应为 `B=C(A)`，而不再只与 A 比较。后续若获准加入局部残差，零头必须回到 B，并与相同预算的 score-only 适配比较。应同时报告 OA/BA、类间净救回/误伤和原决策余量变化，不能只以风险降低宣布增量成立。是否执行由主线决定，本报告不启动新臂。

这一步的公平性比换一个池化名字更重要：冻结相同基校准、loss/selection、split、seed 和预算，不给局部臂额外选择空间。若新增修正头在 C(A) 上训练，必须交代 C 对这些训练事件是否也为 OOF，不能用“基础 Ridge 已 OOF”掩盖二级校准对相同标签的重用。局部统计没有稳定增量时，保留 score-only 强基线即可；负结果不自动授权新的网络。

### 校准后局部方向是否匹配：仅作假设

若 `A=W h+b`，可微校准后的局部特征敏感度满足链式法则：

`d C(A)/d h = J_C(A) W`。

类别 c 对 b 的决策余量方向则是 `(J_C(A)[c,:]-J_C(A)[b,:]) W`。这给出一个可检验假设：当 C 改变实际类别竞争时，旧 W 上的局部矩是否仍与新的决策边界匹配？它不是“校准一定导致 W 失效”的结论。若 C 只是统一正温度缩放，argmax 边界不变；若只是加类偏置，边界位置可变但 Jacobian 为 I、方向仍是原 W。因此方向错配不能仅由校准有效来推断。

若 C 非线性，`J_C(A(x)) W` 是样本依赖的一阶敏感度，不是全局固定分类器，也不能把有限局部扰动后的校准分数等同于一阶近似。Jacobian 必须覆盖完整 C（含恒等残差、输入标准化/类中心化），并遵循原极化/C4积分；不能只微分末层或忘记原 A 路径。

更重要的是，`sigma(Jv)` 一般不等于 `J sigma(v)`，kappa 亦然。类方向混合需要相应空间联合响应信息，不能拿保存的逐类 sigma/kappa 直接乘 J 就声称得到新矩。这里仅说明假设与可达限制，不重提图、不计算 Jacobian、不更换方向。主线原图 kappa 贡献分析可提供依据，但本报告不预判其结果。

### 源参考标定的当前位置

前述“逐类参考分位坐标”仍与普通二阶池化不同，但现在是次级假设：只有在强校准基线之外出现可利用的局部增量、且原始尺度的共享效用映射确有问题时，才值得比较。需以逐类位置/尺度标定为必要对照，不能将普通标准化或更大类专属 MLP 的效果命名为参考分布创新。

参考仍须配套各 reader：inner held 只能使用 inner fit 的同 reader 源参考，outer 推理只用 outer source train；不得纳入 held/validation/test，也不能混用未经定义的分位轴。参考分位不是正确概率，没有跨港口保证；改变条件坐标也不应悄然改变 raw kappa 载体的分数单位。

有意义的贡献只能在证据成立后表述为“超越强校准均值基线的局部分布增量及其适用条件”。校准方向配套或源参考坐标是可能解释与方法选择，不是现阶段已经确立的贡献，更不是首次负证据、首次高阶池化或物理去噪。

## 来源与阅读边界

均为原始会议论文，已通过出版社/CVF PDF 实际读取上述方法页，无二手博客依据。下列为可追溯原文地址：

1. WILDCAT：`https://openaccess.thecvf.com/content_cvpr_2017/papers/Durand_WILDCAT_Weakly_Supervised_CVPR_2017_paper.pdf`
2. B-CNN：`https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf`
3. OpenMax：`https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf`

本轮早前还读取了 WELDON 与 iSQRT-COV 的方法段，按最新指令将主比较收在上述三篇，不继续扩展检索或罗列 SAR 标题。此有界比较不证明不存在更接近先例，不作 first claim。原论文数据主要为自然图像，不视为 SAR/身份隔离跨场景收益证据。

本地依据：父 PROTOCOL_ZH.md、train_signed.py、extract_signed.py、FINDINGS_ZH.md；source_utility_diagnosis/FINDINGS_ZH.md；source_utility_alignment/PROTOCOL_ZH.md；八类 spatial_std_transfer/weight_norm_diagnosis/FINDINGS_ZH.md。最新旧 split 60轨迹及新 split score-only OA比较来自本轮用户回传，明确作为结果更新而非本支线重跑；主线四例 kappa 贡献图的机制结论未代写。

仅创建并更新本文，无新训练、统计拟合、Jacobian计算或后台进程；父代码/数据未改，无 Git。
