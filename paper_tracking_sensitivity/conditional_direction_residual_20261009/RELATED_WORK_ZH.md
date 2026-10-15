# 条件方向残差：相关工作与可检验的贡献边界

核验截止：2026-10-09。独立文献笔记，不包含实验结果，也不改变主实验代码。

**判断：方向值得继续，贡献应围绕“强 C4 适配基线上，哪些被平均消去的方向信息仍对目标任务有用，以及怎样用少量 support 稳定恢复”展开。** 可学习 TTA 权重、闭式 ridge、冻结编码器后的 support 残差和 PSD 核均有直接先例；这些先例不否定具体方向，但决定了必须比较什么。

## 1. 候选方案与比较口径

按任务约定，冻结预训练/源分类器，以 C4 平均 R50 表征加同一批 `same64` 目标标签的 ridge 适配为 `effectivebase`；这里不把 `same64` 擅自解释成每类 64-shot。额外利用原始朝向相对 C4 均值的表征差：

\[
z_r(x)=f(R_r x),\quad \bar z(x)=\tfrac14\sum_{r=0}^3z_r(x),\quad
\delta(x)=z_0(x)-\bar z(x).
\]

支持集标签用于学习方向信息的分类作用；elongation / relative-center 描述符由图像无标签提取。残差强度由 source episodes 选择并锁定，原有 5% expert handoff 规则保持不变。本文将其视为待验证设计，未声称已经优于 effectivebase。

## 2. 最近方法：已读正文的六篇

下表依据一手 PDF 的方法与实验章节，公式为保留机制的统一记号；完整入口见文末。没有依据摘要推断未读方法。

| 工作与正文定位 | 实际机制、公式和设定 | 与候选方案重叠 | 可成立的具体差异 |
|---|---|---|---|
| **[1] Better Aggregation in Test-Time Augmentation**, ICCV 2021；§4–5，式(1)–(2)，补充材料§5 | 冻结分类器，得到各增强的类别 logits 矩阵 \(A\)。ClassTTA 为 \(g_c=\sum_m\Theta_{mc}A_{mc}\)，AugTTA 为 \(g=\theta^TA\)。权重非负但不要求和为 1；交叉熵训练，更新后截断负值。实验把发布的 test split 再分为 40% 拟合、10% 选择、50% 评价；涉及 ImageNet、Flowers-102 等。补充材料还试过 **image-dependent augmentation weights** 和 **原始/平均预测的学习混合**，在所测小数据条件下未显著获益。 | “均匀增强平均会损伤一部分样本”“利用标签学习哪些增强有用”，以及输入条件化，都已有先例。 | 当前拟学习的是独立的 \(\delta\) 表征分类作用和 support 相似性，不是重加权已有类别 logits；必须实验证实这一区别产生增益。低维描述符条件化的价值应体现为比直接图像 gating 更稳定，而不能只说首次条件加权。 |
| **[2] Meta-learning with differentiable closed-form solvers (R2-D2)**, ICLR 2019；§3.2、3.4、4，式(4)–(6) | \(W=(X^TX+\lambda I)^{-1}X^TY=X^T(XX^T+\lambda I)^{-1}Y\)，输出校准为 \(\hat Y=\alpha X'W+\beta\)。support 闭式求解，query 交叉熵反传学习 CNN 和校准/正则参数。评测 Omniglot、miniImageNet、CIFAR-FS 的未见类别任务；不是冻结通用 R50 后仅拟合目标集的原始设定。 | support 标签条件化的线性分类器、少样本高维特征、dual ridge 均属已有方法。 | 差异在输入被拆为不变均值与有方向信息的补充块，以及该块的条件化和约束。若未反传穿过 ridge，则不应以“可微闭式元学习”描述当前实现。 |
| **[3] Tip-Adapter**, ECCV 2022；§3.1，式(3)–(4)，§4 | 用归一化 support 特征 \(F\) 和 one-hot 标签 \(Y\) 建缓存；\(A=\exp[-\beta(1-qF^T)]\)，\(\ell(q)=qW_c^T+\alpha AY\)。这是向冻结 CLIP 的 logits 添加 support 检索结果。Tip-Adapter-F 继续训练缓存 keys，保留标签 values 和 CLIP 编码器。评测 11 个分类数据集、1/2/4/8/16-shot。 | 冻结大模型、support 标签、相似性加权、残差加入与少参数适配均直接重叠。 | 当前没有文本原型，基线本身已使用目标 support；增量信息来自 \(\delta\) 而非再次检索完整原始表征。要控制“只是又加一个 support 分类器”的解释。 |
| **[4] ProKeR**, CVPR 2025；§3.1–3.7、4.1，式(13)–(14) | 最接近 staged residual：\(\min_\phi\sum_i\|\phi(s_i)-y_i\|^2+\lambda\|\phi-f_{\rm clip}\|_{\mathcal H}^2\)，表示为基预测器加核展开，系数依赖 \(Y-f_{\rm clip}(S)\)。正文支持任意 base learner、RBF/多项式等核和与线性分类器联合训练的版本。11 数据集上还评测了 **ImageNet 选超参后迁移** 的协议。这里引用其目标和残差结构，不将正文式(14)的系数缩放直接当作本项目求解器定义。 | “PSD/RKHS 中保留基预测器并学 support 残差”已有直接先例；换成强基预测器、源域选超参本身也不足以独立构成贡献。 | 有意义的新增应是 **残差的内容与几何**：只从 C4 丢弃的方向块建立由无标签形态条件化的核，并证明比在均值/完整特征上做一般核残差更合适。 |
| **[5] TAG-Net**, Remote Sensing 2024；§3.2–3.5、4.1，式(1)–(3)、(15) | ResNet50/FPN/CenterNet 上做舰船检测与分类。旋转框生成 TAA/前景监督图；预测支路得到 \(F'\)，两个卷积分支生成 \(\alpha,\beta\)，再做 \(F_{out}=(1+\alpha)\odot F+\beta\)。TAA loss 使用 smooth L1。LSSDD+ 含 11 类、4683 舰船，94/22 幅大图划分训练验证/测试；端到端训练。**不是 few-shot 分类论文。** | SAR 中利用姿态相关信息缓解类内变化、条件式特征调制已有依据。 | 当前从已有四视图特征提取方向残差、冻结骨干、只用类别 support；没有 TAG-Net 的旋转框派生角度/前景监督。任务和标注资源不同，不能直接用检测 mAP 比分类准确率，也不能把它说成“额外人工标角度”的方法。 |
| **[6] ST-PN**, Remote Sensing 2022；§3、4.1–4.4，式(1)–(5) | 四层 CNN 后接 spatial transformer，localization network 预测变换，按 \((x_s,y_s)^T=\Theta(x_t,y_t,1)^T\) 生成采样网格并双线性采样；使用旋转形式来对齐特征。support 类原型 \(c_k=|S_k|^{-1}\sum_{i\in S_k}f_\theta(x_i)\)，query 按 \(\operatorname{softmax}[-\|q-c_k\|^2]\) 分类，交叉熵从头训练整个网络。MSTAR 上是 5-way 1/5-shot；训练 5 类为 17° 俯视角，测试另外 5 类为 15°，每类 15 query。 | **少样本 SAR 中处理方向差异、在特征层而非输入层操作** 已有直接先例，而且不依赖 TAG-Net 的检测框监督。 | ST-PN 学习对齐后做原型距离，目标是减小方向差异；候选保留冻结均值基线，再检验方向差异中是否还有任务所需信息。差异是“何时应恢复方向性”，不是泛称首次特征级方向处理。 |

## 3. 必须说准确的机制边界

以下是针对候选设计的代数分析与建议，不是声称上述论文提出了本模块。

1. **独立特征块才有空间上的增量。** 若只做 \(\bar z+\gamma\delta\)，它等于 \(\frac{1+3\gamma}{4}z_0+\frac{1-\gamma}{4}\sum_{r=1}^3z_r\)。再接同一个固定线性头，就是视图 logits 的加权组合；当 \(\gamma\in[0,1]\) 时还是非负凸组合。相反，\([\bar z,\sqrt\eta\,\delta]\) 允许学习相互独立的均值与方向权重。两者不能混称同一种创新。
2. **PSD 的明确实现示例。** 设无标签描述符为 \(u\)，\(\kappa\) 为 PSD 核，可定义 \(k_\delta(x,x')=\langle\delta(x),\delta(x')\rangle\kappa(u(x),u(x'))\)，于是 \(K=K_0+\eta K_\delta\) 在 \(\eta\ge0\) 时 PSD。理由是 Gram 矩阵的 Hadamard 积与非负和保持 PSD。该构造本身是标准核运算；需要验证的是其方向信息与条件变量的实际价值。若 \(\kappa\equiv1\)，就退化成普通方向块拼接的核形式。
3. **标签条件化的对象要分清。** 上述 \(k_\delta\) 只依赖特征和无标签描述符；标签进入 ridge 系数。因此准确说法是“support-label-conditioned predictor / descriptor-conditioned residual kernel”。只有额外用 support 标签学习核参数或表征映射，才可进一步称 kernel/representation 本身 label-conditioned。
4. **联合增量核与分阶段残差不同。** 在 \(K_0+\eta K_\delta\) 上重新求 ridge 会联动改变基分支系数。固定现有预测器 \(g_0\) 后则可以定义 \(g(x)=g_0(x)+\eta k_\delta(x,S)(K_\delta+\lambda_\delta I)^{-1}[Y-g_0(S)]\)。后者与 ProKeR 的思想尤其接近。两个版本都应使 \(\eta=0\) 恢复实际 effectivebase，特征归一化、截距和既有融合规则沿用主实验。
5. **LOO 是稳定化选择，不是收益保证。** 已用同一 support 拟合的 base 可能几乎插值，直接用 \(Y-g_0(S)\) 会低估未见样本上的误差；support LOO 可检验这一点。若采用 LOO 误差作残差训练目标，这是区别于原始 in-sample proximal 目标的新估计步骤，需单独消融。评估一个 held-out support 时，依赖其标签的映射/门控也要在该 fold 内重拟合。全局 \(\eta\) 仍由 source episodes 锁定；任何 target-support shrink 规则及其参数应先在 source episodes 确定，不能顺手改成查询标签调参。
6. **方向残差的含义有限。** 不要平均四个 \(z_r-\bar z\)，其和恒为零；保留指定 \(z_0-\bar z\) 不存在这个问题。该差向量也不自动正交于 \(\bar z\)，不能直接称为“独立子空间”。离散图像旋转敏感性不等于真实 SAR 观测方位变化的散射模型；elongation/relative-center 是候选条件变量，不应未经验证就称为物理方位估计器。

## 4. 主实验最有信息量的对照

全部使用同一骨干、同一 `same64`、相同 source 超参选择资源和原有 5% handoff 规则；这里仅建议对照，不要求本 sidecar 执行实验。

| 对照 | 可以区分的解释；结果如何影响结论 |
|---|---|
| effectivebase；`[mean4, delta]` 普通 ridge；条件方向核 | 第一处增益检验方向块是否补充信息，第二处增益检验描述符条件化是否必要。条件化无增益时保留较简单的方向恢复结论。 |
| 同一 effectivebase 上，用 mean4 或 z0 建普通核残差，与方向核残差比较 | 排除泛化收益仅来自再次核适配/更强分类器。该组是最贴近 ProKeR 的机制对照，但不冒称完整复现 CLIP 版 ProKeR。 |
| 四视图的 AugTTA/ClassTTA 式 score weighting，与独立方向块比较 | 检验表征空间学习是否超出已有预测加权。适配其数据协议到 same64，并说明不是照搬原论文的大验证集协议。 |
| 去掉描述符；分别只用 elongation、relative-center；一次固定置乱描述符 | 区分有意义的形态条件关系和额外自由度。若置乱后无下降，削弱“形态引导”解释；这属于消融，不是可靠性的普遍证明。 |
| staged in-sample residual；预先规定的 LOO/shrink 版本 | 区分 base 已拟合 support 导致的残差不足，以及残差拟合本身的过拟合；保留失败结果，不持续按目标查询集调整方案。 |

报告采用配对 support splits 的增量及波动，并列出相对 effectivebase 的纠正数和损坏数；仅有净 accuracy 会隐藏方向恢复是否重新引入大量错误。分别报告 handoff 前分类结果和固定规则后的系统结果；不能把增加专家预算当作模块增益。

## 5. 可用于论文的贡献表述：以实验成立为前提

> 我们研究冻结 SAR 识别器经离散旋转平均与少量目标标签适配后，仍可恢复的方向判别信息。相较于学习增强预测权重或在完整特征上施加通用 support 残差，我们保留有效的不变适配基线，将原始朝向相对旋转均值的表征差作为独立补充，并用无标签形态描述符构造条件相似性，由目标 support 学习其分类作用。若配对实验表明，方向块优于一般核残差、条件化优于无条件方向块，且 source 选强度及固定专家预算下仍稳定受益，则贡献是对“不变性与任务所需方向性如何共存”的具体推进。分阶段收缩只有在进一步降低小样本退化时才构成有证据的稳定化机制。

这段表述不宣称首创残差适配、核岭、SAR 方向感知或条件 TTA。若最终只有方向块有效，贡献应收缩到方向信息恢复；若一般核残差已解释全部收益，应如实将方案定位为有效工程组合，不据此否定保留强基线与分析负结果的价值。

## 6. PGRA：真实文献，全文比较尚有缺口

**[7] PGRA: Physics-Guided Rotation Adaptive Attention for Few-Shot SAR Target Recognition** 确有 DOI `10.1109/TAP.2026.3684830`。出版方登记的 Crossref 记录给出 Binze Hu、Xiayang Xiao、Xu Zhang、Haipeng Wang；出版年月为 **2026-07**，记录创建时间为 **2026-04-22**。两者均早于本笔记截止日；记录创建日期不等于首次公开日期。

本次 IEEE 官方文章页返回 HTTP 202 且未提供正文，未取得可阅读的一手方法全文；无法确定是访问限制、付费限制还是页面获取问题。**未核实其旋转建模、attention 公式、support conditioning、数据划分或训练协议，不能根据标题宣布它与本方案相同或不同。** 在补齐该文之前，任何“首次 few-shot SAR 条件方向建模”断言均不成立；这不影响继续检验这里已经明确的机制差异。

## 7. 核实书目与阅读证据

1. Divya Shanmugam, Davis Blalock, Guha Balakrishnan, John Guttag. **Better Aggregation in Test-Time Augmentation.** ICCV, 2021, pp. 1214–1223. [CVF 正文](https://openaccess.thecvf.com/content/ICCV2021/papers/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.pdf)；[CVF 补充材料](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Shanmugam_Better_Aggregation_in_ICCV_2021_supplemental.pdf)。已获取正文 PDF 第 1–7 页，重点读印刷页 1218–1219 的方法/协议，以及补充材料第 3 页§5。年份以 CVF 会议版本为准。
2. Luca Bertinetto, João F. Henriques, Philip H. S. Torr, Andrea Vedaldi. **Meta-learning with differentiable closed-form solvers.** ICLR, 2019. [作者 arXiv 全文](https://arxiv.org/pdf/1805.08136)。已获取并阅读 PDF 第 1–9 页中的方法与实验设定，重点第 4–7 页；PDF 标注 ICLR 2019，不能把 arXiv 编号中的 2018 当作会议年份。
3. Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, Hongsheng Li. **Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification.** ECCV, 2022. [作者 arXiv 全文](https://arxiv.org/pdf/2207.09519)。读取版本标明 2022-07-19；重点核对第 5–6 页缓存、式(3)–(4)与 Tip-Adapter-F，及正文实验设置。
4. Yassir Bendou, Amine Ouasfi, Vincent Gripon, Adnane Boukhayma. **ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language Models.** CVPR, 2025, pp. 25092–25101. [CVF 正文](https://openaccess.thecvf.com/content/CVPR2025/papers/Bendou_ProKeR_A_Kernel_Perspective_on_Few-Shot_Adaptation_of_Large_Vision-Language_CVPR_2025_paper.pdf)。已读正文 PDF 第 1–8 页，重点印刷页 25094–25097 的完整方法与两种超参协议。
5. Dece Pan, Youming Wu, Wei Dai, Tian Miao, Wenchao Zhao, Xin Gao, Xian Sun. **TAG-Net: Target Attitude Angle-Guided Network for Ship Detection and Classification in SAR Images.** Remote Sensing, 16(6):944, **2024-03-07**. [DOI](https://doi.org/10.3390/rs16060944)；[出版方全文 PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-16-00944/article_deploy/remotesensing-16-00944.pdf)。已读第 1–16 页，重点第 8–9 页 TAFM、13–14 页损失与14–15页数据/训练；出版日期以 PDF 首页为准。
6. Jinlei Cai, Yueting Zhang, Jiayi Guo, Xin Zhao, Junwei Lv, Yuxin Hu. **ST-PN: A Spatial Transformed Prototypical Network for Few-Shot SAR Image Classification.** Remote Sensing, 14(9):2019, **2022-04-22**. [DOI](https://doi.org/10.3390/rs14092019)；[出版方全文 PDF](https://mdpi-res.com/d_attachment/remotesensing/remotesensing-14-02019/article_deploy/remotesensing-14-02019-v2.pdf)。已读 PDF 第 1、5–11 页，重点第 5–8 页完整方法和第 8–9 页任务/类别划分；年份与日期由 PDF 首页核实。
7. Binze Hu, Xiayang Xiao, Xu Zhang, Haipeng Wang. **PGRA: Physics-Guided Rotation Adaptive Attention for Few-Shot SAR Target Recognition.** IEEE, 2026. [DOI](https://doi.org/10.1109/TAP.2026.3684830)；[IEEE 文章页](https://ieeexplore.ieee.org/document/11493636/)；[出版方登记的 Crossref 记录](https://api.crossref.org/works/10.1109/TAP.2026.3684830)。仅核实题名、作者、DOI 与登记出版年月，未读方法；不将其计入六篇已完成全文方法比较。

检索范围为上述方法族的一手论文与官方出版/作者全文，未做穷尽式文献覆盖；未读内容不用于支持排他性 novelty 判断。下载和文本提取均在内存完成，唯一交付文件为本笔记。
