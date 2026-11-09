# 近期直接相关 SAR 船型分类文献阅读笔记

核查日期：2026-11-09。独立文献侧线，仅新增本文件；未修改主稿、引用编号、模型或其他文件，未执行 Git 操作。

## 1. 本轮结论与阅读边界

建议补入下列四篇，覆盖双极化表征、同模态跨分辨率域适应、光学到 SAR 分类域适应及目标区域引导表征。三篇为 IEEE TGRS，一篇为 IEEE TASE。四篇均不是仅做检测，也不重复用户列出的已有文献。没有为凑齐 ISPRS JPRS/Pattern Recognition 或 2026 年份而加入相关性不足的论文；本轮不是穷尽检索，也不声称四篇是截至今日的最新全部工作。

| 标识 | 文献与正式年份 | 纳入理由 | 实际阅读范围 |
|---|---|---|---|
| R1 | Xie et al., DBDN, TGRS 2024 | 最直接的双极化船型分类表征对照 | 作者摘要、IEEE 元数据对应的 Crossref 记录；未取得方法全文 |
| R2 | Liu et al., MSADAN, TASE 2025 | 最直接的多传感器、多分辨率 SAR 船型跨域分类 | 作者摘要及 Note to Practitioners、Crossref 正式书目；未取得方法全文 |
| R3 | Shi et al., Progressive Feature Alignment, TGRS 2024 | 明确处理船型分类中的跨域偏差和类别结构 | IEEE 摘要检索结果、作者公开摘要、通讯作者主页与 Crossref；未取得方法全文 |
| R4 | Chen et al., PGMNet, TGRS 2025 | 可读正文，直接涉及 R50、区域引导和特征更新 | 公开期刊全文的 Sec. III-A–D、式(1)–(11)、图2–4图注、Sec. IV-A–D 的实验设置与消融文字；不是逐页完整精读 |

四篇 DOI、题名、作者、卷、年份及页码均逐条查询 Crossref 的 IEEE 注册记录；这与从搜索摘要猜测出版信息不同。IEEE 网页/全文访问遇到验证或读取失败，因此没有把失败访问写成已读全文。R4 的公开全文可读，文章号和具体出版日期由期刊正文首页核实。其余缺失文章号不补猜。

比较基准为当前 MANUSCRIPT_ZH.md：VV/VH 分别前向、共享 ResNet-50 卷积；M1 使用极化专属候选类别响应及局部交叉统计，并将响应用于 M2；M2 保留 OBB 框内、轴端、侧向身份，区分区域读出与内部回写；M3 从带源标签参考库形成类别证据，保留参考选择与边际聚合，并向中间骨干回写。当前 M3 使用固定锚点残差，不能把文献差异包装成新版三个模块已实时联合学习。

## 2. R1：DBDN，双极化分类最直接对照

**已核验书目**

Nishang Xie, Tao Zhang, Weiwei Guo, Zenghui Zhang, and Wenxian Yu, “Dual Branch Deep Network for Ship Classification of Dual-Polarized SAR Images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024. DOI: **10.1109/TGRS.2024.3372021**.

- DOI：https://doi.org/10.1109/TGRS.2024.3372021
- IEEE：https://ieeexplore.ieee.org/document/10457854/
- 官方注册元数据：https://api.crossref.org/works/10.1109/TGRS.2024.3372021
- 实际读取的作者摘要：https://www.researchgate.net/publication/378677189_Dual_Branch_Deep_Network_for_Ship_Classification_of_Dual-polarized_SAR_Images

**核对到的方法。** 输入构造模块形成极化特征引导和纹理特征引导的两种伪 RGB 图像；特征提取使用迁移层与 ConvNeXt-Attention block；融合分类部分包含增强倒残差块和通道空间注意。实验对象明确为 OpenSARShip2.0 船型分类。上述为作者摘要可确认的模块级描述，尚未核查伪 RGB 各通道的公式、双分支权重共享方式和具体数据划分。

**与本稿的关系与差异。** 最适合放在双极化表征段，与共享 R50 加极化专属 M1 比较：DBDN 的两分支由两种构造表示组织，不能直接等同于本稿的 VV/VH 两路。本文可强调候选类别响应如何继续指导 M2，而不是把“双分支”“极化注意”本身当作差异。摘要未报告本稿的 OBB 三区域读出或源参考类别证据回写；由于未读全文，不据此断言作者全文完全没有任何区域或参考机制。其公开摘要也不构成严格跨场景划分的证据。

**建议引用位置。** 第2.1节双极化船型分类，紧接已有 DPIG/CPINet 的讨论；第4.1节不宜用它证明本文的共享权重设计更优。

## 3. R2：MSADAN，跨分辨率 SAR 到 SAR 分类

**已核验书目**

Zhunga Liu, Kun Li, Longfei Wang, and Zuowei Zhang, “Multi-Scale Alignment Domain Adaptation for Ship Classification in Multi-Resolution SAR Images,” IEEE Transactions on Automation Science and Engineering, vol. 22, pp. 4051–4062, **2025**. DOI: **10.1109/TASE.2024.3407130**.

- DOI：https://doi.org/10.1109/TASE.2024.3407130
- IEEE：https://ieeexplore.ieee.org/document/10547709/
- 官方注册元数据：https://api.crossref.org/works/10.1109/TASE.2024.3407130
- 实际读取的作者摘要及实践说明：https://www.researchgate.net/publication/381176510_Multi-Scale_Alignment_Domain_Adaptation_for_Ship_Classification_in_Multi-Resolution_SAR_Images

**年份处理。** Crossref 的正式卷页记录是 2025、卷22、4051–4062。DOI 含 2024、聚合页标题栏显示 2024，不应因此把最终卷页书目写成 2024。本轮未独立核实具体 early-access 日，故不填写。

**核对到的方法。** 作者把多传感器 SAR 的尺度/分辨率差异作为跨域分类难点，设计尺度感知特征提取器、多层双线性融合驱动的对抗对齐，并以类别对比损失减少类间混淆。阅读限于摘要和 Note to Practitioners；未核实网络层数、对齐公式、目标域样本使用细则及身份/过境划分，不能据二手综述补齐这些信息。

**与本稿的关系与差异。** 这是引言中“已有工作已经专门处理 SAR 船型跨域问题”的直接证据。与 M1 的联系是判别表征学习；与 M2 的联系是尺度变化下的证据组织，但多尺度分布对齐不等于 OBB 相对区域池化。M3 的源参考类别证据回写与摘要所述域级对抗对齐属于不同作用环节。本文可以讨论保留区域来源及类别证据的研究问题，不能声称以 OBB 或原生像素裁剪就已解决跨传感器物理分辨率差异，也不能仅凭摘要宣称两个方法的训练信息条件相同。

**建议引用位置。** 引言跨场景动机和第2.1节跨域分类；可在输入尺度说明附近作为问题背景，但不作为本文尺度处理有效性的证据。

## 4. R3：Progressive Feature Alignment，跨模态分类域适应

**已核验书目**

Yu Shi, Lan Du, Yuchen Guo, Yuang Du, and Yiming Li, “Unsupervised Domain Adaptation for Ship Classification via Progressive Feature Alignment: From Optical to SAR Images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–17, 2024. DOI: **10.1109/TGRS.2024.3458937**.

- DOI：https://doi.org/10.1109/TGRS.2024.3458937
- IEEE：https://ieeexplore.ieee.org/abstract/document/10679175
- 官方注册元数据：https://api.crossref.org/works/10.1109/TGRS.2024.3458937
- 通讯作者机构主页：https://web.xidian.edu.cn/dulan/paper.html
- 实际读取的作者摘要：https://www.researchgate.net/publication/383999418_Unsupervised_Domain_Adaptation_for_Ship_Classification_via_Progressive_Feature_Alignment_From_Optical_to_SAR_Images

**核对到的方法。** 任务为有标签光学域到无标签 SAR 域的船型分类。域级 FCDA 用重建与同图不同扰动版本的一致性缓解光学源偏差；类级 FECA 用伪标签跨域对比和一致性学习改善目标域类别结构。只读到完整作者摘要和 IEEE 对应摘要内容，未读损失定义、训练调度或方法正文。

**与本稿的关系与差异。** 其类别结构约束与 M1/M3 的判别目标有关，但伪标签跨域对比属于训练目标，不能直接等同于推理期参考证据回写。其设置使用无标签目标域参与适应，且源域是光学；本稿按当前描述使用源 SAR 标签和源参考库开展跨场景分类，不能把前者称为源端独立训练的域泛化方法。M2 的 OBB 区域身份不由该文域/类对齐摘要直接覆盖。可以引用它说明“跨域学习不仅需要域级统计，还关注类别结构”，不能由此推导本稿 M3 必然优于域适应。

**建议引用位置。** 第2.1节跨域分类，明确标注“光学到 SAR 的无监督域适应”。不要与同作者2022年题名相近的船舶检测论文混用。

## 5. R4：PGMNet，已读方法正文的区域与表征对照

**已核验书目**

Liang Chen, Jianhao Li, Honghu Zhong, Hao Shi, Zhu Yang, and Wei Li, “PGMNet: A Prototype-Guided Multimodal Network for Ship Recognition in SAR Images,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, Art. no. **5215517**, pp. 1–17, 2025. DOI: **10.1109/TGRS.2025.3583282**.

- DOI：https://doi.org/10.1109/TGRS.2025.3583282
- IEEE：https://ieeexplore.ieee.org/document/11052669/
- 官方注册元数据：https://api.crossref.org/works/10.1109/TGRS.2025.3583282
- 实际读取的公开期刊全文：https://www.researchgate.net/publication/393081846_PGMNet_A_prototype-guided_multimodal_network_for_ship_recognition_in_SAR_images

正文首页明确给出 publication date 为 **2025-06-26**，current version 为 **2025-07-21**。不要沿用聚合页面的“January 2025”。公开全文页面标注 CC BY 4.0；此次阅读其全文文本及图注，没有完成页面渲染或全部图表逐项复核。

**核对到的方法。** Sec. III-B 由 Lee 滤波、阈值分割及形态学处理生成当前图像的目标掩膜。R50 的 PGAM 以图像特征生成 Q/V、掩膜生成 K，注意聚合后与原特征拼接（III-C，式1–5）。III-D 从关键区域强散射点估计长宽，编码成文本；CLIP 预训练文本编码器配合动态通道矩阵迭代更新池化后的视觉向量（式9–11）。IV 部分核查了数据集、训练设置和模块消融文字，未将跨数据集分别评价当作跨域迁移协议。

**与本稿的关系与差异。** 与 M1 同样涉及表征调节，但 PGAM 的引导量是目标掩膜，不是极化专属候选类别响应。与 M2 最直接：PGMNet 重点组织目标关键区域，本稿显式保留框内和框外轴端/侧向身份，并区分区域读出与内部回写。与 M3 的关键区别是“prototype”不指带源标签的参考样本集合；其文本更新也不是局部参考类别证据场向中间骨干的空间回写。此差异由已读方法支持，但不自动证明本稿更有效。本文原生幅度 chip 与该文的关键区域/尺寸文本处理也不能直接视为同输入预算。

**建议引用位置。** 第2.2节目标区域引导表征，补充第2.3节参考方法讨论时的一句术语区分。优先用它检验“已有 SAR 方法是否已经使用区域指导和特征更新”，而不是按 prototype 名称归入少样本原型分类。

## 6. 建议引用段落（仅供主代理整合，未写入主稿）

以下 R1–R4 是本笔记内部标识，不改变主稿现有参考文献编号。前三篇仅摘要级阅读，因此段落仅使用已核对的高层方法事实。

### 6.1 引言或2.1节：已有跨域船型分类进展

跨域 SAR 船型分类已有针对性研究。Liu 等通过尺度感知特征、多层双线性融合对齐及类别对比约束处理多分辨率 SAR 数据的分布变化[R2]；Shi 等则面向有标签光学图像到无标签 SAR 图像的迁移，结合域级校准与类级对齐改善目标域类别结构[R3]。这些方法表明，场景差异需要结合成像条件与类别判别共同考虑。本文进一步研究局部证据的极化、几何区域来源及源参考类别关系如何参与预测，并据配套实验限定各环节的实际作用。

整合提示：若下一句声称“训练时完全不接触目标域”，主代理应以主稿实际开发/选择协议核实措辞；本笔记没有审计该声明。R3 的无标签目标域适应条件应明确保留。

### 6.2 第2.1节：双极化表征

DBDN 将双极化信息组织为极化特征和纹理特征引导的两种伪 RGB 表示，再通过双分支特征提取与融合完成船型分类[R1]。本文采用共享 ResNet-50 的逐极化前向，并在极化专属模块中构造候选类别响应；比较重点是这些响应如何继续指导区域取证，而非双分支或注意机制本身。

### 6.3 第2.2–2.3节：区域引导与参考证据的区别

PGMNet 以当前 SAR 图像生成的目标掩膜指导视觉特征对齐，并融合尺寸描述的文本表征[R4]。与此相关，本文也显式组织目标相关信息，但 M2 保留 OBB 框内、轴端和侧向的区域身份，M3 则从带源标签的参考样本形成类别证据并回写中间特征。因而，两者需要区分掩膜引导、几何区域组织与跨样本类别证据三种信息来源。

## 7. 收束记录

- 优先补引顺序：R1（双极化直接性）、R2（SAR跨域直接性）、R4（可读正文和区域关联）、R3（补全跨模态域适应背景）。若只补三篇，保留 R1/R2/R4。
- 用户补充的 DBDN 已纳入。HDSS-Net 和相位研究未继续扩展；相位输入与当前幅度 chip 存在观测信息差异，不能未经输入与协议核查就写成可直接替换的同预算方法。
- SAAN、BiEnFRN、PASTA/FGFL 等通用关系与干预文献留给主代理，本轮未重复检索。
- 不据“摘要未提及”得出全文不存在某机制的结论；未补造前三篇的损失公式、支持协议、权重共享、划分或消融数值。
- 未作同协议复现，因此不横向排序这些论文和本稿的 OA/BA，也不从四篇阅读推导全领域首次性。
