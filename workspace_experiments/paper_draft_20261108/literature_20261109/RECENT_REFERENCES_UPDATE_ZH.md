# 近期高质量参考文献补充

核查日期：2026-11-09。用户要求更新过旧的参考文献。本轮新增9篇2023–2025年正式发表工作，原有SCRC-Net（2026）保留；参考文献从27篇增至36篇，其中14篇为2023–2026年。不以影响因子或年份代替相关性，不宣称穷尽最新研究。

## 纳入与阅读范围

| 工作 | 正式来源 | 阅读范围 | 对应问题 |
|---|---|---|---|
| DBDN | TGRS 2024 | 作者摘要、IEEE注册书目 | 双极化与纹理构造表示，不能简单等同于VV/VH各一路 |
| MSADAN | TASE 2025 | 作者摘要及实践说明、IEEE注册书目 | 多分辨率SAR的尺度感知、域对齐和类别区分 |
| Progressive Feature Alignment | TGRS 2024 | 官方/作者摘要、作者机构目录、IEEE注册书目 | 光学到SAR域适应，不与源端独立训练混为同一协议 |
| PGMNet | TGRS 2025 | 公开期刊正文III-A–D、公式、图注及IV实验/消融文字 | 目标掩膜引导和尺寸文本融合；prototype不是源参考库 |
| Selectively Augmented Attention Network | TCSVT 2025 | 作者接受稿III-B–E、图3–6及消融段 | 选择查询—支持匹配增强视图，再计算空间/通道注意 |
| BiEnFRN | TPAMI 2024 | 正式书目、作者项目说明、摘要；未获得方法全文 | 双向/自重构及集成，区别于类别证据场回写 |
| PASTA | ICCV 2023 | CVF正文3.1–3.3、算法1、表7及对应讨论 | 从数据频谱差异出发构造增强，不是SAR物理因果证明 |
| FGFL | ICCV 2023 | CVF正文3.1–3.3、图3–4、表1与表4及讨论 | 任务相关频率掩膜指导空间学习，不等于统一删高频 |
| Contrastive-ACE | TIP 2023 | 作者机构正式目录、作者预印本摘要；未完整核查期刊方法全文 | 以平均干预效应组织跨域约束，区别于输入干预选样 |

前三篇SAR工作的摘要级事实只用于高层方法比较，未补写未读到的公式、训练协议或消融数值。BiEnFRN与Contrastive-ACE同样只引用已确认的核心框架，不声称逐式复核。详细SAR笔记见RECENT_SAR_READING_ZH.md。网上发布的作者论文属于论文内容来源，不以聚合站概述替代方法证据。

## 与本文的实质关系

**M1/M2。** SAA从增强样本中选择参考相关内容，再估计注意；DBDN在输入表示层组织极化与纹理；PGMNet由目标掩膜和尺寸描述引导表征。本文保留逐极化候选类别响应、OBB框内/轴端/侧向身份，并检查响应指导读出是否优于配套均值与内容注意。这里的差异是取证依据和消息接口，不是首次注意、首次双分支或首次区域操作。

**M3。** BiEnFRN以重构关系增强分类；SAA以重加权特征计算原型距离。当前M3把相对源参考类别的判别关系编码为类别证据场并回写中间骨干。不能仅凭操作不同声称优于这些方法；已有内部聚合/匹配对照与实际跨场景结果继续作为本稿的效用依据。

**干预训练。** PASTA根据合成—真实频谱差异设计频率相关扰动，FGFL生成任务相关频率掩膜，Contrastive-ACE学习效应稳定性。本文操作单位是极化×目标相对区域×尺度，并追踪正确类与竞争类分差；M1–M3仍不使用显式频域输入。新增文献提供问题与方法参照，不把干预筛选在当前组合上尚未兑现的附加收益改写为已验证成功。

## 主要原始入口

- DBDN：https://doi.org/10.1109/TGRS.2024.3372021
- MSADAN：https://doi.org/10.1109/TASE.2024.3407130 （正式卷页2025，不按DOI中的2024误写年份）
- Progressive Feature Alignment：https://doi.org/10.1109/TGRS.2024.3458937
- PGMNet：https://doi.org/10.1109/TGRS.2025.3583282
- SAA：https://doi.org/10.1109/TCSVT.2024.3480279 （2024在线、2025卷期）
- SAA作者稿：https://discovery.ucl.ac.uk/id/eprint/10198870/
- BiEnFRN：https://doi.org/10.1109/TPAMI.2024.3376686
- BiEnFRN作者目录及项目：https://www.dongliangchang.cn/publications/ ； https://github.com/PRIS-CV/BiEnFRN
- PASTA：https://openaccess.thecvf.com/content/ICCV2023/html/Chattopadhyay_PASTA_Proportional_Amplitude_Spectrum_Training_Augmentation_for_Syn-to-Real_Domain_Generalization_ICCV_2023_paper.html
- FGFL：https://openaccess.thecvf.com/content/ICCV2023/html/Cheng_Frequency_Guidance_Matters_in_Few-Shot_Learning_ICCV_2023_paper.html
- Contrastive-ACE：https://doi.org/10.1109/TIP.2022.3227457 （正式卷页2023）
- Contrastive-ACE作者正式目录：https://www.eee.hku.hk/~ycwu/publications.html
- Contrastive-ACE早期稿：https://arxiv.org/abs/2106.00925 （早期稿作者列表与期刊版不同，以期刊版列书目）

## 处理原则与本轮不改动

保留ResNet、CAM、SE、原型网络、FRN、ROAR等与实际算子或分析直接相关的经典来源，不为抬高近年比例删除必要归属。新增文献已进入引言和2.1–2.4的比较，而非仅添加文末清单。未增加未跑过的基线结果，未修改模型、公式、实验数值、中文摘要和图7单栏要求。

下载的作者稿和期刊/会议PDF仅留在recent_fulltext_local供本机阅读，不随Git归档发布。书目与阅读记录、主稿及生成稿可归档。
