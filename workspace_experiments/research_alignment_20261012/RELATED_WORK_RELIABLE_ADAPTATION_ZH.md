# 可靠少样本适配路线：前人基础与当前差异

核对日期：2026-10-12。定向近邻调研，不是穷尽式 novelty 检索，不是新增实验或论文录用判断。依据作者原文与会议页面；方法级核对重点是 CLAP、SS-Text+、Tip-Adapter、SUR、R2-D2，WiSE-FT用于原则性参照。

## 一句话结论

有明确基础：少量案例辅助预训练模型、保留先验以缓解适配偏移、按任务选择表示、利用可微闭式求解器，均有前人方法。最接近当前“不完整支持案例可能带偏模型”的两篇是 CLAP 与 SS-Text+。我们应在它们基础上推进 SAR 适配机制，而不是宣称首次发现支持偏差或首次使用内部验证。

## 六篇方法与可借鉴点

| 文献 | 原方法要点 | 对我们有用的部分 | 不能混同 |
| --- | --- | --- | --- |
| Silva-Rodriguez et al., A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models, CVPR 2024，CLAP | 分类头保持接近原零样本文本原型；逐类约束强度由支持信息估计 | 不借额外验证标签控制先验保留，避免一个强度用于所有类别 | 不是四候选分组CV；主设置每类均衡K例，缺类公式不能直接照搬 |
| Silva-Rodriguez et al., Few-Shot, Now for Real: Medical VLMs Adaptation without Balanced Sets or Validation, MICCAI 2025，SS-Text+ | 不平衡/缺类支持设定；按类支持量结合视觉信息与文本先验 | 缺少同类支持时保留先验；总标签预算与类别组成分开评价 | 医学VLM文本先验不是我们的SAR源分类器；不是已证明我们的选择器有效 |
| Zhang et al., Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification, ECCV 2022 | 支持特征为键、独热标签为值，检索项与原CLIP分数相加；F版可学缓存键 | 案例关系用于校正原预测，是直接结构参照 | 不是先验残差的带权岭求解；训练免费不等于超参数选择免费 |
| Bertinetto et al., Meta-learning with differentiable closed-form solvers, ICLR 2019，R2-D2 | 支持岭回归闭式求解；外层通过求解器学习表示和超参数 | 改变表示时配套求解，训练目标可贯通实际分类读出 | 我们当前冻结骨干的源先验残差适配不等于完整R2-D2元训练 |
| Dvornik et al., Selecting Relevant Features from a Multi-domain Representation for Few-shot Classification, ECCV 2020，SUR | 多域表示分块加权，利用支持集NLL优化权重，最近类中心分类 | 表征贡献依赖任务，低维选择可替代复杂适配 | 不是我们的双路配方或MMSI分组CV；不证明任意双骨干必胜单骨干 |
| Wortsman et al., Robust Fine-Tuning of Zero-Shot Models, CVPR 2022，WiSE-FT | 插值原模型和微调模型权重，平衡目标性能与分布偏移鲁棒性 | 原能力应保留为可用参照，强更新不总是理想 | 是权重插值，不等价于我们的概率融合/核修正，亦非纯少样本支持研究 |

## 最值得读透的细节

CLAP 的式(7)为分类损失加逐类原型偏离惩罚；式(12)的类权重来自零样本模型在该类支持样本上的正确类概率均值，不是每类正确率，也不是CV搜索。作者只作一次外层乘子更新以减少支持过拟合。原文还讨论同总标注预算下留验证集与全用于适配的取舍。原型与分类器的归一化、温度和固定训练设置不能忽略。

SS-Text+ 区分每类均衡、自然不平衡可缺类、不平衡但各类至少一例三个设定。其闭式更新只最小化所拆损失的一项，是完整正则交叉熵的近似；不能写成完整目标的精确解。缺类时仍保留文本先验。作者代码与正文的温度/原型后处理写法有待复现核对，不在未运行时认定二者数值等价。

Tip-Adapter 的支持检索输出是非负相似度加权独热标签后加原先验；我们支持岭解算得到的残差系数与此不同。R2-D2支持通解算器的训练设计，但不能单凭引用认定我们的任务损失训练一定有效。

## 我们现在有什么，缺什么

- 已完成：同一64标签内按已知MMSI分组的3x4折验证，选择 original/resolve/retain_half/retain_full。它是先验更新接法选择，不是本轮新增支持方向修正/强度匹配选择。结果见 `target_domain_gap_20261012/support_cv_selection/FINDINGS_ZH.md`；更广覆盖已见 `frozen_candidate_coverage_20261012/readout/FINDINGS_ZH.md`，不要重做为“尚未开展”。
- 已完成：支持方向与解算系数交叉干预、强度替代、缺席类别的先验/校正分数分解。F_B16有范围明确的正信号，但方向修正相对逐类强度对照的额外收益仍依赖抽样。见 `support_mechanism_20261012/FINDINGS_ZH.md`。
- 未完成：新方向修正候选的支持侧可靠选择，以及相对上述文献启发的匹配基线的收益验证。不把前人结果当成本研究证据，不把已有先验更新选择器误写成尚无。

## 可发展的贡献论述（候选，不是已证实结论）

面向中低分辨率SAR闭集跨港识别，将研究对象从“有多少目标标签”进一步落实到“这些标签对各类证据提供了什么约束”。在保留源分类先验的基础上，区分支持关系重组、系数重求解和分数强度的作用，利用不新增标注的支持侧信息决定适配方式，并通过缺类/不平衡、救回误伤及完整系统评价，验证何时有净收益。

这段定位只有配套方法与比较成立后才能成为贡献。不能声称“不平衡缺类”“保留先验”“用支持集决定权重”“闭式求解”本身是我们首次提出，也不应仅以换成SAR数据说明创新。MMSI分组是协议必要性，不自动构成算法创新。支持内CV不新增标注，但并非完全不做验证。

## 下一步借鉴顺序

先对照 CLAP/SS-Text+ 的先验保留思想设计最小SAR适配基线，复用原支持/查询、源模型、信息与选择预算；优先检验缺类时的保护作用与已有类别收益的权衡。不要机械移植文本编码器，也不把每类样本量直接当作可靠性的真值。先证明相对原模型及简单缩放有可利用净收益，再决定是否加入新选择器。当前未启动这些新实验。

## 原文入口

1. CLAP 原文：`https://arxiv.org/html/2312.12730v2`；CVPR 2024，非NeurIPS。
2. SS-Text+ 原文：`https://arxiv.org/html/2506.17500v1`；官方：`https://papers.miccai.org/miccai-2025/0334-Paper4796.html`；DOI `10.1007/978-3-032-04981-0_23`。不要误用2506.18731。
3. Tip-Adapter：`https://arxiv.org/html/2207.09519`；ECCV版 `https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf`。
4. R2-D2：`https://arxiv.org/html/1805.08136`；作者ICLR代码 `https://github.com/bertinetto/r2d2`。
5. SUR正式题名以ECCV版为准：`https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550766.pdf`；不要混用预印本Universal与正式Multi-domain题名。
6. WiSE-FT：`https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html`。
