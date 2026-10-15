# 频率选择与有限转交：本轮相关工作收尾

日期：2026-10-10。范围：五篇最接近的原始论文；正文之外附比较表和可追溯阅读记录。实验数值与最终取舍由主任务报告，本文件不宣告新实验结果。

## 相关工作与本轮定位

本项目冻结 SAR 主头和四个频带头，用每个目标域的 64 个标签研究跨港口选择，覆盖 23 个 leave-port 设置及 F0921、OSS2。内部频带选择与外部专家转交是两层决策，后者预算不超过 5%。应借鉴正确性估计和相对收益决策，而把贡献放在配对频率证据、少标签适配及失效机制上，不把校准本身包装成新方法。

Mozannar 与 Sontag [R1] 将分类和转交共同写成系统损失，再化为代价敏感分类。其式 (6) 可记为 $L=-\log q_y-\mathbf1[m=y]\log q_\perp$，其中 $q$ 是类别与转交动作的联合 softmax。专家正确时两个动作都得到训练信号，故转交并非一般的低置信度拒识。命题 2 比较专家正确率与 Bayes 分类正确率；第 5.1 节说明受限函数类下，单独训练分类器会错失互补性。这支持比较双方能力，但不能把联合优化的一致性直接移到冻结五头的有限样本校准器。

Verma 与 Nalisnick [R2] 用 OvA 二元适当损失处理专家正确性，式 (9) 为 $P(m=y\mid x)\approx\sigma(g_\perp(x))$。这样避免将转交 softmax 值误认成正确率；但第 5 页明确指出数据稀缺或严重不平衡时可能退化。本轮共享低参数拟合可据此获得动机，却只是受启发的应用，并非完整复现其类别与转交联合损失，也没有自动获得正式校准保证。

Verma、Barrejón 与 Nalisnick [R3] 扩展到多个专家，式 (7) 在主分类器与各专家正确率间选择。其式 (12) 的 softmax 正确率估计为 $t_j/(1-\sum_e t_e)$，共享分母使误校准可能传播；式 (13) 则逐专家使用 sigmoid。多个头可以同时正确，不能将五个正确率强制归一化为互斥分布。该文第 5 节另外构造专家集合，其覆盖对象和查询机制不同于本项目的外层 5% 配额，不能用来声称跨港口风险受控。

Cruz 等 [R4] 的原始 META-DES 已把“某分类器在某样本上是否正确”作为元标签。五组证据包括邻域硬正确性、后验支持、局部准确率、相似输出轮廓的正确性和边界距离；总维数为 $2K+K_p+2$。MLP 选择合格分类器后多数投票。这是能力元学习的直接先例，不能把样本与头配对训练宣称为首创。可借其监督单位，但不照搬邻域搜索、MLP 和投票。我们的推论是：64个支持案例展开五头后仍只有64个样本单位，不能当作320个独立标签；源/目标样本本身也可能相关。本轮LOO指生成五头适配预测时排除该支持案例，不是对校准器做第二层嵌套LOO。

Jitkrittum 等 [R5] 的固定级联最优规则为 $r^*(x)=\mathbf1[P(h_2(x)=Y\mid x)-P(h_1(x)=Y\mid x)>c]$。第 4.2 节把正确性差或真类概率差作为回归目标；第 4.3 节把有限样本误差区分为近似与估计误差。因此专家较好并不足够，相对优势还须能从可用证据预测。本项目内部已有频率输出，信息与成本条件不同于只用第一模型输出预测下游收益的级联，不能直接沿用其性能界。

本轮唯一预设实验臂用目标64个支持案例的五头严格LOO分数作输入，再用全部64个支持标签拟合 pooled L2 logistic：两个共享 score/confidence 斜率、四个收缩频带偏置和一个截距，共七参数，主头偏置置零。每行权重为真实类别平衡权重除以五；逐头估计 $\widehat P_w(\mathrm{correct})$，仅在最佳频带相对主头的估计差大于零时内部切换，外层路由及5%预算不变。这里 $P_w$ 表示类别重加权目标下的正确性估计，不能直接解释成自然类别比例下的校准概率。产生支持案例的类别预测时，残差拟合与类别权重排除该案例；校准器元拟合则使用这64个案例各自的OOF分数和标签。目标查询标签完全不进入这些步骤，也不报告支持集内拟合准确率作为泛化结果。

原有对照保留，采用 source-only 接受或拒绝规则，不扫阈值网格。这次目标域独立拟合与此前源域 33/43 特征 HGB 选择器不同；既有稀疏效用、连续 margin 和配对效用的正负结果均保留，不能据此预言新实验成功。真实类别仅用于训练标签与训练权重，推理特征必须在预测时可得；不得把真类概率差这个监督目标直接当作测试输入。逐头正确率差是本轮实施方式，直接差值回归是对照文献机制，两者不应混称为同一训练目标。若收益仍不稳定，本轮即按预设规则收尾，不追加选择器搜索。

可发展的 novelty 表述是：在冻结 SAR 多频率预测与严格少标签跨港口适配条件下，研究同一样本的配对频率证据能否支持相对优势识别，并把内部选择与有限外部转交分离，通过对照解释互补性何时可兑现、何时失效。新增价值应由跨域证据和机制研究支撑，不声称首创校准、已证明频率不变性、普遍提升或共形保证。这五篇已足以界定本轮借鉴范围，暂不扩展综述。

## 方法比较

| 文献 | 决策对象与监督 | 与本项目的接近点 | 本轮借用及边界 |
| --- | --- | --- | --- |
| [R1] ICML 2020 | 联合学习分类器/单专家转交；类别及专家预测 | 比较双方正确性而非只看主头置信度 | 借系统损失视角；冻结头不继承联合学习结论 |
| [R2] ICML 2022 | 类别 OvA 与专家正确性二元损失 | sigmoid 正确率估计 | 借二元监督；七参数事后拟合不是原损失复现 |
| [R3] AISTATS 2023 | 多专家选择；每位专家正确性及可选集合推断 | 多个候选可同时正确 | 借逐候选估计；集合覆盖不等于外层预算/风险保证 |
| [R4] Pattern Recognition 2015 | 样本-分类器元特征，MLP 筛选后投票 | 小样本下学习条件能力 | 借配对监督；不新增邻域模型、投票或独立样本计数 |
| [R5] NeurIPS 2023 | 冻结级联；正确性差/概率差回归 | 直接围绕切换相对收益 | 借差值决策；输入可用性、加权目标及成本条件不同 |

## 核读记录与一手来源

页码均从 PDF 第一页起算，非会议论文集页码。R1/R3/R4 为本 sidecar 重点方法阅读；R2/R5 由主任务主读，本 sidecar 复核下面所列原文与公式。未声称逐行复核全部附录证明。

- **[R1] `mozannar2020consistent`**：Hussein Mozannar、David Sontag，ICML 2020，PMLR 119:7076-7087。读第 3-4 节，PDF pp.2-4；命题 2、式 (5)-(7)、第 5.1 节，p.4。正式信息：[PMLR](https://proceedings.mlr.press/v119/mozannar20b.html)；[原文 PDF](https://proceedings.mlr.press/v119/mozannar20b/mozannar20b.pdf)。
- **[R2] `verma2022calibrated`**：Rajeev Verma、Eric Nalisnick，ICML 2022，PMLR 162:22184-22202。复核第 4.1-4.2 节，PDF pp.4-5；式 (8)-(10)、定理 4.1、p.5 稀缺/不平衡警示。正式信息：[PMLR](https://proceedings.mlr.press/v162/verma22c.html)；[原文 PDF](https://proceedings.mlr.press/v162/verma22c/verma22c.pdf)。正确标识为 `verma22c`，不是 `verma22a`。
- **[R3] `verma2023multiple`**：Rajeev Verma、Daniel Barrejón、Eric Nalisnick，AISTATS 2023，PMLR 206:11415-11434。读第 3-4.2 节，PDF pp.3-5；式 (6)-(13)、定理 3.2/推论 3.3；第 5 节 p.5 的专家集合设定。正式信息：[PMLR](https://proceedings.mlr.press/v206/verma23a.html)；[原文 PDF](https://proceedings.mlr.press/v206/verma23a/verma23a.pdf)。PMLR HTML 将姓氏写作 Barrejon，PDF 写作 Barrejón；BibTeX 保留重音。
- **[R4] `cruz2015metades`**：Rafael M. O. Cruz、Robert Sabourin、George D. C. Cavalcanti、Tsang Ing Ren，Pattern Recognition 48(5):1925-1935，2015；DOI `10.1016/j.patcog.2014.12.003`。读作者接受稿第 3.1-3.2.3 节，PDF pp.9-17；重点为算法 1-2、表 1、图 2 及元标签/维数定义。该 PDF 含封面，正文印刷页比 PDF 页少一页；2014 接受日期不是正式发表年。正式信息：[出版商页面](https://www.sciencedirect.com/science/article/pii/S0031320314004919)；[作者托管接受稿](https://www.cin.ufpe.br/~gdcc/papers/2015-pr-cruz.pdf)。出版商直开返回 403；卷期、页码、作者、出版年以[出版商提交的 Crossref DOI 记录](https://api.crossref.org/works/10.1016/j.patcog.2014.12.003)核对，方法依据上述完整接受稿，未以二手综述替代。
- **[R5] `jitkrittum2023confidence`**：Wittawat Jitkrittum、Neha Gupta、Aditya Krishna Menon、Harikrishna Narasimhan、Ankit Singh Rawat、Sanjiv Kumar，NeurIPS 2023，Advances in Neural Information Processing Systems 36:9891-9906。复核第 3.2 节命题 3.1，PDF p.4；第 4.2 节和表 1，p.6；第 4.3 节引理 4.2 及其前面的失效条件，p.7。正式信息：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1f09e1ee5035a4c3fe38a5681cae5815-Abstract-Conference.html)；[原文 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/1f09e1ee5035a4c3fe38a5681cae5815-Paper-Conference.pdf)。作者全名采用 PDF，卷号、页码及 DOI 来自官方页面的 citation 元数据。

范围限制：这是最近邻方法比较，不是穷尽式 novelty 检索；未开展独立实验、复现这些论文或核验主任务指标。网页浏览工具未返回可读正文，后续通过 HTTP 获取原始 PDF、在内存中提取文本核读，未额外写文件。未加入可选的 Conformal Risk Control，以免把未使用的保证机制混入本轮结论。方法阅读参考 `paper-reading` 技能，篇幅与文体服从本轮交付要求。
