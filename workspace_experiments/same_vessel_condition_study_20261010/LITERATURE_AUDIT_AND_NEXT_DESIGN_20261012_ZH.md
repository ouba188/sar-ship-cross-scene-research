# 已参考文献审计与下一步可借鉴设计

日期：2026-10-12。范围：用户贴出的13篇清单、当前同船条件研究的代码/结果，以及既有完整模型的文献记录。本轮补读相关论文的方法正文，未启动新实验，未改动主模型。这是针对当前瓶颈的近邻方法阅读，不是穷尽性 novelty 检索。

## 1. 结论

不是没有参考文献，而是要把“列在推荐清单”“已经读过”“借鉴并实现”“按原方法复现”分开。范围内没有找到这13篇的官方实现完整复现记录，不意味着没有借鉴。最明确的已实施借鉴是同船分数一致性及训练期特权信息教学。

下一步优先级建议：样本关系/支持核教学（RKD、SPKD）高于继续压缩一个更稳定的隐空间；学生实际收益驱动的教学权重（Ren、MetaDistil）次之；FactorCL 用于保留共享之外仍有用的信息。先验证收益空间与可利用性，再开发模块，不一次叠加所有论文损失。

## 2. 13篇清单到底用到了哪里

“未找到”仅指本轮检索的本地研究目录、代码、记忆与报告，不能断言从未阅读。表中历史状态与本轮补读分开。

| 文献 | 此前可核实状态 | 本轮判断/启发 |
| --- | --- | --- |
| MatchDG: Domain Generalization using Causal Matching，ICML 2021 | 已读对象匹配方法；未复现迭代匹配算法 | 同船配对有依据，但有真实MMSI不等于已做MatchDG；学习对象关系仍值得保留 |
| CoRe: Conditional Variance Penalties and Domain Shift Robustness | 有原文笔记和明确的CoRe-inspired同船中心化分数一致性实现 | 已尝试，不能重新当新模块；BA 37.274低于task 37.345。期刊年份本轮未确认，暂按arXiv标识引用 |
| SIE: Self-Supervised Learning of Split Invariant Equivariant Representations，ICML 2023 | 范围内未找到具名实施记录；已有的是ISR，不是SIE | 本轮补读：不变分支与变换条件化等变分支。观测条件不完整时，不能直接照搬已知变换假设 |
| FactorCL: Factorized Contrastive Learning: Going Beyond Multi-view Redundancy，NeurIPS 2023 | 未找到对应互信息目标的实现记录 | 本轮补读：共享之外的视图特有信息也可能与任务有关，值得保留；先测增量，不急着引入整套互信息估计器 |
| Multi-View Causal Representation Learning with Partial Observability，ICLR 2024 | 未找到对应实现记录 | 本轮补读：不同视图可能只共享部分潜因子；支持局部共享假设，不支持无条件找出全局纯因果坐标 |
| Weakly-Supervised Disentanglement Without Compromises，ICML 2020 | 有原文方法及假设阅读；未实现Ada-GVAE/Ada-ML-VAE | 自然同船观测未必只改变少数独立因素，不可直接套用其识别结论 |
| Unifying Distillation and Privileged Information，ICLR 2016 | 已读；训练期使用历史/知识、预测不新增这些输入的设定已实施 | 是当前同船教师路线的直接设定依据，不保证教师一定有增量或学生一定能学到 |
| Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style，NeurIPS 2021 | 有正文假设与理论边界笔记 | “同船共享内容”仍可能包括身份等信息，不自动等于船型因果因素 |
| Variational Autoencoders and Nonlinear ICA: A Unifying Framework，AISTATS 2020（iVAE） | 清单推荐，未找到本项目实施证据 | 保留为理论候选；本轮不声称已完成其正文复核，不优先上生成式分离模块 |
| Weakly Supervised Causal Representation Learning，NeurIPS 2022 | 清单推荐，未找到本项目实施证据 | 保留为理论候选；本轮不把自然重复观测等同于其干预设定，不宣称已复现 |
| FiLM，AAAI 2018 | 未找到该具名模块的对应实施证据 | 条件调制可作实现工具，但不是条件有效性的证据；当前不因“可加调制”就开发 |
| Effect of 6-DOF Oscillation of Ship Target on SAR Imaging，Remote Sensing 2021 | 有物理原文阅读和条件代理实验 | 运动影响成像的依据，不是分类必涨的依据；SOG不等于角运动 |
| Analysis of Image Domain Characteristics of Maritime Rotating Ships for Spaceborne Multichannel SAR，Remote Sensing 2026卷期 | 有原文阅读和代理交互试验；实际发表于2025-12-23 | 没有复现其多通道成像/补偿；长宽、SOG和方向乘积的失败不能否定真实物理链 |

本地证据：
- `directional_candidate_space_20261011/RELATED_WORK_ZH.md` 与 `train_directional_candidate.py`。
- `CAUSAL_RELATION_SCOPE_20261011_ZH.md`。
- `knowledge_train_only_design_20261011/EXPERIMENT_PLAN_ZH.md`。
- `interaction_feasibility_20261012/LITERATURE_ZH.md` 与同目录 `FINDINGS_ZH.md`。

清单外也已经读过ISR、VICReg、ICP。它们分别帮助界定子空间恢复、避免塌缩和条件分布不变性，但本地诊断或自定义投影不等于相应论文复现。

## 3. 完整模型另有一套前序，不能只看因果清单

已有模块底稿列明：ResNet/SimCLR用于编码器；岭回归、R2-D2用于少标签闭式适配及求解器反传；RTN是源目标分类器残差关系的相关工作；PCA、SimpleMKL、DSN用于比较保留/补空间与核组合。固定核权重不是SimpleMKL，正交补也不是DSN语义分离。

此前还读了SimpleShot、TADAM，研究后处理、度量尺度与任务适配；以及Learning to Defer、校准多专家转交、META-DES和固定级联相对收益规则。它们已经支持过我们对选择器、双路互补和读出失配的分析，不是本轮新发现。

本地索引：
- `E:/Hermes/paper_tracking/sensitivity/module_evidence_20261008/MODEL_MODULE_DOSSIER_ZH.md`。
- `E:/Hermes/paper_tracking/sensitivity/research_progress_20261011/BOTTLENECK_REFERENCES_ZH.md`。
- `E:/Hermes/paper_tracking/sensitivity/frequency_closeout_20261010/RELATED_WORK_ZH.md`。

以上是历史来源审计，不把10月8日的预算化版本误当作10月12日融合实验的当前成绩。

## 4. 为什么现在调整阅读优先级

最新实际完整模型训练：三港，九次R18 layer4的64步续训，一个训练序列种子、五组64标签支持抽样。并非全23港确认。

| 完整融合臂 | OA（%） | BA（%） |
| --- | ---: | ---: |
| A：匹配任务续训 | 48.469966 | 43.391317 |
| B：等权同船教学 | 48.517164 | 43.579092 |
| C：知识加权同船教学 | 48.533977 | 43.600854 |

B相对A的BA为+0.187775个百分点；C相对A为+0.209537；知识相对等权的独立增量仅+0.021762，对应3艘船的3个救回样本，没有误伤。不能把整体+0.209537全部归于运动知识。

固定先验/核的干预回放中，单独替换先验没有提升，替换支持核及二者交互贡献主要正信号。教师自身BA却在三港中的两港下降。因此当前最有证据的问题是“怎样改善适配中的案例关系”，不是“更自信的教师必然教得更好”。这是计算路径证据，不是物理根因或纯因果因素证明。

完整证据：`full_teacher_integration_20261012/FINDINGS_ZH.md`。

以下不能重新包装成未尝试的新方向：
- 只教正确类别关系、单侧扩大间隔：修正后相对完整分数MSE的BA仍为-0.706/-0.615个百分点（完整/条件匹配队列）。
- 完整蒸馏加可靠有界类别关系：平均有小正信号，但两个种子一正一负，尚不能归因于可靠排序。
- 分类证据与案例核分离：相对同结构task的BA仅+0.131，两个种子一正一负；加强方向一致性及梯度定标后仍不及普通分离版本。

对应证据：`local_relation_distillation_20261011/angle_matched/FINDINGS_ZH.md`、`reliable_relation_distillation_20261011/FINDINGS_ZH.md`、`split_evidence_kernel_20261011/FINDINGS_ZH.md`。

## 5. 最值得新增借鉴的文献

### 5.1 RKD与SPKD：教会比较案例，而不只模仿类别分数

Park等，Relational Knowledge Distillation，CVPR 2019。读方法第3节：迁移不同样本之间的归一化距离及三元组角关系，而非逐样本特征逐坐标复制。距离用批内平均距离归一化，再以Huber损失比较。

Tung与Mori，Similarity-Preserving Knowledge Distillation，ICCV 2019。读方法公式(2)-(4)：特征矩阵形成样本Gram矩阵，按行L2归一化，再比较教师与学生的相似结构。它不要求二者特征维数相同。

对我们的新增启发：教师利用同船历史形成更可靠的案例表征；单次SAR学生学习“该图像与哪些支持案例更相近”。这不同于已经尝试的“同一图像的八个类别分数之间的关系”。简单把中心化分数MSE改写成类别对间隔误差没有新增机制。

采用前提：先证明这种教师案例几何经过我们的真实64标签残差求解器确有优势。SPKD的行归一化Gram用于教学损失，不能直接不加处理地充当部署核；部署仍保留由同一映射产生的合法K_SS和K_QS。

来源：`https://arxiv.org/abs/1904.05068`；`https://arxiv.org/abs/1907.09682`。本轮阅读原始PDF方法正文，不只看摘要。

### 5.2 Ren与MetaDistil：让学生学完后的表现决定教学价值

Ren等，Learning to Reweight Examples for Robust Deep Learning，ICML 2018。读第3节：对样本权重做虚拟更新，通过独立干净验证集损失计算元梯度，得到非负归一化权重。原任务是噪声和不均衡学习。

Zhou、Xu与McAuley，BERT Learns to Teach: Knowledge Distillation with Meta Learning，ACL 2022。读第2节：通过学生试学及独立quiz集反馈优化教师。原任务是NLP，不是SAR因果识别。

对我们的启发：教师自己预测更准/更自信，不等于其信息对当前学生有增量。权重应参考“学生经过支持适配后的源端留出任务损失是否降低”。先用少数固定权重或一次虚拟更新检验，不立即开发完整元教师。

使用源端留出港/船构建验证任务，不使用目标查询标签来训练权重。支持/查询划分与外层验证不能混在一起。不要把可微损失代理的改善直接写成OA/BA改善。

来源：`https://proceedings.mlr.press/v80/ren18a.html`；`https://aclanthology.org/2022.acl-long.485/`。

### 5.3 FactorCL：共同信息之外的变化，也可能有用

Liang等，Factorized Contrastive Learning: Going Beyond Multi-view Redundancy，NeurIPS 2023。读第2-3节：分别刻画共享与视图特有的任务信息，使用互信息上下界以及标签/增强假设约束。并非简单把特征切成两半或把残差全部保留。

对我们的启发：共同信息可能负责跨条件匹配；某次观测才显露的结构可能仍帮助分类。下一步应测“已有共享表示之后，剩余信息有没有额外任务贡献”，而不是只测残差自己能否分类。先用匹配容量的轻量读出比较，再考虑正式分解目标。

两个网络都看同一张SAR图像，不自动满足多模态理论前提；同船多次观测也不保证独有部分都是因果信息。完整互信息估计器暂低于支持关系教学优先级。

来源：`https://arxiv.org/abs/2306.05268`。

### 5.4 DKD：不要把非真类关系全部视作无用

Zhao等，Decoupled Knowledge Distillation，CVPR 2022。读第3节公式(1)-(6)：把标准KL蒸馏分成真类对其余类别的知识及归一化非真类之间的知识，并分别加权。原文指出后者可能十分重要。

启发：一艘船不仅要“更像真类”，它与其他相似船型的混淆结构也可能有价值。因此此前仅保留部分正确关系的损失可能删掉有用教学内容；这是待验证解释，不能据此宣布旧实验根因已确定。

当前中心化MSE不是原文KL，不能直接断言具有同一耦合问题。DKD适合作为少量、固定配置的蒸馏强基线，不宜只换公式就宣称新增模块。

来源：`https://arxiv.org/abs/2203.08679`。

### 5.5 DOT：仅在优化失配有持续证据时借鉴

DOT: A Distillation-Oriented Trainer，ICCV 2023。读第3节：为任务与蒸馏设计不同动量，蒸馏用更高动量；不是简单删除负梯度或PCGrad。

我们仅看到首批教师/任务梯度夹角为负，不足以认定整轮冲突为根因。只有多个批次及源端留出损失显示持续失配，才值得针对性试验。它是优化参考，不是因果机制来源。

来源：`https://arxiv.org/abs/2307.08436`。

### 5.6 新近候选VRM：关系也可能传播错误

Zhang等，VRM: Knowledge Distillation via Virtual Relation Matching，ICCV 2025。官方摘要及引言指出关系蒸馏中的过拟合和虚假响应传播，构造虚拟视图的跨样本/类别/视图关系，并动态断开冗余、不可靠边。官方实现已公开。

启发：不能因为支持核有正信号，就默认越密集的关系监督越好。真实同船历史与人工增强不是同一种视图；后续可研究哪些真实观测关系值得传递，但先验证关系教师的收益，不立即加入图网络或复杂剪枝。

阅读状态：本轮核对官方论文摘要、引言与官方仓库说明；PDF直开受限，尚未完整核读剪枝公式，列为下一轮精读候选，不声称已完成其方法复核或复现。原文：`https://arxiv.org/abs/2502.20760`；代码：`https://github.com/VISION-SJTU/VRM`。

## 6. 清单中仍值得保留但暂不优先开发的方向

SIE：第3节、公式(3)-(6)，使用已知变换驱动等变分支的条件预测器，同时保留不变分支及方差/协方差约束。可借“不要求所有有用信息都不变”的思路。但本项目此前条件变化预测有失败记录，先证明未见船上的关系可预测且有任务收益，再加超网络。原文：`https://proceedings.mlr.press/v202/garrido23b/garrido23b.pdf`。

Partial Observability：第3-4节研究视图只覆盖部分潜变量时的共享块识别，并依赖相应生成和信息保持假设。可借“不同条件只约束真正共享的部分”，不是给每种观测硬套同一个全局不变量。暂不为此新增大规模分组搜索。原文：`https://arxiv.org/abs/2311.04056`。

MatchDG：第4节对象匹配与迭代学习有助于区分“同类”与“同对象”。我们已知MMSI，不需要先重做全套对象发现，但需要验证同船知识是否比同类聚合更有增量。原文：`https://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf`。

CoRe：组内预测方差可作有意义的弱约束，但本地强一致性失败应保留，不继续默认权重越大越好。原文：`https://arxiv.org/abs/1710.11469`。

## 7. 下一轮最小实验：先确认可用，再教给学生

本节是建议，尚未运行。所有方案保持训练使用知识、预测不新增AIS/MMSI/历史观测输入；既有SAR采集入射角是否保留须与现有基线一致，不能混称为严格仅像素。

### 第一步：教师关系到底有没有收益空间

在源端留出船/港的少标签任务上，固定现有先验和部署求解器，比较单观测案例表示、等权同船多观测表示，以及已有知识加权表示。教师多观测只用于训练期或源端可行性诊断，不当作可部署学生成绩。

需要有一致的支持-支持K_SS和查询-支持K_QS，不能只替换一块矩阵制造不一致核。教师汇聚排除焦点观测所在产品；支持与查询按船分开，历史供体不能跨出对应许可数据池。没有可用供体的样本保留并单独报告覆盖，不只挑覆盖子集报整体成绩。

匹配核能量或固定比较有效正则，区分“邻近关系更好”与“只是尺度改变”。通过真实残差求解和最终融合报告OA、BA、救回/误伤及船只分布，不能以教师特征更稳定、教师分数更准或核拟合误差更小替代任务收益。

这一步能区分：根本没有值得教的关系，还是有优势但此前教错了对象。若关系本身无收益，不开发相似性蒸馏模块；先检查核接入及尺度是否一致，再接受当前知识不足这一可能。

### 第二步：有收益后，学生能否学会

保持旧任务续训作为匹配参考，只加紧凑三臂：现有等权分数教学；样本关系教学；二者联合。源端验证选择少量固定强度后冻结，先做短训练及第二种子，不追加大网格。

输出“教师上限 -> 单次SAR学生 -> 完整适配融合”的同链条结果。关系更像教师但OA/BA不涨，只算传递了几何，不算传递了有用知识。小幅可复现正增益可以保留，不要求预先达到某个大幅度。

### 第三步：确有必要才学习权重

若等权关系有用、当前知识权重无明显附加价值，再使用源端独立任务的学生收益指导权重。先做固定候选权重/一次虚拟更新的可行性对照，确认收益后才考虑MetaDistil式教师更新。

有同类且条件可匹配的样本时，比较同船教师与同类不同船教师，回答是否真的用了对象重复观测的额外价值；不重启用户此前暂缓的全局置乱大扫描。真类仅可作训练监督或事后分析，不进入目标查询的推理特征。

## 8. 可发展的贡献表述

待验证的核心不是“首次使用蒸馏/因果学习”，而是：利用训练期同船多时相及条件知识，构造服务于少标签跨港适配的案例关系监督；区分可共享的匹配依据与仍有分类价值的条件特有证据，并以学生经过实际适配后的增益决定教学信息是否值得使用，部署不新增AIS和历史观测依赖。

相对RKD/SPKD，新增重点应落在知识如何构造、如何对齐少标签支持求解及何时有效；相对FactorCL，重点不是复用共享/特有名词，而是匹配任务和分类任务中的信息分工。只有对应消融成立后，才能把这些差异写为贡献。

目前仍没有证明“纯因果因素已被单独分离”。当前可工作的定位是知识引导的关系学习与适配；可继续研究因果动机，但不能让命名代替识别证据。
