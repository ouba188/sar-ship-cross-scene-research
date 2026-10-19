# 平均前区域卷积：有界近邻工作精读、贡献边界与本轮实现审查

日期：2026-10-19。范围：5篇指定方向的主源论文及公开核心实现；不是截至2026年的穷尽检索，也不据此判断“首创”。本报告仅做阅读、比较和只读审查，不执行训练、不修改实验代码。论文、文本提取和作者代码副本位于本目录 `literature/`，仅本地阅读，不随研究归档上传。

## 1. 结论与本轮结果

**可保留的研究方向，是在区域汇总丢失邻域关系之前，用明确的区域归属与观测条件组织内容，再让源端跨过境任务学习这些内容如何影响实际支持分类。它目前是一条有条件的设计路径；本轮没有证明普通卷积优于平均后读出，也没有证明方向接近有助于船型识别，不晋升主线。**

最接近的既有技术已经覆盖：掩膜后有效数量归一化、可学习采样偏移、图像条件化旋转卷积核、中心差分与强度混合、多维动态卷积。因此，增加一个3×3卷积、把角度输入路由器、或者把几个既有模块串起来，均不足以独立支撑新贡献。可能站得住的是：从本任务已观察到的错误支持关系出发，说明为什么要保留/隔离哪些局部内容，观测条件如何约束而非直接决定相似性，以及这种组织如何在新的过境中改善完整分类决策。

本轮实际结果已经限制了叙事。以下数值由 `summary.csv` 直接读取核对，百分点均为前者减后者；个案原图事实来自本轮主线分析/用户提供，本阅读任务未重新查看该组原图：

| 配对 | OA变化，百分点 | BA变化，百分点 | 可支持的结论 |
|---|---:|---:|---|
| 跨过境B16，conv − base | 0 | +0.9259 | 附加读出相对旧基线有局部信号，不能归功于卷积 |
| 跨过境B16，conv − pool | 0 | 0 | 该条件未体现卷积的额外top1收益 |
| 跨过境B32，conv − pool | −0.1389 | −0.4872 | 当前空间卷积分支有净损失 |
| 跨过境B16，drop高权重 − 面积匹配低权重 | −0.2778 | −0.6481 | 删除效应不等于attention值的单调用途 |
| 跨过境B32，同上 | −0.1389 | −1.5185 | BA降低同时Brier改善，不能择一讲“更有用” |

四个同/跨过境×预算条件中，pool的Brier均优于conv。B16细格内置换未改变任何top1是主线提供的逐预测结论，汇总表与之相容；不能只由均值相同推出逐样本不变。置换仍可能改变核和连续风险，top1不动不等于完全没有学到排列，但没有证成可利用的分类收益。

主线个案197：同类支持45与查询方向不同，错类63方向更接近；conv相对pool将错类63相似性约0.208推到0.300，将同类45约−0.201推到−0.282，并误伤查询。**已定位到错误关系被加强；尚不能由一个案例证明方向是因果变量。** 当前卷积分支没有输入雷达几何，也不能把该误伤描述为“雷达角度对齐失败”。它足以否定把方向接近预设为同类证据的设计前提。

## 2. 实际阅读范围和访问情况

本轮采用literature-review技能的来源记录、方法抽取与贡献对照流程；按用户要求做有界叙述性精读，没有展开系统综述或另行安装依赖。已读项目AGENTS、`codex-brain/INDEX.md`、`04-workflows.md`相关科研段落，以及上一轮 `learned_geometry_crosspass_20261019/FINDINGS_ZH.md`。沿用“实际数据/图像线索 → 具体失效位置 → 针对性改变 → 配对证据”的要求。

`web.run`联合检索下列4组词，本次没有返回可读条目：ARC ICCV2023 GitHub、Deformable Convolutional Networks ICCV2017、Partial Convolutions ECCV2018、CDC CVPR2020与ODConv ICLR2022。随后直接从CVF官方目录/论文页面、作者仓库、arXiv获取主源。未使用博客结论补写未读方法。

| 文献 | 实际读取的主源范围 | 已核对的作者代码 | 本地文件 |
|---|---|---|---|
| Pu等，ARC，ICCV2023 [1] | CVF PDF第3–5页，§3.1–3.4，式(1)–(4)，图2–3相邻说明；核对§4的任务范围 | `RountingFunction`、`_get_rotation_matrix`、`batch_rotate_multiweight`、`AdaptiveRotatedConv2d.forward` | `ARC_ICCV2023.pdf/.txt`，`ARC_routing_function.py`，`ARC_adaptive_rotated_conv.py` |
| Dai等，DCN，ICCV2017 [2] | CVF PDF第2–3页，§2.1–2.3，式(1)–(6)，采样/插值与offset学习说明 | 作者MXNet ResNet101检测符号中的offset生成和DeformableConvolution调用；未审阅CUDA算子 | `DCN_ICCV2017.pdf/.txt`，`DCN_resnet_v1_101_rcnn_dcn.py` |
| Liu等，Partial Convolution，ECCV2018 [3] | CVF PDF第3、5–6页，已有masked convolution脉络、§3.1–3.3，式(1)–(2)及U-Net/损失说明 | `PartialConv2d.forward`中的mask计数、偏置处理和更新 | `PConv_ECCV2018.pdf/.txt`，`PConv_partialconv2d.py` |
| Yu等，CDC，CVPR2020 [4] | CVF PDF第3–4页，§3.1–3.2，式(1)–(4)、深度监督与损失；不是只看纯差分式(2) | `Conv2d_cd.forward` | `CDC_CVPR2020.pdf/.txt`，`CDC_CDCNs.py` |
| Li、Zhou、Yao，ODConv，ICLR2022 [5] | arXiv作者全文2209.07947v1，第3–6页，§3.1–3.2，式(1)–(2)、实现细节；PDF标注ICLR2022 | `Attention`、`ODConv2d._forward_impl_common`，并读作者README | `ODConv_ICLR2022.pdf/.txt`，`ODConv_odconv.py` |

页码均为PDF文件页序，非混用会议页码。阅读主要依赖PDF文本提取、公式文本与代码交叉核对；未宣称逐页视觉审读所有图表，也未运行这些作者代码。

访问限制如实保留：OpenReview的ODConv PDF出现SSL失败，已改用同作者arXiv全文；NVIDIA、CDCN、ODConv的GitHub树API出现403限流，公开raw文件仍可读取。曾试探一个不匹配的arXiv编号，标题核对后排除，未用作证据。ODConv作者README明确说明公开训练/评估实现为重新实现，部分结果与原论文有差异，不能把仓库数字直接替换论文数字。代码读取的是访问日公开分支快照，未固定上游commit；本地副本用于追溯本次阅读。

第五篇选ODConv，因它直接覆盖空间核位置和通道的条件化，比再读一篇仅讨论核混合的论文更有辨别力。Dynamic Convolution/CondConv仅通过ODConv的方法回顾了解，**未独立精读，不能列作本轮完成的第六篇**。Partial Convolution引用的segmentation-aware和sparsity-invariant工作同样未展开精读；这也说明本报告不能替代后续投稿前的完整近邻检索。

## 3. 五篇工作对具体设计的约束

### 3.1 Partial Convolution：区域卷积的最近算子先例

设当前滑窗为X、二值有效掩膜为M、窗口元素数为K，论文式(1)为：

\[
x' = \begin{cases}
W^T(X\odot M)\,K/\sum M+b,&\sum M>0,\\
0,&\text{否则}.
\end{cases}
\]

式(2)把“窗口至少有一个有效输入”的输出位置标为有效，逐层扩大可用区域，以完成缺失像素修复。作者代码用全1卷积计算有效数量；先从普通卷积输出中减去bias再乘有效数量比，最后加回bias，避免偏置也被错误放大。可核对本地 `PConv_partialconv2d.py` 第61、64、70、74行。[3]

**为何有用：** 它是“卷积前掩膜、按所属区域有效数量重归一化”的直接技术依据，不能把这两步包装成新算子首次提出。论文自己还明确承认更早masked/reweighted convolution的使用，不应反向把PConv也说成所有掩膜卷积的起点。

**真正不同的任务含义：** 本项目的框内和框外都是观测到的内容，不是“有效像素与需要填补的洞”。要保留的是各自归属及可比较的区域关系，不能机械使用PConv逐层扩张mask，否则框外位置会逐步变成框内有效位置。可复用masked renormalized convolution基础操作，在每层维持原区域身份，并明确哪些跨区域交互被允许、在哪一步发生；这种改动首先是任务适配，其价值仍取决于配对结果。

局部卷积的分母与最终区域平均的分母也不是一回事：前者按每个卷积窗口可用输入数归一化，后者按最终输出格的区域质量归一化。只有池化除以区域面积，并不能消除此前卷积已经混入的另一侧内容。

### 3.2 ARC：最直接的方向卷积比较对象

论文§3.1将离散卷积权重视作可插值的核空间，通过旋转坐标和双线性插值产生旋转权重。§3.2–3.3的核心是：

\[
(\theta,\lambda)=f(x),\quad W'_i=\operatorname{Rotate}(W_i;\theta_i),\quad
y=\left(\sum_i\lambda_iW'_i\right)*x.
\]

路由器为depthwise 3×3卷积、LayerNorm/ReLU、平均池化及两个线性分支；角度用softsign并缩放，核混合权重用sigmoid。每个输入样本预测一组角度与混合权重，核先合并再卷积。**这是权重空间旋转，不是逐像素自由offset，也不是必须旋转原图。** 论文面向旋转目标检测，在DOTA及HRSC2016评价；不能把其船只检测结果当作SAR细粒度跨过境分类证据。[1]

作者公开实现与理解相符：`ARC_routing_function.py` 第22行默认角度幅度40度；`ARC_adaptive_rotated_conv.py` 第9行构建3×3权重旋转矩阵，第157行起按样本生成权重并用分组卷积计算。这里的“40度”是该构造器默认值，不表示所有实验只能使用这个范围。正文初始化标准差写0.2，所读代码为0.02；若将来正式复现应明确版本，本任务未据此推断性能差异。

**为何有用：** ARC已覆盖“内容决定旋转角度并调整卷积核”。我们的差异不能只写“自适应方向卷积”。更具体的候选，是OBB无向轴与原雷达观测几何提供可追溯约束，内容学习在约束下如何读取不同区域，并以源端实际支持任务训练。相对于ARC，需要区分内容自由路由、只使用OBB、以及加入真实雷达观测条件分别改变了什么；正确元数据优于匹配替代才支持观测条件的独立用途。

尤其不应设定“方向越近相似性越高”。方向应改变内容读取或关系表达，而不是直接把同向查询/支持拉近；197/45/63正是这种捷径的反例线索。

### 3.3 DCN：自由采样与受约束读取的区别

原始DCN的式(2)为：

\[
y(p_0)=\sum_{p_n\in R}w(p_n)x(p_0+p_n+\Delta p_n).
\]

offset由同一输入特征上的卷积产生，每个位置输出2N维二维偏移，分数坐标按式(3)–(4)双线性插值。权重与offset一起接受下游任务梯度。§2.2还提出deformable RoI pooling，按RoI尺寸缩放偏移。原始2017版不是后续带调制mask的DCNv2，不能混用两者技术描述。[2]

在所读官方检测符号中，第696行生成72通道offset，第698行传入 `DeformableConvolution`，对应3×3采样与4个deformable groups。它确认了“特征预测局部offset再采样”的调用关系；本轮没有核查算子底层CUDA实现。

**为何有用：** 若候选只是给采样位置加可学习位移，DCN已经覆盖核心思想。可能有意义的区别是：自由偏移改成观测条件约束的低维自由度，并在采样前维护区域身份，以减少无关方向/邻域内容被当成类别证据。不能先验宣布受约束一定胜出，自由offset可能更合适，固定几何也可能限制真实有用内容。

ARC与DCN不可简单写成同一算子：前者在核权重空间旋转，后者在输入特征空间移动采样；离散插值、边界及mask处理下不能默认数值等价。若未来采用旋转/偏移采样，双线性插值前就应明确区域归属。先插值完整特征再乘采样点mask，可能已混入另一侧内容；应明确是否采样 `M_r F` 与 `M_r` 并按有效质量归一化。这是拟议方案真实可达的边界问题，不是要求本轮增加新实现。

### 3.4 CDC：不是所有形式都丢掉强度

论文式(2)是纯中心差分，但实际广义CDC使用式(3)，等价式(4)为：

\[
y(p_0)=\sum_{p_n\in R}w(p_n)x(p_0+p_n)
-\theta x(p_0)\sum_{p_n\in R}w(p_n),\quad 0\leq\theta\leq1.
\]

θ=0退化为普通卷积；θ=1才是纯中心差分。原文默认θ=0.7，同时保留强度与差分贡献。官方 `Conv2d_cd.forward` 先做普通卷积，再用空间求和的权重执行1×1卷积，返回两者的加权差；本地第40–60行可核对。原任务是人脸反欺骗，训练还有深度监督及contrastive depth loss，不能移植其“鲁棒纹理”表述为本项目SAR物理机制。[4]

**为何有用：** 它为中心与邻域关系提供清晰对照，也提醒我们避免不必要的固定减法。用户提出“保留中心与邻域内容、不预设减法”合理，但不是相对广义CDC的充分新颖性论证。更准确的差异是保留中心和区域邻域作为可分别访问的信息，再让任务决定它们的组合，而不是使用与同一组核权重绑定的固定差分项。

普通卷积本来也有中心权重及邻域权重。只有证明分开保留的信息对实际支持关系有用，才超出一般参数化选择；拼接后再线性相似度也不自动产生跨区域条件交互。

### 3.5 ODConv：条件化不止核混合

论文式(2)可概括为：

\[
y=\left[\sum_i \alpha_{w,i}(x)
\left(\alpha_{f,i}(x)\odot\alpha_{c,i}(x)\odot
\alpha_{s,i}(x)\odot W_i\right)\right]*x.
\]

乘法沿各自维度广播：空间核位置k×k、输入通道、输出通道、核索引。路由来自输入GAP、降维和四个输出头；实际实现跨多个核共享空间/输入通道/输出通道注意力，核注意力单独区分各核。公开代码把输入通道权重乘在输入、输出通道权重乘在输出，并在卷积前合并带空间及核权重的参数，利用等价计算节省开销。[5]

特别注意：这里的spatial attention指卷积核的k×k偏移维度，**不是为特征图每个H×W位置都独立产生一套卷积核**。只根据一个汇总向量调制核也不能直接称为逐位置内容交互。

**为何有用：** “内容控制通道/方向/核系数”已有强先例。候选有机会贡献的是外部观测条件、稳定区域归属和跨场景支持目标如何协同工作，而不是再给路由器增加一个输入字段。若ODConv式普通内容条件化在相同接入、训练与预算下已能得到相同收益，则应把贡献收在更简单的读出/任务组织上，而不是强行宣称雷达约束必要。

## 4. 已有技术与可能新增的任务关系设计

| 设计部分 | 已有技术边界 | 可能有意义的任务变化 | 当前证据 |
|---|---|---|---|
| 平均前处理邻域 | 普通卷积、DCN、ARC等均早于本工作 | 避免本任务细格平均先抹掉有用的同区域局部关系 | 本轮conv未优于pool，暂不成立 |
| 卷积前区域mask与局部归一化 | PConv及其前序已覆盖操作 | 固定框内/框外归属，保留两者并显式安排交互，避免把外部内容当缺失洞 | 本轮尚未实现该约束 |
| OBB+原雷达元数据约束方向 | 旋转卷积、变形采样、条件化均已有 | 区分无向船轴与有向观测几何；条件影响读取而不强制同向相似 | 待验证，现有误伤反对方向捷径 |
| 保留中心和邻域 | 普通卷积及广义CDC已有强度信息 | 分开保留、让任务学习组合并识别何时邻域误导 | 不能单独当创新 |
| 源episode学习可迁移关系 | episodic learning与可微求解并非本轮新技术；本项目上一轮已使用 | 使区域/方向组织在真实支持中心化、SS/QS重解和完整融合下接受训练 | 当前runner已有这条训练链；不能重复认领 |

**可写的暂定贡献表述：**

> 我们研究检测后SAR细粒度船型分类中局部证据如何在跨过境支持关系中被误用。基于持续误判与误伤案例，候选方法拟在区域汇总前维护框内/框外归属，并用可追溯的目标轴与雷达观测条件约束内容读取；中心与邻域证据保持可访问，由源端跨过境episode通过实际支持求解学习其组合。贡献需由区域归属、观测条件和支持关系改善之间的配对证据建立，而不把旋转、掩膜卷积或平均前非线性本身作为新发明。

以上用“拟”“候选”，因为当前实验没有完成该方法，更没有建立净收益。发表层面的完整新颖性判断还需要相邻的SAR几何条件化、支持关系和区域自适应卷积工作；本轮5篇足够界定基础算子，不能证明整个研究谱系不存在相同设计。

## 5. 与正在完成的3arm验证准确对齐

只读核对 `PROTOCOL_ZH.md` 与 `conv_model.py`：

- pool_first：对32×32特征按4×4格及区域平均，再256→32投影、LayerNorm/GELU。
- point_first：相同逐点投影、LayerNorm/GELU移到区域平均之前，没有空间卷积。
- conv_first：投影后增加零初始化的3×3 depthwise残差卷积，再LayerNorm/GELU与区域平均，真实增加288参数。

当前 `conv_model.py` 第44–49行在完整特征图上卷积，mask到 `region_pool` 才参与运算。因此它是“普通全图卷积后区域池化”，**不是本报告讨论的“卷积前区域掩膜/所属区域归一化”**。这符合已写协议的空间验证用途，不是三组对照无效的bug；但不能拿其结果证明区域隔离已测试。当前分支不输入雷达角度，同样没有验证几何约束卷积。

冻结layer1本身已经有跨区域感受野。未来在这个特征层加mask，只能限制新增算子的直接混合，不能声称原始像素层的背景影响全部移除。OBB内部也不等于纯船体，框外也不等于纯海面。若最后有收益，应按实际输入层和归属定义解释。

另一个数学边界：空间不变的纯线性卷积在周期边界、完整平均等条件下可交换为核系数和作用于输入均值。普通“先卷积再平均”并不天然保存局部排列。当前模型确有非线性、残差及区域边界，不满足简单等价条件，但仍需比较point_first/pool_first与置换诊断，不能只凭算子顺序说明空间信息被有效利用。

## 6. 有限可行下一步，不自动启动

**当前决定：保留本轮负结果，不追加普通卷积角度搜索，不晋升conv，不将高attention当单调有用区域。**

唯一值得保留的下一候选，是先判断错误关系来自跨区域混入，还是区域内也主要在学习方向/外观捷径。原图197/45/63只能提出这一分歧，尚未回答它。可复用现有缓存及主线原图，在同一个查询、同类45、错类63及少量持续判对/错案例上，定位当前卷积窗口实际读入的区域组成、改变的带符号支持贡献；不进行新的全量审计。

若这份定位显示有明确的跨区域混入线索，可做一次固定配置的区域归属候选：在新增局部卷积前分别使用框内/框外mask，按局部有效区域质量归一化，区域身份不随层扩张，保留中心内容和旧主通道。与现有全图conv和pool使用相同任务、源episode、支持预算及完整求解；各组覆盖、非线性与输出尺度保持可解释。它检验的是“新增局部聚合的归属是否更合理”，不是预设空间卷积必须胜出。无需把所有文献模块同时实现。

若没有这种线索，而错误主要在同区域内方向接近的异类被拉近，先处理源任务监督/关系读出的这一失效；单纯按船轴旋转可能进一步强化捷径。此时可以保留观测条件约束作为后续候选，但须明确：

1. OBB长轴按π周期处理，不能把任意正轴叫船首；轴向表示可用cos(2φ)、sin(2φ)，或显式处理两种等价轴方向。有向LOS及入射角来自原产品几何、转换到相同图像坐标；不要直接相减不同坐标系方位。未知观测条件不伪装成0度实测值。
2. 条件调制局部内容读取，不直接把方向相近映射成大相似性。完整同类/混淆类支持共同约束任务目标，保留无同类支持时的原先验作用。
3. 正确条件与匹配替代条件的比较要区分训练即错配和仅测试错配；后者可能只是分布改变。根据具体主张只选能区分解释的对照，不机械重跑一套全部算子。

候选是否保留，看同查询/同支持下的净分类收益、连续风险、同类与异类带符号关系和原图案例是否相互支持。OA/BA与Brier方向相反要分别报告；改善不必遍及所有条件，但不能只选某个总分掩盖关系误伤。不承诺涨点，不把现有开发集当独立确认集。

## 7. conv_model/run_probe只读快速审查

**结论：在本次静态阅读及现有小清单核对范围内，未发现足以使3arm对照失效的实际bug、目标标签泄漏或极化/区域维度混淆。无需因本审查重跑或改代码。** 这不是对所有上游数据生成过程的重新独立认证。

审查对象为本目录 `conv_model.py`、`run_probe.py`，并追读其直接依赖的上一轮 `model.py` 的Reader/centered/residual_probability、`runner.py` 的context/fit/collect，以及本轮 `extract_maps.py`。没有执行模型、训练或改代码。

- 维度：输入为N×2极化×256通道×32×32。`region_pool`广播得到N×极化×2区域×通道×H×W，池化后排列为区域优先的128个token，再还原为2区域×64位置；与position及mass一致。depthwise卷积把N和极化合并为batch，groups=32用于通道，不把VV与VH当卷积通道混合。格内置换同时移动特征与mask，保持区域质量与逐点值集合。
- 可比性：三个mode使用同一归一化、position、父类Reader参数与后端。conv额外卷积零初始化，起点与point相同；逐seed重设随机种子，源episode训练日程相同。额外288参数已经披露，不能称严格等参数，但不使既定比较无效。
- 数据路径：`run_probe.Study.__init__`只用 `maps[self.fit]` 求均值/标准差；全体标签虽然在内存中，预测的残差求解只索引实际support标签，query标签在fit任务中用于合法源损失，在目标任务中用于评价。源验证决定checkpoint，目标评价发生在fit返回之后。
- 清单实查：fit 64行、validation 6行，交集0；37个记录为32训练episode和5验证episode。逐记录核对未见目标任务的支持/查询行进入源训练或源验证，训练episode的source/support/query均在fit中，验证query均属于validation。该核对针对实际行清单，不声称额外完成全局MMSI/产品元数据审计；父context已有MMSI分离断言。
- 求解与评价：新表示按实际支持中心化，核矩阵SS与QS随新表示一起更新；旧基线来自 `ct['base_prob']`，不会误把pool预测叫base。`receipt.json`记录9次训练完成，旧基线回放最大误差9.71445e−17。此处读取现有记录，没有重复运行验证。

以下是解释限制，不当作虚构bug：

- 三组都沿用了几何可用门槛，虽然本轮不用角度。这维持当前配对覆盖，但限制结论到当前门槛下的附加读出，不代表所有无几何样本都获同样机会。
- attention删除发生在已有内容交互/注意力形成之后，保留全图context，再对剩余值权重重归一化，并重新中心化、求解支持系数。因此它测量的是完整系统对该干预的响应，不是一个格的独立因果贡献；面积近似匹配也不等于内容信息量匹配。
- 格内置换会改变邻域连接，也可能改变跨格边缘的邻接关系。它是受控特征排列诊断，不是物理方向干预。没有top1变化不等于没有连续响应，也不能宣布卷积分支有收益。
- 源验证只有6艘两类、预算外推和重复开发集仍是已有范围限制。当前没有证据把负结果强行归咎于代码、数据缺陷或“创新方向无效”。

## 8. 主源索引

[1] Yifan Pu, Yiru Wang, Zhuofan Xia, Yizeng Han, Yulin Wang, Weihao Gan, Zidong Wang, Shiji Song, Gao Huang. **Adaptive Rotated Convolution for Rotated Object Detection.** ICCV 2023.

- 官方PDF：https://openaccess.thecvf.com/content/ICCV2023/papers/Pu_Adaptive_Rotated_Convolution_for_Rotated_Object_Detection_ICCV_2023_paper.pdf
- 作者仓库：https://github.com/LeapLabTHU/ARC
- 本轮读取代码：https://raw.githubusercontent.com/LeapLabTHU/ARC/main/mmdet/models/backbones/modules/adaptive_rotated_conv.py
- 路由实现：https://raw.githubusercontent.com/LeapLabTHU/ARC/main/mmdet/models/backbones/modules/routing_function.py

[2] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei. **Deformable Convolutional Networks.** ICCV 2017.

- 官方PDF：https://openaccess.thecvf.com/content_ICCV_2017/papers/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.pdf
- 作者仓库：https://github.com/msracver/Deformable-ConvNets
- 本轮读取代码：https://raw.githubusercontent.com/msracver/Deformable-ConvNets/master/faster_rcnn/symbols/resnet_v1_101_rcnn_dcn.py

[3] Guilin Liu, Fitsum A. Reda, Kevin J. Shih, Ting-Chun Wang, Andrew Tao, Bryan Catanzaro. **Image Inpainting for Irregular Holes Using Partial Convolutions.** ECCV 2018.

- 官方PDF：https://openaccess.thecvf.com/content_ECCV_2018/papers/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.pdf
- 作者仓库：https://github.com/NVIDIA/partialconv
- 本轮读取代码：https://raw.githubusercontent.com/NVIDIA/partialconv/master/models/partialconv2d.py

[4] Zitong Yu, Chenxu Zhao, Zezheng Wang, Yunxiao Qin, Zhuo Su, Xiaobai Li, Feng Zhou, Guoying Zhao. **Searching Central Difference Convolutional Networks for Face Anti-Spoofing.** CVPR 2020.

- 官方PDF：https://openaccess.thecvf.com/content_CVPR_2020/papers/Yu_Searching_Central_Difference_Convolutional_Networks_for_Face_Anti-Spoofing_CVPR_2020_paper.pdf
- 作者仓库：https://github.com/ZitongYu/CDCN
- 本轮读取代码：https://raw.githubusercontent.com/ZitongYu/CDCN/master/CVPR2020_paper_codes/models/CDCNs.py

[5] Chao Li, Aojun Zhou, Anbang Yao. **Omni-Dimensional Dynamic Convolution.** ICLR 2022.

- 本轮实际读取作者全文：https://arxiv.org/pdf/2209.07947
- 已核对标题/作者的摘要页：https://arxiv.org/abs/2209.07947
- 会议入口：https://openreview.net/forum?id=DmpCfq6Mg39 （PDF访问SSL失败，未冒称读到该入口的PDF）
- 作者仓库：https://github.com/OSVAI/ODConv
- 本轮读取代码：https://raw.githubusercontent.com/OSVAI/ODConv/main/modules/odconv.py

归档交接：仅本报告进入本轮研究文字归档；`literature/`内公开论文、提取文本与作者代码副本保持本地。遵照本次只写指定目录的边界，本阅读任务未操作归档仓库或GitHub，同步由主线统一完成。
