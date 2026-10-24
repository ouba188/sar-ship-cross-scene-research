# 候选竞争局部聚合：近邻文献、实际差异与有限审阅

日期：2026-10-24。范围：文献侧线及已授权的有限实现核对。只新增本报告；未修改 run.py / PROTOCOL_ZH.md，未运行训练或重跑测试，未操作 Git。主线报告训练9次已完成、分析进行中；本文不提前判断分类收益。

## 1. 结论与可发展的贡献

可发展的具体差异是：**对每个候选类对，用交换对称的证据统计与查询内容选择局部位置，再汇总有符号类间证据**。它把“这一位置是否有利于区分类c与类d”变成显式计算对象，而不是给所有候选共享一个局部可靠性分数。这是值得按当前配对实验继续验证的区别，尚不是已证实优势或首创声明。

最接近的四个先例为 DMN4、ATL-Net、TALDS-Net、MCL。它们分别已有跨类相对间隔选择、全支持池关系归一化、跨类置信度驱动的自适应描述子选择、全候选局部图上的双向传播。因此不能把“参考比较之后、空间汇总之前引入竞争”本身写成新增贡献。[1–4]

当前代码的更准确定位是：**共享跨类中心化之后，比较逐类独立门控与显式类对条件门控**。并非严格的“无其他类信息 vs 有其他类信息”。此外，最终节点求和会消去纯循环边分量；不能用非零三角环测试作为“循环推理带来收益”的证据。两点详见第5节。

## 2. 阅读证据与访问边界

本轮先读指定 NEXT_DISCRIMINATIVE_OPERATOR_ZH.md、Codex Brain 索引及04-workflows的相关研究约定，复用本地 local_evidence_transport/LITERATURE_ZH.md 的局部匹配记录、query_evidence_pooling/LITERATURE_ZH.md 的DMN4及聚合记录，以及 reference_relative_metric/LITERATURE_ZH.md、edge_difference/LITERATURE_AND_ALGORITHM_ZH.md 的既有边界。随后完整读取本目录 run.py / PROTOCOL_ZH.md，以及已有 checks.json / receipts.csv；有限查看其复用的 episodic/run.py 的参考日程。

直接 web.run 本轮返回空内容，本机 requests 访问 arXiv 出现 TLS EOF。备用 search_service 接口随后取得以下一手PDF可读正文，故不能笼统写成本轮完全无法联网。IJCAI正文存在间歇性重定向失败，下表明确成功阅读范围。没有执行作者代码，没有重现论文结果，也没有宣称逐页读完附录或证明。

| 编号 | 论文与版本 | 本轮直接读取的范围 |
|---|---|---|
| [1] | Liu等，DMN4: Few-shot Learning via Discriminative Mutual Nearest Neighbor Neural Network；arXiv:2103.08160v3，2021-09-08；本地已核验正式发表为AAAI 2022 | PDF pp.3–4，§3.1–3.3及后续DMNN方法段，式(1)–(12)、图2–3；本文节号按预印本，不挪作正式版节号 |
| [2] | Dong等，Learning Task-aware Local Representations for Few-shot Learning，IJCAI 2020，pp.716–722 | 官方PDF首页动机、p.718的§3.1–3.2、式(1)–(5)，以及§3.3开头的自适应阈值动机；未取得§3.3全部后续公式，不自行补齐其实现细节 |
| [3] | Qiao等，TALDS-Net: Task-Aware Adaptive Local Descriptors Selection for Few-shot Image Classification；arXiv:2312.05449v2，2024-09-03 | PDF pp.1–3，§2.1–2.3、式(1)–(9)；不以预印本修订日期断言会议年份 |
| [4] | Liu等，Learning to Affiliate: Mutual Centralized Learning for Few-shot Classification，CVPR 2022；arXiv:2106.05517v4，2022-03-18 | PDF pp.2–5，§3.1–3.6、式(1)–(13)；CVF会议条目核对题名与年份 |

一手来源，编号与下文相同：

- [1] `https://arxiv.org/pdf/2103.08160v3`；正式版入口沿用本地记录：`https://ojs.aaai.org/index.php/AAAI/article/view/20076/19835`。
- [2] `https://www.ijcai.org/Proceedings/2020/0100.pdf`；会议条目 `https://ijcai.org/proceedings/2020/100`。
- [3] `https://arxiv.org/pdf/2312.05449v2`。
- [4] `https://arxiv.org/pdf/2106.05517v4`；会议条目 `https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Learning_To_Affiliate_Mutual_Centralized_Learning_for_Few-Shot_Classification_CVPR_2022_paper.html`。

## 3. 四个真正接近的机制

下面统一用i表示查询位置，c表示候选类，s表示支持描述子。公式为必要的统一符号转写；比较判断属于本报告分析。

### 3.1 DMN4：相对最强异类的间隔决定局部保留

DMN4在合并全部类别的支持池中找近邻。令 t_ic=max_(s in S_c) cos(q_i,s)，赢家c*=argmax_c t_ic；其判别选择量为 tau_i=t_i,c*−max_(d!=c*) t_id。当多个查询描述子争取同一个支持近邻时，以相对间隔决定保留者；选出的查询子集再用于各类NBNN累加。[1，式(5)、(11)–(12)，图2–3]

这是直接的“先竞争、再聚合”先例。区别是硬选择一个任务相关查询子集，围绕局部赢家和最强异类；本实现为每个无序类对分配不同的连续位置权重。仅把DMNN软化、或仅使用top1−top2间隔，不足以建立本轮更具体的差异。DMN4选中的相对优势也可能属于错误类别，不能把间隔自动解释为正确语义证据。

### 3.2 ATL-Net：关系先在全候选支持池分配权重

ATL-Net构造原始余弦关系R_is及经共享变换的关系U_is。其TL-Net基础式为 A_is=I(U_is)/sum_(s' in 全支持池) I(U_is')，I将低于阈值的关系置零；随后按类累加 A_is R_is。ATL-Net进一步学习适合不同查询局部的阈值。[2，式(1)–(5)、§3.3开头]

分母跨类别，竞争发生在局部关系汇总之前；不能归为“只看本类的普通注意力”。但原式竞争在查询—支持关系边上，以跨类归一化分配份额。本实现先得到每位置每类E，再由类对统计决定在位置轴如何聚合差值。两者的归一化轴和保留的对象不同。只用跨类softmax调权，仍不足以超出这里的已有机制。

### 3.3 TALDS-Net：先选支持，再按跨类置信度选择查询局部

支持描述子先计算对各类kNN余弦和gamma，取R=max_c softmax_c(gamma)，再用学习阈值和sigmoid选择支持。查询描述子对选出的各类支持子集重复计算跨类R，阈值读取查询及支持子集，得到局部权重M_q；类别分数为局部gamma的加权和。[3，§2.3，式(2)–(8)]

它已覆盖“跨类匹配剖面→学习局部筛选→聚合”的主干。区别在于所读公式将剖面压成最大置信度，未显式构造每个(c,d)专属的交换对称门控和反对称差值。原文阈值下标及gamma类轴有省略，本文不据此断言其全部代码张量形状。最大置信度可以很高却支持错误类；本实现也不会因改用类对就自动避免这种情况。

### 3.4 MCL：全候选局部图上的双向竞争与传播

MCL对查询和全部支持局部建立二部图：P_Sq按全部支持节点归一化，P_qS按查询节点归一化。支持稳态满足 pi_S=P_Sq P_qS pi_S，类别概率累加所属支持节点质量。其可训练实现用Katz中心性闭式解 ((I−alpha P)^−1−I)1近似；也可将中心性用于局部加权池化。[4，§3.2–3.6]

它已有跨类归一化、多步局部交互及汇总前重加权，不能说前人都是类内独立匹配。区别是全图可达性/中心性；本实现是局部证据上的无序类对门控，再作一次节点求和，没有多步传播。MCL按单张查询的局部构图，不能误称必须联合一批测试查询的传导方法。

## 4. 与当前实际实现的精确对照

### 4.1 代码真正输入的E

依据 run.py:38–51、146–151，先对每类有效参考船及其有效token等权计算：

```text
L_igc = log mean_(有效参考船r) mean_(有效tokenu in r)
                  exp(sqrt(32) * dot(q_ig, s_cgru))
E_igc = (L_igc - mean_(有效类k) L_igk) / scale_g
```

scale_g是原参考库下源utility训练样本的组内RMS，固定后用于全部参考库；下限0.05。无参考的类不参与均值，查询无效位置不参与池化。局部是否确实为单位向量取决于复用的source_data上游，本文未追溯全部PCA生成链，不仅凭变量名补作验证。协议应明确sqrt(32)、预中心化和RMS，因为它们都进入非线性门控，不能当作任意可删常数。

### 4.2 成对路径与own路径

省略组g，当前pair为：

```text
h_icd = [(E_ic+E_id)/2, abs(E_ic-E_id), E_ic*E_id,
         (E_ic-E_id)^2, q_i]
w_icd = softmax_i MLP(h_icd)             # 仅在有效位置上
B_cd = sum_i w_icd * (E_ic-E_id), B_dc = -B_cd
n_c = (1/K) sum_(d有效,d!=c) B_cd        # K为该组有效类数
logit_c = full_c + sum_g alpha_g * n_gc
```

own用 [E_ic,E_ic²,E_ic³,E_ic⁴,q_i] 输入同样36→16→1、Tanh的共享MLP，位置softmax后得到t_c，再计算 t_c−mean_k t_k。两臂均609个MLP参数加6个alpha，共615；无固定类ID参数。alpha初始为零，而MLP随机非零，因此初始完整预测相同，不代表初始局部通道或初始alpha梯度相同。

最有意义的比较是：共享同一E、同一查询内容和训练机会后，**是否需要显式读取竞争候选的局部证据，才能选择更有用的位置**。四个文献机制提供近邻背景，当前own对照只回答这一局部实现问题，不是对四篇完整方法的排名。原论文训练特征与新类别few-shot设定，也不能直接等同于本地冻结特征、源参考与跨场景闭集分类。

与已有M1的区别依据本地方案：这里门控读取真实查询—参考比较之后的类对证据；不以“用了注意力”区分模块。本文未复审M1实现，相关定位沿用指定方案，而非新增代码审计结论。

## 5. 有限代码审阅：两项解释边界及一项协议遗漏

### 5.1 [P2] own已含跨类均值，不能称完全无竞争信息

定位：[run.py:50](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:50)、[run.py:80](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:80)。

证据：own读取的是预中心化E。固定原始L_ic、改变其他类L_id时，E_ic仍会改变，从而改变own的非线性门控。协议“只看本类E”按代码变量字面成立，但若用来解释为“无跨类信息基线”，就不成立。pair的均值/乘积输入也受到其他类经中心化产生的共同偏移，因此它并非只依赖两类原始L。

建议本轮保留已跑实验，写成“共享跨类中心化后的本类门控 vs 显式类对门控”；若将来要回答严格的无跨类信息问题，再另行定义原始L上的own对照，不能事后把这轮结果改称该对照。这里没有发现目标标签进入门控的问题，也不据此要求重训。

### 5.2 [P2，条件性主张风险] 非零环不能证明最终分类利用循环分量

定位：[run.py:74](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:74)、[run.py:123](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:123)。

原测试正确证明生成的边不总能写成 B_cd=t_c−t_d。协议将“非加传递关系”限定到边时是正确的。但当前输出只使用行和 n_c=sum_d B_cd/K。任何反对称R只要sum_d R_cd=0，就完全不影响输出。特别是三节点循环 B_12=B_23=B_31=b 的环量为3b，但所有节点输出为零。

进一步，sum_c n_c=0，构造 B'_cd=n_c−n_d 会产生完全相同的节点输出；B−B'的每行和为零。这是本报告根据实际节点归约作出的代数分析，不是文献结论或新实验。它不说明pair等价于当前own：n_c仍可依赖整个竞争集合；它只限定了“环测试”能证明什么。

因此应把已有测试称为“边级非加性示例”，将潜在收益归于类对条件权重改变了节点净证据，不能归于纯循环本身。无需为此更改当前求和或引入图求解器。若主线本来只主张边级非加性，则代码与该主张一致。

### 5.3 [P2，复现文档] 协议未记录门控前的预处理

定位：[run.py:47](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:47)、[run.py:146](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/competitive_evidence_operator/run.py:146)。协议只写log-mean-exp余弦匹配，没有记录sqrt(32)、逐位置有效类中心化及训练组RMS。本报告第4.1节补录实际操作，避免以后从协议重建时得到不同的E及门控。其RMS仅由tr求得，代码未使用evaluation拟合该尺度。依用户要求不改协议文件。

### 5.4 已一致的部分与审阅限度

静态读取确认：无序对统计交换对称；差值使边反对称；按有效类数K归约；无效查询组输出零；own/pair结构参数数匹配；优化器与200步及检查点日程符合协议。复用schedule以10001..10200随机种子抽源fit_train参考并保持既有可用性签名；当前文件没有按evaluation选择checkpoint。

已有checks.json记录零门控重放误差不超过4.45e-16、类/位置置换误差不超过1.39e-16；有限差分误差own约6.31e-8、pair约2.64e-9；边级环量约0.08139。本文只读取记录，没有独立复跑。这些是合成输入和特定梯度坐标的检查，不是全部数据链验证。receipts.csv的9条记录确认own/pair均615参数；pair种子29选择step0，所以不能把三个选中模型都描述成实际启用了学习后的非零第三通道。

未审计完整上游MMSI/pass划分和特征生成链，未重算指标，未查看本轮原图，不作全链无泄漏或实验完成认证。原图与收益分析仍由主线负责。除上述解释和协议记录边界，所读核心Pool代码未发现需要立即修复的功能错误。

## 6. 本轮结果可以怎样支持方法差异

以下是供主线解读已有实验的条件，不是本侧线要求追加训练：

- pair优于own时，先表述为“显式类对条件聚合相对共享中心化后的独立门控有增量”；相同参数量仍不等于相同输入函数族，四个多项式与四个对称统计的差别是本次比较内容。
- uniform同样有效而pair未胜出时，保留普通聚合的收益，不能将新增通道整体收益都归到竞争门控。历史普通匹配不是本轮同输入的精确替代，需保持其原有定位。
- 固定一组和一个类对，w非负且和为1，B_cd必在该组有效位置差值的最小值与最大值之间。如果所有位置都偏错，单靠该类对的正权重重选不能把边符号翻正；跨组alpha、其他类对与full仍可能改变最终判断。31546这类案例应据此分析，不能许诺“找到关键位置”必能救回。
- 权重图、非零环量或更大的真类单独分数都不足以证明完整模型获益；应结合主线已有的最强错类差距、净救回/误伤和实际参考变化。源selection已用于选点，485历史开发集也不能改称全新独立确认。

建议的贡献表述：本研究在冻结既有分类器与参考证据生成流程后，对局部多候选响应引入共享的交换对称类对门控，使不同候选对采用不同的位置权重，并以反对称差值汇总为类别增量。它相对已有硬间隔选择、全支持池关系归一化、最大置信度选择及图中心性聚合的具体区别，是**候选对条件的位置选择与有符号证据归约**。其是否形成跨场景净收益、何种参考与观测条件下有效，仍由本轮配对结果和原图证据决定；本文不宣称首次，也不提前宣布机制已成立。
