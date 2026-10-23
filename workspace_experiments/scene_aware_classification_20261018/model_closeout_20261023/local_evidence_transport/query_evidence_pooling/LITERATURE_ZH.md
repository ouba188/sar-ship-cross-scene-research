# 查询局部证据聚合：四篇原始方法的边界与一个可深化缺口

2026-10-24。独立文献侧线；未训练、未运行新分类实验、未操作 Git，只写本文件。使用 deep-read 梳理证据、discover-innovation 收敛缺口。文献限定 DN4、DMN4、FewTURE、FRN；不据此宣称完成全领域首创检索。

## 结论先行

1. **逐 query token 形成类条件证据、再平均 log，是值得做的普通聚合替代，不是强创新。** DN4 的逐位置近邻累加及 DMN4 的 NBNN 推导已给出直接基础。
2. **共享非判别响应混合空模型，属于合理的稳健化基线。** 四篇所读方法未给出下文完全相同的混合式，但这不足以使普通混合模型、软门控或 log 比值成为原创核心。
3. **跨 token 一致性必须具体定义。** 相同预测、平均概率、方差惩罚不是新的关系机制；DMN4 已处理多查询响应竞争同一支持描述子的冗余。FRN 的“整图重建”却仍对 query 各行独立求解，不是跨位置联合相容性。
4. **唯一建议继续深化的缺口：在保持每个局部的类别匹配强度不变时，辨别这些匹配是否组成有类别区分力、能跨采集复验的联合证据。** 这比再学一个权重有明确的不同研究对象，但现在只是可检验缺口，不是已成立的第三强创新。

## 1. 阅读范围与可复查来源

“全文”在此表示取得原始完整论文并直接精读相应方法段，不等于逐页精读参考文献与所有附录。页码为 PDF 从1开始；DMN4 另给正式印刷页码。

|论文|本轮实际阅读|定位和来源|
|---|---|---|
|DN4，CVPR 2019|取得作者 arXiv 全文，精读方法；另核对作者分类器代码。不是摘要推断|PDF pp.3–4，§3.2–3.3，式(1)–(2)；[论文](https://arxiv.org/pdf/1903.12290)，[CVF条目](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Revisiting_Local_Descriptor_Based_Image-To-Class_Measure_for_Few-Shot_Learning_CVPR_2019_paper.html)，[作者 classifier.py](https://github.com/WenbinLee/DN4/blob/master/models/classifier.py)，`ImgtoClass_Metric.cal_cosinesimilarity`|
|DMN4，AAAI 2022|读取 AAAI 正式全文及作者 arXiv 方法，精读正式版 Methodology 至 DMNN；仅查看仓库首页，未声称逐行复现代码|PDF pp.2–4／印刷pp.1829–1831，式(2)–(12)、图2–3；[AAAI原文](https://ojs.aaai.org/index.php/AAAI/article/view/20076/19835)，[作者预印本](https://arxiv.org/pdf/2103.08160)，[作者仓库](https://github.com/LouieYang/DMN4)。正式版方法小标题未编号；预印本节号不能直接移作正式版节号|
|FewTURE，NeurIPS 2022|复用已完成的本地全文方法阅读及固定提交代码笔记；本轮重读笔记、核对作者README。不声称本轮重新下载并精读PDF|此前原文pp.3–5，§2.1–2.4，式(1)–(2)；[作者论文入口](https://arxiv.org/abs/2206.07267)，[会议入口](https://openreview.net/forum?id=p_g2nHlMus)，代码提交`c67d4c1fe01d683abcc46eb409ce8eed6fcf1b76`；详细定位见下方本地笔记|
|FRN，CVPR 2021|取得作者 arXiv 全文，精读§3.1–3.7；核对作者重建及均值代码。不是摘要推断|PDF pp.3–5，式(1)–(13)，尤其§3.3；[论文](https://arxiv.org/pdf/2012.01506)，[CVF条目](https://openaccess.thecvf.com/content/CVPR2021/html/Wertheimer_Few-Shot_Classification_With_Feature_Map_Reconstruction_Networks_CVPR_2021_paper.html)，[作者 FRN.py](https://github.com/Tsingularity/FRN/blob/main/models/FRN.py)，`get_recon_dist/get_neg_l2_dist`|

FewTURE复用笔记：[FewTURE_CODE_NOTES_ZH.md](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/FewTURE_CODE_NOTES_ZH.md)。本轮其PDF接口返回体积限制，沿用允许复用的已核查记录，不把失败访问说成新全文阅读。DN4/FRN的CVF PDF接口受限，方法依据为作者 arXiv 全文及作者代码。DMN4仓库首页提示已弃用并指向后续工作；本轮不扩展第五篇，也未把后续方法混入DMN4。

## 2. 四个算子实际做了什么

统一记查询描述子为 q_i，类c的支持描述子为 s_cj，余弦为 t_icj。下式是统一符号转写，不冒充原文逐字公式。

### DN4：每个查询位置先找类内近邻，再累加

原文§3.3式(2)：

\[
L_c=\sum_i\sum_{j\in\mathrm{top}k(t_{ic:})}t_{icj}.
\]

作者代码沿支持描述子轴 `topk`，再对邻居和query位置求和。支持图像的边界被类内池合并；不要求各局部来自同一参考图。不对query位置做全局max，所以“让各query位置都有一票”已有基础。固定有效位置数时，sum改mean主要改变尺度；可变有效质量时才另有数量处理意义。[DN4原文及代码，表1]

### DMN4：已有逐局部log证据、竞争类比较和冗余抑制

正式版式(2)–(4)从条件独立得到：

\[
\log p(Q\mid c)=\sum_i\log p(q_i\mid c),\qquad
p(q_i\mid c)\approx |S_c|^{-1}\sum_j\kappa(q_i,s_{cj}).
\]

再用近邻近似。MNN在全类别支持池选query子集；DMNN对多个query竞争同一支持描述子时，用式(12)的最佳类与最佳异类近邻余弦差决定保留谁。最终图2仍为选中描述子的NBNN分类，不能把用于选择的相对分差说成额外的新网络。其优势针对任务无关、重复响应；不保证保留下来的匹配在空间或参考图身份上联合相容。[DMN4正式版pp.3–4，式(5)、(10)–(12)、图2–3]

### FewTURE：支持token权重，图像级全配对LSE

固定提交的`eval_FewTURE.py/PatchFSL`行为为：

\[
L_c(Q)=\log\sum_{j:y_j=c}\sum_i\exp((t_{ij}+v_j+M_{ij})/\tau).
\]

v由支持图像互相预测优化，推理时跨query共享；不是query条件权重。多shot内环屏蔽自身整图。官方v未作类内质量归一，因而还能产生类别偏置。本地冻结R50/PCA32、区域及VV/VH、归一化质量等改变了设定，只能叫FewTURE-inspired基线。其公式和代码依据已完整记录于上述本地笔记§2–6。

### FRN：类内线性重建，不是查询位置间联合推理

原文§3.1–3.2式(1)、(8)–(9)：

\[
W_c^*=\arg\min_W\|Q-WS_c\|_F^2+\lambda\|W\|_F^2,
\quad \widehat Q_c=\rho QS_c^T(S_cS_c^T+\lambda I)^{-1}S_c,
\quad L_c=-\gamma\|Q-\widehat Q_c\|_F^2/r.
\]

\(\lambda=(kr/d)e^\alpha\)，\(\rho=e^\beta\)。作者代码先求各query位置重建误差再`mean(1)`；§3.3明确各query特征独立重建。系数不是非负运输质量。§3.5的跨类别支持特征正交辅助损失作用于表征训练，也不是查询位置一致性。只在冻结PCA上换重建算子，不等于复现其训练过的完整FRN。[FRN原文§3及作者代码]

## 3. 本地聚合诊断：哪些改动真的改变了什么

以下为本地代码与代数分析，不是上述论文声称的实验结论。暂省略极化/区域g，令查询质量a_i和类内支持质量rho_cj分别归一，定义：

\[
f_{ic}=\sum_j\rho_{cj}\exp(t_{icj}/\tau)>0.
\]

当前`support_transfer_learning/run.py`先在`kernels()`消去query轴，再在`Readout`消去支持轴，等价于：

\[
L_c^{\mathrm{pair}}=\log\sum_i a_i f_{ic}.
\]

它对\(\log f_{ic}\)的梯度是\(a_if_{ic}/\sum_k a_kf_{kc}\)：高响应可获得多数贡献，但**本轮尚未测出三个案例是否真的由少数query位置支配**，不能提前写成根因。

### A. 逐query平均log：推荐做基线，不作创新宣称

\[
L_c^{\mathrm{meanlog}}=\sum_i a_i\log f_{ic}.
\]

它是加权几何平均的log，符合NBNN类独立证据累加的已有路径；由Jensen不等式不大于原pair分数，但两类的下降量不同，排名可能改变。它抑制跨query位置的指数赢家通吃，却不消除类内近邻偏好，也不消除重复query描述子的多票效应。无信息位置同样进入平均，可能稀释真正有用的少数位置。

重要等价：令\(p_i(c)=f_{ic}/\sum_d f_{id}\)，则\(\sum_i a_i\log p_i(c)\)与meanlog只差一个与c无关的常数，在相同有效位置和质量下，最终softmax/argmax完全相同。不要把二者作为两个独立创新。相反，先逐位置CE再平均与对聚合logit做一次CE的训练梯度并不等价；前者强迫每个局部单独识别整船，不能默认适合海面或不完整船体片段。

### B. 共享非判别响应空模型：有效的稳健化对照，不是自动的原创

设b_i是各候选类共享的响应参考，eta_i在[0,1]且不读query真标签：

\[
\widetilde f_{ic}=(1-\eta_i)f_{ic}+\eta_i b_i,\qquad
L_c^{\mathrm{mix}}=\sum_i a_i\log\widetilde f_{ic}.
\]

它改变弱位置的贡献，能避免强迫所有位置有类别证据。但这仍是普通混合/稳健估计；若eta由一个小MLP给出，就是普通门控，不能靠名称升级为强算法。b可以是类平衡支持池核密度，**不能把它叫纯背景或纯噪声**：它可能含船体共有结构和当前表示分不开的类别信息。缺失VH或无效区域应从有效集合移除，不是交给b补成观测。

两个容易做成无效改动的地方：

- 仅做\(\sum_i a_i\log(f_{ic}/b_i)\)，共同的\(-\sum_i a_i\log b_i\)会抵消，不改类别排名或最终softmax；它不是有效的新增校正。
- 换成\(\log\sum_i a_i f_{ic}/b_i\)则会改变相对query权重；效果来自重加权，不能借同一个“似然比”名称混为一谈。

若称其为概率密度混合，需要统一特征域、核归一常数和带宽；否则准确称为“正核响应混合”。四篇未直接给出此混合式，不等于广义文献中没有。

### C. 跨token一致性：普通版本已有；联合证据才是待检验问题

|具体做法|判定|
|要求各token预测同一类别，或最小化其概率方差|普通一致性正则，且可能误压制互补局部和极化差异|
|多个query反复使用同一支持响应时降权|与DMN4明确重叠；软化为可微权重不自动成为强创新|
|一类支持池重建所有query位置|FRN已有，但各行独立，不提供联合布局或参考身份约束|
|固定单点匹配分布，检查多个匹配是否来自可复验的联合响应关系|四篇所读算子未显式回答；适合构成下节研究问题，尚不能宣称领域首创|

**实现提醒：现有`kernel.npz`已经对query位置做LSE，不能从该缓存恢复meanlog、逐位置贡献或联合关系。** 必须从已有`tokens.npz`和冻结`projection_layer1.npz`重新计算、保留query轴；不需要重跑骨干。六个极化×区域组要保留原可用质量，不能将无效slot当成低响应负证据。

## 4. 本地证据起点：为什么不是简单“给general参考降权”

本轮直接查看原生160裁补、未缩放的三个VV/VH案例面板及实际参考；读`analysis/case_margins.csv`和`reference_contributions.csv`。本节比较固定seed17的uniform与token_pass，不混入mask_factorial或别的seed。

|案例|可核查事实|还不能推出什么|
|13372，general，救回|uniform判fishing，token_pass判general；general-fishing分差-0.04971变+0.04428。query为小型条状亮响应，同类33307/7884进入展示的高贡献参考|不能把general普遍定义为强散射；也不能仅凭图证明“多个独立部位共同支持”|
|1677，bulk，误伤|同一组general参考7884/33307也被采用；bulk-general分差+0.03365变-0.03149。query同样有紧凑条状亮响应|不能据此宣布参考错误；它们对13372有用。也不能从船轴接近推出类别因果|
|220，general，持续错|仍判bulk；general-bulk分差-0.11491变-0.09000。VV周围显示响应较强，VH不同；同类与混淆类都含亮条状目标|不能认定无同类信息；分差略改善但没跨界，也不能把显示亮度直接当校准RCS|

13372的full相对展示fishing虽有+0.03093分差，full实际判的是类别3，不是正确；“相对某个展示rival的正分差”不等于top-1正确。每类实际用8参考，面板只展示贡献较大的2个，不是本次只检索2例。

可见事实支持的问题是：**同一参考响应对不同query的价值不同；全局抬高或压低general参考都可能有损失。** 它尚不能区分“极少位置偶然匹配”“多位置错误但一致”“表征本身分不开”三种解释。下一步聚合诊断的价值正是把三者分开。

## 5. 唯一建议深化：有配对空模型的联合证据，而非强迫一致

### 5.1 研究问题与现有工作的具体差别

问题不是“亮点多的类是否得分高”，而是：**保持每个位置分别有多像某类不变后，这几个位置在同一参考实例中共同出现的相容性，能否额外区分真类与混淆类？这种相容性是否能在不同船、不同采集中再次得到支持？**

DN4/DMN4逐局部累加；FewTURE将所有配对指数相加；FRN各query行独立重建。它们都不足以直接回答这一特定联合增量。不能笼统说“前人不考虑局部关系”，DMN4已有竞争关系，FRN也已利用支持特征矩阵。

### 5.2 一个可证伪的最小数学入口

保留参考图身份r及区域身份。令\(h_{icr}>0\)为query位置i对类别c、参考图r的类内局部核响应，所有参考采用共同归一质量pi_r。对两个不同有效位置i、k，比较：

\[
\mu_{ic}=\sum_r\pi_r h_{icr},\qquad
J_{ikc}=\sum_r\pi_r h_{icr}h_{kcr},\qquad
J^0_{ikc}=\mu_{ic}\mu_{kc},
\]

\[
D_{ikc}=\log\frac{J_{ikc}}{J^0_{ikc}}.
\]

J保留两个响应在同一参考中的共同出现；J0允许独立抽取参考，保持两个单点边际完全相同。D因此衡量**超出单点强度的参考内依赖**，不是单纯把两个高分相乘，也不是新发明概率比或二阶统计。其核响应、参考权重及有效集合均应固定于配对比较中。

这里至少有三个必须正视的边界：

- D大也可能只因某张参考整体更亮、背景更接近，不能直接叫“正确船体组合”。各类都共有的依赖不应被包装成类别专属；应比较真类与竞争类D，并在跨采集、不同船上复验，而非给所有候选各自寻找最有利解释。
- i和k若是同一感受野信息的重复副本，也会显得一致。先用不同区域或较少重叠的槽位诊断；不能把相邻大量token算成独立支持，也不能以两个极化必同类预测为硬条件。
- 类内单图联合可能过严，否定DN4有价值的跨实例组合。**不要求整船所有部位必须由同一参考解释**，仅检验少数局部关系的额外证据。若需要把所有位置绑到一个r才涨点，必须排除退化成普通image-to-image检索。

真实可用的算法深化点应落在：**联合关系相对于独立空模型何时应被采用，以及如何通过跨船/跨采集复验学习其可迁移判别效用，同时保留单点基线。** 仅把D加到分数、加一个背景gate或换成均值，并不够强。

### 5.3 针对三个案例的可检验预测，不预设答案

- 13372：检查使general得分上升的位置是否形成可在多个独立general参考中复验、相对fishing更明确的联合关系。若只是所有位置普遍抬高general，属于类别偏好或单点度量，不支持联合机制。
- 1677：同样检查7884/33307。希望区分其对本query的匹配是否只在少数局部成立、其他局部并不相容；**若错误general匹配也联合一致，应承认该设计无法凭一致性识别错误**，需要判别表征或支持覆盖，而非再添一致性罚项。
- 220：检查general是否已有强单点但被别处抵消，以及joint是否提供额外区分；若各有效位置及其联合都更像bulk，继续换聚合不应被许诺能救回它。

这三个案例用于解释与开发，不作为优化专用目标或独立确认集。

### 5.4 最少但有辨识力的验证边界

先完成当前三种普通聚合对照及逐query贡献诊断。若存在联合信号，再比较保留J与固定边际的J0，使用同一套单点h、同参考、同有效槽位和同残差接入；不能同时改骨干、支持选择与最终倍率。

建议的机制干预是**只打散局部之间的参考身份配对，保持每个局部的响应边际、有效性与参考质量不变**。等权且共同有效参考可以置换；非等权时普通置换可能改变边际，应使用保持质量的耦合空模型，或先限定共同有效等权子集。VV/VH缺失不能由置换意外改变可用参考数。相同参考的多个部位不算跨船复验证据。

至少检查有益/有害联合关系能否在留出参考船和留出采集中重现，而不是只重复抽同一组船；按这些单位统计，不把token对数当独立样本量。训练只用源标签；query真类仅在诊断汇报中用于标记正负，不进入部署选择器。

已有本地`relational_kernel/FINDINGS_ZH.md`中，错误依附对照没有输给真实依附，故**正确区域身份重要尚未被证明**。本提案不能把那轮换名复跑成创新；其必要新增是保留单点边际的联合空模型与跨采集可复验性。若仍无法区别，收回联合机制主张，保留有效的普通聚合即可。

## 6. 交给主线的明确方法边界

- 可立即比较：全配对LSE、逐query平均log、共享非判别核混合；名称按普通基线写，有收益照样可用。
- 不要同时宣称：平均log是新贝叶斯算法、减共享log背景改变了分类、FRN已有跨query耦合、MNN等于物理对应、重复token投票等于独立证据。
- 最值得深化的单一缺口：**可跨采集复验、相对固定边际空模型有独立判别增量的局部联合证据**。
- 当前新意等级：问题定位与可验证设计，尚未达到已证实原创模块。若普通meanlog/混合就取得同等收益，应将其归为简洁有效改进，不硬保联合复杂性。
- 四篇的实验多围绕新类别few-shot；本地为冻结特征上的闭集跨采集识别。匹配头比较可说明本地算子适用性，不能代替四篇完整训练协议的强方法排名。

### 本地证据定位

- [当前算子](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/run.py)：`kernels`、`Readout.__call__`。
- [案例分差](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/analysis/case_margins.csv)、[参考贡献](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/analysis/reference_contributions.csv)。
- [13372原图面板](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/analysis/case_rescued_13372.png)、[1677原图面板](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/analysis/case_harmed_1677.png)、[220原图面板](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/support_transfer_learning/analysis/case_WW_220.png)。
- [已有二阶区域关系及错配结果](E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/relational_kernel/FINDINGS_ZH.md)。

证据分层：四篇公式/作者代码、本地预测表是直接依据；原图只说明可见响应差异；“偶然单点匹配导致误伤”“联合证据能跨采集救回”仍是待检验机制。本文未新增实验证据。
