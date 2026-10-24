# 参考敏感性、协方差聚合与模式保留：限定文献侧线

2026-10-24。仅新增本文；只读本地报告、实现及已完成源probe，不运行代码、训练或Git。本文按主线最新结果更新方向，不把原先设想的最小方差模块当作已获支持的下一步。检索是有界机制比较，不是全领域系统综述或首创认证。

## 1. 先给判断：能预测波动，不等于能预测错误

直接读取本目录 `probe_summary.csv`、`PROTOCOL_ZH.md` 和 `probe.py`：

| 已有源证据 | 能支持什么 | 不能支持什么 |
|---|---|---|
| 12个role×group的删参考预测SD与六库实际SD的Spearman为0.4143–0.7160 | 当前参考删除敏感性与换库分差波动有排序关联 | 校准的不确定度、错误概率、跨场景失效概率 |
| 类别内再按局部间隔二分，高/低不确定性准确率差在10/12组为正，范围+2.24至+11.44个百分点；selection VH_body为-1.36，utility VH_side为-0.46 | 当前分层描述通常不支持“越不稳定越容易错” | 高敏感性导致正确，或任何组显著优于另一组；这里没有独立重复/置信区间论证 |
| `offdiag_median`约0.609–0.821 | 协方差矩阵绝对值总量中非对角项占比大 | 平均相关系数很大，或利用非对角项必有分类收益 |

两个诊断的候选对不同：换库相关性使用冻结full配置17的top2；准确率分层使用各组局部读出的winner/runner。不能将前者的相关性直接解释为后者的错误可预测性。类别调整具体是类内间隔二分、各单元中按local_sd中位数分高低、再等权平均；并非消除了全部混杂。六库共享源池，query×group也不是独立样本。

**当前结论是停止从“方差存在”直接跳到“惩罚方差”。** 更合理的待解问题是：删去某参考会否移除对当前查询有用的类内模式，而全8参考的混合是否误用了模式质量。稳定但错误与敏感但正确都必须保留；不能因文献有成熟QP就忽略这里的直接负证据。

收口时主线已完成同LME、同位置读出的nearest2/random2/all8源对照。已读取 `reference_scope_metrics.csv`、`reference_concentration.csv` 和 `reference_scope.py`；selection各组局部BA如下，单位%：

| 组 | all8 | nearest2 | random2（固定711） |
|---|---:|---:|---:|
| VV_body | 35.81 | 34.28 | 30.21 |
| VV_side | 28.09 | 29.55 | 26.13 |
| VV_end | 27.68 | 26.22 | 21.24 |
| VH_body | 25.56 | 23.16 | 25.27 |
| VH_side | 25.33 | 25.25 | 21.86 |
| VH_end | 23.35 | 24.46 | 21.65 |

**nearest2没有统一优势。** 正负差异依赖区域和极化，不能因VV_side获益就将全部参考改成top2；random2也不是多个随机种子的均值。这里评价局部分类，不是新完整模型收益；对历史oldlocal优势的归因仍未完成。

12个role×group中，删除方差与有效参考数的Spearman为-0.9396至-0.7325。具体是**真类**参考责任 `p_r` 的逐位置 `1/sum_r p_r^2` 再按位置平均，对比真类均匀读出删除方差；不是部署时无标签的类对风险。它强烈提示该量主要跟踪少数参考依赖，但不等于独立证明噪声、估计了因果贡献比例或解释了跨场景错误。两者共享相同响应剖面的代数结构，见第4节。

## 2. 本地已有机制，及其与本候选的关系

以下均据实际文件读取，不仅依据方法名称。路径相对于上一级 `local_evidence_transport/`。

| 本地实现/证据 | 已经做过什么 | 本轮需要区分的对象 |
|---|---|---|
| `competitive_evidence_operator/run.py` 的 `evidence`、`Pool.forward`；对应FINDINGS/LITERATURE | 有效参考船等权、船内有效token等权的LME；跨类中心化；交换对称类对MLP选择query位置；反对称边后节点求和 | 当前门控不读删参考协方差。pair没有稳定超过uniform/own；同系数uniform更好，不能以变权证明选择有用 |
| `support_transfer_learning/run.py` 的 `Readout`；`FewTURE_CODE_NOTES_ZH.md` | 支持互相预测，学习参考或支持token的共享权重，类内质量归一 | 普通支持权重和留出自身预测已存在；新的参考路由不能只换名称 |
| `reference_dependence/run.py` 的 `dependence/prepare`；`query_evidence_pooling/ALGORITHMIC_TARGET_ZH.md` | 同参考上跨区域核响应的归一化协方差型统计，保边际置乱对照 | 新删参考协方差与旧响应依赖统计有一阶联系，见第4节；“考虑相关性”不是新贡献 |
| `conditional_relation_operator/run.py` 的 `transfer/raw_scores`；`reference_relative_metric/run.py` | 岭条件重建、真实/随机配对风险差、相对identity读出 | 这些矩阵是特征/配对关系的算子，不是query token类间分差对删船的敏感性矩阵 |
| `conditional_relation_operator/reference_resampling/episodic/PROTOCOL_ZH.md` | 每步重采样源64参考，并用六库源selection选点 | 参考组成变化已用于训练组织；不是首次引入参考扰动 |
| **`match_readout.py` 的 `make_bank/main/operations`** | 每类KMeans选8个独立MMSI参考；每张query按主体描述相似度从每类选2船，各匹配臂共用这2船；independent做区域内带dustbin的OT | **已确认selected2与当前all8不同。** 原方法不是全8船局部证据的同权版本；还同时有OT/LME、面积质量/有效token均值、聚合轴等差别，不能只凭历史分数将收益归因于top2 |

历史8参考本身由聚类代表选择，并非独立同分布随机抽样。它们可能覆盖不同外观；但KMeans簇不能直接命名为船向、散射机制或港口模式。主体top2是查询条件的参考组织，也不能先验称“去噪”。

## 3. 三条最接近的机制线

### 3.1 组合优化与JMA：联合风险已有成熟数学，但风险目标必须正确

Markowitz [1，pp.79–83、89–90] 明确区分组合均值与组合方差：统一符号下，收益为 `w^T mu`，风险为 `w^T Sigma w`，且非负权重和为1。关键是交叉协方差，不能靠增加高度同动的项来分散风险。当前“query位置作组合分量、删参考作扰动来源”的simplex QP，是该结构的任务适配，不是新优化原理。

Hansen与Racine的JMA [3，§2–3、式(5)–(6)] 更接近“留一+QP”：构造各模型对被留出观测的预测残差矩阵R，最小化 `w^T(R^T R/n)w`，权重在simplex上。**R是有真值的预测误差，不是固定query删除支持后的中心化响应。** 未中心化误差二阶矩保留偏差信息；删参考协方差会消掉稳定偏差。JMA的随机独立观测、线性估计器及离散权重理论限制，不能移植为本候选的最优性保证。

有意义的差别只能是本任务的扰动单位、保留的证据对象及任务效用；不能把“jackknife+协方差+QP”当贡献。源probe目前未把该敏感性连接到错误，故纯最小方差训练没有足够依据。

### 3.2 最小方差波束形成：抑制变化必须另有可信的信号保留条件

Lorenz与Boyd [2，§I、§I-C、§II，式(4)、(17)–(22)] 在最小化输出功率时约束目标响应不被消除；进一步对响应向量的不确定椭球要求最坏情形增益不低于1。实值转写为 `min w^T R w`，满足 `a0^T w - ||A^T w||_2 >= 1`，对应二阶锥约束。它不是一般simplex QP，也不要求权重非负。

可借鉴的是**保留想要的信号，再抑制有根据的扰动**。困难在于这里没有已知的“正确类别信号方向”：若把当前赢家或大绝对间隔当作steering vector，就可能保护稳定错误。若用源真类构造方向，只能用于源监督或诊断，部署时不能读取query真类。删参考椭球也不是已经校准的跨场景不确定集合。

这条线不支持把所有参考异质性当干扰；它反而说明，仅凭相关波动不足以决定应压掉哪部分。不得把删除匹配输入的算法干预称为SAR生成机制的因果干预。

### 3.3 IMP：类内多模式需要保留，训练目标也不能把它们都压向同一证据

Allen等的Infinite Mixture Prototypes [4，§3、Algorithm 1、式(6)–(7)] 从类均值起步，根据距离阈值增加簇，用软分配更新簇中心；一类可以有多个原型。查询以每类最近簇比较，训练也只保留每类最近簇，避免同时惩罚全部簇而压制多模态。这里的“最近簇”及多模态训练是具体机制，不是笼统增加注意力。

这与主线当前问题更贴近：同类参考可能是不同的有效解释，query未必应与所有模式同时匹配。历史主体top2已具有局部参考选择，故“按query挑参考/多个模式”并非空白。可能值得发展的差异是：**在当前VV/VH及区域证据中保留有用参考模式，并区分模式替换造成的变化与模式内重复参考的变化**；只有源配对结果能说明这种区分是否有用。当前all8的LME也不是单一均值原型，不能用IMP对单原型的批评直接给本地机制定罪。

补充近邻：UAFS [5，§3.2、式(3)–(8)] 已通过全候选query–prototype图学习相似度方差，并以高斯采样进入分类目标。因此“匹配不确定性辅助分类”已有先例；本文讨论的可观测删船剖面、token间联合协方差与它的学习标量方差不同，但区别尚不等于优势。

## 4. 数学定位：可写出的QP，与它尚不能解决的问题

省略query和组g。令 `h_cir>0` 为query位置i对类c参考船r的船内平均指数相似度，`L_ci=log(mean_r h_cir)`；当前类对向量为 `m_cd=(L_c-L_d)/s_g`，尺度s_g固定。每类按有效参考船分别删一：

```math
\ell_c^{(-r)}=\log\!\left(\frac{\sum_{s\ne r}h_{cis}}{n_c-1}\right)_{i=1}^{T},\qquad
\bar\ell_c=\frac1{n_c}\sum_r\ell_c^{(-r)},
\quad
\widehat\Sigma_c=\frac{n_c-1}{n_c}\sum_r(\ell_c^{(-r)}-\bar\ell_c)(\ell_c^{(-r)}-\bar\ell_c)^T.
```

按两类分别扰动的设计，`Sigma_cd=(Sigma_c+Sigma_d)/s_g^2`。这是当前probe的构造，不是现实两类环境变化独立的事实。不要将两类第r船人为配对，也不要把全部64船删除结果当64个独立重复；每类n通常至多8。n<2是不可估，缺失极化不是零风险。删船、换库和按过境共同删船对应不同问题，不相互冒充。

**与旧协方差统计的联系（本文推导）。** 记 `z_cir=h_cir/mean_r h_cir`，则单位置精确删除变化为：

```math
L_{ci}^{(-r)}-L_{ci}
=\log\frac{n_c-z_{cir}}{n_c-1}
\approx-\frac{z_{cir}-1}{n_c-1}.
```

一阶展开给出 `Sigma_c,ij ≈ sum_r[(z_cir-1)(z_cjr-1)]/[n_c(n_c-1)]`。旧 `reference_dependence` 的 `log E[h_i h_j]-log E[h_i]-log E[h_j]` 同样源于归一化核响应协方差。因此这里主要变化是token/类对尺度、非线性删船敏感性及读出用途；不能宣称发现全新统计信息。n小或某参考主导时一阶近似不可靠，应使用已有精确删除，不因近似另起训练。

进一步，单位置 `p_r=h_r/sum h`、`n_eff=1/sum p_r^2` 时，同一近似的对角项为 `(n_c/n_eff-1)/(n_c-1)`。所以方差与有效参考数的强负相关部分源于共同构造，并非两个独立测量相互验证“噪声”。本地报告比较的是跨位置聚合方差与平均有效数，两者还受非对角项影响，不能把这条单位置近似当成已测统计的精确等式。

供方法边界讨论的候选QP可以写为：

```math
\min_{w\ge0,\;\mathbf1^Tw=1}
w^T\widehat\Sigma_{cd}w+\lambda\|w-u\|_2^2-\eta a_{cd}^Tw,
\qquad B_{cd}=w^Tm_{cd}.
```

这里u为有效位置均匀权重，lambda、eta非负；a必须是候选交换对称、部署可得的效用。该式**只是明确原设想，未被当前结果批准为训练方案**。若 `a=|m|`，只是偏好强证据，不保证方向正确，也不保证正负位置不会抵消。若以 `a=sign(u^T m)m` 保留当前方向，就会偏好既有赢家；仍可能保护错误。

需保留以下直接影响解释的限制：

- **类对交换。** `m_dc=-m_cd`，`Sigma_dc=Sigma_cd`。u、a、约束和选解必须交换不变，才能有 `w_dc=w_cd`、`B_dc=-B_cd`。直接最小化 `w^T Sigma w-eta*m_cd^T w` 不满足这一要求。lambda>0提供唯一解，但属于正则选择，不是新增信号。跨类共同中心化在差值中抵消；固定尺度时删除第三类也不应改变该差值。
- **稳定错误。** 若所有位置对真类的差值均为-1且删船不变，则Sigma=0，任何simplex池化仍为-1。反之，只能由少数参考支持的正确模式可能很敏感。一般有 `min_i m_i <= B <= max_i m_i`；纯正权重不能把全负边改正。
- **方差不等于误差。** 固定w时，误差二阶矩含 `w^T Sigma w+(w^T b)^2` 的偏差项。当前诊断没有观测到校准真分差，也没有估出b。相关性预测成功只验证敏感性用途。对自适应w，`w^T Sigma w`还是固定读出下的删除风险，不能直接称重新求解权重后的整个算法方差。
- **共同波动未必可消。** 若 `Sigma=sigma^2*11^T`，所有simplex权重的风险相同；若 `Sigma=0` 或近零，结果可能主要来自u锚定。对均匀u，`||w-u||^2=||w||^2-1/T`，与对角加载等价到常数。这是普通正则，不是发现了新的判别线索。每类删船矩阵秩至多n_c-1，强求逆没有必要；simplex QP与逆矩阵后截负归一不是同一个解。
- **信号保留不能自我封死。** 若硬约束 `m^T w=m^T u`，则该边输出与均匀池化完全相同；在现有仅使用边值的节点归约中，权重再复杂也不改变预测。放松约束才可能改变输出，但同时失去完全保真保证。节点求和仍会消去纯循环边，不新增循环推理主张。

## 5. 收口：停止单调惩罚，保留尚未检验的上下文用途

selected2/random2/all8的匹配源对照已经完成，不再作为待办建议。实际结果没有给出统一参考范围修法，也没有证明全部异质性有害。下一步由主线原chip和实际参考案例分析区分：敏感但正确是否依赖少数有用参考，稳定但错误是否缺少可分证据，以及nearest2救回/误伤具体对应什么参考改变。本文不执行或预写这些案例结论。

“保模式”与“压波动”可用概念分解区分：`Cov(Z)=E[Cov(Z|M)]+Cov(E[Z|M])`。M若代表可重复、与query匹配有关的参考模式，后一项可能是有用差异；把两项一并惩罚可能抹去它。这里是分析视角，**尚未从本地数据识别M**，不把既有KMeans簇或图像外观直接当成物理模式，也不假定删除剖面已经服从该潜变量模型。

两条可以保留的差异空间如下，都是定位边界，不是新模块或追加训练建议：

1. **模式保留的参考组织。** 相对旧top2与普通支持权重，差异需要落在当前主体、区域和可用极化条件下如何保留参考解释，而不是再提出一个统一top-k。IMP提供先例；本地配对结果显示组织作用有条件，但尚未证明这些条件能被模型可靠识别，也未识别真实模式。
2. **敏感性作为上下文，而非惩罚。** 当前结果否定了单调降权的直接依据，却不逻辑排除响应剖面、参考责任集中度与query内容共同提供条件效用。由于敏感性与有效数高度重叠，任何未来“新增信息”主张都须超过已知间隔、有效参考数和普通内容条件读出所能解释的部分。尚无这种增量证据；不据此继续给现有MLP添加输入或启动训练。

报告中的“新增信息”应指相对当前仅看LME均值的读出，恢复了它丢掉的参考身份/响应剖面，而非凭空增加观测。删除剖面是原query与参考数据的确定函数。需要证明的是这些已有但未利用的细节具有任务增量，而不是矩阵非零、权重改变或方差降低。源probe本轮已经给出了一个清楚边界：**停止把敏感性当噪声的单调惩罚路径；上下文效用仍是未证实问题，不是继续训练的既定理由。** 不预支novelty、分类收益或因果解释。

## 6. 一手阅读记录与访问限制

直接web.run本轮无返回；备用search_service成功取得下列原始论文正文。本文只概括已读的方法范围，不声称逐页读完证明/附录，也未运行作者实现。论文公式采用统一符号转写；第4–5节的本地联系与候选设计是本文推导。

| 编号 | 一手来源与本轮实际读取范围 |
|---|---|
| [1] Harry Markowitz. *Portfolio Selection*. Journal of Finance 7(1):77–91, 1952. DOI `10.1111/j.1540-6261.1952.tb01525.x`。大学托管原文PDF：`https://traders.berkeley.edu/papers/Markowitz.pdf`；读取pp.79–83及89–90的均值、方差、非负组合与相关性论述。未将金融收益解释为分类真值 |
| [2] Robert G. Lorenz, Stephen P. Boyd. *Robust Minimum Variance Beamforming*. IEEE TSP 53(5):1684–1696, 2005. DOI `10.1109/TSP.2005.845436`。作者PDF：`https://web.stanford.edu/~boyd/papers/pdf/rmvb.pdf`；读取pp.1684–1687、§I–II。PDF部分公式文本抽取缺符号，正文中的实值椭球约束由定义转写推导，未声称页面截图核验成功 |
| [3] Bruce E. Hansen, Jeffrey S. Racine. *Jackknife Model Averaging*. Journal of Econometrics 167:38–46, 2012. DOI `10.1016/j.jeconom.2011.06.019`。作者PDF：`https://users.ssc.wisc.edu/~behansen/papers/joe_12.pdf`；读取pp.38–40、§2–3及导言理论限制 |
| [4] Kelsey R. Allen, Evan Shelhamer, Hanul Shin, Joshua B. Tenenbaum. *Infinite Mixture Prototypes for Few-Shot Learning*. ICML/PMLR 97:232–241, 2019。官方PDF：`https://proceedings.mlr.press/v97/allen19b/allen19b.pdf`；读取PDF pp.3–5、§2.3–3.3、Algorithm 1及式(6)–(7)，重点为多模式及训练时的最近簇选择 |
| [5] Zhizheng Zhang等. *Uncertainty-Aware Few-Shot Image Classification*. 作者预印本 `arXiv:2010.04525v2`，2021-06-03。`https://arxiv.org/pdf/2010.04525`；读取PDF pp.3–4、§3.1–3.2、式(3)–(8)。不以预印本日期代替正式会议元数据 |

检索记录（2026-10-24，备用网页检索）：`Jagannathan Ma 2003 risk reduction large portfolios wrong constraints pdf`；`Hansen Racine 2012 jackknife model averaging pdf`；`robust minimum variance beamforming Lorenz Boyd 2005 pdf`；`covariance aware pooling jackknife reference uncertainty few shot classification minimum variance`；主线结果更新后追加 `Infinite Mixture Prototypes few shot learning Allen NeurIPS 2019 paper`（检索词中的会议猜测经PMLR核对纠正为ICML）。

访问边界：Jagannathan–Ma正文403，检索还发现2019勘误，未取得全文，不依赖其命题或证明；Efron–Stein扫描PDF只取得封面文本，未将其定理作为本地抽样保证；CASR检索命中但正文429，不凭摘要判断其与本任务相同或不同。本文的经典组合结构以已读取的Markowitz原文为依据。DMN4/ATL-Net/TALDS-Net/MCL及FewTURE复用前轮本地阅读记录，本轮没有冒充重新读完这些论文。
