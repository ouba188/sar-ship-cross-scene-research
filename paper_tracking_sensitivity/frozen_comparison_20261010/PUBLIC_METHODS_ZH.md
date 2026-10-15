# 公开方法核读与冻结模型比较适配

日期：2026-10-10。状态：**论文及官方关键源码已核实；未复现、未运行训练、未修改主代码。** 本文给主代理提供实施协议，不能作为已完成实验的证据。

主代理完成状态补记：上句描述文献核读支线的工作。随后主代理已跑完两个冻结元ridge头，各25任务/125抽样，数值见同目录FINDINGS_ZH.md及各子目录；完整图像R2D2、完整OvA和SARATR-X仍未完成，不能据文献核读状态混淆实验状态。第二个头补齐官方求解公式，但正则初值、优化轮次与冻结骨干设定仍属显式协议适配。

本地依据：`frequency_closeout_20261010/COMPARISON_SCOPE_ZH.md`、`RELATED_WORK_ZH.md`，以及主代理本轮补充。范围仍为23个合格港口任务、F_0921、OpenSARShip 2.0，同五组64标签。主代理已实现同预算简单强基线和23种路由；其通报称全量softmax均值融合的三集OA/BA均高于5%主方案。该结果应保留，本支线未独立核算，不据此改模型。

## 1. 立即执行的选择

**第二批优先实施 `R2D2-inspired frozen-feature meta-ridge (learned lambda/alpha)`。** 在源episode上用query交叉熵学习正值lambda及alpha；目标端只用既定64标签解ridge。骨干与主方案均保持冻结。这是有明确原文依据的元学习头适配，不能写成完整R2D2复现，也不同于主代理已完成的普通ridge源验证选参。

| 方法/变体 | 是否需要新的训练 | 本轮定位 |
| --- | --- | --- |
| 冻结特征普通ridge，源验证选择lambda | 不需要梯度元训练；目标支持集闭式拟合 | 已有强基线；不是R2D2 |
| 冻结特征lambda/alpha元学习ridge | **需要源端轻量episodic元训练**；不重训骨干 | 推荐第二批；R2D2-inspired协议改版 |
| 完整图像R2D2 | 源端episodic训练编码器及调整层；lambda是否学习按配置 | 不在本次立即实施建议内 |
| 冻结特征OvA，8+1输出 | 需要拟合类别与专家正确性轻量头 | 完整OvA损失的冻结表征适配，可后补 |
| 固定基础预测OvA预算路由 | 需要拟合OvA头，但保留原基础决策 | 路由适配；不能当作原OvA完整系统 |
| SARATR-X官方分类分支 | 可复用公开预训练权重；仍需64标签拟合分类头 | 可获得的SAR分类候选，不阻塞第二批 |

此处“源端”指本项目允许的源训练任务；原OvA论文本身并没有强制“源域到目标域”设定。已有合规训练好的元头时可直接复用；本次没有核实存在这样的工件。

- **主代理实施状态：初版 frozen head 已运行并保留**，无常数/bias列、无sqrt(n)缩放，lambda初值1、alpha初值1，30 epochs、Adam lr=0.01；属于冻结骨干元ridge改版，不是官方默认配方或完整R2D2复现；本行不报告成绩。
- **主代理实施状态：第三臂 normalized_head 的源训练正在运行**，加入官方常数列与平均平方损失缩放，lambda初值1/64（n=64时仅正则尺度等效于未归一化lambda=1，加入截距后不代表与初版模型完全等价）、alpha初值1，同30 epochs、Adam lr=0.01；冻结骨干、不按目标选配置，仍非官方默认配方或完整复现，成绩未知。

## 2. R2D2：公式及官方实现

### 2.1 已核实的原文

Bertinetto等，*Meta-learning with differentiable closed-form solvers*，ICLR 2019。核读原文§3.1-3.4、§4.1，PDF pp.4-6。[R1]

令支持特征为X，one-hot标签为Y，query特征为Xq：

\[
W=(X^TX+\lambda I)^{-1}X^TY
 =X^T(XX^T+\lambda I)^{-1}Y,\qquad
\widehat Y_q=\alpha X_qW+\beta.
\]

分别对应式(4)、式(5)及式(6)。式(2)和§3.4说明：外层在episode query标签上计算损失，通过求解器反传，训练表征参数及所选元参数。测试episode的支持集求解W，不等于重新训练编码器。

**完整R2D2的关键是episodic外层元学习与可微闭式求解结合，不是仅仅使用ridge，也不是必须学习lambda。** 原论文允许学习lambda、alpha、beta；官方发布配置默认固定lambda，仍然称R2D2。[R1-R4]

### 2.2 直接影响复现的源码事实

已读 `fewshots/models/r2d2.py` 的 `RRNet.loss/rr_standard/rr_woodbury`，以及 `adjust.py`：[R2]

1. 向支持特征追加常数1列，ridge同时拟合类别截距；该常数列也被正则化。
2. 令n为支持样本数（含augmentation倍数）。追加常数列后的X和one-hot Y均除以sqrt(n)。因此实际是平均平方误差尺度，不能把lambda=50直接塞入未归一化的求和平方误差实现。
3. query特征不除以sqrt(n)，使用 `zq @ w + b`，再经标量调整层。
4. alpha/beta在官方默认 `adj_base=1` 时直接是可训练标量；不是强制指数参数化。lambda在 `lambda_base=1` 时也直接使用。
5. 编码器在源码中处理拼接后的support/query。移植到缓存特征时，须说明特征产生及归一化统计是否访问query；本次没有完整核验官方推理时所有归一化行为，不能仅凭拼接代码断言其一定是transductive。

上述缩放可等价写成：

\[
\widetilde X=[X,\mathbf1],\quad A=\widetilde X/\sqrt n,\quad B=Y/\sqrt n,
\quad W=A^T(AA^T+\lambda I)^{-1}B,
\quad L_q=\alpha[X_q,\mathbf1]W+\beta.
\]

对应目标为 `(1/n)||X_tilde W-Y||_F^2 + lambda||W||_F^2`。**ridge里的类别截距向量和最后共享beta不是一回事。** 共享beta对softmax及CE无影响，可以在适配中省略；不能因此删除ridge的常数特征列。

### 2.3 官方超参数，不冒充项目最优值

| 项目 | 已核实值 | 来源/解释 |
| --- | --- | --- |
| lambda初值 | 50 | `scripts/train/conf/fewshots.yaml` |
| 学习lambda | 默认False | `base_learner.learn_lambda` |
| alpha初值 | 默认1e-4 | miniImageNet实验也显式使用1e-4 |
| CIFAR-FS及Omniglot alpha初值 | 10 | `scripts/train/experiments.sh` |
| lambda/adjust底数 | 均为1 | 默认直接标量，不是log参数 |
| 外层优化器/学习率 | Adam / 5e-3 | 默认配置；不保证适合当前缓存尺度 |
| epochs / patience | 1200 / 200 | 默认配置，不要求机械照搬到轻量元头 |
| 学习率衰减 | 每20轮乘0.5 | 默认配置 |
| meta-train way | miniImageNet 16；CIFAR-FS 20 | 实验脚本；原评估多为5-way 1/5-shot |

来源[R3-R4]。本项目是闭集八类跨域，源目标类别语义相同；原标准few-shot benchmark是meta-train/meta-test类别不相交。这是任务设定差异，不应为“忠实”强行改掉项目八类标签体系。

### 2.4 最小冻结特征适配

下面是**建议协议**，不是原论文声称已经验证的SAR方案。

1. 复用已确定的一种缓存表征及其预处理；不新扫均值/phi/核的组合来挑目标结果。若用项目专属phi或核，明确列在“相同表征的机制比较”，不当作独立图像系统优势。
2. 使用源episode的support/query及真实标签。源support解W，源query CE训练全局共享lambda/alpha；query不是目标查询集。元训练/元验证按独立源任务或原有源隔离协议拆分，不能同一批源query既训练元参数又报告为独立验证。源验证选择训练轮次，目标查询标签只评价。
3. 推荐用 `lambda=exp(log_lambda)`、`alpha=exp(log_alpha)` 保持正值；这是对官方直接参数及默认固定lambda的显式改动。lambda可从50开始；alpha初值必须按源特征尺度预先固定。1e-4和10都是真实官方设置，不能宣称其中一个在当前标准化缓存上必然合适。若沿用官方默认外层Adam 5e-3，记录其为起始配置，不承诺已验证收敛。
4. 一次源元训练得到共享lambda/alpha；五组目标支持抽样各解一次W。不对每个目标域再选lambda、alpha或checkpoint，不使用目标query进行梯度更新。
5. 维持原64个样本和原八类顺序。64总标签不等于均衡8-way 8-shot；官方规则张量采样器需要换成已有样本索引。支持缺类时，保留八列one-hot、缺类列为零，不补采目标标签；若采用源先验补齐，必须另标“带先验改版”。
6. 最小元ridge采用不加类权重的原始平方损失及CE；类平衡版本若实施须另标，不把BA收益与原损失混称。
7. 几乎只学习lambda/alpha两个元参数，但成本还包含每个源episode的求解和反传。alpha为正时对单个episode的argmax没有直接影响，它通过外层CE影响lambda学习，也影响最终概率/融合；不能把alpha单独学习描述为必然提升硬分类准确率。

**可直接移植的PyTorch核心**（文档示例，未运行；使用现有环境，不要求新安装）。这保留官方support缩放与正则化截距；现代线性方程求解替代旧 `torch.gesv`/显式逆。输入必须是同一已固定预处理下的二维特征，y取0..7：

```python
import math
import torch
from torch.nn import functional as F

def meta_ridge_logits(xs, ys, xq, log_lambda, log_alpha):
    n = xs.shape[0]
    a = torch.cat([xs, xs.new_ones((n, 1))], dim=1)
    a = a / math.sqrt(n)
    b = F.one_hot(ys.long(), num_classes=8).to(xs.dtype)
    b = b / math.sqrt(n)
    eye = torch.eye(n, device=xs.device, dtype=xs.dtype)
    dual = torch.linalg.solve(a @ a.T + log_lambda.exp() * eye, b)
    w = a.T @ dual
    aq = torch.cat([xq, xq.new_ones((len(xq), 1))], dim=1)
    return log_alpha.exp() * (aq @ w)

# Create log_lambda/log_alpha once and optimize them across SOURCE episodes:
# loss = F.cross_entropy(meta_ridge_logits(xs, ys, xq, ll, la), yq)
# optimizer.zero_grad(); loss.backward(); optimizer.step()
# Select the epoch on held-out SOURCE episodes, then freeze ll and la.
# On each target draw: xs/ys = its 64 supports; xq = its query features.
```

骨干特征可detach，**不要detach求解W或log_lambda/log_alpha参与的计算**。否则外层无法学习。全局元参数不能在每个episode重新初始化；同一次目标query预测不得使用其yq求loss更新。这两点比更换骨干更值得实施时检查。

缓存接入依据主代理提供的字段：`frequency_preserve_20261010/source/{dataset}/{task}/episode_*.npz` 有 `indices, learned, expected`；原 `learned_frequency_residual_20261009/tasks/.../episodes/00.npz` 有 `phi/prior/mean/conditions/support/query/labels`。**本次没有读取这些数组的shape及语义，不假定learned/expected就是特征、概率或标签。** 由主代理沿已有解析代码选择X，并用indices对齐样本；不能按两个文件的物理行号默认一一对应。已经运行成功的普通ridge输入最适合作为同表征元ridge的X，以隔离“源验证选参”和“可微元学习”的差异。

建议表格行名：`Meta-ridge (frozen features; source-episodic lambda/alpha; R2D2-inspired)`。若只对lambda做源验证网格、没有外层梯度训练，仍叫 `Ridge (source-selected lambda)`。

## 3. OvA learning-to-defer：可实施，但区分两个问题

### 3.1 已核实的方法

Verma与Nalisnick，ICML 2022，PMLR 162:22184-22202，正确页面标识 `verma22c`。[O1]

式(8)在logistic损失下等价于8个类别二元CE加1个专家正确性二元CE：

\[
\ell=\sum_{k=1}^{8}\mathrm{BCEWithLogits}(g_k,\mathbf1[y=k])
 +\mathrm{BCEWithLogits}(g_e,\mathbf1[m=y]).
\]

每样本先求9项之和，再对样本平均，原论文实验无类别重加权。`m`是专家预测；代码中的 `m` 参数通常已经转成“专家是否正确”的0/1，接入时不要混淆。原规则 `h=argmax_k g_k`，转交条件 `g_e >= max_k g_k`；正确率估计为sigmoid(g_e)，类别置信度用独立sigmoid，不能把9维做softmax。[O1-O2]

**更正候选范围中的一个潜在误读：预算截断并非本文新增。** 原文附录E（PDF p.18）已经规定按 `max_k p_k - p_e` 升序，转交 `min(floor(b*N), n_eligible)` 个，其中eligible为 `p_e >= max_k p_k`。因此b=0.05可直接使用该规则。概率差排序和logit差排序不等价，虽然它们的零阈值判定相同。[O1]

### 3.2 两种移植的准确命名

**A. `OvA-L2D / frozen-feature classifier`：** 冻结同一编码器或输入特征，拟合9输出线性头（也可用已有小网络，但记录容量），严格采用上式。专家使用固定模型在源留出案例上的预测。最终未交接样本用新OvA类别头的argmax。这保留完整OvA损失与原决策，改变的是表示、数据和训练协议；适合完整系统比较。

**B. `OvA-based fixed-base routing`：** 固定当前已适配基础/专家预测，以同样9输出损失估计类别概率和专家正确率；实际基础决策仍为h_B，所以用 `p_e - sigmoid(g[h_B])` 排序及非负资格条件，而不是 `p_e - max_k sigmoid(g_k)`。只有当h_B等于新OvA argmax时两者一致。该路由变化是明确的协议适配，不继承原系统的一致性保证，也不能说是“完整OvA复现”。

若只拟合基础正确性及专家正确性两个BCE，再取差值，是更简洁的双正确率估计，**属于OvA启发的路由，缺少原8类别OvA项**。现有七参数多头校准器也不应改名为完整OvA。

最小可执行核为 `Linear(d,9)` 加 `binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(1).mean()`；targets前8列为one-hot，第9列为专家预测是否正确。使用源验证选择优化轮次/正则；不需要重训图像网络，但确实需要拟合头，不能只重排现有分数就称已完成OvA。

源训练应模拟目标适配后专家能力：用源伪目标episode的独立query预测生成正确性监督；不能用专家刚拟合过的support内准确性冒充外推正确率。若将目标64标签用于再拟合路由，则正确性监督要来自支持集OOF预测，且须记录此目标适配步骤；源训练版本和目标再拟合版本不混成一行。

排序若限制到项目共同候选集，另标constrained；预算分母仍为预先规定的完整query数，实际转交率照实记录。全query批量排序属于预算批处理访问，不等于用query标签训练。若输入包含全量专家特征或分数，费用应计入，不能因5%交接宣称节省95%特征计算。

### 3.3 官方源码中实际可达的问题

已读官方 `losses/losses.py`、`main.py`、`validation.py`，并核对论文PDF p.6脚注确实链接该仓库。[O1-O4]

- `LogisticLoss` 对零logit作原位无穷替换，且直接计算exp，不能作为新实现的稳定数值模板。按原文使用BCEWithLogits/softplus即可；官方使用log2，若要数值一致可除以ln(2)，并注意正则相对权重。
- 两个evaluation函数写成 `ouputs = F.sigmoid(outputs)`，后面仍使用outputs。零成本argmax受单调变换影响不大，但概率解释和加成本判定不同；移植时明确分开原始logits与sigmoid概率。不要把sigmoid后的值再喂入BCEWithLogits。
- `validation.py` 的test loader是 `drop_last=True`。项目全query评估应为False，避免丢掉最后不足一批的样本。
- 原 `main.py` 面向CIFAR-10和模拟专家，不能将允许接收labels的模拟专家接口原样当SAR推理接口；SAR专家预测必须在不知道query标签时生成。

这些是已读源码的具体事实，不是本次运行发现的报错；没有修改官方或项目代码。原文的有限样本校准、跨域泛化、类别重加权和受约束预算排序不能由一致性定理直接保证。

## 4. 一个可获得的SAR分类基线：SARATR-X

**选择官方分类分支的SARATR-X，而不是其检测器。** 官方README、`classification/MIM_linear.sh`、`classification/trainers/MIM_linear.py` 和配置均已读取，Hugging Face公开文件列表也可访问。[S1-S5] 未下载约1GB权重、未加载检查、未训练。

匹配理由：是SAR目标表征及少样本分类方法，公开预训练集包括C波段双极化OpenSARShip（表III为2.3-17.4m）；分类分支实际含船舶细分类评估。**这支持把它作为中低分辨率SAR的可适配候选，不证明它已经验证本项目八类、23港跨域任务或是本任务SOTA。** 原论文分类类别及划分与当前八类不同。[S1-S2]

可用入口及真实限制：

- 官方README写的是 `classification/finetune/MIM_linear.sh`，当前仓库实际可读入口是 `classification/MIM_linear.sh`，后者调用 `scripts/MIM_linear/main.sh`；不要盲用README旧路径。[S1,S3]
- `MIM_linear.py` 实际使用HiViT（512维、depths=[2,2,20]、8 heads）；`vit_b16`是配置文件名，不代表模型就是普通ViT-B/16。单通道输入在forward中复制为三通道。[S3]
- 默认源码加载本地 `checkpoint-200.pth`，冻结大部分骨干，但解冻 `fc_norm` 和含BN的head，所以严格说不是仅一个纯线性层。[S3]
- 公开权重含 `weight/186K_all/checkpoint-200.pth` 及400/600/800版本，也有 `weight/186K_notest/checkpoint-600.pth`。目录列表已核实；`notest`名称不等于已证明与本项目OSS2或自建港口无重叠。[S5]
- 官方分类配置是224x224，batch=25，AdamW lr=1e-3、weight_decay=1e-4、betas=(0.9,0.95)、30 epochs、cosine、1轮warmup。论文附录A表X相符。[S2,S4]
- 原预印本附录B另有600轮预训练、解冻最后6个block的识别比较。因而当前 `MIM_linear.py` 路径不能声称逐项复现原论文所有主结果；选择它时标为官方LP路径适配。[S2-S3]

建议最小协议：复用公开预训练权重，统一当前图像裁剪和单通道幅度预处理，采用官方分类头路径输出八类；把官方按类shots采样器替换为同五组64标签索引，禁止生成额外目标验证集。训练轮次用源验证预先确定。若输入为VV+VH，不能直接把双通道塞入官方三次拼接；最小版选择预先固定的单极化幅度，或另标双极化改版并单列输入信息差别。

**预训练暴露必须单列：** OpenSARShip在公开预训练集内已核实，但与本项目OpenSARShip 2.0是否有样本重合未核实，不能反向断言已经泄漏。可先作为“公开预训练、暴露待核查”的独立表项；严格无目标暴露的主表需进一步按已有样本/场景清单核对。不会因此恢复本项目已排除的FUSAR或OSS1实验。

它有可获得实现，不是现成可直接套八类的模型；新数据适配、checkpoint加载兼容及当前环境能否运行仍未验证。无需等待这条支线才启动元ridge第二批。

## 5. 阅读范围、版本与来源

网络情况：`web.run`检索和打开未返回可读内容，GitHub匿名API触发限流；随后直接HTTP获取官方页面/raw源码、在内存提取原始PDF文本。没有借助二手综述替代方法核读，也未声称完成穷尽检索。下面的提交ID仅用于标识已阅读代码版本，不生成或使用额外校验和。

- R2D2：`bertinetto/r2d2`，master，页面currentOid `fc0c13ec991bb9f84395cb12de57cb150ce76f8d`。
- OvA：`rajevv/OvA-L2D`，main，`0998898feba805183b8fb37a88e4405dd86ecf8c`。
- SARATR-X：`waterdisappear/SARATR-X`，main，`8f110555e32e5777034c5ab6729d23bcf9741936`。

### R2D2一手来源

- [R1 原论文PDF](https://arxiv.org/pdf/1805.08136)，§3.1-3.4、§4.1，PDF pp.4-6；[正式OpenReview](https://openreview.net/forum?id=HyxnZh0ct7)。
- [R2 官方ridge源码](https://github.com/bertinetto/r2d2/blob/fc0c13ec991bb9f84395cb12de57cb150ce76f8d/fewshots/models/r2d2.py)；[Adjust/Lambda层](https://github.com/bertinetto/r2d2/blob/fc0c13ec991bb9f84395cb12de57cb150ce76f8d/fewshots/models/adjust.py)。
- [R3 官方默认配置](https://github.com/bertinetto/r2d2/blob/fc0c13ec991bb9f84395cb12de57cb150ce76f8d/scripts/train/conf/fewshots.yaml)。
- [R4 官方实验脚本](https://github.com/bertinetto/r2d2/blob/fc0c13ec991bb9f84395cb12de57cb150ce76f8d/scripts/train/experiments.sh)；[官方README](https://github.com/bertinetto/r2d2)。

### OvA一手来源

- [O1 PMLR正式页面](https://proceedings.mlr.press/v162/verma22c.html)；[PDF](https://proceedings.mlr.press/v162/verma22c/verma22c.pdf)，§4 pp.4-5、§6 p.6、附录E p.18、附录F p.19。
- [O2 官方损失](https://github.com/rajevv/OvA-L2D/blob/0998898feba805183b8fb37a88e4405dd86ecf8c/losses/losses.py)。
- [O3 官方训练](https://github.com/rajevv/OvA-L2D/blob/0998898feba805183b8fb37a88e4405dd86ecf8c/main.py)。
- [O4 官方评估](https://github.com/rajevv/OvA-L2D/blob/0998898feba805183b8fb37a88e4405dd86ecf8c/validation.py)。

### SARATR-X一手来源

- [S1 官方仓库](https://github.com/waterdisappear/SARATR-X)。README给出正式TIP引用：2025，34:869-884，DOI `10.1109/TIP.2025.3531988`；[IEEE入口](https://ieeexplore.ieee.org/document/10856784)。本次未核读付费最终排版版全文。
- [S2 作者预印本PDF](https://arxiv.org/pdf/2405.09365)，§III、表III、附录A/B，重点PDF pp.6-7及15-17。所获PDF仍有预印本页眉；不将其措辞等同于最终出版版逐字内容。
- [S3 官方分类实现](https://github.com/waterdisappear/SARATR-X/blob/8f110555e32e5777034c5ab6729d23bcf9741936/classification/trainers/MIM_linear.py)；[入口脚本](https://github.com/waterdisappear/SARATR-X/blob/8f110555e32e5777034c5ab6729d23bcf9741936/classification/MIM_linear.sh)。
- [S4 官方分类配置](https://github.com/waterdisappear/SARATR-X/blob/8f110555e32e5777034c5ab6729d23bcf9741936/classification/configs/trainers/MIM_linear/vit_b16.yaml)。
- [S5 作者Hugging Face权重目录](https://huggingface.co/waterdisappear/SARATR-X/tree/main/weight)；[公开文件列表API](https://huggingface.co/api/models/waterdisappear/SARATR-X/tree/main?recursive=true)。

状态边界：公式、关键源码、配置及文件可获得性已核实；源缓存语义、数值运行、训练收敛、目标性能、权重实际加载及预训练样本交集未验证。本文不新增实验结果，不宣称SOTA，也不要求解冻主模型。

## 6. SARATR-X可执行性复查（2026-10-10补充）

**结论：当前不能直接补出这条公开SAR分类基线的成绩。** 本地已有官方源码、现有CUDA环境可以导入官方HiViT，但检查范围内没有找到SARATR-X权重，当前HF账户也未获该模型访问授权。因此这是具体的权重获取阻塞，不是方法只剩一个标题；取得一个官方预训练checkpoint后，存在不重做预训练的有限适配路径。目录可列出文件不等于文件可以直接下载，本节修正此前仅核实文件列表所能支持的可获得性表述。

### 6.1 本地资产与环境

| 项目 | 本次只读结果 | 对实施的影响 |
| --- | --- | --- |
| 官方仓库 | `E:/Hermes/paper_tracking/repos/SARATR-X`，git HEAD `e1eb57f2881596a587f60898cb20e7f7372eb9ca` | 不需要重新克隆；本地版本与前文在线版本不同 |
| 论文及历史核查 | 本地 `papers/saratr-x_2405.09365.pdf/.txt`、`2026-06-26_saratrx_learning_card.md` 已存在 | 历史核查也明确记载未取得权重；不能把此前GF模块smoke当作已跑分类模型 |
| 权重 | 在项目仓库、checkpoints、models、hf_home、cache_home及用户HF默认缓存未发现对应checkpoint/snapshot；广泛文件名搜索亦未命中SARATR-X权重 | 是“在这些位置未找到”，不声称穷尽其他磁盘或改名文件 |
| 可用解释器 | `E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe` | 此路径真实存在；环境是venv布局，不是根目录python.exe |
| 依赖 | torch 2.12.1+cu132、torchvision 0.27.1+cu132、timm 0.4.12、numpy 1.26.4；CUDA可用 | 与官方旧配方版本不同，不能直接宣称训练兼容 |
| 关键导入 | 用上述解释器、`-B`，成功导入官方 `classification/model/hivit.py` 的HiViT类及所需timm符号 | 未构建/加载训练权重，未做forward，不等于完整模型运行验证 |
| Dassl | 该环境未安装 | 直接官方train.py入口尚不可用；独立HiViT特征提取及分类头适配可避开该训练框架依赖 |
| GPU | RTX 5050 Laptop，8151 MiB总显存；核查时空闲7145 MiB | 可以设计小batch冻结特征提取；本次未实测吞吐或容量上限 |

默认 `D:/Program_files/anaconda3/python.exe` 的torch/timm联合导入实际触发OpenMP Error #15退出；没有用忽略重复OpenMP的环境变量掩盖它，也没有修改环境。已有项目venv的上述导入成功，因此无需为了该问题更改系统环境。

### 6.2 官方下载是否真的可用

本次只读取元数据、HEAD和带 `Range: bytes=0-15` 的请求；请求失败，没有下载大权重，也没有输出任何凭据。

| 官方文件 | API给出的大小 | 真实访问结果 |
| --- | --- | --- |
| `weight/186K_all/checkpoint-200.pth` | 1,044,098,136 bytes，约1.044 GB / 0.972 GiB | 匿名Range请求401，正文明确要求授权与登录 |
| `weight/186K_notest/checkpoint-600.pth` | 1,044,098,904 bytes，约1.044 GB / 0.972 GiB | 匿名HEAD/Range为401；使用当前已有HF凭据仍403、`GatedRepo`，明确不在授权名单 |

官方模型API当前为 `private=False, gated=auto, disabled=False`。这里的auto表示仓库门控类型，不表示匿名下载可用。没有替用户提交访问申请、接受条款或联系作者。

官方百度备选分享入口 HTTP 200，但返回提取码页面，没有得到实际文件列表或下载直链；README给出的提取码为 `7b3u`。因此百度路线只能记为“备选入口存在，实际下载未验证”，不能据此宣称当前限制已经解除。[S1,S5]

访问依据：[HF模型元数据](https://huggingface.co/api/models/waterdisappear/SARATR-X)、[notest文件列表](https://huggingface.co/api/models/waterdisappear/SARATR-X/tree/main/weight/186K_notest)、[notest权重入口](https://huggingface.co/waterdisappear/SARATR-X/resolve/main/weight/186K_notest/checkpoint-600.pth)、[all权重入口](https://huggingface.co/waterdisappear/SARATR-X/resolve/main/weight/186K_all/checkpoint-200.pth)、[官方百度入口](https://pan.baidu.com/s/1EuTo0WCbb69F_bGV6Iwd8A?pwd=7b3u)。

### 6.3 与OSS2/F0921的监督及预训练暴露

**必须分开“监督标签暴露”和“无标签图像/场景暴露”。** 本建议采用ImageNet SSL加SAR SSL的预训练checkpoint，不使用作者的下游检测/分类监督checkpoint；这不能自动排除SAR预训练图像与目标集重叠。[S1-S2]

| 对象 | 已知证据 | 可写结论/不能写的结论 |
| --- | --- | --- |
| OSS2监督标签 | 原文预训练使用未标注SAR图像；未发现其训练当前八类OSS2标签的证据 | 不能直接指控目标标签泄漏；也不能用“自监督”抹去图像暴露 |
| OSS2图像/场景 | 官方预训练含OpenSARShip；本地旧OSS2场景清单含2016年Sentinel-1 GRD产品。该旧shared3清单只能证明数据来源，不能替代当前八类样本索引 | 与同系列公开数据存在待核实重叠，**当前不能认定严格无暴露**；未拿到预训练样本清单，不能给出具体交集数 |
| `186K_notest`含义 | 原文脚注5称移除其下游任务测试集；该声明并未列出当前项目OSS2/F0921的冻结支持/查询索引 | 文件夹名和该脚注都不足以证明排除本项目目标集 |
| F0921 | 本地 `f0921_chips/index.csv` 首行是2021-05-26北美/夏威夷海域AIS芯片；scene_port记录含2021年纽约、休斯敦、洛杉矶等采集。官方14数据集清单未列自建F0921 | **未发现直接纳入F0921或其八类标签的证据**；但没有官方完整源场景清单，仍不能证明Sentinel-1原始场景完全不重合 |

F0921实际芯片资产也已存在：`E:/Hermes/paper_tracking/f0921_chips/F0921_VV.npy` 与 `F0921_VH.npy` 各19,350,374,528 bytes，伴随index.csv。仅检查文件元数据及索引前两行，未加载大数组。后续必须沿当前target_manifest回到这些原始行，并核对预处理；不能直接拿整个历史芯片池替代当前冻结查询集。

本地证据：`E:/Hermes/paper_tracking/2026-06-26_saratrx_learning_card.md`；`papers/saratr-x_2405.09365.txt` 的OpenSARShip说明及脚注5；`external_os2_fusar/EXTERNAL_OS2_FUSAR_PREREG_v1.md`；`projects/cross_port_knowledge_opensarship1/run_logs/opensarship2_rc_pckt_scene_strict_20260701/opensarship2_environment_manifest.csv`；`f0921_chips/index.csv`、`f0921_scene_port.json`。旧预注册仅用于追溯数据来源，未将旧3类协议当成本轮8类协议。

### 6.4 本轮的具体实施决定

**本次权限范围内：不能完成SARATR-X实验，暂不列为已完成公开基线。** 原因是权重不在本地、现有账号无下载授权，而本轮明确不做大下载、不训练。获取授权/从官方百度取得文件后再安排，不能为了凑表改用随机HiViT、ImageNet-only权重或检测checkpoint冒充SARATR-X分类结果。

拿到权重后的最小路径已经明确，不需要重新做SAR预训练：

1. 优先固定 `186K_notest/checkpoint-600.pth`，记录该选择及仍未知的OSS2暴露，不在目标结果间挑checkpoint。检查权重与本地HiViT分类架构对应关系后才称加载成功。
2. 从当前target_manifest及源验证清单读取同一批芯片，采用预先固定的单极化灰度幅度和224输入，复制三通道；每个样本提取一次，五组64标签复用特征。输入强度/幅度转换必须承接现有图像来源，不能仅靠Resize声称与官方预处理一致。
3. 若要贴近官方LP路径，保留冻结主体、可训练fc_norm和BN+八类分类头；缓存fc_norm之前的表示以支持这一步。只用现成ridge/logistic读出最终特征也是可运行的更小方案，但应叫 `SARATR-X frozen features + ridge/logistic`，不是官方LP完整复现。
4. 支持/查询与本轮完全一致，不另采每类shots，不借用目标query选轮次或预处理；超参数由源验证确定。无需安装整套检测依赖；若自行实现LP训练循环而不使用Dassl，记录框架移植。
5. OSS2在暴露未排除时放入单独“公开预训练、目标暴露未核实”比较；F0921可先作为较清楚的数据来源适配对象，但不得标“已证实无原始场景重叠”。23港任务同样不能因自建名称就自动排除预训练场景交集。

未给出预计完成时间：未实测官方权重推理吞吐，F0921查询量较大，不能从GPU名称推算本轮必然来得及。当前可确认的是源码和导入基础已经具备，下载授权、权重加载及特征提取尚未完成。
