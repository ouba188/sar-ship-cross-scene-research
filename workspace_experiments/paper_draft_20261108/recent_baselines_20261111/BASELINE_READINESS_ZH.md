# 近期强基线可落地性核查：SCRC-Net / CPINet

日期：2026-11-11。对象：当前OSS2四分类论文。结论基于本地原论文、已有实验/代码文本和在线primary来源；未运行模型、GPU训练或推理，未安装/下载依赖与权重，未读取或上传凭据。主代理M2两种合并替换×三种子训练按用户告知已启动，本任务未检查、操作或修改该任务。

## 1. 可执行结论

**先准备CPINet的论文复现及共享R50适配；SCRC-Net列第二候选，暂不排训练。两者目前均没有经核验可直接运行的官方完整实现。** 这不是性能排序，也不代表本轮已开始复现。

| 候选 | 有无官方实现 | 能否迁入当前同协议 | 最少还缺什么 |
| --- | --- | --- | --- |
| **CPINet，2024：优先** | 论文、共同作者出版物页及定向检索均未找到可归属该文的官方代码地址；不能断言作者没有代码 | 可按原文做独立复现；配对VV/VH、全监督类别标签与当前任务兼容。共享R50版本须标为适配，不是原版整网 | 多级特征/通道映射；MO-SE/FBC与四路集成细节；GradNorm及单路不可用处理。部分细节依赖论文[26]/[30]，仍需查清 |
| **SCRC-Net，2026：第二** | 官方仓库存在，但当前公开main递归树只有CITATION.cff、LICENSE、README.md；无模型/训练/评价文件 | 方法上可以适配，现阶段不足以承诺忠实整网复现。换ConvNeXt为共享R50、增加双极化融合均须披露 | 官方实现或等价实现说明；“VV+VH”到底怎样组织；动态核尺寸与卷积权重/梯度的对应关系 |

CPINet优先的理由：**双极化任务匹配更直接，主要算子与损失公式较完整，实施不确定性相对较少。** 但若要求“本周只接官方代码、不做独立复现”，这两个候选目前都不满足条件。当前表IX/XV结果继续留空。[P1–P5, L1–L3]

**重要排除：本地 `visual_reliable_baseline/e207_cpinet.py` 不是文献CPINet。** 它是2026-09-27自拟的共享R18门控融合/极化差值/排序损失模型，没有文献的IMDFF、MO-SE、FBC及四路GradNorm；旧代码、权重和成绩均不能充当He等人的基线。[L5]

## 2. 文献身份与方法底线

### SCRC-Net

**Yuhang Qi, Lu Wang, Chunhui Zhao, P. Takis Mathiopoulos, Tomoaki Ohtsuki, Fumiyuki Adachi.** *SCRC-Net: A structure-constrained and representation-consistent network for SAR ship classification*. **Pattern Recognition, 180, 114294, 2026**. DOI：`10.1016/j.patcog.2026.114294`。[P1, L1]

本地PDF首页：2026-06-16接收，**2026-06-22在线发表**；出版商卷期为2026年12月，后者在核查日2026-11-11之后，但论文已经在线发表。读了本地PDF第3–7页方法/设置和第12页极化讨论，并实际查看第4页图3渲染；复用10月23日已读消融记录。[L1–L2]

核心为HFDM（细节/结构特征重构）、SCFM（预测核高宽、共同旋转角、局部伸缩并动态采样）、CICS（尺度间/尺度内门控协作）。图3两条路径是**原图多尺度路径与重构特征层级路径，不是VV/VH两路**。SCFM不要求检测OBB；HFDM使用语义背景，不能概括成“全部背景都是噪声”。只装一个SCFM应叫模块对照，不能称完整SCRC-Net。

原设置：四阶段ConvNeXt卷积块，256×256、batch32、500 epochs、SGD momentum0.9/weight decay1e-4、LR0.0125/cosine、分类CE。OSS2原文选六类；不是当前四类或跨船只/过境划分。已读设置未要求相位、目标域标签、干净图监督或源参考库。[L1，第3–7页]

**阻塞点：** 第12页表11列VV、VH、VV+VH，正文又说每种单极化仅占总样本一半；不能据此确认逐船配对双流融合。动态奇数核的组批、参数对应和梯度实现亦未由代码核实。取得实现或作者说明前，不能把自选融合方式和近似采样写成原版。

### CPINet

**Jinglu He, Ruiting Sun, Yingying Kong, Wenlong Chang, Chenglu Sun, Gaige Chen, Yinghua Li, Zhe Meng, Fuping Wang.** *CPINet: Towards A Novel Cross-Polarimetric Interaction Network for Dual-Polarized SAR Ship Classification*. **Remote Sensing, 16(18), 3479, 2024**，发表于**2024-09-19**。DOI：`10.3390/rs16183479`。[P3]

读了MDPI官方HTML第3节/式(1)–(9)、第4.1–4.5节/表1–7及数据声明。PDF直接访问失败，未声称核对过PDF图中每一细节或官方实现。

原方法是配对VV/VH **IW-GRD图像的全监督分类**。采用SAR-DenseNet-v1 pseudo-Siamese骨干；原文明示共享的是四分支FC，不能据此称骨干共享。必须保留的主体：

- IMDFF多级融合；MO-SE用一阶GAP及二阶分组双线性统计生成跨极化门控。
- FBC低秩双线性编码、软阈值和空间max pooling；式(7)可直接计算，不需要每批另解外部LASSO。
- 中间融合、MO-SE融合、VV自融合、VH自融合四路CE与共享FC；GradNorm在共享FC上计算梯度、更新并归一化损失权重，最后集成四路预测。

原设置为OpenSARShip三/五类、64×64、逐图均值/标准差归一化；300 epochs、batch32、SGD momentum0.9、LR0.01，150/200/250轮降十倍；weight decay5e-4/dropout0.2。FBC为k2048、lambda0.001、rank1；GradNorm alpha0.12。原文随机划分及汇总方式不等于当前固定划分三种子；原分数不可回填当前四分类表。[P3]

## 3. 官方代码的核验结果

**SCRC-Net：** 地址由论文给出：`https://github.com/luwang-Lucia/SCRC-Net`。2026-11-11实际读取公开网页、raw README、未认证递归树API；返回`truncated=false`，仅三个根文件。README说训练/评价脚本仍在整理待发布；它展示的`train.py`、`eval.py`、配置和requirements命令不对应已发布文件。结论限于**当前公开main**，未穷尽历史/非main分支。旧“Coming soon”记录已用本次目录证据更新。[P2, P5, L2]

**CPINet：** MDPI数据声明只说明数据来源；作者Zhe Meng出版物页的该条只有PDF，没有代码入口。检索过`"CPINet" "SAR" "github"`、`"CPINet" site:github.com`、作者名加code及完整方法名加code，未找到可验证的官方仓库。**不填写猜测地址，也不采用同名本地模型。** 此结果是“未找到”，不是“证明从未开源”。[P3–P4]

## 4. 同协议复现必须固定什么

当前实际协议来自`staged_ablation_20261109`，不是直接照搬原论文划分。[L3–L4]

| 条件 | 可执行要求 |
| --- | --- |
| 类别与队列 | 四类顺序：散货、普通货、化学品/成品油轮、集装箱。训练719=fit478+adapt241，683艘船/13过境；验证228；评价378，369艘船/5过境。评价类数132/26/83/137。复用原ID、船只/平台绝对轨道隔离，selection194不并入训练 |
| 图像与监督 | 当前OSS2 IW-GRDH、native160原尺度裁剪/补零；同一VV/VH配对、预处理、footprint和available。仅使用719训练标签；验证选参，冻结后评价。不用原论文六类/五类Tanker名称替代当前标签映射 |
| 极化可用性 | 评价348双路、12仅VV、18仅VH，指当前处理后可用性，不是传感器原始缺极化。主评价保留378，预先定义候选的缺路门控/损失掩码；另报共同双路348。只支持完整配对的版本必须与所有对照同子集比较，不能拿348与旧378直接比 |
| R50与命名 | 共用当前ImageNet起点`resnet50-11ad3fa6.pth`。CPINet改骨干/共享方式，SCRC改骨干/极化组织，均写`adapted to shared R50 / our reimplementation`。原架构同划分对照可以另做，但不能同时宣称严格原版和完全同R50 |
| 优化机会 | 当前Full/Plain/SE仅微调指定四个卷积，非全骨干微调。不能机械套Full五阶段冻结到候选。候选若全量解冻，应披露并补同机会Plain，或明确只作不同训练配方的系统比较；现有成绩不能证明等预算结构优势 |
| 选择与预算 | 沿用17/29/43；每方法独立在验证集获得可比调参机会，记录训练范围、步数和曝光量。若沿用干预阶段，各自生成教师选样，不能借Full标记。不能依评价378挑设置；该队列已历史使用，不称新盲测 |
| 额外信息 | Full的既有OBB与32个源参考须披露，参考来自训练集，不增标签权限；候选无须硬接M3。成本须包含各自真实推理路径，不用已知Full时延代填 |
| 评价 | OA、四类BA、macro-F1、逐类及配对预测；实际完成后看VV/VH持续对错、救回、误伤，必要时同时看Full合法同类/竞争类支持chip |

**CPINet最小实施决策：** 定义五级特征到R50的映射和通道投影、MO-SE分组/压缩、FBC嵌入维度及归一化、四路集成、GradNorm更新和缺路策略。当前`early_x`仅为layer2内部128×20×20特征，不能用它或最终区域特征冒充原方法全部多级输入。部分细节仍需查[26]/[30]或作者代码；不得用随意默认值冒充已核实设置。[P3, L4]

**后续最小验证：** 实施后做CPU小批前向/反向，检查跨极化对应、四路损失、共享FC梯度、GradNorm权重更新及单路可用输入；失败就修接口。通过后才安排同划分训练。本轮仅给建议，未执行这些检查或测量显存/速度。

## 5. 可复用的历史证据

| 材料 | 可复用内容与边界 |
| --- | --- |
| staged协议、cohort_counts、selected及summary [L3] | 原划分、17/29/43、验证选择和旧对照。Full/Plain/SE的OA均值61.023/55.908/54.233、BA55.630/52.176/53.497可作同队列系统比较锚点；不是本次新成绩 |
| Full三种子evaluation.npz [L3] | 已只读确认文件存在；新预测可按ID配对，无需重跑旧模型。本轮未加载分数重算指标 |
| manifest/groups及records()/arrays() [L4] | 原ID/类映射、分组与缓存入口。只能复用权重/预处理完全相同且确实冻结的前缀输出；训练穿过该段或需要缺失层级时不能用旧缓存 |
| CASE_NOTES及原图索引 [L3] | `002309`两极化空间响应差异、`000261`框外强响应支持比较动机；`000133/000132`保留误伤/持续错误；`000283/000230`说明缺路处理实际需要。旧案例不是新基线有效的证据 |
| 10月23日SCRC比较 [L2] | 可复用SCFM与OBB读出的技术区别和消融出处；旧代码发布状态须更新，旧局部读出成绩不能当整网基线 |
| 本地e207“CPINet” [L5] | 只保留为历史自拟方案，不能复用为本篇基线实现/成绩 |

**主任务下一步只需作一个实施选择：是否接受“CPINet共享R50独立复现”作为适配基线。** 接受后补齐上述具体实现决策即可推进；若坚持完整官方实现，则两候选均先不排GPU训练。SCRC-Net等待代码/关键说明，不做近似整网来占结果行。

## 6. 来源与唯一交付物

在线来源均于2026-11-11核查：

- **[P1] 出版商：** `https://www.sciencedirect.com/science/article/pii/S0031320326012598`；DOI `https://doi.org/10.1016/j.patcog.2026.114294`。
- **[P2] 官方仓库及raw README：** `https://github.com/luwang-Lucia/SCRC-Net`；`https://raw.githubusercontent.com/luwang-Lucia/SCRC-Net/main/README.md`。
- **[P3] CPINet官方全文：** `https://www.mdpi.com/2072-4292/16/18/3479`；DOI `https://doi.org/10.3390/rs16183479`。方法第3节、实验第4.1–4.5节、表1–7；作者以CC BY 4.0发布。
- **[P4] 作者出版物页：** `https://www.zhemeng.site/publications/`，2024年第17项。
- **[P5] 公开目录API：** `https://api.github.com/repos/luwang-Lucia/SCRC-Net/git/trees/main?recursive=1`，无认证请求，完整main树只有3文件。

本地证据：

- **[L1]** `E:/Docms/无监督域适应/translated/0917/related_papers/SCRC-Net.pdf`。文本与图3渲染均在内存/终端读取，未保存派生文件。
- **[L2]** `E:/临时会话/scene_aware_classification_20261018/eight_class_evidence_20261023/obb_dynamic_readout/SCRC_COMPARISON_ZH.md`。
- **[L3]** `E:/临时会话/paper_draft_20261108/staged_ablation_20261109/` 下的`PROTOCOL_ZH.md`、`protocol.json`、`ROUND_FINDINGS_ZH.md`、`cohort_counts.csv`、`selected.json`、`CASE_NOTES_ZH.md`及其预测/案例索引；Full预测位于`full_s17_lr1`、`full_s29_lr1`、`full_s43_lr1`的`evaluation.npz`。
- **[L4]** `E:/临时会话/paper_draft_20261108/staged_ablation_20261109/staged.py`；`E:/临时会话/paper_draft_20261108/experiment_completion_20261109/common_ablation.py`、`raw_current.py`；`E:/临时会话/m1_coverage_20261101/experiment.py`及同目录`manifest.json`、`groups.json`。只读代码文本并确认清单存在，未import模型或全量审计。
- **[L5]** `E:/临时会话/visual_reliable_baseline/CPINET_PREREG_v1.md`、`E:/临时会话/visual_reliable_baseline/e207_cpinet.py`。
- **占位约束：** `E:/临时会话/paper_draft_20261108/EXPERIMENT_TABLE_PLACEHOLDERS_20261110.md`。

唯一写入：`E:/临时会话/paper_draft_20261108/recent_baselines_20261111/BASELINE_READINESS_ZH.md`。未修改正式代码、稿件或主代理M2目录，未另开chat，未提交/推送GitHub；本任务仅写指定报告的约束优先于一般同步约定。

## 主代理复核与实施决定

主代理随后复核了SCRC公开根目录及CPINet官方方法正文。近期基线仍未进入训练；优先准备CPINet独立复现，但不要求用户再次批准常规实现。骨干适配和输入处理必须在实验命名中说明，不能把共享R50版本标成原论文整网。

为确认是否有更快的官方实现入口，另快速检索了DPIG-Net（Remote Sensing 2023, doi:10.3390/rs15082138）及A Lightweight and Real-Time Dual-Polarization Fusion Framework（Remote Sensing 2026, doi:10.3390/rs18081129）。二者官方HTML未出现GitHub链接；这仅是入口检查，不是完整复现审查，也不表示代码一定不存在。未扩大候选训练矩阵。

M2实验期间不并发启动另一GPU训练。近期基线缺口明确保留，不能将本次文献核查写成已经完成对比。
