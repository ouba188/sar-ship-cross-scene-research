# 同输入、同容量的类别响应attention替换计划

2026-10-23。状态：**仅设计及只读缓存/接口检查完成，未提图、未训练、未评价新模型。** 本目录独立，不改父文件、旧结果或他人目录，不Git。后续实现需主流程安排，不将本计划视为已执行结果。

## 1. 唯一建议及所回答的问题

建议只增加一个候选：**同一源尺度、双向类别响应attention摘要，替换additive的σ_e/κ_e摘要**，其余均值支、强B、两支结构和训练协议保持不变。简称`two_sided_attention`。若获准执行，3fold×seed17/29/43共9个新头；原full additive9头直接复用，不再增加单向/多温度/多宽度臂。

问题是：在相同源监督类别响应、原尺度图像、极化/footprint/C4及相同266参数读出条件下，当前两侧波动矩摘要是否比一个合理的softmax加权响应摘要更有任务效用？这不是挑战整个CSRA系统，也不是官方CSRA复现。

选择双向而不是随意取两个温度，原因是原σ_e/κ_e在每视图描述总波动与两侧能量不平衡。只用一个正向attention值并复制两槽虽然参数数目相同，却少一个可变统计维度；引入两个任意温度又增加未经约束的尺度选择。这里对r与-r用**同一温度尺度**，得到正、负尾部的两个可变摘要，与原两统计的维度及符号结构相配。反向是本对照的明确改编，不冒充原CSRA自带的第二头。

## 2. CSRA原文实际阅读与取舍

复读`module_novelty_review/READING_RECORD_ZH.md`后，本轮又请求官方CVF PDF，web返回空内容；随后用requests+pypdf在内存读取arXiv:2108.02456的10页PDF，实际读第3–5页，含完整第3.1–3.4节及第4.1节。没有下载持久论文文件，没有扩大文献搜索。论文为Zhu与Wu，*Residual Attention: A Simple but Effective Method for Multi-Label Recognition*，ICCV 2021。

- 第3.2节式(2)–(6)：m_c投影位置特征，softmax(T·m_c^Tx_u)得到类别空间权重；类别加权特征与GAP按g+λa_c相加，再由m_c分类。
- 第3.3节式(8)：等价于mean(z_c)+λΣsoftmax(Tz_c)z_c。T是乘数，越大越尖锐；T趋于无穷得到max。
- 第3.4节：单头H=1固定T=1；多头采用各自分类器和预定T序列，其中最后一个是max。原文还归一化分类权重为单位范数。不能把该多头结构偷换成固定ridge W后任选两个T填槽。
- 原训练版使用端到端BCE/SGD，也有无需训练的测试时mean+max插件。本轮不照搬其数据、任务、训练或λ；保留的是“由类别响应产生空间softmax权重，再读出同一响应”的核心操作。

本轮原文入口：`https://arxiv.org/pdf/2108.02456`。会议及既有详细阅读记录见上级`module_novelty_review/READING_RECORD_ZH.md`。

## 3. 唯一冻结的聚合定义

### 相同响应，改变聚合

严格复用`calibrated_signed_transfer/extract_moments.py::weighted_moments`的输入与中心化：各pol/C4用同W投影冻结layer3图，按归一化footprint权重w去空间均值，再去类公共项并校正加权空间均值，得到r_c(u)。不添加reader截距、不改变W范数、不使用真实类别选图。

正向softmax对单纯减去该类空间常量μ不变；但去除逐位置的类公共响应一般会改变注意力。因此本轮明确使用与full相同的r，而不是把原始z与r的差异混入聚合对照。等价方向为U_c=W_c-mean_class(W)，由同一已冻结W确定，没有新拟合。由此，结论限于**相同中心化类别响应上的attention与矩摘要比较**，不能直接写成原始CSRA的优劣。

### 每fold、每pol唯一源尺度

定义τ_{f,p}为该fold源OOF训练对象、有效pol及全部8类的**缓存C4平均σ_e的中位数**。不读validation/test来定尺度，不按类分别调温、不按单图σ_e归一化、不用验证J选温度。这样保留逐图离散度大小，而不是先将其全部变成单位方差。

有效温度乘数为T_{f,p}=1/τ_{f,p}；在无量纲r/τ上固定T=1。它只控制softmax权重，聚合的value仍是原响应单位r。不能把后续descriptor scaler当作温度修正：输入scaler发生在attention之后，无法逆转此前softmax的饱和。也不能随意将ridge W单位范数化，否则改变了与full相同的类别方向尺度及类间相对信息。

本次仅从合法源OOF缓存实际算得：

|fold|train|validation|τ_VV|τ_VH|
|---|---:|---:|---:|---:|
|0|4392|310|0.6916389141|0.6977549929|
|1|4419|286|0.7235525871|0.6577786590|
|2|4418|319|0.7353725509|0.6793553104|

均为正，无零尺度阻塞。使用的是源OOF全部训练tokens的总体预处理，和当前source scaler同层级；不是新增全流水线OOF保证。内层W与outer W尺度可能不同，这是应记录的源/验证分布差异，不据其结果另调温度。即便统一缩放W时τ可相应缩放，实际不同读出器的变化也未必只是标量，不能宣称完全尺度不变。

### 双向attention及两槽

对每pol、每C4、每类定义

\[
\alpha_\pm(u)=\frac{w(u)\exp(\pm r_c(u)/\tau_{f,p})}
 {\sum_vw(v)\exp(\pm r_c(v)/\tau_{f,p})},\qquad
q_\pm=\sum_u\alpha_\pm(u)r_c(u).
\]

实现用`softmax(log(w) ± r/τ)`，w=0位置严格屏蔽；先减合法位置最大logit保证数值稳定，不人为截断r或改变温度。fractional footprint必须进入归一化，不能只将非零格等权。理想算术下因E_w[r]=0，有q_+≥0、q_-≤0。

唯一两槽固定为

\[
h=(q_+-q_-)/2,\qquad b=(q_++q_-)/2.
\]

h描述两尾的响应跨度，b描述两尾的不对称；此二元变换可逆，不额外增加信息。反转r后h不变、b变号，且|b|≤h。它们不是σ_e/κ_e的估计量或物理能量。对每个C4视图先算h,b，再平均；VV/VH分别保留，不先把图逆旋转平均或将极化相加。

这与双温度寻优不同：两个方向共用唯一|T|，分别读同一分布的两侧，没有第二个尖锐度超参数。它也不是未经改编的单向CSRA，报告应始终称“同容量双向CSRA式摘要替换”。如果它不如full，只能支持本固定改编下的差异，不能推导所有attention弱于矩。

## 4. 同容量、同协议的接入

将当前σ_e槽替换为h、κ_e槽替换为b，并按原方式生成各自类中心化值。共同保持：

- mean支的7维输入Bcenter、两pol的μ及类中心化μ、valid不变；源OOF/outer选择不变。
- distribution/attention支仍11→12 GELU→1；mean支仍7→12 GELU→1；共266参数，两支类共享并一起训练。
- 原full与新attention支的权重张量形状、初始随机种子与初始化顺序完全一致，末层零，0步严格等于B。不是把训练后的full参数作为attention warm start，也不冻结已训练mean支。
- 新descriptor按本fold源OOF训练tokens重新拟合scaler；不能把σ_e/κ_e的scaler套给h,b。共同B/μ/valid槽的scaler应与full相同，改变的只应是统计槽。
- 残差仍为center_class(f_mean+f_attention)，最终s=B+残差。没有新增λ、门控、温度学习或独立类别bias头。
- 同seed17/29/43、200步全批次AdamW、lr=.003、wd=.01、jointCE；每5步按source J选点，含0步、严格改善才更新、平局最早。源身份及采集划分不变。

两者在原始输入、辅助W、摘要数目、维度和网络容量上匹配；**统计信息本身不同正是研究变量**。σ_e/κ_e是均方波动及符号二次矩，attention是指数倾斜后的响应期望，可受更多分布细节及尾部影响；两者均不编码空间邻接，对响应与权重共同置换不变。相同参数并不意味着相同函数类或相同优化难度，温度仍是本对照的建模选择，需保留这个解释范围。

完整additive直接复用原冻结9头与源验证输出。两臂先比较同源协议；若后续明确授权目标评价，全部9个新checkpoint按源选定、冻结后一次评价，不能按test改温度/选点。T校准头并非二级OOF这一原有边界不变。

## 5. 缓存与提取接口：已检查的事实

已只读`extract_moments.py`、`run_polarization.py::load_pol`、`run_joint.py::inputs/load_features/Head`以及实际NPZ和extraction.json：

- `signed_features.npz`为108319407字节，含15个reader、15251个ID；`moments_pol`形状(15,15251,2,8,2)，`mean_pol`为(15,15251,2,8)，float64；valid为(15251,2)。完整分布/局部图没有保存在其中。
- `validation_maps.npz`仅四视图各有(15,2,8,100)的r及对应权重，是首批两个极化输入校验片段，不能覆盖源数据。
- 有限的μ/σ_e/κ_e一般不能恢复Σsoftmax(r/τ)r，故必须重新执行冻结R50前向，不能从已有矩拟造attention值。
- 所有15份原reader W/intercept/fit/held已存baseline/models。源train从各inner-held装配，每训练对象覆盖一次；validation用对应outer。**不重训reader、A、T，不重新提检测或邻船。**
- 三fold源train/validation并集实际8750对象。源阶段只需要这些原图；每个对象的F一次计算，可同时供其合法fold/reader使用。一个对象在某fold源、另一fold为目标时，只计算前者合法源角色，不把后者目标预测带入本轮。

后续独立提取函数应复用`crop_pad`、`masks_for`、原coverage/disabled_alignment、valid、W reshape及`weighted_moments`的r定义，但**不能直接执行父extract_moments.main**，它会写父目录并全量提取。同理不能直接运行父run_joint.main或修改其HERE/global后运行，因为其流程含旧输出及目标路径。

原输入是160 crop/pad，而非本轮另定160缩放；保持既有实际裁补来源、C4、ImageNet归一化、冻结R50层和fp32前向，响应及聚合沿用float64。batch以原16对象分块为上界，仅保留所需源输入。无效pol继续置零，不让padding成为attention候选。

可执行的最小新缓存契约：`reader_names/instance_ids/valid/attention_pol`，attention_pol按原(15,N,2,8,2)存C4平均h,b；另以`computed[15,N]`明确已提取的合法源角色，禁止把未计算零槽当有效值。缓存只含摘要和少量验证map，不保留全F。组装时复用原held映射；不需要为配合父入口复制或覆盖原moments缓存。

## 6. 成本、检查与判读

原全15251对象、29034有效pol图、15reader提取记录为219.39秒，峰值GPU已分配约391.7MB；这是旧运行实测，不是本方案耗时承诺。8750源对象只前向一次，主要成本仍是R50×C4，另有双向softmax及投影；预计同类分钟级工作，实施后实际记录wall time/峰值显存，不能将小MLP训练成本当部署成本。正式推理仍仅需对应outer W，不是15reader都部署。

执行获准后只做以下直接相关检查，不扩审计：

1. 在实际源样本复算μ/σ_e/κ_e与原缓存，核对GAP及mean+intercept读出一致。按旧提取已有1e-4边界报告实际误差和top1；若发生不符，先查预处理/映射，不能放宽阈值掩盖。首批保存map也用于操作级核对。
2. 合法角色映射、held覆盖、缺pol零值及source-only τ/scaler；这能发现误用outer训练内响应或未计算槽。
3. attention概率和为1、零footprint质量为0、常量响应中心化后h=b=0；反转r时h不变/b变号；joint位置置换不变。小型已知数组另核对权重公式，不能只用同一实现自证。
4. 记录源train/validation的r/τ范围、attention相对footprint的集中程度及h/b范围，用于判断是否几乎均匀或近max；**只诊断，不据结果改T或增加温度臂**。单一源尺度对稀有类/长尾的适配有限，这是本轮允许的负结果解释。
5. 同初始化、0步=B、全9保存checkpoint重放及选点J一致；full缓存源分数作为固定参照。以后若评价test，训练入口不得预先调用父代码的test检查/评估。

预期交付为source预测、曲线、选点、paired OA/BA/NLL/Brier、逐类及pass/有效pol分层、固定seed17四结局原VV/VH及读出响应；无翻转就如实记录。原图是核查实际作用，不由attention图强行命名物理散射。先验不要求任一臂处处赢，也不因单个显著性结论否决方向。

若attention相当或更好，应收窄“σ_e/κ_e特殊性”，保留类别方向分布读出的一般用途；若full更好，则说明在该固定源尺度、双向聚合和同容量协议下二次矩摘要有优势，仍不是所有CSRA/attention的普遍否定。源验证参与选点及多轮开发，不是独立确认。

## 7. 本次收口

接口与源尺度均可用，唯一实际新增成本是重提合法源局部响应；未发现需要改父流程的阻塞。已明确采用上述双向、同|T|、同266参数的一个替换，不保留温度搜索或额外结构备选。本次只写PLAN；没有新GPU前向、训练、目标评价或Git操作。
