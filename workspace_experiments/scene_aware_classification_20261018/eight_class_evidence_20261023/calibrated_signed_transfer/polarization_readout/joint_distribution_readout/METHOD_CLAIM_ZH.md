# 方法与贡献定位：校准分离的类别响应均值与分布读出

## 后续解释补充（2026-10-23）

本文主体仍定义sigma/kappa版本，不将其公式或归因移作attention实现。后续已完成同266头的双侧softmax替换、18个单侧限制头、本图摘要匹配替换和attention自身4例响应归因/16次干预，见 `matched_attention_readout/EXPLANATION_CLOSEOUT_ZH.md`。双侧相对B净45/34/85但相对本文full净-1/-11/+10，作为有用替代保留，不替换/叠加原候选。双侧相对low平均OA相同、BA+.3983pp；相对high平均OA+.1827、BA+.2611pp。模型层面的可解释净效用足以保留，不要求其完全对应物理散射机制。

本文full版本自己的4例固定模型响应归因也已完成，见 `fixed_model_spatial_attribution/FINDINGS_ZH.md`。4919的真实分布相对零响应使分差-.06699到+.19431，VH路径贡献为主；7310零响应时已误判，真实分布进一步加重。该闭合归因不是所有增益的物理因果证明。下文较早的“尚缺κ独立作用/正式比较”应结合后续组成对照与匹配attention结果阅读；仍未证明普遍跨数据集优势。

2026-10-23。本文只梳理当前已实现的24港开发队列additive方法及证据，不新增模型、实验或文献搜索。复用实际核心方法阅读记录，不将本轮复读笔记写成重新精读原论文。sigma-only/kappa-only的18头源端训练与评价、同18个已选checkpoint的固定test推理均已完成，完整additive 9头及mean-only 9头直接复用，没有新增fit或改选点。test是主流程基于用户持续推进研究的总体授权，在看到源结果后追加的组成验证，不是用户当场逐项明确追加授权，也不是盲测。

## 1. 结论先行

**可发展的贡献核心是：以源监督辅助类别方向作为统一的响应坐标，将整图均值与均值抵消后仍保留的正负波动能量分开读出，在冻结的强校准识别器之上学习可核对的类别竞争增量，并通过同图像均值参照、分支删除及条件化信息替换辨别其用途。** 贡献不是发明方差、带符号矩、双极化融合、MLP或残差相加，而是对一个有实际分类取舍的问题作具体且可验证的方法组织。

当前最强实证是完整additive相对强B的三seed OA/BA均正；分布分支具有不同于均值分支的类别作用。更窄的“给定B与均值后，本图分布仍有样本特异效用”已有小幅同方向证据，但不足以解释全部系统增益。组成消融中full相对sigma-only源端三seed OA/BA均高，test则OA三seed更高、BA两高一略低；full相对kappa-only的BA在源/test六组均高，但OA在源端三seed较低、test两高一低。这支持保留完整输入及其类别均衡取舍，**不能据此宣称κ_e绝对必要或两项联合普遍最优。** 数学可辨识性、实际模型使用与分类净收益是三层不同命题。

本文统一将局部类别响应离散度写为 **σ_e**，带符号归一二次矩写为 **κ_e**。代码中的`sigma/kappa`对应这两个量。σ_e不是SAR后向散射系数σ⁰（sigma0），κ_e也不是物理散射极性或标准三阶偏度；其量纲来自学习到的类别响应，不能标成RCS或标定散射功率。

## 2. 真实计算对象及能量解释

### 2.1 从像素到类别响应

每个有效极化p及C4视图t，按既有native PNG来源作160 crop/pad，冻结ImageNet R50 layer3，得到空间特征F_{p,t}(u)。真实图像footprint在特征网格上提供非负面积权重；归一化后记为w(u)，满足Σ_u w(u)=1。它不是船体语义mask，也不会消除卷积感受野混入的框外或padding信息。

仅源数据拟合的辅助ridge给出8类方向W_{p,c}，局部响应为

\[
z_c(u)=W_{p,c}^{\mathsf T}F_{p,t}(u),\qquad
\mu_c=\sum_u w(u)z_c(u).
\]

这里不向局部图分摊reader截距，也不把W说成非线性旧A+D的精确分解。源训练对象使用对应inner-held W，源验证和目标使用outer W；不按对象真实类别选方向，八类同时计算。

先去每类的加权空间均值，再去各位置的类公共响应，并按实现再次去除数值上的空间均值：

\[
v_c(u)=z_c(u)-\mu_c,\quad
r_c(u)=v_c(u)-\frac1{8}\sum_jv_j(u),\quad
r_c\leftarrow r_c-\sum_u w(u)r_c(u).
\]

精确算术下最后一次中心化为冗余校正。等价地，令U_c=W_c-mean_j(W_j)，则r_c(u)=U_c^T(F(u)-E_wF)。所以“正/负”指**该类相对类平均方向、并相对自身空间均值的偏离**。它不是“真类支持/错误类反对”的标签；错误类别也可有正响应，真类也可有负响应。当前onehot+intercept ridge的类公共W可已接近零，因此不能把去类公共项独立宣称为主要有效模块。

### 2.2 σ_e、κ_e恰好保留哪些能量

以下恒等式针对**一个极化、一个C4视图、一个类别**。定义非负的两侧响应能量：

\[
E_+=\sum_u w(u)\max(r_c(u),0)^2,\qquad
E_-=\sum_u w(u)\min(r_c(u),0)^2.
\]

若σ_e>0，理想定义为

\[
\sigma_e=\sqrt{E_w[r_c^2]},\qquad
\kappa_e=\frac{E_w[r_c|r_c|]}{\sigma_e}.
\]

因而

\[
\boxed{\sigma_e^2=E_++E_-},\qquad
\boxed{\kappa_e\sigma_e=E_+-E_-},\qquad
E_\pm=\frac{\sigma_e^2\pm\kappa_e\sigma_e}{2}.
\]

σ_e给出总波动能量的平方根，κ_e补充正负两侧平方能量的不平衡；不是正响应位置数量，也不是通常的E[r³]/σ³。由非负性有|κ_e|≤σ_e。σ_e=0时两侧能量均零，定义κ_e=0。实际实现分母为max(σ_e,1e-12)：上述乘积恒等式在σ_e≥1e-12时成立；极小非零值受保护阈值影响，不把数值保护当新的信息。

**这只是学习响应的能量分解。** 两侧均为非负平方量，负侧不意味着负物理功率；符号不能被解读为二面角、尾迹、相干相位或散射机制。对正r进行放大也不保证分类变好。

### 2.3 C4聚合的关键边界

实现对四视图各自计算上述统计，再缓存

\[
\bar\mu=\tfrac14\sum_t\mu_t,\quad
\bar\sigma_e=\tfrac14\sum_t\sigma_{e,t},\quad
\bar\kappa_e=\tfrac14\sum_t\kappa_{e,t}.
\]

不是先对齐/平均响应图再求矩。一般有

\[
\overline{E_++E_-}=\overline{\sigma_e^2}\ne\bar\sigma_e^2,
\qquad
\overline{E_+-E_-}=\overline{\kappa_e\sigma_e}\ne\bar\kappa_e\bar\sigma_e.
\]

因此不能从当前缓存的平均σ_e/κ_e精确反推平均E_+/E_-。当前网络读取的是**逐视图能量摘要的平均表征**，不是显式两侧能量输入；本报告的等式用于说明每视图统计意义，不将其偷换为已实现的能量网络。

### 2.4 一个可辨识性例子，不是SAR实验

三个等权位置取r=(-1,-1,2)/√2，或取其相反数。两者均值均为0，σ_e均为1，但前者(E_+,E_-)=(2/3,1/3)、κ_e=1/3，后者交换两侧能量、κ_e=-1/3。若要求同时满足逐位置类中心化，可令另一类响应为-r，其余六类为0。

此例证明相同均值/方差不决定符号能量不平衡，**不证明这些差别在SAR数据里与标签相关或能泛化**。相反，joint permutation同时置换响应及其权重时，这些统计均不变；当前方法不保留邻接、船体部位或空间排列。

## 3. GAP丢失了什么，完整additive实际如何用

在固定特征及W下，μ_c=W_c^TE_wF。GAP只传递一阶平均，围绕均值的正负响应可相互抵消；相同μ可对应不同总波动能量，相同μ/σ_e又可对应不同κ_e。若C_w是特征加权协方差，则σ_e²=U_c^TC_wU_c：这是类别方向上的二阶投影，不是新协方差原理；只取这些方向也未保留所有类对同位置交叉项。

这并不意味着旧B“完全不知道局部信息”：编码器特征本已含局部计算，旧A+D还含其自身图像/邻域路径。正确问题是辅助图像读出的这些摘要，是否在实际B之外可用，而非由GAP不可逆性直接推出模型收益。

当前B=T(oldA+D)冻结，辅助W、图像编码器也冻结。对每类共同读取以下原值与类中心化值，按仅源OOF训练tokens拟合的scaler标准化：

- mean支：center(B)_c；VV/VH各μ_c及center(μ)_c；两个valid，共7维，7→12 GELU→1，109参数。
- distribution支：center(B)_c；VV/VH各σ_e、center(σ_e)、κ_e、center(κ_e)；两个valid，共11维，11→12 GELU→1，157参数。

两支类共享、零末层初始化、一起训练，共266参数。无效极化统计置零并带valid。设两支输出为f_μ,c及f_D,c，则

\[
s_c=B_c+\operatorname{center}_{class}(f_{\mu,c}+f_{D,c}).
\]

不是相加此前两个独立训练的209参数头；没有额外κ载体、tanh cap或逐位置attention。两支均可使用B，因此分支名称不保证纯信息分解，σ_e/κ_e全零也不保证f_D为零。固定类别分差可以分解成B、mean支和distribution支三项，但非线性f_D不能直接按E_+/E_-逐位置精确归因。

训练200步，jointCE为ordinary CE与balanced CE各半；源验证J=(OA+BA)/2，每5步检查、包含0步、平局保留早期。辅助W及基础预测的OOF不等于整个T及增量流水线的OOF；T不是二级OOF，源监督重复使用这一边界必须保留。

## 4. Claim与已完成证据对应

24港是开发项目范围；本轮test实际为4744观测、2973 MMSI、85 pass、18港、8类。源验证915折暴露对应816唯一对象。下面17/29/43均是同队列训练seed，不是三个独立数据集；具体模型按源验证选点，但方案冻结参考过反复开发的test，不冒充盲测确认。[D1–D4]

|拟保留的claim|已完成对照/结果|不能越过的边界|
|---|---|---|
|完整分路读出在当前强B上有实际效用|additive相对B净+46/+45/+75，OA+.970/+.949/+1.581pp，BA+1.009/+.413/+1.127pp；源验证OA/BA亦均正。[D1]|非每港每类全胜；pass BA区间seed29跨0，保留pass及MMSI两种区间，不择优。|
|新增辅助图像证据不能只与弱校准比较|additive相对extra-score三seed OA/BA/NLL/Brier点估计均更好；同辅助W的mean-only是强参照。[D1,D2]|不同输入/容量不构成排除所有校准器的证明；两支仍读B。|
|均值与分布对应不同类别取舍，不能以一个代替所有|mean-only相对B净+77/+50/+84；additive相对mean-only OA低.654/.105/.190pp，BA高1.207/.731/.724pp，mean-only的NLL/Brier也更好。[D1,D3]|不称“分布全面优于GAP”或“必须用分布才能提高OA”。|
|分路相加是当前可用组织，不需要以联合隐藏层交互立论|相同信息的joint273对照中，additive266 test三seed OA/BA均高；source排序并非一致。[D1]|非同参数化，不证明非线性交互普遍无用，亦不把相加/GELU作为原创。|
|当前固定模型的distribution支承担了可复核的不同决策作用|完整相对固定只留mean支，test BA+1.173/.586/.759pp、OA净-3/+2/+9；source BA也均正。local加回对crude/offshore部分保护或救回，同时伤chemical。[D1]|删除不是重新训练；包含B依赖及共同优化，不能当纯局部分布因果效应。完整模型对offshore相对B仍净-7/-15/-3。|
|本图分布在近似给定B与μ后仍有小幅条件性效用|冻结additive，仅替换σ_e/κ_e：B+μ匹配时，本图相对借用source净+12/+11/+9，test+7/+6/+7；test OA+.148/.126/.148pp，BA+.297/.377/.163pp。[D4]|两种块bootstrap区间均跨0；近似匹配非真实条件采样，有分布偏移；这不是新部署模型，也不足以解释全部46/45/75例收益。|
|κ_e含σ_e不含的符号能量不平衡，完整输入在固定预算下保留较稳的类别均衡取舍|第2节恒等式；源18头中full相对sigma-only三seed OA/BA均高；固定test中full相对sigma-only净+20/+26/+31，BA+.516/−.018/+.403pp，相对kappa-only净−18/+7/+5，BA+.947/+.213/+.392pp。[D6]|两新臂相对B仍有OA/BA正增量；重新训练及各自选步包含优化因素，不能声称κ_e独立必要、充分或独占收益；追加历史test非盲测。|
|可迁移性有明确边界而不是普遍优势|OSS2同方法复验additive净+2/+1/+1，BA-.496/+.014/+.462pp，NLL/Brier均改善；local净-4/-8/-18。[D5]|OSS2基线B=T(native224 layer3 GAP ridge)，不同于24港旧AD；不是单因素数据替换或同权重直接迁移，不能抹去自建队列收益，也不能声称普遍稳定。|

原图证据亦有对应边界。既有报告已实际查看8面板、5唯一对象，本次复用记录而非声称重新查看：4919固定最终竞争类的分差为B -0.13193、mean +0.12557、distribution +0.20066，后者帮助翻转；7310两支同向误伤；3494分布支降低分差但仍CC；3504分布支抵消部分均值负作用仍WW。它们证明固定模型的具体作用位置，不证明亮体/纹理属于哪个散射机制，更不能把旧226乘法头的空间贡献图挪作当前266非线性头的精确分解。[D1]

新增组成验证的视查与上述旧分支案例分开记录：本支线agent已实际查看两新臂共8幅test面板，覆盖6个唯一对象4919/7310/5025/7819/3494/3504；主流程也已实际查看这6个唯一对象。5025的VH无效，不能把其κ-only救回解释为双极化协同；7819的full为epoch0，保留B正确不等于学习到了sigma保护。案例和最终竞争类分差见D6，不把8面板当8个独立对象。

## 5. 与实际近邻方法的差异

以下复用`module_novelty_review/READING_RECORD_ZH.md`和Salerno阅读记录。CSRA、GSoP记录包含核心方法全文段落，不是摘要猜测；本轮未重新下载论文。[R1–R5]

|近邻及实际方法|应承认的继承|当前有意义的具体差异|
|---|---|---|
|CSRA，ICCV 2021：分类方向m_c投影空间特征；softmax(T·m_c^Tx_u)聚合，加到GAP；多head可有各自W，另有无需再训插件。|类别响应图、GAP之外的局部聚合及残差已有。|这里用辅助源W而非直接等同主分类器；先分离空间均值，再摘要两侧波动，结合冻结强B读出。无空间softmax。已有3类CSRA-inspired试验不是官方复现，也不构成8类击败CSRA的证据。|
|GSoP，CVPR 2019：通道/位置协方差经映射产生特征缩放，网络末端可接GAP或矩阵归一化协方差。|二阶依赖、全局统计及高阶增强不是空白。|当前取辅助类别方向的协方差二次型与符号二次矩，位于末端增量读出，分开均值路径和校准参照；不学完整协方差或中间特征attention。|
|CPINet，Remote Sensing 2024：GAP/分组双线性统计跨极化调制，低秩对应位置双线性交互、带符号非线性与池化。|双极化、一二阶混合、带符号运算均已有近邻。|各极化先在自己的footprint形成同类轴摘要，再融合；不要求跨极化逐像元相乘，不恢复相干极化矩阵。差异有输入适配意义，但不自动等于优势。|
|Garaiman/Radoi，Remote Sensing 2026：两路MobileViTv2 GAP向量拼接分类，并比较简单融合。|第一阶双极化融合是强参照，不能贬低成天然不足。|同W的mean-only实验证实其竞争力；我们的研究问题落在强校准及均值之后的分布效用，而非拼接首次出现。|
|Salerno，LGRS 2022：从校准浮点裁片提footprint/轴向分段剖面，幅度、峰/位置与几何特征交给随机森林。|SAR中幅度分布、分段统计及尺寸本来就是合理信息源。|这里是学习到的类别响应，不是物理强度/船轴剖面。原论文四类、随机平衡训练及AIS尺寸筛选，与本轮身份/采集隔离任务不同；不能把其幅度排序直接作为κ_e有效证据。|

有意义的差异不要求每个算子都新。当前工作把“主分类分数重映射”“辅助均值带来的额外类别信息”“相同均值附近的波动分布作用”组织成可分别干预的对象，在真实SAR类别取舍中检验它们。这里可推进的是**任务类别坐标下的分布表征与作用分离**，而不是把已知矩重命名。

## 6. 当前必要性消融与仍缺的证据

旧novelty报告的“唯一下一方向、不再同cohort小头搜索”是当时阶段判断，不是覆盖当前授权的永久禁令。现在的sigma-only/kappa-only 18头是对**已有效、已冻结方法的统计组成必要性消融**，full9复用；目的在于确定已声称的表征分工，而不是搜索新网络、以test挑赢家或增加超参数寻正。

源端结果已读取`component_isolation/source_pooled.csv`的validation行：每seed为915折暴露、816唯一对象、629 MMSI。两新臂复用原source scaler，在标准化后将被移除统计槽置零，保留B、均值与valid，从头训练并按各自源J选点；名义266参数相同，有效输入信息/权重自由度不同，并非固定full推理时删槽。[D6]

|源验证差值，seed17/29/43|ΔOA pp|ΔBA pp|ΔJ百分点（100×J差）|
|---|---|---|---|
|full减sigma-only|+.546/+.437/+.109|+.803/+.161/+.071|+.675/+.299/+.090|
|full减kappa-only|-.437/-.109/-.546|+.143/+.125/+.324|-.147/+.008/-.111|

因此，在固定学习预算与选择规则下，σ_e单独未复现full的源端OA/BA，支持加入κ_e的补充价值；但κ_e-only保留了更高OA，加入σ_e后换得更高BA，full仅seed29的源pooled J略高于κ_e-only。这是联合表征的类别取舍证据，不是“κ_e绝对必要”或“full源J全面最佳”的证明。κ_e本身仍带响应幅度信息，不是纯符号开关。源端已参与选择及多轮研发，pooled J没有用于重新选头。

主流程见源结果后追加的固定test推理已完成，保持这18个checkpoint及scaler，没有新增fit。以下差值同样按seed17/29/43排列：[D6]

|固定test差值|净判对|ΔOA pp|ΔBA pp|
|---|---|---|---|
|full减sigma-only|+20/+26/+31|+.422/+.548/+.653|+.516/−.018/+.403|
|full减kappa-only|−18/+7/+5|−.379/+.148/+.105|+.947/+.213/+.392|

sigma-only相对B净+26/+19/+44，κ-only净+64/+38/+70，OA/BA均为正，不能称单统计无效。full的test OA/BA为49.325/42.554、50.084/42.509、49.199/41.775；sigma-only为48.904/42.038、49.536/42.528、48.546/41.372；κ-only为49.705/41.607、49.937/42.296、49.094/41.383。full相对sigma-only的test NLL/Brier三seed均更低；κ-only相对full的NLL差+.01859/+.00398/−.00249、Brier差+.00445/+.00112/−.00102，未统一支持只留κ。原mean-only的test OA仍三seed高于full及两新臂，BA三seed较低，保留这一强参照的真实取舍。

类别变化而非名称解释了主要差异。sigma-only相对full的bulk净变化，源为−4/−5/−9、test为−10/−16/−29；κ-only相对full的chemical源为+9/+3/+9、test为+55/+4/+16，而test container为−22/−4/−9、crude为−11/−7/−2。高频chemical的增加可以支撑OA，却不保证类均衡BA更高。κ-only源端较高OA并未在test三seed全部保留；full的BA优势则保留了。这里只定位实际决策取舍，不把统计赋予物理类别含义。

两新臂各自重新学习、选步不同，且共有2个epoch0回退；删输入改变有效信息及学习路径，并不是固定头的独立因果消融。历史test已经反复开发，本次又是见source结果后追加，因此不作为盲测或独立外推确认。源结果保留，不据test重训、换选点、调力度或挑融合；κ_e独立必要性的边界没有因补完test而消失。

尚缺的关键证据不是更多名称或网络：κ_e的独立任务作用、开发队列之外的可重复收益、与最接近方法在同协议下的正式比较。现有统计也未保留类对交叉项或真实空间排列，但信息在数学上缺失并不等于应加入；已完成的OSS2源端pair/topology试验未兑现独立分类增量，不能写成已解决的新贡献。当前辅助编码路径有实际成本，不能用266参数宣称系统几乎免费；热I/O batch16约8.27ms/对象不含旧AD/T完整推理。[D1]

## 7. 可用于论文的方法贡献段

本研究面向跨采集SAR船舶分类，将额外图像读出的价值区分为已有分类分数的重新校准、辅助类别均值和均值之外的响应分布。在冻结强校准识别器之上，我们以源监督辅助类别方向统一组织各有效极化的空间响应，将一阶均值与去均值后的波动摘要分路读出，再形成可核对的类别竞争增量。后一表征以响应离散度σ_e及带符号归一二次矩κ_e分别描述每视图正负响应平方能量的总量与不平衡，而不赋予其物理散射或预测正确性的含义。不同于直接按类别响应进行注意力聚合，或在特征层构造通用高阶交互，该组织将均值与分布的任务作用放在独立强校准参照下检验。当前开发队列中，完整分路模型相对强B在三seed取得OA/BA双正增量；均值参照、固定分支删除和保持B及均值后的分布替换进一步显示，有益作用包含类别均衡取舍及小幅本图对应信息，不宜简单等同于仅用B的再校准，也不能全部归因于局部形状。贡献候选因此是类别坐标中的均值与波动分离及其可验证用途，而非新统计量、首次双极化或新MLP；κ_e单独必要性、广泛外推及物理机制仍待对应证据。

## 引用与本地证据索引

- [D1] 本目录`FINDINGS_ZH.md`、`run_joint.py`、`paired_intervals.csv`、`branch_deletion_metrics.csv`、`review_cases.csv`、`COST_ZH.md`。本次核对真实输入、分支、训练/选点及完成报告。
- [D2] `../strong_calibration_increment/FINDINGS_ZH.md`；[D3] `../mean_only_control/FINDINGS_ZH.md`。数值与当前D1/既有novelty汇总交叉对应，不将早期弱B结果冒充强B对照。
- [D4] 本目录`conditional_distribution_diagnosis/FINDINGS_ZH.md`。区别于早期oldAD上的centered匹配；本轮引用的是新additive、保持B与μ的固定替换。
- [D5] `oss2_distribution_transfer_20261023`已完成方法复验；本文数值复用`../MODULE_METHOD_ZH.md`最新汇总，未新跑或改变其协议。
- [D6] 本目录`component_isolation/PROTOCOL_ZH.md`、`TEST_ADDENDUM_ZH.md`、`SOURCE_FINDINGS_ZH.md`、`FINDINGS_ZH.md`及`source_pooled.csv`、`test_pooled.csv`、`test_paired.csv`、`test_paired_groups.csv`、`test_review_cases.csv`。源18头与同18个已选头的固定test推理均已完成；本次只更新指定文档的授权出处、完成状态、真实结果与视查记录，不改实验。
- 真实矩定义：`calibrated_signed_transfer/extract_moments.py::weighted_moments`；本次核对空间/类别中心化与分母保护。
- [R1] Zhu & Wu, *Residual Attention: A Simple but Effective Method for Multi-Label Recognition*, ICCV 2021, 184–193，arXiv:2108.02456。既有记录实际读第3.1–3.4节及训练设置。
- [R2] Gao et al., *Global Second-Order Pooling Convolutional Networks*, CVPR 2019, 3024–3033。既有记录实际读第3.1–3.4节及实现表。
- [R3] He et al., *CPINet: Towards A Novel Cross-Polarimetric Interaction Network for Dual-Polarized SAR Ship Classification*, Remote Sensing 2024, 16, 3479，DOI:10.3390/rs16183479。复用已有1–13页、重点9–13页方法笔记。
- [R4] Garaiman & Radoi, *A Lightweight and Real-Time Dual-Polarization Fusion Framework for SAR Ship Classification*, Remote Sensing 2026, 18, 1129，DOI:10.3390/rs18081129。复用已有1–13、17–23页笔记，未称全篇读完。
- [R5] Salerno, *Using low-resolution SAR scattering features for ship classification*, LGRS 2022，DOI:10.1109/LGRS.2022.3183622。复用`oss2_distribution_transfer_20261023/next_mechanism_review/SALERNO_READING_ZH.md`中实际5页原文阅读，不把本轮学得响应与其校准幅度混同。
- R1–R4官方来源及实际阅读范围完整记录于`../module_novelty_review/READING_RECORD_ZH.md`。本次文档收口无新增原文阅读、无训练、无Git，仅修改本文件及component_isolation内FINDINGS_ZH.md、TEST_ADDENDUM_ZH.md。
