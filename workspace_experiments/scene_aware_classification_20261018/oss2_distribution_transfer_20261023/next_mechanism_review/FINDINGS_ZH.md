# 下一机制：同一类别证据图上的邻接增量

2026-10-23。独立结论：**支持已批准的 source-only 三臂9头，而不是先换骨干层级、修padding或继续扫小头。** 它检验一个此前没有检验的具体信息缺口：在保留B、均值、幅度分布和支持数后，真实四邻接是否还有分类增量。支持的是做这个有边界的实验，不是预判空间排列重要。

已读 brain INDEX、02-user-preferences、04-workflows；按 brainstorming skill 的问题优先、失败边界与简化对照收敛。本文只作方案审阅，未训练或提取；不覆盖同目录主代理的 `SALERNO_READING_ZH.md`。以最新授权的 zero/complete/local 为准，不采用此前讨论的随机图或类别方向替换支线。

## 已支持的失败位置，与尚未分清的解释

- **24港有效范围保留。** additive相对强B净+46/+45/+75，OA/BA均升；但条件保持B+mu的本图/借用矩对照净仅+7/+6/+7、区间跨0。系统收益不能全部算作独有局部分布，亦不能因此否认其有效性。
- **OSS2的增量任务不同。** 合计mu已精确组成同一辅助A；分极化mu仍可含求和后未显式保留的信息。local没有稳定新增分类收益；paired9相对marginal三seed源验证top1完全相同。它们没有检验位置排列，只检验了给定源W的分布/同位置类间联合统计。
- **统计压缩的确定性缺口：** 对完整类别响应向量及其footprint权重作共同位置置换，mu/sigma/kappa和pair矩均不变。pair矩保留的是“同格两类如何共同响应”，不是“相邻格如何排列”。数学不变性是真实缺口，但缺口不等于有可用类别信息。
- **编码层的另一种可能：** 当前layer3理论最大RF267超过多数原chip，且格间RF重叠，不能把格点当独立船体部位。RF上限不等于实测有效影响，也不证明所有位置相同；当前非零矩与源图已经排除完全常数图，但未证明细部排列仍可恢复。
- **监督方向是第三个问题：** W为源GAP分类目标拟合，局部 `W F_p` 没有逐部位语义监督。源内合法OOF不保证inner/outer方向的局部效用相同，更不保证跨采集可迁移。本轮固定W，故不直接验证“换方向即可解决”。已有固定头outer矩换inner均值的源诊断也未改善，不能把读出器替换当已证实修法。
- **padding不是当前主修点。** 已完成224→288干预保全raw像素、相位和pool；279源验证feature相对差中位1.733%，B仅2改判、净-1。存在敏感性，不存在此次扩画布有效的证据，不据此包装padding创新。

## 原图与实际可辨识覆盖

本轮实际再次view三张既有源面板，未新渲染：

|案例|可见事实及已有模型事实|对方案的支持与限制|
|---|---|---|
|000556，bulk，41×41|VV有分开的亮响应，VH更连续向下延伸；B判general。有效图16格、24边，n_eff=9.240。|确有可问的相邻/分离差异，但原图亮点不等于类别图局部，不能据外观保证救回。|
|004609，chemical，77×77|中央近水平集中亮结构，边缘另有亮响应；B已正确，旧pair probe只是救回marginal错误。36格、60边，n_eff=28.582。|可检验邻接是否保护已有证据，不称新方法发现B没有识别的类别。|
|000569，container，89×89|VV/VH有上端亮点和延展结构；224/288都正确。36格、60边，n_eff=35.099。|画布外围响应变化明显，而pool内部/分类变化较小；不能将粗图颜色直接叫物理背景。|

面板来自 `../pair_local_evidence/panels/pair0_4_class0.png`、`pair0_3_class3_rescued_review.png` 和 `../native_canvas_probe/case_CC.png`。前两图零pool权重格已灰色遮罩，不能再用旧图无效角落解释模型。

Plato当前 `spatial_topology_probe/coverage.csv`：source共1821，train1542/val279；两极化几何相同，不把3642极化行当独立对象。训练正权格数/有效格数/边数中位25/16.99/40；验证36/21.08/60，所有行至少4正权格且有边。仍有训练211、验证27条 n_eff<4；验证分别为chemical12、general6、offshore5、fishing3、tug1。offshore为5/6、tug为1/3，少数类单例容易改变BA。

因此无需据此删小窗，但必须保留全队列及预先固定的 n_eff<4 分层。正权格“够四格”不是四个独立空间证据；fractional权重质量占比中位训练39.08%、验证37.57%，不能忽略权重。

## 单个实验与公式审阅

**原计划可实施：** source fold0的1542训练/279验证，原B、源OOF矩与W分配不变；三臂共同15槽 `B+mu/sigma/kappa+valid`，加两pol n_eff，再加两pol ratio，均19→12→1共享253参数、同初始化、零末层。固定3seed、200步、jointCE/sourceJ含0与earliest tie，仅新增9头；不访问test，不加其他臂/尺度/层级。B的T使用过源标签，不把整个训练栈称全OOF。

对一pol、一C4、一类别，设 `p_i=w_i/sum(w)`，r已加权空间中心化，`v=sum_i p_i*r_i^2`；无向边集合G的能量定义为：

`E_G = sum_{(i,j) in G} w_i*w_j*(r_i-r_j)^2 / (2*sum_{(i,j) in G} w_i*w_j)`。

三个新增输入分别为：

1. **zero：** ratio=0，仍保留全部共同信息与geometry。
2. **complete：** 全部不同节点对，`E_complete/v = 1/(1-sum_i p_i^2) = n_eff/(n_eff-1)`。
3. **local：** 仅正权格水平/垂直邻接，ratio=`E_local/v`。

complete不是第二种空间信息，而是n_eff已给定后的一个显式非线性函数。因此local对complete是主比较；complete对zero能揭示几何函数/优化便利带来的收益。它不是“额外维度无用”的证明，三臂名义参数相同也不代表有效自由度完全相同。

公式确实能区分排列：等权2×2的行块 `[[1,1],[-1,-1]]` 与棋盘 `[[1,-1],[-1,1]]` 的mu/sigma/kappa相同；complete比值均4/3，local分别1、2。这只证明可达信息差别，不是实际SAR效果。ratio对同一图的非零标量缩放和全体符号反转不变；它不是新的signed统计，也不能单独表达哪类应加分，类别方向由共同B/mu/kappa与监督读出提供。

### 实现中真正会改变解释的四点

1. **逐C4先算比值再平均。** 缓存sigma是逐旋转标准差的平均，`(mean sigma)^2`不是每旋转v，不能用于新比值。每pol每C4先计算E/v再等权平均；complete应逐样本闭合上述恒等式。
2. **固定footprint与因子。** 无向边只计一次，E含1/2；不包括纯padding格、不含自环，VV/VH不作像素相乘或强配准。按各旋转实际footprint构边，避免奇数窗相位错置。三臂共用完全相同的旧输入、合法样本和valid条件。
3. **零方差处理一致。** v=0时实际E/v未定义，不能一臂填0、另一臂仍填解析几何值而额外引入可用性信息。记录这类数量，按共同规则置0，不因局部ratio无效就抹掉有效B或mu。仅处理数值除零，不由结果挑epsilon或截断尺度。
4. **小窗与幅度假设不越界。** 不把local低值当“更正确”；邻差能量虽除以全局v，仍随尺寸、覆盖和RF混合变化。geometry提供对照但不是完全控制这些混杂。完整图比值是有限支持修正；当前源最大约1.344，并非恒为1。无需另加随机图或高阶统计。

三臂先固定并保存全部source选点再汇总；用同seed的local-complete、local-zero以及相对B的逐类救回/误伤报告。重点看源端OA/BA和NLL分别如何变化；源验证参与选点，不称独立确认。epoch0是B回退，不是学到连续性。各seed重复对象，不把9头当9份独立数据。原图仍按实际翻转/CC/WW取例，缺组如实记录。

**成本与缓存：** 已有mean/moments无法逆推出邻接；旧缓存只保存少量示意类别图。若无新增完整源图缓存，需要一次1821源图的原224、C4、双pol冻结前向，流式计算E即可，无需保存整套1024通道图或提取test。9头为既定小型CPU训练，本文未估算或实测耗时，不能引用旧缓存训练时间当端到端成本。

## 最接近前人与可保留差异

**Geary类统计[1]：** DCluster官方式为 `(n-1)*sum_ij W_ij(Z_i-Z_j)^2 / (2*sum_ij W_ij*sum_k(Z_k-Zbar)^2)`。本试验是带footprint加权总体方差的邻差比；等权时其ratio与标准Geary C相差确定的 n/(n-1) 系数，不冒称全等或新发明。贡献候选是同类证据/同分布/同支持数条件下的任务增量与适用边界，不是邻差公式。

**Salerno 2022[2]：** 已通过web核对作者版方法与特征排序：校准浮点chip的footprint按惯性轴生成剖面，强端假定stern，读分段均值及峰幅/数/位置。分段/峰幅较重要，峰位置/数较低，恰提醒不能先验把空间排列当核心。它使用尺寸误差筛选、四类随机划分与RF；不与本八类pass/MMSI隔离的显示PNG/冻结深层响应比准确率。本方案不估计首尾、不将W响应称物理散射，仅检验固定网格关系。详读记录保留在主代理的 `SALERNO_READING_ZH.md`。

**BagNet[3]：** 该论文第2节通过替换卷积限制9/17/33像素RF，然后对局部类别分数平均；其明确的局部性来自结构约束。我们的layer3未限制RF，不能借“类别热图”称局部可解释；本试验改变池化后的关系读出，没有改变生成特征的RF。BagNet也说明无显式大范围布局可有效，不能从池化丢位置直接推出必须修复。

**Zhang & Zhang 2021[4]：** 本轮读取主代理的 `INJECTION_READING_ZH.md`，未重复下载、未独立声称读完32页。主代理已读范围至p21（p1-10初次输出部分截断）；p14-20比较HOG/NGF/LRCS/PAF等在CNN末端注入，推荐手工特征L2→MLP等维→加权拼接；p21因OSS2复杂背景下最小船框提取困难而不用OSS2，OSS1则用SLC、三类平衡训练，不是本GRDH八类身份/采集隔离。当前类别证据图不依赖可靠首尾或精确船体mask，在同一复杂背景输入中控制幅度与邻接，是可发展的任务/方法差异；它不等于首次融合局部特征，且必须以本轮邻接独立效用成立为前提。本文不比较其准确率、不把任务更难当作贡献已成立。

以上仅为可检验的推进方向，不是首创主张。SC RC Net只有摘要线索，本报告不引用其方法、不声称已读全文。

## 结果如何直接改变下一步

- **local优于complete且zero：** 支持当前W、layer3与源验证范围内的真实邻接增量；检查是否仅小类单例/零步取舍。可继续发展同层关系读出，尚无需前移层级；仍不证明身体部位或物理机制。
- **complete与local相近或更好：** 不支持排列独有收益；优先将变化解释为支持数函数/优化或已有共同特征利用，不为local换名称追正结果。
- **三者分类持平或local仅训练损失更好：** 保留概率改善与源负结果。不能据此认定浅层也无信息；但也不能自动启动layer1/2扫层。前移须有新的同样本证据表明当前类图的排列确实不可辨，而更早特征保留了可用差别。本轮未改变W，阴性仍无法拆开W方向限制与RF混合。

最终选择是**先执行已批准的这一组三臂，而非同时更换空间统计、源方向、编码层级**。这给出一个能改变设计的结果：在当前可用表示中，空间邻接究竟是否值得保留。不是要求先证明必涨，也不把源小pilot当八类跨域系统胜利。

## 来源

[1] DCluster官方 `gearyc` 文档，已web读取公式；Geary1954全文未独立读取。
`https://search.r-project.org/CRAN/refmans/DCluster/html/gearyc.html`

[2] Salerno, Using low-resolution SAR scattering features for ship classification, 2022, DOI10.1109/LGRS.2022.3183622，作者版第2-3节。
`https://iris.cnr.it/bitstream/20.500.14243/416954/1/prod_468769-doc_189566.pdf`

[3] Brendel & Bethge, Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet, 2019，第2节及局部解释边界。
`https://arxiv.org/html/1904.00760v1`

[4] Zhang & Zhang, Injection of Traditional Hand-Crafted Features into Modern CNN-Based Models for SAR Ship Classification: What, Why, Where, and How, Remote Sensing 13:2091, 2021, DOI10.3390/rs13112091。方法依据主代理上述有界精读记录，不宣称独立全文核验。

本地证据：`../native_canvas_probe/FINDINGS_ZH.md`、`../geometry_coverage_diagnosis/FINDINGS_ZH.md`、`../reader_substitution/pooled.csv`、`../pair_local_evidence/antisymmetric_readout/analysis/FINDINGS_ZH.md`、`../spatial_topology_probe/MECHANISM_NOTE_ZH.md`及coverage表；24港以既有joint/conditional报告为准，不重复该诊断。本轮仅新增本文件，无训练、父文件改动或Git操作。
