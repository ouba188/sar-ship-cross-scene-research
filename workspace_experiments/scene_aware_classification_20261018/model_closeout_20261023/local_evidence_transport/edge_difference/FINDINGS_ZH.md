# Edge Difference：源端18 fits完工及有界诊断

2026-10-24。训练运行和缓存诊断均已完成，两个进程退出码均为0。主实验约13.14秒，18个头各按协议运行200步，源selection选择0/20/50/100/200中的checkpoint。只写edge_difference目录；没有外推、Git、原图分析、追加拟合或参数/阈值搜索。

## 结论

**真实四点差分mixed没有在本轮源协议下取得新增分类收益。** 三个seed均由selection选择step0；additive也均选择step0。保留阴性结果，不扩外部，不把数学恒等式正确包装为有效创新。

只有body对照存在小幅平均收益，净救回+4/0/-1；它是相同合法边入口下仅用船体端点的对照，不是四点关系成立的证据。两个置乱臂在seed43被源selection选中，但evaluation均净-3，也不支持真实配对优势。

有界诊断没有发现“差分向量普遍接近零”或“训练没有起作用”。更明确的现象是：mixed对真实类和固定竞争类都能给出较强匹配，但多数区域的两类分离弱于body；训练及selection的连续损失均下降，却没转成selection的J收益。尚不能进一步认定是表征、配对、聚合或损失与分类边界哪个因素主导。

## 完整结果

evaluation485，三个冻结基础配置均值，OA/BA为百分数。

| 臂 | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| full |58.832|40.122|1.348249|0.593532|
| anchor |59.038|40.556|1.336449|0.588747|
| body |59.244|40.589|1.301538|0.575758|
| additive |59.038|40.556|1.336449|0.588747|
| mixed |59.038|40.556|1.336449|0.588747|
| mixed_shuffle101 |58.832|40.253|1.338740|0.590203|
| mixed_shuffle211 |58.832|40.324|1.337218|0.590180|
| mixed_shuffle307 |59.038|40.556|1.336449|0.588747|

| 臂 | selected step17/29/43 | 对anchor净救回17/29/43 |
|---|---|---|
| body |20/0/20|+4/0/-1|
| additive |0/0/0|0/0/0|
| mixed |0/0/0|0/0/0|
| mixed_shuffle101 |0/0/20|0/0/-3|
| mixed_shuffle211 |0/0/20|0/0/-3|
| mixed_shuffle307 |0/0/0|0/0/0|

step0表示新系数为零、保留anchor，并不表示没有训练后续checkpoint。

## 实现与协议

源3387条的四组raw分数全部缓存；1741条按644utility/612selection/485evaluation拟合与评价。源参考固定64条。输入复用同一PCA32单位token，差分向量没有再次单位化。

侧边(b=i,n=8+i)，端边(b=0,1,6,7;n=16,17,18,19)。四组为VV_side、VV_end、VH_side、VH_end。所有臂合法边等权，mass只用于两端为正的合法性；每个有效reference等权，其内部合法边等权。query合法边平均log证据。没有使用sqrt面积权重。

shuffle101/211/307分别在每图、每组合法边内置换neighbor端点；保存实际端点映射。它保留原合法边集合和所有边权，不混入重算面积质量。各组有效类别中心化后按各arm/group的utility RMS缩放，缺组为0。四项类别共享系数叠加冻结anchor。

模型选择沿用既有fit：Adam .01，0.5CE+0.5balancedCE+.01平方和，源selection严格J提升才更新checkpoint。未使用evaluation标签选边、尺度、温度或checkpoint。

## 有界诊断一：raw分差在哪里变化

每个seed对每张query固定r=argmax_{c!=y} anchor_c；以下raw分差是新局部读出的真实类分数减该固定竞争类分数，不是重新挑对新模块有利的竞争类。缺组时仅在双方均有效的查询上统计；原始明细保留有效性。

### Seed17 evaluation均值

| 组 | 有效query | body raw差 | additive raw差 | mixed raw差 | body标准化差 | mixed标准化差 |
|---|---:|---:|---:|---:|---:|---:|
| VV_side |485|+0.235153|+0.219844|+0.066781|+0.320475|+0.179326|
| VV_end |485|+0.267649|+0.225705|+0.019132|+0.362543|+0.031269|
| VH_side |416|+0.074798|+0.057112|+0.000568|+0.135511|+0.002276|
| VH_end |416|+0.081316|+0.075549|+0.014603|+0.150424|+0.026257|

mixed的分数本身不弱：四组真实/竞争均值依次为3.5398/3.4730、4.1902/4.1711、4.1176/4.1171、4.9768/4.9622。问题表现为两类都匹配，而不是所有响应都归零。由于各算子数值范围不同，不只比较raw差，也保留源RMS标准化后的差。

分位数也显示边界：mixed四组raw差中位数为+0.1313/+0.0065/-0.0209/-0.0032，正分差比例约57.94%/51.34%/46.15%/49.76%。不能用均值略正认定每张query获得了可用增量。

逐类不一致。seed17的渔船，body在VV_end/VH_end分差均值+1.2122/+0.9473，mixed则为-0.4139/-0.3047；普通货船VV_side从body -0.0448到mixed +0.00055，但VH_side从+0.0168变为-0.1687。这是任务相关区别，不据此删类或选区域。所有seed、三个role、八类、四组及10/25/50/75/90分位数均在raw_margin_diagnosis.csv；逐query来源在sample_raw_margins.csv。

## 有界诊断二：是否过拟合

直接复用已保存checkpoint记录，比较同一种连续目标：训练joint CE，selection joint CE=0.5NLL+0.5balanced_NLL；另列selection J。没有根据诊断重新选择模型。

### mixed的step200相对step0

| seed | utility joint CE变化 | selection joint CE变化 | selection J变化 |
|---|---:|---:|---:|
|17|-0.033661|-0.017304|-0.031499|
|29|-0.020827|-0.031301|-0.008931|
|43|-0.027049|-0.026111|-0.008327|

mixed三个seed的所有非零checkpoint，训练CE均改善、selection joint CE也均改善，但selection J均低于step0。因此不能概括为“训练降、验证损失升”的典型整体过拟合，也不是训练未更新。较准确的描述是：连续目标改善未转化为本协议要求的OA/BA综合决策收益。局部过拟合或类别取舍仍可能存在，当前缓存不足以分清其全部成因。

其他臂的边界：additive seed43在50/100/200步出现训练CE下降、selection joint CE分别上升约0.00482/0.00246/0.00223，具有局部验证退化；其他additive seed并非同样情况。body seed17/43在20步selection J有正增量，后续CE继续改善但J回落，所以源选择20步而不是最终步。全部18条轨迹、90个checkpoint在checkpoint_diagnosis.csv，不一概归为同一种过拟合。

## 有界诊断三：是否普遍弱差分

只统计原协议合法边，不加入新threshold过滤。范数为原单位token的`||q_b-q_n||`，理论范围0到2；不是重新单位化后的范数。边数是局部端点组合数，不是独立船数。

### evaluation485合法边

| 组 | 合法边数 | 均值 | P10 | 中位数 | P90 | 精确零数 |
|---|---:|---:|---:|---:|---:|---:|
| VV_side |3879|1.3527|1.0139|1.3823|1.6504|0|
| VV_end |1938|1.4172|0.9512|1.4832|1.7657|0|
| VH_side |3328|1.4151|1.1294|1.4304|1.6827|0|
| VH_end |1664|1.5172|1.1293|1.5900|1.7899|0|

对应mixed自相似`.5||q_b-q_n||²`均值约0.945/1.053/1.025/1.187。参考bank64四组范数中位数为1.327/1.364/1.448/1.489，utility为1.330/1.385/1.397/1.504，均未表现为普遍接近零。

全3387条的最小合法差分范数仍存在较小值（四组约0.308/0.249/0.339/0.354），但不能用最小值推断整体塌缩。完整各role、参考库、按类分布已保存。当前证据不支持“多数差分太弱导致模型失效”这一笼统解释；也不能从范数不小推导差分方向具有类别信息。

## 协议内检查与重放

- 四项展开与差分内积误差0；任意行/列偏置不变性误差8.88e-16。
- 按原槽位拓扑翻转OBB轴，真实边集合重排、三种真实算子分数误差0。这是坐标轴符号翻转，不是重渲染图像任意旋转。
- 缺VH/无效端点仅保留合法边，shuffle保持原合法mask及邻域端点多重集。
- NumPy四点/参考/边逐层trace与GPU缓存最大误差7.09e-7。
- 保存预测与四组系数贡献重放最大误差3.24e-7。

没有扩大全库审计。上述只对应协议列出的数学、索引、缺失、轴翻转及贡献重放风险。

## 交付与后续接口

- run.py：实现和训练入口；六臂18fit已完成，不需重跑。
- diagnose.py：只读现有缓存和源token的有界诊断，没有fit。
- features.npz：全部3387行每臂8类×4组标准化特征，idx指向1741条开发队列。
- localcache.npz：全部3387行raw分数、有效性、逐query边log分数、neighbor端点映射、bank、instance_id。
- predictions_17/29/43.npz和weights.npz：本机完整表索引预测及已选/各checkpoint系数；idx之外保持旧anchor。
- training_receipts.csv：selectedsteps；selection_curve.csv：训练和selection原记录；metrics/class_results/paired_results/coefficients/scales/summary.csv：完整结果。
- raw_margin_diagnosis/sample_raw_margins/difference_norms/checkpoint_diagnosis.csv：新增有界诊断。
- receipt.json、diagnosis_receipt.json：完成记录。

主代理做原图时可导入run.py，调用`trace_query(table_index, arm)`。返回每组四项点积（顺序bb、nn、bn、nb；数组轴为4×class×query_edge×reference×support_edge）、相似度、逐reference log贡献、逐query边分数、reference/edge责任占比及端点映射。已保存mixed的两个示例trace，source table index8和280（后者缺VH）。需要具体案例时重放即可，不需要重训。trace是匹配分数的数值解释，不是物理因果归因。

**本轮收口：有界源验证完成，保留mixed阴性和body有限收益。不能推出四点方向必然无效，也未获得足以继续包装强创新的正证据。后续是否改变表征、训练目标或匹配方式，由主代理结合原图和理论决定，本任务不自行扩大。**
