# Centered配对修正：固定模型局部矩条件效用

日期：2026-10-23。主代理在设计一致性检查中发现实现遗漏原先约定的逐行中心化。这是子实现偏差，不是用户更改标准。本目录为设计一致性修正；上级目录uncentered预测、统计和图保留，不覆盖。无训练、选点、阈值或模型变更。

## 结论与实际修正

对oldA+D基线上的已冻结free_real，修正后source validation及test三个seed均为本图矩优于匹配借用矩，OA/BA和两项概率风险方向一致。保留的是当前冻结模型、当前近似条件匹配下的样本特异效用，不是所有样本本图矩更好，不是真实条件采样或像素因果证明，也未证明局部分布优于同辅助读出的普通均值。

B先减每行8类均值，再按同fold源OOF训练行的逐类均值/标准差标准化。头输入B、descriptor scaler和全部头保持原样；仅匹配坐标修正。同fold/seed/B预测类/valid组合、不同MMSI、确定性最短边贪心不变；无标签参与配对，整组极化/类别/矩一起互换，未匹配保持原样。

逐行均值并非常数：源端最大range=0.215068，评价端=0.103974；合法候选边距离最大变化0.162248。共有918个donor行改变，validation各seed为11/32/23，test为228/275/349；匹配覆盖数相同但对象不完全相同。借用输出top1相对旧版变化validation 0/7/3、test 24/35/61例。因此不能说中心化无影响。

冻结B及original输出与旧版逐元素完全相同；original对上游保存分数最大误差8.88e-16，B误差4.44e-16，top1不一致为0。见`version_checks.csv`、`replay_checks.csv`；预检查与最终centered donor一致。修正匹配/重放约5.13秒，不含统计作图。

## 配对结果

下表均为original减swapped；OA/BA为百分点，NLL/Brier为原尺度，风险负值有利于original。seed顺序17/29/43。净例数是本图独有正确减借用独有正确。

|范围|seed|旧uncentered净例|centered净例|ΔOA|ΔBA|ΔNLL|ΔBrier|
|---|---:|---:|---:|---:|---:|---:|---:|
|source val|17|28|28|3.0601|2.7675|-.018711|-.009583|
|source val|29|24|23|2.5137|2.4103|-.017666|-.008911|
|source val|43|24|26|2.8415|2.6937|-.023863|-.009813|
|test|17|23|21|.4427|.5991|-.012210|-.004266|
|test|29|23|23|.4848|.5318|-.014859|-.006666|
|test|43|42|43|.9064|.8903|-.012109|-.004335|

Test本图独有正确/借用独有正确为169/148、226/203、235/192。original对B净+62/+79/+54不变，swapped对B为+41/+56/+11。original仍不是相对B的BA升级：ΔBA=-.4283/-.1943/-.3758pp，不能把本图胜借用改写成全面胜B。source val曾参与头的选点，并非独立确认集。`pooled.csv`给出三版本绝对OA/BA/NLL/Brier，`version_metrics.csv`给出历史与修正的逐项比较。

两种固定配对块bootstrap均保留，2000次，不择优。Test original-swapped 95%区间如下：

|seed|pass ΔOA|MMSI ΔOA|pass ΔBA|MMSI ΔBA|
|---:|---|---|---|---|
|17|[-.2247,1.1504]|[-.3158,1.1730]|[-.0400,1.3275]|[-.0845,1.3171]|
|29|[-.3751,1.3208]|[-.4075,1.3128]|[-.5638,1.5737]|[-.3923,1.4398]|
|43|[.0526,1.7999]|[.1045,1.7691]|[.0606,1.7579]|[.0309,1.7942]|

Test两种块下三seed NLL/Brier区间均在0以下；source OA/BA多数区间在0以上，但seed43 pass OA下界为0，不写成全部严格显著。source pass重采样有87/2000次缺少类别，BA按有效次统计。区间条件于固定头、固定配对，不包含重新训练、选点和匹配不确定性；donor联系可能跨pass/MMSI块，故不作为独立因果检验。完整风险及source区间见`paired_intervals.csv`。

## 覆盖、距离与类别

Test每seed4744对象、2973 MMSI、85 pass、18港、8类；source val为915折内暴露、816独立对象、629 MMSI、32 pass、12港，不把重复源暴露当独立新样本。匹配source为890/894/890行(97.27/97.70/97.27%)，test为4724/4728/4714行(99.58/99.66/99.37%)，均覆盖全部8类。未匹配source25/21/25、test20/16/30，原样保留；主要因候选已配对，source另4/5/6、test另0/2/0行是条件组单例。

source距离中位数1.2670/1.2689/1.3303，p95=3.0513/3.2121/3.0466；test中位数.9936/1.0098/1.0542，p95=2.4559/2.4937/2.5020，最大9.3676/10.7216/8.8306。未加事后距离阈值，长尾并不十分相近。事后同真类率source41.12/38.70/39.33%，test39.37/40.19/39.67%；test同pass约5.38/6.09/5.81%，同港28.92/29.86/28.85%。因此不能称为完全同场景或同分布替换。

|test类|n|MMSI|original-swapped净例17/29/43|
|---|---:|---:|---|
|bulk|1290|938|6/8/12|
|fishing|145|130|1/0/2|
|general|439|334|6/6/7|
|product/chemical|1480|768|-1/-6/12|
|container|519|325|7/12/6|
|crude|405|220|0/5/2|
|tug|202|122|2/-1/0|
|offshore|264|136|0/-1/2|

类别并非处处改善。`class_intervals.csv`含source/test×3seed×8类×2块共96行；test逐类recall区间均包含或触及0。`composition.csv`、`matching_summary.csv`、`unmatched_summary.csv`保留MMSI、港、pass、类、有效极化与距离构成；`validation_unique_metrics.csv`仅作去重描述，不重新配对。

## 四图实际回看

已实际查看本目录四张短名panel，每张同时展示query/donor的native VV/VH及实际160 crop/pad输入，合计8个对象。固定seed17按original→swapped四结局首ID选例；真类仅用于事后结局分类，未用于选donor。修正后这四对ID未变，距离更新；不是新的空间贡献图。

- `panels/matched_rescued.png`：3504←7484，均bulk，Busan/Fujairah，距离1.0028。query为倾斜亮条及较亮纹理背景，donor倾斜方向不同、背景较暗；query由general错判变bulk，真类margin -.1932→+.0208。是借用矩救回，不是本图救回，说明本图矩并非逐例占优。
- `panels/matched_harmed.png`：4708←returned_20260915_0001368，fishing/bulk、Busan/PortHedland，距离2.9875，高于本seed test p95。query VV为紧凑纵向亮体，donor为较小亮点及右下亮区；两者VH均无效，不用于推断。query由正确fishing变general，margin +.1994→-.2396；donor仍错。不是严格相同条件下的物理对照。
- `panels/matched_CC.png`：3494←32742，均bulk、Busan/Ningbo，距离1.7187。query为横向亮条、背景纹理较强，donor细长纵向且亮度不均；query margin 2.1114→1.7276，donor1.3181→1.4795，均持续正确。原窗分别64与72/80，未伪称统一native160。
- `panels/matched_WW.png`：3594←returned_local_knowledge_0012526，bulk/crude、Busan/PortSaid，距离1.3293。两图均可见纵向亮区但背景不同；fold1 seed17选中epoch0，头为零，query margin -.9225、donor+.6523均不变。这例没有矩作用，不能据此否认信息价值。

完整ID、双方类别/预测/margin、真实路径与donor信息见`review_cases.csv`。可见亮体、纹理和尺度差异不等于散射机制；粗深层读出混合目标与周围区域，未做像素因果遮挡，也不命名尾迹或特定散射机制。

## 范围与停止点

近似score条件化后本图矩仍有可保留的经验效用，但尚不能区分辅助W的普通均值与局部分布形状。主流程的score-strength诊断与此并不矛盾：替换损失不意味着本头是最佳校准模型。沿用主代理已读2008条件重要性正文的本地阅读记录；本匹配不是该论文的树网格实现，更不是真实条件采样。

本轮只修匹配定义，原uncentered结果保留。无新训练、图像提取、超参数搜索或Git操作；四图及报告完成后停止执行实验。
