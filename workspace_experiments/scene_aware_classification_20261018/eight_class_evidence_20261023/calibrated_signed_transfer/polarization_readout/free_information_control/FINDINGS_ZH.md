# Free读出的信息替换：18头完成

2026-10-23。**逐图矩/分路信息可保留为有边界的候选，但本轮不能证明它们普遍必要，也不能认定仅廉价校准已经充分。** 冻结真实free相对aggregate三seed OA均略高，BA两seed较低；相对constant三seed BA均略高，OA一seed较低。两种块bootstrap下，这些真实信息与替换对照的OA/BA差均跨0，不是等效性证明。constant复制了相当部分OA收益，不能再把真实free相对B的净收益直接称为极化信息增益。

## 执行与范围

- 按先写的PROTOCOL，仅free_aggregate/free_constant各3fold×3seed17/29/43，共18头、3600步。CPU训练19.95秒，含加载/重放总22.76秒；无特征提取/Reader拟合，无真实free重训，无额外臂、超参搜索或test选赢家。
- 直接复用standard_controls.Head('free_shared')，209参数，11→16 GELU→1、类均值中心化；无tanh cap、无κ乘法载体。相同seed初始化逐参数一致、末层0，全部零头误差0。每头200步jointCE、AdamW .003/.01，源J每5步含0、严格改进/平局最早。
- aggregate调用父organized聚合重复，所有描述符矩同步替换，保留总矩，结构误差0；constant取匹配inner-held源OOF train的各pol有效行均值[2,8,2]，再按逐图valid遮罩。各fold VV/VH有效均值行数为4392/3905、4419/3926、4418/3913。常量逐类σ/κ和计数见source_constants.csv。
- 两臂各用替换后源train描述符拟合相同规则的scaler。逐图B和flags不变。constant不是无图像模型：B仍来自图像，固定类别矩参照也可帮助类条件校准；只是没有本图σ/κ输入。名义容量相同，有效输入自由度不同。
- 全18选点保存后才统一test。保存头的validation J在训练float32和最终float64 B+残差下均精确一致；B与冻结真实free缓存最大差1.78e-15，分数加法闭合误差8.88e-16，全部基线top1一致。选0步5/18，其中aggregate为fold0 seed43，constant为fold0 seeds29/43、fold1/2 seed29。constant seed29整路严格回B，并非缺失实验。
- 每seed同4744 test对象、2973 MMSI、85pass、18port、8类；308对象VH无效。sourceval为915折暴露/816唯一对象、629MMSI、32pass。A与局部矩为匹配的源OOF；C冻结但不是第二层OOF，**不能称整流水线OOF**。这里只比较B之外增量，既有A/B/旧A+D各自协议不变。

## Test结果

OA/BA单位%；Δ相对同seed B。seed17/29/43的B OA分别44.5616/47.3229/46.9435，BA38.7095/38.5059/38.6456。真实free来自已冻结standard_controls预测。

| 信息路 | seed | OA | BA | 净救回 | ΔOA pp | ΔBA pp | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real/free_shared |17|45.0885|38.3420|+25|+.5270|−.3675|1.48122|.69274|
| real/free_shared |29|47.3440|38.7878|+1|+.0211|+.2819|1.43499|.67687|
| real/free_shared |43|47.4073|39.0757|+22|+.4637|+.4302|1.44119|.67727|
| free_aggregate |17|44.6669|38.4599|+5|+.1054|−.2496|1.47989|.69167|
| free_aggregate |29|47.3229|39.0869|0|0|+.5810|1.43349|.67612|
| free_aggregate |43|47.1543|38.9440|+10|+.2108|+.2985|1.43735|.67660|
| free_constant |17|45.1728|38.2179|+29|+.6113|−.4916|1.48754|.69421|
| free_constant |29|47.3229|38.5059|0|0|0|1.43709|.67796|
| free_constant |43|47.1965|38.8216|+12|+.2530|+.1760|1.43787|.67757|

三seed描述性均值：real OA46.6133/BA38.7352，aggregate46.3814/38.8303，constant46.5641/38.5151。不是将seed当独立测试对象。real相对aggregate的NLL和Brier三seed均更差，不能只汇报其OA优势；相对constant的Brier三seed略好，NLL seed43较差。

## 信息对照与区间

以下均为real减control；95%区间采用2000次配对块重采样、固定随机种子20261023。pass/MMSI均列出，不择优。括号为[下限,上限]，单位pp；缺类重采样次数均0。

| 对照 | seed | 净差 | ΔOA；pass CI；MMSI CI | ΔBA；pass CI；MMSI CI |
|---|---:|---:|---|---|
| real−aggregate |17|+20|+.422；[−.316,1.261]；[−.254,1.084]|−.118；[−.930,.643]；[−.842,.595]|
| real−aggregate |29|+1|+.021；[−.517,.519]；[−.482,.479]|−.299；[−.871,.222]；[−.840,.214]|
| real−aggregate |43|+12|+.253；[−.251,.740]；[−.240,.763]|+.132；[−.391,.773]；[−.456,.737]|
| real−constant |17|−4|−.084；[−.663,.479]；[−.682,.546]|+.124；[−.652,.773]；[−.644,.904]|
| real−constant |29|+1|+.021；[−.322,.345]；[−.360,.432]|+.282；[−.035,.722]；[−.064,.664]|
| real−constant |43|+10|+.211；[−.285,.665]；[−.253,.666]|+.254；[−.145,.662]；[−.218,.746]|

real与aggregate的“仅real正确/仅control正确”为137/117、66/65、77/65；与constant为105/109、43/42、66/56，净差小不等于逐例预测相同。完整逐例配对在paired_predictions.csv。

相对B的两种区间亦完整保存在paired_intervals.csv：constant seed17 OA区间均为正（pass[.129,1.117]，MMSI[.103,1.125]），但BA负点值区间跨0；aggregate seed29 BA区间均为正（pass[.130,1.099]，MMSI[.074,1.112]），OA净0。real seed43 OA的pass区间跨0而MMSI不跨，必须同时保留。所有区间只条件于已选固定预测，不含训练/选点不确定性，未做多重比较校正。

## Source与样本组成

源val pooled ΔOA/ΔBA（pp，按seed17/29/43）：real为+1.421/+.314、+.437/+.539、+.219/+.002；aggregate为+.874/+.237、+.656/+.227、+.874/+.665；constant为+.984/−.530、0/0、+.219/+.375。constant seed17的OA换BA已在sourceval出现，不是只在test发生。每fold J合法选点，并不承诺pooled BA不降。816唯一对象按最早fold保留的描述性表另存validation_unique_metrics.csv，不替代原fold选点。

sourceval上real−constant BA为+.844/+.539/−.373pp，并非源端三seed均支持真实矩；real−aggregate也不是源端普遍优势。因此test的弱正方向是范围内描述，不宜反过来作为新设计选型依据。全部源val OA/BA/NLL/balNLL/Brier、选点曲线已保存。

| 类别 | test n | real−aggregate平均净差 / 召回差pp | real−constant平均净差 / 召回差pp |
|---|---:|---:|---:|
| bulk_carrier |1290|+1.67 / +.129|−5.67 / −.439|
| fishing_vessel |145|−.33 / −.230|+1.00 / +.690|
| general_cargo |439|+2.33 / +.532|−1.00 / −.228|
| product_chemical_tanker |1480|+11.67 / +.788|+2.33 / +.158|
| container_ship |519|−.67 / −.128|−1.00 / −.193|
| crude_oil_tanker |405|+2.33 / +.576|+6.00 / +1.481|
| tug_towing |202|−1.33 / −.660|+.33 / +.165|
| offshore_supply |264|−4.67 / −1.768|+.33 / +.126|

real−aggregate平均OA净差主要由较大类product_chemical_tanker贡献，同时offshore_supply召回下降，解释OA和BA方向分离的具体构成；并非证明物理极化机制。real−constant的BA小优势较多来自crude_oil_tanker，而constant更偏bulk/general。两种控制也都降低container召回，简单校准不是没有代价。

real在VH有效4436对象上的净收益为+26/+2/+25，无效308对象为−1/−1/−3；aggregate对应+5/+1/+11和0/−1/−1；constant对应+29/0/+11和0/0/+1。此分层受对象类别和场景组成影响，不能称VH因果效应。尤其只有VV有效时aggregate输入与原矩相同，但训练头不同，预测仍可不同。逐港/逐pass结果在stratified_metrics.csv，非每港获益；例如real平均净收益较集中于Port Klang +6.67、Jebel Ali +6，而Port Hedland −1.67。

## 已实际查看的8张原图面板

固定seed17，每臂各结果取首ID，不择漂亮案例；图标题明确B为冻结校准。8图仅5个唯一对象，全部fold0，不能当作8个独立场景，也不能代表8类比例。图中仅原VV/VH、实际160crop/pad与记录OBB，**没有constant的逐图κ贡献图**。

| 臂 / 结果 | ID | 类别/港 | B→本臂 | 对实际最强错类的margin |
|---|---|---|---|---|
| aggregate 救回 |pair_0032894|bulk / Ningbo-Zhoushan|2→0|−.00531→+.04939|
| aggregate 误伤 |pair_0010487|fishing / Fujairah|1→7|+.04933→−.17059|
| aggregate CC |pair_0003494|bulk / Busan|0→0|+.38865→+.96255|
| aggregate WW |pair_0003504|bulk / Busan|3→3|−.96513→−1.95721|
| constant 救回 |pair_0005025|general / Busan|1→2|−.33570→+.00866|
| constant 误伤 |pair_0010487|fishing / Fujairah|1→7|+.04933→−.17177|
| constant CC |pair_0003494|bulk / Busan|0→0|+.38865→+.56879|
| constant WW |pair_0003504|bulk / Busan|3→3|−.96513→−1.03136|

- 0032894：VV72×72内为较亮的近纵向长条，VH88×88同方向响应更弱、分段明显，双路有效。仅聚合矩也能把近边界bulk/general竞争翻转；不证明是哪一路或哪个位置造成救回。
- 0010487：双路64×64，中央紧凑白亮块，周围纹理明显，OBB角度有差异。两臂都将近边界fishing误推offshore，说明此误伤不必依赖逐图矩；外观不能确认类别，也不能据此归因具体散射机制。
- 0003494：双路64×64，近水平亮条；VV周围整体更亮，VH上下轮廓有差异。两臂都强化既有正确bulk判决，对同一crude rival，aggregate增量较大；只能描述分差，不把亮条当成已证明的模块依据。
- 0003504：双路64×64，斜向连续亮条及不均匀周围纹理；两臂均未纠正bulk→product/chemical，aggregate反而扩大错误margin。它与CC同港且外观同为长条，粗外观不足以解释成败。
- 0005025：VV64×64内竖向白亮块；VH80×80显示较细的纵向响应，但**VH valid=False，本轮不使用该路矩**。constant仅靠逐图B、flags及固定源参照便将general/fishing翻转；不能把图中VH形态描述为此次救回依据。图中的该路160窗口是来源展示，不是有效模型输入。

这组事实支持保留“校准本身可纠正/误伤”的解释，同时留下真实矩相对常量的类别权衡缺口。原窗、OBB和预测不足以建立像素因果或船体/背景机制；本轮没有进行遮挡或空间归因。图表路径、原图路径、实际尺寸、valid、footprint、旧/新rival和分差均在review_cases.csv。

## 可保留的结论与停止

1. **分路信息**：real对aggregate的OA弱正方向可以保留，但平均BA略差、NLL/Brier三seed较差、信息对照区间均跨0。不能宣称分路普遍必需，也不能把它证明成全无价值。
2. **逐图矩**：real对constant三seed BA小幅更好，特定类别有正差异；但constant已复制接近的平均OA，且seed17 OA更高。现阶段更准确的表述是“有类别取舍的候选补充”，不是已排除廉价校准解释。
3. **廉价基线**：固定源矩+B+flags是必须保留的强对照，但不是纯B-only或完全无局部数据的控制；source均值仍由局部矩构成。不能从这次相近结果推出所有局部信息可删除或校准一般充分。无需因此继续试臂或按test选择赢家。

训练、统计和8图实际回看均完成。一次图像渲染ID格式错误（pair_前缀被误转整数）已仅在本目录修正并重跑渲染；未重训/改选点。代码、PT、CSV、报告均只在本目录，不改父、不Git。**本轮到18头停止。**

主要文件：run_information.py / analyze_information.py；models/；selection_frozen.json / selections.csv / selection_curves.csv / source_constants.csv；predictions.csv / paired_predictions.csv；pooled.csv / all_pooled.csv / stratified_metrics.csv / validation_unique_metrics.csv / matched_metrics.csv / matched_classes.csv / paired_intervals.csv；checks.csv / baseline_replays.csv / costs.csv / summary.json；review_cases.csv / panels/。
