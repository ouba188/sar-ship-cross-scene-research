# 冻结参考替换与匹配摘要干预：16例原图审阅

2026-10-26。已通过 `view_image` 逐张实际查看本目录16张PNG，VV/VH分开，并核对 `cases.json` 的原始路径、预测、实际参考和native放置记录。仅复用既存预测，无训练、无新推理实验、无结果导向改选方案。

## 结论先行

冻结模型对参考库和区域匹配摘要都有可测的依赖，但依赖不是每个样本都受益，也不证明某种物理散射机制。指定的draw101在两个方向均使OA下降，反向BA却上升；no_end/no_side的总体负结果与局部救回同时保留。极端案例说明作用位置与剩余缺口，不代表总体占比。

本报告与 `module_replacements/FINDINGS_ZH.md` 的问题不同：后者是仅s1/s12的 **conditional matched head comparison**，不是完整M3链重训；本报告是既存M3冻结推理的bank替换或匹配摘要干预，不能混成同一种消融。

## 固定协议与数据来源

- 只取cross队列，seed17。OSS1训练对应OSS2查询1619观测/1347船/27pass；OSS2训练对应OSS1查询921观测/742船/21pass。观测并不全部独立，未把同船多次观测或重复参考当独立机制证据。
- original为修复缓存的 `frequency`；reference仅取本轮审阅预先指定的draw101，不从多个draw挑最好结果。source identity已由上游隔离，本图集沿用该队列，不重新拆分或训练。
- reference改变冻结模型的bank及实际检索参考，不是重新抽样训练。mechanism分别用OSS1的no_end和OSS2的no_side，保持原actual references和像素。
- 定义固定rival为原分数中最高的非真类，固定分差为 `score[true]-score[original_rival]`。每组rescue/harm/CC/WW中取绝对分差变化最大的一例，相同值按表序。分组按完整四类argmax，不按固定分差是否大于0。
- 标签仅用于事后评价、分组和案例挑选；不用于模型拟合、超参、检查点或干预臂选择。此处每组最大变化属于有意的极端案例，不能当随机样本。
- no_end/no_side在local匹配张量中同时清零对应区域的相似度与质量字段，重算极化头及门控中随分数变化的前11维；原geometry/content描述、频率输入、availability及训练权重不变。它不是把原图的端部/侧部物理擦除，更不是独立物理因果实验。零值输入与门控重算的联合作用也不能拆成单个物理解释。

生成入口为本目录 `build_gallery.py`。运行环境为 `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B -u build_gallery.py`，使用现有 `extract_frontend.prepare_inventory/prepare`；未安装或修改环境。

读取链路：以 `public_m123_cross_dataset_20261026/normalization_repair_20261026/方向/target_table.csv` 为cross顺序，用同目录 `table.csv` 定位原source/target行，原 `seed17/predictions.npz` 提供frequency、refs及query_rows。新参考读取 `reference_sensitivity/源/draw101/seed17.npz`，区域干预读取 `m3_interventions/源/seed17.npz`，两者均先按within长度截取cross。运行时已核对obs_id顺序、source行与原stack_table一致、interventions.original与修复frequency逐值一致，避免串队列或把source-row当obs_id。

`cases.json` 保存完整四类前后分数、固定/动态分差、old/new真实参考各类两张的source-row及obs_id、原极化权重、local true/rival摘要，以及每个显示chip的原始绝对路径、native尺寸、crop/pad、target_valid。图中reference为2x5，只显示每类旧/新第一张真实参考；第二张仍参与原预测并在JSON记录，不能把图中四张当全部推理输入。mechanism为query加同类两张、固定rival两张原实际参考。图中参考属于M3，未声称MLP使用任何参考。

## 整体配对结果

仅seed17、对应cross；差值为百分点，不是三seed结论。changed还包括错类之间翻转，因此不等于rescue+harm。

| 源→查询 | 指定干预 | 救回 | 误伤 | 总预测改变 | OA变化 | BA变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OSS1→OSS2 | draw101 | 69 | 126 | 306 | -3.5207 | -2.3485 |
| OSS1→OSS2 | no_end | 167 | 279 | 594 | -6.9179 | -4.9643 |
| OSS2→OSS1 | draw101 | 50 | 61 | 130 | -1.1944 | +2.6077 |
| OSS2→OSS1 | no_side | 83 | 192 | 345 | -11.8350 | -7.0228 |

反向draw101的OA/BA分歧说明按观测占比和按类等权的结论不同，不能省略任一指标；这里未进一步扩展类别实验。局部改善不推翻总体下降，总体下降也不抹去确实获救的样本。

## Native显示边界

从原始chip经原提取器native160裁切/补零，输入scale=1，无按目标缩放。每格使用相同0..255显示范围，绿色是检测目标mask边界，不是注意力、分区边界或人工真值轮廓。显示尺寸统一不意味着物理地面尺度相同，也不能跨chip把亮度直接解释为经过标定的后向散射系数。

15个查询native边长为37至109像素，均补零；obs58为177x177裁到160x160，VV裁切起点(7,8)、VH(8,8)，各自检测target像素464/485全部保留，但周边背景有所裁去。其余查询检测target也全部保留。16例双极化target_valid均为true，这不保证定位语义正确。尤其obs927的VH绿色目标与另一明显亮结构分离，不能把亮结构直接当成已被模型正确采用的船体。

## OSS1→OSS2：reference draw101

以下ID均为查询所在OSS2的obs_id，参考ID为source OSS1的obs_id。参考数组按 `[true两张; 固定rival两张]` 记录。

### Rescue：611，container，general→container

图 `oss1_reference_draw101_rescue_611.png`。57x57查询在VV/VH均为斜向小目标，VH的绿色覆盖比VV更长；旧同类174横向，新同类1198斜向。旧参考 `[174,748;1235,159]`，新 `[1198,135;786,40]`。固定分差-2.1731→+1.4242，最终相对最强非真类仅+0.3434。冻结替换确实救回；方向与外观更接近只是可见共现，不能归因于1198单张，因为两个类的参考、第二参考及其他类都可能改变。

### Harm：927，container，container→bulk

图 `oss1_reference_draw101_harm_927.png`。65x65查询VV为近横向小亮段，VH检测框在另一亮结构上方，双极化目标区域明显不一致；缓存仍将两极化判为有效。旧 `[797,174;880,165]`，新 `[1198,1295;507,128]`，分差+0.1668→-4.9330。与611相同的新第一同类1198在这里没有带来收益，不能笼统称它“更好参考”。定位/跨极化对应是有图像依据的疑点，但本次只换bank，未检验该疑点是否造成翻转，不修改标签或检测结果。

### CC：2520，bulk，bulk→bulk

图 `oss1_reference_draw101_CC_2520.png`。81x81查询呈斜向长框，VV局部亮点突出，VH沿框内分布较连续。旧 `[1294,1245;820,11]`，新 `[507,916;804,551]`，固定分差+0.9009→+4.5407，最终动态分差+2.6446。图中旧bulk1294与chemical820的外观均有相似细长形态，新参考姿态也明显改变；结果支持置信证据改变但不支持肉眼形状就是检索准则。

### WW：1159，container，general→chemical

图 `oss1_reference_draw101_WW_1159.png`。41x41查询仅占少量native像素，VV/VH均斜向；旧同类748明显更大，新1198较紧凑。旧 `[748,174;1235,1151]`，新 `[1198,814;1324,40]`。固定container-general分差-3.2957→+0.8536，然而container对新的最强非真类chemical仍为-0.6007。此例直接说明“固定分差转正”不等于救回；图中rival依协议仍展示general，不能伪称已经展示新的chemical赢家参考。

## OSS1→OSS2：mechanism no_end

### Rescue：525，container，chemical→container

图 `oss1_mechanism_no_end_rescue_525.png`。69x69查询近竖直，VV/VH的亮区分布不同，VH中部亮点突出；实际参考 `[797,174;522,820]`，chemical522框内有明显局部强亮块，尺寸也大于查询。固定分差-4.5352→+0.0497，是很窄的救回。只能确认端区匹配摘要在既定网络中影响了相对读出，不能据亮点位置命名为某种端部散射或证明去掉真实端部更好。

### Harm：2604，bulk，bulk→chemical

图 `oss1_mechanism_no_end_harm_2604.png`。69x69查询在两极化均细长斜向，VH框内强响应更连贯；实际 `[1294,1245;820,11]`，不同类别都出现细长、局部亮点结构。固定分差+1.3542→-2.5008。与525的救回方向相反，表明同一摘要置零并非普遍降噪；未改变原图，不能把误伤解释成真实端部结构被移除。

### CC：1780，bulk，bulk→bulk

图 `oss1_mechanism_no_end_CC_1780.png`。109x109查询VV目标较暗，VH左端相对突出，图像上方另有框外亮结构。实际 `[1294,155;156,820]`，bulk155和chemical156都可见一侧局部亮区。分差+5.0723→+0.1164，虽未翻转却几乎耗尽原优势；不能只统计正确率而说该样本“不依赖端区”。框外亮结构是否进入有效匹配区域和其物理性质，图集不作未经验证的解释。

### WW：2347，bulk，container→container

图 `oss1_mechanism_no_end_WW_2347.png`。97x97查询较宽，双极化均含多个强亮点，VH更连续；实际 `[155,880;122,764]`，同类与混淆类参考的native尺度、姿态、亮点分布都不同。分差-9.1647→-4.8534，改善明显但仍远离翻转。端区摘要不是唯一剩余混淆来源；不能以改善推断其他区域物理含义，也不能以仍错否认该干预改变了数学证据。

## OSS2→OSS1：reference draw101

此组query ID属于OSS1，reference ID属于OSS2。

### Rescue：1300，bulk，container→bulk

图 `oss2_reference_draw101_rescue_1300.png`。61x61查询斜向，VV/VH均有紧凑亮段；旧bulk1926和新bulk2525都较小，旧container2730较长，新container1484较横向且VH较弱。旧 `[1926,2517;2730,1409]`，新 `[2525,1932;1484,2008]`。分差-0.0575→+1.6966，动态分差+0.7732。原本接近边界，冻结bank变化足以救回，但无证据将收益唯一归给“更相似的bulk参考”。

### Harm：515，bulk，bulk→container

图 `oss2_reference_draw101_harm_515.png`。85x85查询斜向长框，VV一端突出、VH更弱；旧bulk1960长而近竖直，新bulk1932较小，新container692也呈竖向长结构。旧 `[1960,1995;249,2730]`，新 `[1932,2621;692,1994]`。分差+1.2445→-0.8594。外观尺度差异是可见事实，但多个参考共同变化，不能宣称缩小同类支持是已证实原因。

### CC：560，bulk，bulk→bulk

图 `oss2_reference_draw101_CC_560.png`。57x57查询较短，VV局部亮、VH较弱。旧 `[1995,1960;249,2730]`，新 `[1932,2621;2008,692]`；旧bulk1995较长，新1932更小，新container2008也较长。分差+2.7806→+0.4079。它与515共享新同类两张而结局不同，不能把固定新bank的某个参考笼统认定有效/无效。持续正确不意味着证据稳定。

### WW：469，bulk，container→container

图 `oss2_reference_draw101_WW_469.png`。65x65查询斜向，两极化都有局部强响应，VH外观范围稍宽。旧 `[1926,1995;2730,249]`，新 `[1932,2621;2008,692]`，与560的新同类和混淆类两对完全一致。分差-8.1205→-6.0224。相同参考组合对不同query仍产生不同结果，说明不能只以bank组成解释全部误差；现有可见细长/亮点结构不足以分开两类。

## OSS2→OSS1：mechanism no_side

### Rescue：404，general，chemical→general

图 `oss2_mechanism_no_side_rescue_404.png`。37x37查询为本组最小chip，VV/VH都只有紧凑斜向目标；实际 `[274,1597;2571,2716]`，同类274较横向，chemical2571框外还有亮点/条状响应。分差-6.1871→+0.3441。小目标下区域摘要对读出的影响很大，但“侧区含背景污染”仍是假设；本操作没有删除像素，也没有重算目标检测或区域几何。

### Harm：7，bulk，bulk→general

图 `oss2_mechanism_no_side_harm_7.png`。73x73查询斜向，VV/VH均在框一端有亮响应；实际 `[1926,2517;1409,2730]`，container1409明显更宽且周边纹理较强。固定bulk-container分差+5.7517→+0.0620，但bulk对新的最强非真类general为-0.4235，所以属于误伤。不能因固定分差尚正而误标CC，也不能把图中container参考当作最终general赢家参考。

### CC：58，container，container→container

图 `oss2_mechanism_no_side_CC_58.png`。177x177原chip按native裁到160，检测目标完整保留；查询横向、双极化均有多个沿长轴排列的强响应。实际 `[1985,2017;2571,1914]`，container1985同样横向且有多个亮段，2017近竖直，chemical参考明显更小。分差+54.1715→+47.6386，绝对下降为该组最大但剩余优势仍大。只能说此例对这次扰动的判别稳定；原始分数不是校准概率，也不能将亮段解释成已识别的货箱或证明某物理结构造成优势。

### WW：540，general，bulk→chemical

图 `oss2_mechanism_no_side_WW_540.png`。61x61查询近横向，VV/VH均呈数个亮段；实际 `[274,1597;1995,255]`，同类274/1597与救回404相同，但query姿态和尺度不同。分差-5.7777→-1.4325，动态分差仍-2.0748，赢家从bulk变chemical。该侧区摘要干预改变了竞争关系，却没有恢复真类优势；本图依固定rival规则展示bulk参考，不是新的chemical参考。

## 可保留的结论与未解释缺口

1. 在同一冻结模型与同一查询上，实际bank或匹配摘要的改变能改变类分数、门控读出及最终预测，这是计算管线层面的证据。两个方向的指定区域干预总体均负，但同时存在明确rescue；不支持“该区域始终有益”或“该区域都是干扰”。
2. 小native目标、VV/VH的检测区域及强响应差异、同类/异类支持外观重叠，都是原图可见约束。obs927定位对应疑点、obs1159/7/540的竞争类更换应保留，不用单一形状故事抹平。
3. 未分离单张参考、参考检索、局部相似度/质量、极化头及门控更新各自的贡献；未证明具体物理散射来源。冻结替换结果也不能预测重新训练后的模型表现。这里不据结果新选bank、不追加训练或干预实验。
4. 同时保留正负结果与全部16例，和within8图报告及M1/M2中间层对照一起交付；主代理负责原cross8图、24港对照与总报告，本支线不重复或扩展。

输出索引：`build_gallery.py`、16张PNG、`cases.json`、`summary.csv`、`receipt.json`、本文件。`receipt.json` 的实际审阅完成标记仅对应本次生成版本；重新运行脚本会重置为待审阅，避免将旧人工审阅冒充新图审阅。
