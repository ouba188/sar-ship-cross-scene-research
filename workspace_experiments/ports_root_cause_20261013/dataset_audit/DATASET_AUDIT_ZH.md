# 本地公开数据与 source24 审计

2026-10-13。完成只读来源追溯、队列/重叠统计、XML恢复和有界图像检查。本分支未拟合、未编码、未用GPU、未改标签或其他工作流。**主线已完成有界两臂几何对照，取代此前临时收尾决定；本分支随后独立核对fold 2的R/T expert及配对指标，840项通过。详见GEOMETRY_IMPLEMENTATION_AUDIT_ZH.md；唯一物理根因仍未被识别。**

## 发行版与本地队列

论文事实来自主线已核验的 Huang DOI `10.1109/JSTARS.2017.2755672`、Li DOI `10.1109/BIGSARDATA.2017.8124929`；官方入口 `https://opensar.sjtu.edu.cn/DataAndCodes.html`。本分支在线工具未返回可读页面，另直接读取本地发行ReadMe，不冒称独立联网全文核验。

| 层级 | OSS1 | OSS2 |
|---|---|---|
| 论文发行版 |11,346 chips、41产品、5港|34,528 chips、87产品（52GRD/35SLC）、10港|
| 本地目录/XML |41产品、5,673船舶记录|46产品、10,024船舶记录|
| 本地双极化导入 |5,673对、41产品|9,120对、38产品|
| 当前八类评估 |1,349对、22GRDH产品|3,745对、38产品；2,746GRDH+999SLC|

OSS2是扩展，论文增加23,182 chips，21幅场景含两类产品，不是独立发行。本地OSS1确有5,673VV+5,673VH=11,346个uint8文件，但ReadMe没有明确论文总数单位，**不单凭乘二关系定论论文按极化计数**。OSS2的9,120对是部分产品且要求VV/VH的导入结果，不能当完整发行版或总数除二；本地构建收据有904条未匹配注释、0转换错误。

## 实际来源链

OSS1 `opensarship1_data.py` 按scene+XML中心坐标连接 `D:/Documents/project_root/OpenSARShip_1/*/Patch_Uint8`，直接读TIFF。3,030条原生GRDH经兼容标签映射保留1,349；原生SLC不评估但参与源身份purge。全部最终标签来自elaborated，无质量过滤。

OSS2 `D:/Documents/project_root/project_data/scripts/build_public_opensarship2.py` 同样按坐标连接TIFF，直接转PNG，无resize。共享 `opensarship2_labels.py` 从9,120对保留3,745，排除coarse3,997、ambiguous154、unsupported1,224；不把Cargo/Tanker强行细分。最终3,720条来自elaborated，25条来自上游AIS回退的primary（渔船22、拖船3）。`type_level=elaborated_or_ais`、`quality_flag=ok`是构建器常量，不能分别证明细标签来源或无噪声。

source24来自 `visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`。`prepare_eight_class_adaptive.py` 合并旧缓存25,656+新增12,435=38,091；实际来源为legacy21,195+returned16,896，不能把旧缓存全算legacy。图像来自merged/returned的UTM_8bit裁切。已读legacy-relabel skill，但其七类流程不能套成本八类数据的处理史。

16,896条returned与server manifest唯一连接，AIS绝对时间差中位26秒、P95为142秒、最大180秒；position_method、uncertainty_m、human_review_required为空。legacy未被假定同质量。源VV/VH路径各38,091个唯一；已知MMSI+product无重复，同MMSI+datatake有3组6行、标签一致；未做全量像素去重。

## 组成与年代

| 类别 | source24 | OSS1 | OSS2 | OSS2_GRDH |
|---|---:|---:|---:|---:|
| bulk_carrier |11128|816|868|590|
| fishing_vessel |8599|35|490|352|
| general_cargo |5929|121|316|231|
| product_chemical_tanker |5827|116|818|611|
| container_ship |2208|218|973|770|
| crude_oil_tanker |1757|13|149|118|
| tug_towing |1478|18|76|48|
| offshore_supply |1165|12|55|26|

全为IW。公开目标全S1A；source24为S1A32,184/S1C5,907，全部GRDH、841产品、766datatake、24港、18,766已知身份、198行身份缺失。OSS1/OSS2分别1,110/2,698身份、22/31datatake。source24日期2025-01-01至2026-02-24；OSS1为2015-08-23至2016-10-18；OSS2为2015-04-03至2016-07-12。年代差异仅描述，不是根因证明。

来源与类别混杂：全部2,208集装箱来自returned，8,459/8,599渔船来自legacy。前一轮来源内对照仍有外观反转，不能只归因来源比例。

## 重叠与配对

| 比较 | 共享MMSI | 两侧涉及行 | 共享datatake | 同船同datatake |
|---|---:|---:|---:|---:|
| OSS1/OSS2 |419|532/647|4|162对|
| OSS1/OSS2_GRDH |274|344/384|1|0|
| source24/OSS1 |198|410/232|0|0|
| source24/OSS2 |427|971/592|0|0|

OSS1/OSS2产品名无重叠、起始秒仅1个共同键，不能据此称独立。162对均为GRDH/SLC对应，标签、AIS时间、AIS长宽全部一致；159对SAR时间也一致。159对SAR中心距离中位28.598米、最大340.612米；人工SAR长度绝对差中位30米、最大117米，无一完全相等。**同次观测不是像素重复**，不同产品x/y也不是同一网格。

历史source purge针对全部native身份，OSS1/OSS2保留37,343/36,996行，共同36,610行。五折source数量23,872/21,223/31,335/27,417/27,745，当前目标身份残留均0；此操作不撤销历史预训练暴露。

原B64五次支持/query仅行互斥：OSS1查询共享支持身份21–35行，OSS2为50–64行，其中同船同datatake11–17行。主线固定 `public_no_other_release_vessel` 后，OSS1每次773–783条query、OSS2为3,043–3,050，均保留八类；另有 `query_unseen_support_vessel`。计数和类分母已导出，主结果不改，场景/预训练关联仍可能存在。

**`oss2_grdh_slc_pairs.csv` 有234对，其中233对标签一致**：同MMSI+datatake+SAR时间，每组恰好1GRDH和1SLC。两端均query数228/224/225/223/224，逐draw掩码另存。370对仅datatake匹配另存；6个多条歧义组不任取一条。使用完整OSS2相同支持/模型缓存，而非不同GRDH子任务的另一次拟合；宏指标按实际出现类别计算。

唯一标签冲突：MMSI305127000，2016-02-09 09:54:35UTC，`pub2_opensarship2_003058`为container_ship、`..._007723`为general_cargo。XML与各自manifest一致，是输入标注冲突；不重标、不从主队列删除。

## 噪声、AIS和字段缺失

OSS2干扰阳性483/3,745（GRDH276/2,746、SLC207/999）；多船独有386、岸线独有58、两者均有39。阴性3,262行子字段空，OSS1三个噪声字段全缺，不能将缺失当零。类别干扰率不同，例如集装箱41/973、原油45/149。

OSS2 `source_time`是SAR时间。XML恢复的AIS−SAR范围：OSS1 −625至539秒，OSS2 −651至575秒；绝对差>300秒分别111/1,349、418/3,745，该阈值仅描述，不新增过滤。

两个最终公开队列的incidence、头尾、人工SAR长度、AIS时间及中心坐标均已恢复：旧manifest未携带不等于原数据没有。AIS长度0值40/439行，宽度0值44/483行；OSS2 `length_m` 对非正AIS长度回填人工SAR长度，不能当同源真值。XML四角是chip范围，不是船体框。5个XML非法UTF-8仅在Callsign，内存替换并记录；原件未改，坐标唯一连接和MMSI一致性通过。

## 输入和2×2的解释

本地ReadMe区分raw `Patch`、定标sigma0 `Patch_Cal`、伪彩 `Patch_RGB`、增强0–255显示图 `Patch_Uint8`。**当前original是既有uint8，不是raw或sigma0。** 140文件抽查均L/uint8；OSS2抽查64个PNG与源TIFF数组一致。

R50将VV/VH双线性缩放128×128；R18为VH转RGB再Resize(128)，不是使用伪彩图。display合并双极化128图的q01/q99，应用共享affine到native图，再clip/round/uint8；不是纯对比度或物理定标。渔船/集装箱native VV宽度中位：source24 64/72、OSS1 29/97、OSS2 25/109；同128输入不是同物理尺度。

2×2需固定source行、目标/支持/query、编码器和配方，仅交叉源/目标original或display；source-display协调PCA、尺度及分类头。**主读数R/T/DD不需要mapper**；只有另行读取重建成分P/E/PE时，才需协调相应mapper。可检验模型与显示处理兼容性，不能推断信息百分比或唯一物理原因。主队列、身份排除、噪声/产品分层均保留，并报类分母；主线模型结果未在本分支独立复算。

## 几何证据的最终状态

**旧版spacing缺失判断已撤回**：OSS2多数元数据名为“产品名.xml”。最终表恢复全部11个入选SLC产品：range_spacing=2.329562米，azimuth_spacing=13.927475–14.028690米，srgr_flag=0、未地形校正。不能用旧论文17.4米或当前官方14.1米代填；GRD分辨率20×22米与采样约10×10米也不同。

ReadMe定义X=range/Y=azimuth。999条SLC的raw/cal/uint8尺寸全部与XML裁切范围一致；每产品1条、两极化共22探针，raw幅度与同位置uint8为单调关系，排序逆序数0。支持native网格尚未校正纵横比，但未恢复全发布增强代码，**不等于完整物理几何认证，更不是性能因果证明**。

主线现已决定执行固定SLC-only局部平面近似 `range_spacing/sin(incidence)/azimuth_spacing` 压缩range后居中补框，与同实际压缩面积的azimuth-sham比较；同插值/补边，原source、完整OSS2原支持，GRDH不变、不调参。本地逐行比例0.2327367–0.3272724，中位0.2505111。控制仍混合插值、补边、分辨率和上下文变化。**本审计分支只交付证据与接口，不执行编码或新拟合；物理机制等待主线结果。**

`oss2_slc_native_grid.csv` 已补充源TIFF、导入PNG、raw/cal路径及预计算比例。`actual_aspect_ratio` 明确定义为局部地面宽高比 `native_width*ground_range_spacing/(native_height*azimuth_spacing)`，是由现有元数据计算的近似，不是直接测量船体宽高比；`native_pixel_aspect_ratio`为像素宽高比，`range_compress_ratio`为range轴压缩系数。坐标X=range、Y=azimuth。

## 交付与范围

核心机器可读文件：`cohort_summary.csv`、`cohort_dimensions.csv`、`overlap_summary.csv`、`coobservation_summary.csv`、`source_purge.csv`、`identity_sensitivity_cohorts.csv`、`oss2_grdh_slc_pairs.csv`、`product_spacing_metadata.csv`、`oss2_slc_native_grid.csv`。逐行证据和五个只读脚本同目录保留。

检查针对实际错配、源残留、导出改变和网格误判；对应范围已通过，抽查不升级为全量像素证明。无哈希、无全盘扫描、无拟合或实验编码。所有owned命令已退出，无后台任务。见 `FINAL_RECEIPT.json`。
