# 最必要补充实验：后两组准备核对

2026-10-29。仅静态阅读代码、现有报告/收据，以及 PowerShell 解析小型 CSV/JSON 和有限路径存在性检查。没有导入 torch、加载 PT/NPZ、运行训练/推理、使用 GPU、下载、操作 Git 或读取进行中的 control 预测。本文只准备第2、3组；第1组同54步 CE+KL 无 margin 对照由主代理执行。

## 结论与最小实施顺序

1. **第2组数据与初始化就绪，训练入口需少量消融适配。** 完整 spatial 模型三个顺序的源拟合及最终端点都已存在，可直接作为主表 full 行；新增 plainR50、去M1、去M2、去M3 各3个训练顺序，共12条配套训练链。必须从同一源拟合前 init 开始，不能在已训练 full 上关模块代替重训。
2. **第3组最小可执行范围是 OSS2 GRDH 历史 fold0 的检测后四类。** 原图、OBB、身份/过境、固定角色划分及32条源参考均有现成清单。保留最终 spatial M1/M2 + 原 M3 架构，改为四类输出，从初始化重训 full 与 plainR50，各3个顺序，共6条链。旧四类算法的成绩不能填到新表。
3. current 公开入口硬编码八类，且没有最终版完整模块开关；目前不能直接一条 current CLI 跑这两组。最小修法是本轮局部运行器适配模块/损失/保存恢复，并做四类实例化；不需要新模型机制、额外位置搜索或全量数据重审。
4. **不把四类结果写成八类复现。** 八类原chip元数据已有2746条；现成检测清单覆盖四类2202条，另四类544条没有这份缓存中的OBB。若主代理明确选八类路线，再只补这544条的检测/prepare与固定split覆盖，实际可用数届时才能确定。本文不执行，也不以 head/tail 线段充当有效 OBB。

以下缩写只用于让路径可读，均为确定的本地绝对目录：

| 缩写 | 绝对路径 |
|---|---|
| W | `E:/临时会话` |
| P | `E:/临时会话/sar_m123_standalone/sar_m123` |
| R | `E:/临时会话/sar_m123_current_20261027` |
| C | `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023` |
| O | `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss2_m123_replication_20261026` |
| D | `E:/临时会话/scene_aware_classification_20261018/oss2_distribution_transfer_20261023` |

## A. 初始化、训练与关闭模块

### 初始化真实来源

`R/weights/init.pt` 已存在。`W/sar_m123_standalone/tools/export_current.py:95` 的 `clean_state()` 与第115-117行说明：由 `joint_polar_m1_20261027/run.py::make('baseline',17)` 的源拟合前模型导出完整 state，仅移除已确认全零的 cross_condition；不是 `teacher.pt` 或 `selected_s17.pt` 的重命名。

可追溯调用链为 `joint_polar_m1_20261027/run.py::make` → `bottleneck_internal_20261027/run.py::make('pre3x3_local',17)` → `selective_m123_20261027/run.py::make('selected',17)`。最后一处构建新模块/头，只加载 ImageNet 骨干；其 `selection.json` 固定原 all_deep2 布局，表示此前源开发选定的结构，不是加载已训练模块。骨干文件为 `W/sar_m123_standalone/assets/backbone/resnet50.pth`，路径存在。上述结论来自导出与构造代码及已完成 refit 收据；本支线没有反序列化权重重验张量。

最终简化版用 `P/current/spatial_m12.py::from_legacy` 从 init 转换；`C/spatial_m12_refit_20261029/run.py::model` 已这样实现。所有保留参数来自同一 init，不采用 `outputs/spatial_m12_20261029/selected_s17_transferred.pt` 作为 fresh init。17/29/43只改变 minibatch 顺序，共享seed17初始化，报告不能称三个独立初始化。

### 哪些参数真的训练

`P/current/model.py::training_policy` 与 `P/current/training.py::train` 已提供完整两阶段训练：

| 阶段 | 更新对象 | 优化与预算 |
|---|---|---|
| fit | M1、M2、M3存活的可训练部分、head；共享 `backbone.model.layer3.5.conv1/2/3.weight` | AdamW，模块/头lr=.001，三卷积lr=3e-5，decay=.01；8轮、batch24 |
| adapt | 仅上述三卷积，共1,114,112参数 | AdamW lr=3e-5，decay=.01；6轮、batch16；模块/头固定但可微 |

其余骨干、BN参数/统计、`anchor_last`、M3的 `reader2/reader3` 固定。M3的selector、branch_bias、project、output、gain可在fit重训。空间M2的均值池化本身没有参数；internal_m2.transforms和deep_output在fit可训练。M1仍训练reader/value/transforms/output/gamma。不能把“能完整重新执行训练流程”写成“所有骨干参数全部解冻”。初始零输出层导致一些上游首步梯度为零是该初始化的已知结构，不据此要求更改模型。

### 最小、明确的模块删除定义

| 训练臂 | 从第1个fit batch起的定义 | fit损失 |
|---|---|---|
| full | `SpatialM12ResNet50`，pre3x3；M1/M2无显式频域，M3保留空间-频域 | CE + .1 aux + .1 relation |
| no_m1 | `BottleneckEvidenceResNet50.intervene` 跳过两路internal_m1，原值送入internal_m2，不产生M1 auxiliary logits | CE + .1 relation |
| no_m2 | 跳过两路internal_m2写回，并移除最终layer2区域读出与deep_pool/deep_output读出；保留M1、M3及它们所需的区域/频率输入 | CE + .1 aux + .1 relation |
| no_m3 | 不执行 `SelectiveBackbone.forward_from` 的reference transport循环，不产生relations；保留M1/M2 | CE + .1 aux |
| plainR50 | 一份共享R50分别处理VV/VH，合法像素加权GAP到2048维/路，拼两路及两个available标志，线性head；无M1/M2/M3、无reference值输入 | CE |

关键落点：

- **去M1不能只设 `m1_stages=()`。** pre3x3的两路模块位于 `internal_m1`，`P/bottleneck_evidence.py:72` 仍会调用。仅把gamma置零也会保留aux监督，且可被训练重新打开。应从执行分支跳过，并从优化器排除禁用参数。
- **去M2不能只设 `m2_select=False`。** 最终 `SpatialEvidencePool` 的pooled本来就等于mean，此设置不会删除M2。`internal_m2[p].enabled=False`只关闭写回，不关闭 `P/selective_backbone.py:99` 开始的浅/深区域读出。上述no_m2采用“完整移除M2”定义；若只关写回，应另名 `no_m2_writeback`，不能充当这一行。
- **去M3不能只改 `m3_count=0`。** 实际循环遍历 `self.m3`；应清空/跳过该 ModuleList，并让关系输出为空。仅将gain设零而保留relation训练也不等价。
- `P/current/training.py::fitting_loss` 对aux与relation都无条件 `torch.stack`。局部适配成“空分支损失为同device标量0”；有分支时原系数、有效reference掩膜和分母保持原样。不要为了避免空列表保留失效模块的假损失。
- `training_policy('fit')`会重新启用所有非backbone/anchor参数。禁用模块应删除，或在policy之后再次冻结并从optimizer排除。no_m3空ModuleList可通过其reader冻结循环；plain模型没有m3属性，不能不加分支就直接调用该policy。
- 禁用布尔属性不进入state_dict，且current loader固定实例化full。checkpoint须记录arm、类名/类数并按arm重建后严格加载，否则恢复时会悄悄重新启用模块。仅加checkpoint元数据和本轮局部加载分支即可，无需迁移框架。

### Head 与纯R50复用

full head 输入为 `2*(2048+3*512)+2=7170`。no_m1/no_m3保留该头；完整no_m2和plain用4098输入。新头从init复制原全局列 `[0:2048]`、`[3584:5632]`、最后2列和bias，而不是复制已训练full头或另抽一个不配对的初始化。

`P/pool_replacement.py::PoolReplacementResNet50(mode='native')` 已有原生无模块、共享VV/VH骨干、合法像素GAP、available标志的前向实现，是plain的可复用主体。复制init backbone和上述全局head；启用同三个LIVE卷积及head，保持BN与其他骨干固定。其 `forward_from` 不接受M3参数，也不返回current损失所需aux/relations；本轮运行器用一个plain分支适配，不直接把 `current.forward_batch` 的全部关键字喂给它。不要对LIVE之后layer4套no_grad，否则会截断LIVE梯度。

纯R50与full使用同一原生160裁剪/补边、相同有效VV/VH、同一标签和排除记录、同训练顺序与步数。基线不重新纳入没有OBB的样本，以免样本集不同；它不消费区域/频域特征作为分类输入。不是两个独立骨干、RGB另一路、旧冻结Ridge、224输入或C4平均版本。

### 第2组固定数据与预算

- `R/manifests/source_fit.json`：1205观测/1004船/88过境；8轮×51步=408步。
- `R/manifests/source_adapt.json`：136观测/128船/20过境；6轮×9步=54步。
- `R/manifests/references.json`：固定16条源fit参考；`current.data.pair_mask`按观测ID与MMSI排除自参考，不能把类别标签作为检索输入。
- `R/manifests/development.json`：160观测/158船/3过境，实际出现7类。
- `C/m2_preadd_expansion_20261028/query_records.json`：3075观测/1950船/85过境/18港/8类，2876双路有效、199仅VV。
- 沿用 `spatial_m12_full_20261029/FINDINGS_ZH.md` 已核实的四队列ID/MMSI/pass隔离；共享港口，历史开发/扩围，不能改叫盲测或严格留一港。

建议预先固定的最小适配口径：三个单模块消融均用**各自fit端点**重算原教师、干预选择、均值对照及E1/K0/F0核对，再用原coefficient=.3683066666126251适配54步；不复用full教师/selection。plain使用自身fit教师继续54步CE+.1正确教师KL，coefficient=0、useful=False。head在这54步同样冻结。

这样full对plain是“完整方法对同预算普通基线”，包含训练目标差异，不声称只隔离架构。第1组正在做的full同54步CE+KL对照完成后，**预先约定**另与plain的CE+KL端点比较即可补上相同后续目标的架构对照，不新增训练、不按control结果选择路线。本支线没有读取该组预测。去模块行回答“各自配套完整流程下删除该模块”的贡献，不能解释为固定所有中间训练材料的单次因果拆分。

源fit class weight沿用每船重复观测补偿；adapt沿用类别平衡。fit现有实现按batch mean，adapt按权重sum归一化，不能在某一臂单独改掉。保存fit与adapt两个固定端点，不按开发精度挑epoch。

## B. OSS2实际输入、类别与划分

### 可复用路径与有效范围

| 文件/目录 | 可直接复用的内容 |
|---|---|
| `D/cohort.csv` | 全2746条八类GRDH行、sample_id、MMSI、scene_id、pass_group、两路PNG/TIF路径、原类id |
| `D/pass_assignment.csv`、`D/membership.csv` | seed20261023、27过境六包与既有各fold身份隔离；当前公共stack固定fold0 |
| `O/data_inventory.csv`、`O/class_scope_exclusions.csv` | 四类2202条及四类以外544条；前者含实际框/prepare可用性 |
| `O/view_inventory.csv`、`O/detection_views.json` | 四类4404路原生检测、raw_xywhr、confidence、有效轴与prepare/crop记录 |
| `O/geometry.csv` | sample_id连接；雷达轴全部无效，不能用heading替代 |
| `O/stack_table.csv`、`O/stack_indices.json`、`O/STACK_SPLIT.json` | 1519条固定行序、五角色、ID/MMSI/过境隔离及计数 |
| `O/stack_pass_coverage.csv`、`O/stack_reference_candidates.csv` | 各过境实际类覆盖与合法参考候选 |
| `O/m3_results/reference_bank.csv` | 已选32个独立MMSI、每类8条，均属于fit_train；本次仅重用参考身份，不重用旧表示 |
| `O/inventory_exclusions.csv`、`O/stack_exclusions.csv`、`O/crop160_impact.csv` | 无框、身份隔离、19个大chip的已验证排除/裁窗解释 |

本地原生PNG根目录是 `D:/Documents/project_root/project_data/public_opensarship2/opensarship2/chips/{vv,vh}/`；上游显示TIF位于 `D:/Documents/project_root/project_data/OpenSARShip_2/{scene_id}/Patch_Uint8/`。本支线抽查stack首行 `pub2_opensarship2_000699` 的两路PNG及两路 `Visual_Cargo_x3612_y3994_{vv,vh}.tif` 均存在。全4404路存在性、尺寸与prepare结论复用 `O/PREPARATION_RECEIPT.json`、`O/DATA_PROTOCOL_ZH.md`；本次解析view_inventory见reported_missing=0、prepare_error=0，没有再扫原图。

这里“原chip”指保留原像元尺度的uint8显示chip，**不是标定后向散射/复数SLC**。已检测四类2202条全部有双路文件，但有效OBB/目标输入不等于文件存在：1839条至少一路可用=1699双路+48仅VV+92仅VH；363条双路无合格框。检测前1738船、检测后1504船，均覆盖27过境。不能把无框行当黑图样本或只凭双PNG判作双路有效。

### 八类与四类不能混用

下表由 `D/cohort.csv` 小文件实际分组取得，与既有summary一致；是检测前GRDH八类数量，不是八类current可运行数量。

| 原id | 类别 | 观测 | 船 | 过境 | 四类缓存情况 |
|---:|---|---:|---:|---:|---|
|0|bulk_carrier|590|509|27|映射新0；至少一路有效565，双路539|
|1|fishing_vessel|352|338|20|不在四类检测清单|
|2|general_cargo|231|207|26|映射新1；有效154，双路124|
|3|product_chemical_tanker|611|456|26|映射新2；有效368，双路313|
|4|container_ship|770|566|27|映射新3；有效752，双路723|
|5|crude_oil_tanker|118|104|22|不在四类检测清单|
|6|tug_towing|48|37|18|不在四类检测清单|
|7|offshore_supply|26|23|14|不在四类检测清单|

原八类总2746观测/2240船/27 scene/27 pass。当前四类缺少的1/5/6/7不是OSS2原数据没有，而是旧公开四类任务预先排除。四类stack的新1=general、新2=chemical、新3=container；**直接用八类current解释，会分别错读成fishing/general/chemical**。`assert_source`允许0..7，不会自动发现该语义错配。

原 `manifest_position/obs_id` 是筛选后目标manifest中的位置；`target_manifest_row`是更早清单行号；`legacy_row_index`是2746行cohort索引；`table_index`是1519行stack索引。例如sample_id尾号000699在stack首行，但obs_id=249。转换器以sample_id持久身份，显式保留各索引，不能按尾号或compact index猜测。scene_id是产品名，pass_group是平台+绝对轨道+datatake；身份/过境仍用于排除而非模型输入。

### 固定角色和实际可用极化

| 原role | 观测/船/过境 | 双路 | 仅VV | 仅VH | 最小current用途 |
|---|---|---:|---:|---:|---|
|fit_train|478 / 448 / 9|444|11|23|source_fit；32参考均来自这里|
|fit_validation|228 / 223 / 4|206|6|16|保留开发监测，不合并入fit|
|utility_train|241 / 235 / 4|227|5|9|source_adapt，教师选样只在此发生|
|selection|194 / 193 / 5|177|5|12|保留开发监测；固定终点，不再搜epoch|
|evaluation|378 / 369 / 5|348|12|18|最终固定评价|

合计1519观测/1468船/27过境，1402双路、39仅VV、78仅VH。角色间sample_id/MMSI/pass隔离直接沿用STACK_SPLIT已有记录。新类0/1/2/3在fit中分别138/47/95/198，adapt为83/26/46/86，evaluation为132/26/83/137；五角色均四类齐全，单过境可缺类。报告类别n和实际覆盖，不以缺类零召回补成八类BA。

### 四类current最小接入，不重跑旧四类算法

1. 只从 `O/stack_table.csv` 和 `O/view_inventory.csv` 导出current格式记录：`id=sample_id`、`label=class_id`(新0..3)、mmsi/pass_group/scene_id和明确split；按role映射上述fit/adapt，两个开发角色保留原角色字段，evaluation不得传入训练。reference_bank.csv通过obs_id连接stack，再保留sample_id。
2. OBB转换复用 `O/build_inventory.py::view_spec` 或 `C/oss1_m123_replication_20261026/extract_frontend.py::prepare_inventory` 的数学逻辑：raw_xywhr原width/height/angle一起旋转成corners，conf>=.25，target_index=0；ship_axis仅用已有axis_valid标记。不要将排序后的length/width配原angle重建框。无合格框的极化置不可用，保留另外一路；radar_axis_rad=None。不导入带实验环境副作用的旧run来获得转换函数。
3. `SpatialM12ResNet50(...,classes=4)` 构造器本身支持四类，M3不依赖固定八类；但 `current.model.load_model/save_model` 固定classes=8/全局CLASSES，`current.data.assert_source`固定上界8，`current.training.class_weights(stage='fit')`分母写死8。局部四类运行器应明确K=4、四类名字与标签断言，fit权重分母4，并按类数/arm保存恢复。margin_floor已有 `score.shape[1]-1`，无需改。
4. 从源拟合前init建立四类初始化：无类别轴参数原值复用；head输出行与internal_m1两路reader输出通道按原类 `[0,2,3,4]` 切取，实例化相应四类模块再严格装入存活参数。plain取同四行的全局head列。不能从训练过的24港教师/最终权重迁移却称OSS2 fresh训练；本方案也不声称完全不曾在历史公开数据上开发结构。
5. 保留32条已有源参考身份，避免新KMeans/新参考搜索；M3从current的原生单朝向空间图重算。该参考预算与自建数据的16条不同，需明示为OSS2协议；full与plain共享同一合法标签池，plain不使用参考值，也不额外获得标签。
6. source_fit：478条、8轮、batch24，共160步；adapt：241条、6轮、batch16，共96步。full用本臂源教师+原选择/分差系数，plain用自身教师CE+.1KL无margin；两者相同样本、顺序、优化器、LIVE和总256步。维持既定轮数，**不是硬凑54步**；54是私有136条协议产生的步数。该表仍是完整方法对普通基线，明确训练目标不同。

四类运行不需要补其它类别OBB，也不需要下载；输入层复用现有prepare行为，19条大chip未按尺寸删除，现有验证目标mask无裁失但外围上下文有裁剪。冻检测器训练来源未审计，不能宣称检测器从未见OSS2。若必须八类，则在独立输出目录沿用相同检测/关联阈值补544条，再按原27过境包和身份预留规则重建八类角色；不能原地覆盖O的四类缓存，不能在检测后重新抽分包求好结果。

## C. 训练矩阵、缓存与旧结果边界

### 预期最小矩阵

| 组 | 臂 | 顺序 | 每条链fit / adapt步 | 新增训练链 | 评价 |
|---|---|---|---|---:|---|
|第2组|full spatial|17/29/43|408 / 54|0，复用3条|development160 + expanded3075|
|第2组|plainR50、no_m1、no_m2、no_m3|17/29/43|408 / 54|12|同上，保留两阶段端点|
|第3组|OSS2四类full、plainR50|17/29/43|160 / 96|6|evaluation378；两开发角色仅监测|

总计新增18条两阶段链，即18个source fit和18个adapt；不再加OSS1、跨库双向、六折、位置候选或参数扫网格。按上述预算是7080个新增optimizer step，不含教师选择/评价。full参考行的6个现成模型路径已逐项检查存在：`C/spatial_m12_refit_20261029/training/spatial_s{17,29,43}/model.pt` 与 `C/spatial_m12_full_20261029/training/spatial_s{17,29,43}/model.pt`。原表及评估已完成，可直接引用，不为凑表重训full。

### 可共享与不可共享

- **私有输入缓存可共享。** `C/spatial_m12_refit_20261029/cache/prefix.npy`、`masks.npy`是本轮修正后的float32，`complete.json`记录实际填充1365行的原索引，包含fit1205+dev160，参考是fit子集。按run.py::load/precise_prefix的ID映射读取，不能把稀疏大数组其它行当有效。旧float16前缀与native分数差.0002378已失败，不退回旧缓存；修正收据误差7.8678e-6可复用。
- **几何/槽位/频率缓存可共享。** 精确路径映射在 `C/spatial_m12_refit_20261029/run.py::load` → `C/selective_m123_20261027/run.py::load`：pool_replacement的valid，joint_readout的slots/energy，separate_evidence的local slots/energy，selective_m123的field2/3。都是固定输入信息；去M1/M2仍需M3频域，不能全删频率预处理。
- **adapt136和expanded3075不在上述1365行有效缓存里。** 可按 `current.prepare_record` 与同一冻结prefix新建一次共享输入缓存，或逐batch原生准备；已有full run复用的是批内prefix，不应宣称它已持久保存所有expanded张量。主代理先看实际收据/文件再选，别重建已有合格缓存。
- **共享边界在冻结layer2输出。** LIVE、M1/M2/M3、参考的live values、最终logit及teacher/selection都依训练臂变化，不能跨臂固定缓存。固定原图Gabor可以共用，干预视图的特征要与干预像素一致；选择后E1/K0/F0核对沿用现有实现。
- **OSS2旧frontend_cache/m3_raw.npz不可当current前缀。** `raw_complete.json`明示C4四次前向、large_maps_saved=false、layer1原256维矩/PCA接口；新current需要单朝向 `[2,512,20,20]` layer2 prefix与空间网格。只能复用原图/框/placement/参考ID；新建一次current输入+prefix缓存供full/plain和全部顺序共享，不重复提旧C4特征。
- 同一init、同类/头映射的共享张量可配对复用；禁用后重新启用、恢复时遗漏arm、类别顺序错位、缓存错行是启动时真正会改变下一步的检查项。

### 哪些旧baseline/ablation不进入新主表

| 现有条目 | 可复用部分 | 不能复用成什么 |
|---|---|---|
|spatial_m12_refit/full_20261029|final full的同初始化三顺序端点、队列、评价、原图记录|legacy/pruned行不是最终空间版逐模块配套重训|
|selective_m123_20261027的drop_m1/drop_m2/drop_m3|损失空分支等实现经验、固定输入缓存|stage布局、旧频域及旧训练定义的成绩不是pre3x3 spatial消融；其drop_m2还保留浅层mean读出|
|pool_replacement_20261026的native / backbone_m2_pilot的none|plain前向主体及数据接口|旧冻结骨干训练/终点不是同408+54步基线|
|joint_polar_m1、shallow_m2、m2_block/residual/stage_depth等名称为baseline的行|各自实验历史比较|这些baseline多数本来就含M1/M2/M3，不能重标plainR50|
|m1_frequency_ablation_side的plain_local|局部M1无通道选择的构造参考|不是无M1、更不是纯R50|
|oss2_m123_replication_20261026 / m3_aligned / new_version_ablations / public_m123_cross_dataset|数据清单、既有参考ID、输入回看和历史方法成绩|冻结C4+读出器/PCA/校准/旧M3头，不能作为current网络full或同协议plain|
|D的八类224窗口Ridge结果|八类manifest和pass/membership|不是本次160输入、可训练LIVE的最终网络|

### 启动前只需解决这些具体失败

主代理落地局部运行器后，用一个含双路和缺一路的实际小batch验证：关闭模块确实不参与前向/损失/优化器，存活模块与LIVE梯度连通；空aux/relations不崩溃；no_m2/plain的全局head列和八/四类映射正确；保存恢复保持arm和输出；缓存行对应实际id，native与缓存一致。若失败，修对应适配再开始，不能靠跳过检查继续。现有full重放收据已合格，无改动不重复全量验证。

实验收尾沿用已有CPU统计和原图画廊流程：同查询配对OA/BA/NLL/Brier、逐类/过境、救回/误伤/持续对错；VV/VH原生尺度分开，支持式臂看实际同类/混淆参考；plain明确无参考作用。当前报告是准备结果，尚无新增baseline/消融/OSS2最终版精度，不预判正负。训练、原图回看和后续Git同步由主代理负责，本支线未操作Git。
