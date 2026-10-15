# 外部数据场景/采集分组接口

2026-10-13。只读有界检查；仅写本文。未运行模型、拟合、重采样或新实验，未计算新性能结果，未改主代码/记忆，未开展文献检索或24港分析。

**结论：OSS2可直接按冻结支持集做场景已见/未见和采集已见/未见的描述性比较。跨采集主分组建议用 `datatake_key`，产品场景 `scene_id` 作补充；GRDH/SLC分层。OSS2现有 `port_id` 全为 `opensarship2`，不能用它声称“同港跨场景”，该问题仍由主线24港研究回答。**

## 精确资产与连接方式

下表路径别名仅用于本文，文件名均相对对应根目录。

| 别名 | 绝对路径 |
| --- | --- |
| A | `E:/临时会话/ports_root_cause_20261013/dataset_audit` |
| P | `E:/临时会话/ports_root_cause_20261013/predictions` |
| I | `E:/临时会话/imaging_coupling_20261013` |
| B | `E:/Hermes/paper_tracking/sensitivity/backbone_ablation_20261010` |
| M | `E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007` |

- **逐样本元数据：`A/public_metadata.csv`**，先筛 `dataset == 'oss2'`，再以 `sample_id` 连接预测 `ids` 并按其顺序排列。实际3745行、3745唯一ID、38产品场景、31采集组；本次全部成功连接。字段：`scene_id`、`product`、`product_type`、`platform`、`orbit`、`datatake`、`datatake_key`、`acquisition_start_key`、`identity`、`sar_datetime`、`ais_datetime`、`class_id`、`incidence_xml`、`heading_deg`、`sar_lon/sar_lat`、`noise_xml/multi_ship_xml/port_noise_xml`。产品列 `product` 实测逐行等于 `scene_id`。
- **原目标清单：`M/target_manifest.csv`**。关联键 `sample_id`；标签 `class_id/class_name`；船只 `mmsi_or_ship_id`；产品 `scene_id`；时间 `source_time`；来源影像 `source_path_vv/source_path_vh`。`target_manifest_row` 是上游清单行号，不能直接当当前缓存下标。`split_group_id` 实测等于船只身份，不是场景组；`port_id` 不是真实港口分层。
- **主模型冻结预测：`B/targets/oss2/oss2/predictions.npz`**。`ids:(3745,)`，`support:(5,64)`，`eval_masks:(5,3745)`，`predictions:(5,10,3745)`，轴顺序为 draw、mode、样本。用 `modes` 查找 `dual_refit_late` 等模式，勿硬编码序号。`support` 是本文件 `ids` 的位置索引；`rows` 实测为0..3744，**不等于**上述 `target_manifest_row`。本文件没有概率数组或标签数组，标签从元数据关联。
- **成像研究冻结预测：`P/original/original/fold_2/oss2/predictions.npz` 与 `fold_3` 对应文件**。`predictions:(5,3,3,3745)`，轴为 draw、arm、head、样本；`arms=['R','T','DD']`，`heads=['prior','expert','c4_late']`。同目录 `probabilities.npy` 为 draw、arm、head、样本、类别轴。两折的 `ids/support/eval_masks` 均与上述B缓存逐项一致，本次已核对；这不表示预测值、模型或完整主系统相同。
- **产品物理元数据：`A/product_spacing_metadata.csv`**，以 `(dataset, product)` 连接；含 `metadata_path`、`range_spacing`、`azimuth_spacing`、`SAMPLE_TYPE`、`srgr_flag`。`A/oss2_slc_native_grid.csv` 用 `sample_id` 连接，含已恢复的原生网格/比例证据，无需重扫XML或图像。

原生OSS2来源目录由既有审计记录为 `D:/Documents/project_root/project_data/OpenSARShip_2`；产品注释为 `<product>/Ship.xml`，产品元数据的实际路径以 `metadata_path` 为准，多数是 `<product>/<product>.xml`。本次复用恢复表，没有重新遍历原生目录。

## 分组选择与实际陷阱

`scene_id` 是完整产品名，适合回答“查询产品是否出现在支持集”。`datatake_key = platform + '|' + orbit + '|' + datatake`，适合把同一次采集的不同处理产品/切片放在一组，回答“查询采集是否出现在支持集”。构造代码见 `A/audit_local.py::parse_product`。

OSS2实际有7个采集组各对应2个产品。例如 `S1A|006718|008FE4` 同时包含：

```text
S1A_IW_GRDH_1SDV_20150708T095424_20150708T095449_006718_008FE4_6186
S1A_IW_SLC__1SDV_20150708T095423_20150708T095451_006718_008FE4_1C72
```

两产品开始时刻差1秒。因此仅比较 `scene_id` 或 `acquisition_start_key` 会把这里的同次采集拆开；`start`/日期也不能替代采集键。反过来，同datatake仅表示共享采集机会，**不保证同船、同观测秒或相同像素**；不同产品的中心x/y不在同一网格。

严格同船同次观测应复用 `A/oss2_grdh_slc_pairs.csv`：先按 `identity+datatake_key` 要求恰好1条GRDH和1条SLC，再要求 `same_sar_datetime`，性能配对还需 `same_label`。已有370个唯一产品配对中，仅234对SAR时间一致，其中233对标签一致；6个非唯一组另存 `oss2_grdh_slc_ambiguous_groups.csv`，不能任选记录。标签冲突保留为数据问题，不自动重标。

比较时保留原完整OSS2的支持和模型，仅对查询按产品分层，优先单列GRDH；另一个 `oss2_grdh` 子任务有不同支持/拟合，不能冒充原任务的单纯查询切片。排除支持样本本身也不等于排除支持船只：既有审计发现每draw有50–64条query与support共享船只，其中11–17条共享船只及datatake。

## 冻结支持集上的可行性

定义：对每draw，query限于原 `eval_masks`；其组键若存在于该draw的64个support中，即“已见”。支持集混合多个场景，故下表不是全支持同场景与全支持跨场景的两种受控任务。

| draw | 支持产品数/采集数 | 同产品query | 未见产品query | 同采集query | 未见采集query |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 25/22 | 2853 | 828 | 3064 | 617 |
| 1 | 26/22 | 2929 | 752 | 3119 | 562 |
| 2 | 29/23 | 3381 | 300 | 3381 | 300 |
| 3 | 29/26 | 3263 | 418 | 3538 | 143 |
| 4 | 26/22 | 2919 | 762 | 3114 | 567 |

每draw总query为3681；上述两种分组的两侧均覆盖8类，但不能据此认为类分母充足。例如draw3未见采集query中渔船、拖船各仅1条，原油船、海工船各3条。应随读数保留类/船/采集分母与产品组成。

同一query在不同draw中两种状态均出现的样本：产品分组1982条/1532船，采集分组1518条/1287船，均覆盖8类。这允许后续从原预测做同样本跨draw的描述性配对；支持集的类别、船只和其余成员也同时改变，不能解释成仅由“同场景支持”造成的收益。五draw不是五个独立数据重复。本文只统计元数据和掩码，不读出新效果。

**可直接做：** 保持主模型 `dual_refit_late` 与原支持不变，按采集已见/未见分层读取现有预测，以产品场景分组和同支持船只排除作解释边界。**现有缓存不能提供：** 固定同一query、同港、同B且类别组成匹配，仅改变支持场景的受控对照；那需要新的支持组织/预测，超出本任务。

## 可复用证据

- `A/DATASET_AUDIT_ZH.md`、`support_query_overlap.csv`、`PAIRS_COMPLETE.json`：元数据恢复、身份重叠、同采集产品配对与分母；原234对的逐draw两端query资格在 `oss2_grdh_slc_pair_query_eligibility.csv`，使用同标签233对时还需该条件，勿直接照搬234对分母。
- `I/product_pairs_frozen.csv`：233个同标签、同船同SAR时间GRDH/SLC配对，232船、仅3个datatake；字段 `pair_id,identity,datatake_key,sample_id_grdh,sample_id_slc`。`I/product_cohort.npz` 含 `ids,support,eval_masks,grdh,slc`，后两者是 `ids` 的位置。`product_scene_summary.csv` / `product_scene_contexts.csv` 中的场景读数对应本研究datatake分组，不能当233个独立场景。
- `I/acquisition_pairs.csv`：`product,identity,class_id,sample_id_a/sample_id_b,datatake_key_a/datatake_key_b,sar_datetime_a/sar_datetime_b,interval_days`；`row_a/row_b` 是成像研究缓存位置，跨资产以ID连接。既有GRDH不同采集相邻观测498对/366船/26datatake；SLC53对/52船/10datatake。分母和边界见 `acquisition_coverage.json`、`acquisition_FINDINGS.md`，不是同港受控证据。
- `I/FINDINGS_ZH.md` 与 `acquisition_FINDINGS.md` 已保留方向边界：同时间产品配对中T相对R的翻转减少，但跨采集GRDH中T的分类改善伴随翻转增加。不能把既有证据改写成普遍条件不变性，也不能把R/T专家结果当完整冻结主模型结果。本次不重算这些指标。

## F0921：仅顺手确认接口

冻结缓存 `B/targets/f0921/f0921/predictions.npz` 有123489个 `ids`、5×64支持、5×123489查询掩码，预测轴为5×10×123489，模式结构同OSS2。

- `E:/Hermes/paper_tracking/sensitivity/f0921_transfer_20261007/target_manifest.csv`：`sample_id,scene,mmsi,class_id,nearest24,dist24_km,in_cluster,lon,lat,target_manifest_row`。`scene` 示例为 `SAR_Ship_Chip_20210526_UTM04N_EPSG32604_10m`，是日期/UTM裁片集合名，不能直接当Sentinel采集键；`nearest24` 不能直接当真实所在港，首条到最近港距离4201.2km。
- `E:/Hermes/paper_tracking/sensitivity/f0921_incidence_recovery_20261010/f0921_incidence.csv`：按 `sample_id` 连接，字段 `scene,product,status,acquisition_candidates,old_tile_product,differs_from_tile_first,metadata_file,method`；已恢复完整产品名可提供采集分组来源，不应一律采用 `old_tile_product`。同目录 `FINDINGS_ZH.md` 记录139条 `ambiguous_acquisition`，不可凭行号任选采集；角度缺失与采集键缺失也不可混为一谈。

F0921本次只读表头、首行、现有恢复报告与NPZ结构，未扫描约58MB逐样本恢复表，也未验证全队列连接、分组覆盖或支持/query可行性。暂不把它作为已就绪的采集对照；OSS2接口和分母已足够作为当前24港主线的外部补充。
