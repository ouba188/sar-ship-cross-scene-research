# 训练专用静态知识可用性审查

日期：2026-10-11。本轮只读核查，不启动训练、不下载、不修改主实验和 memory。

## 结论先行

**有可追溯到 AIS 导出记录的船长、船宽，可以与当前源训练样本联结；不是只有检测框尺寸。首轮尺寸监督具备数据基础。** 但应称为“有来源的 AIS 报告尺寸”，不能把所有非空数字都说成独立核实的真实船体尺寸。原始记录确有异常值、同船冲突及对象 JSON 与跨时段众数不一致。

- 当前训练集 1296 次观测、525 个 MMSI：1244 次/509 艘可获得本轮所读 AIS 记录中唯一的正数长宽对；1140 次/509 艘可获得同产品下唯一的正数长宽对。二者交集 1104 次/496 艘。
- 留出集 823 次观测、302 个 MMSI：对应为 782 次/288 艘、730 次/296 艘；交集 699 次/285 艘。
- “唯一”只描述已读记录的一致性，不保证尺寸数值正确。不能把这些覆盖直接称作最终合格监督覆盖。
- 可以沿用户要求设计“训练时尺寸监督、预测只看 SAR”；留出船知识仅供审查/事后诊断，不进入训练、模型选择或预测输入。
- 已核实返回源栅格的地理采样间距全部为 10×10 米；当前 fit/held 可联结 541/329 次观测。它是真实投影栅格采样间距，不是 SAR 分辨能力，不是检测框大小，也不是 resize 后网络像素间距。本轮没有找到可用于比较的该字段变化，先做几何诊断即可。

## 审查范围与联结基准

基准表：`E:/临时会话/same_vessel_condition_study_20261010/observations.csv`，37893 行、18766 个 MMSI。

原 manifest：`E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv`，38091 行。

分组：`E:/临时会话/same_vessel_condition_study_20261010/incidence_transport_20261011/source_cohorts.npz` 的 `fit_obs_ids`、`held_obs_ids`。

已检查 `observations.manifest_row` 索引回原 manifest 后，`instance_id/mmsi/product/port` 四字段逐行相等；fit/held MMSI 无交集。不能拿 `obs_id` 直接当 manifest 行号。

只使用本地已有清单定位文件。原始 AIS 数据库以 MMSI 主键查询 827 艘源船，共读取 75760 条相关记录；另读取清单中与源 cohort 产品有关的 130 个本地同步 AIS TXT（清单计约 91 MB，含可能重复来源）。没有扫描大图、全盘或全 AIS 数据库。全 37893 行只做了轻量众数表和地理头联结，精确对象及原始 AIS 核对限于 fit+held。

## 来源一：AIS 记录与船级汇总

### 可核实的路径和字段

1. `E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/vessels_to_lookup_dedup.csv`
   - 一行一 MMSI，字段 `mmsi/length_mode/beam_mode/local_label_source/provenance_note`。
   - `length_mode/beam_mode` 是该船各自尺寸字段的众数，不是船型类别均值。两个独立众数也不保证它们曾在同一条记录里共同出现。
   - `master_type/master_source/master_match_confidence` 等列表示外部主数据仍待查，不能当成已获权威船籍核验。
2. `E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/ais_points_dedup.sqlite`
   - `points` 表：`mmsi/epoch/utc/product/ports/length/beam/source`；主键包含 MMSI。
   - 本轮以只读连接按 MMSI 查询，保留出现过的正数 `(length, beam)` 联合值，不分别拼众数。
   - 同产品联结键是 `(mmsi, product)`；这里只验证产品来源对应，不声称已重新校核 AIS 定位、匹配质量或每条记录时间差。
3. `E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/ais_txt_file_manifest.csv`
   - `relative_path/product/ports/bytes/row_count` 可指回 `products_by_primary_port/.../*_AIS_pm3min.txt`。
4. `E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1/private/server*_file_manifest.csv`
   - 以 `kind=ais`、`status in {downloaded,reused_local}`、`product` 定位已有 `local_path`。
   - 已读路径见本目录 `read_source_txt_paths.json`。

### 字段语义来源

`E:/Hermes/paper_tracking/build_port_ais_dedup_tables.py` 读取原始 18 列 AIS TXT：0-based 第 14/15 列分别为 length/beam，然后写入数据库与众数表。

`E:/临时会话/visual_reliable_baseline/filter_safe841_offshore_mmsi_20260917.py` 也将同两列命名为 `length_m/beam_m`。因此“米”有既有项目消费者和对象元数据语义支持；本轮没有取得原始供应商字段规范、船籍证书或独立测量来逐船证明准确性。

本轮未发现这些尺寸由检测框或类别均值生成的证据；它们可回溯到 AIS 导出记录。**“记录来源真实”与“每个物理数值准确”必须分开。**

## 来源二：返回裁片对象 JSON

清单根目录：`E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_return_20260915/metadata/local_knowledge_sync_v1/`。

联结链：`manifest.source_json_vv/vh -> private/server*_file_manifest.csv.remote_path -> local_path -> objects[].obj_uid/source_obj_uid`，用 `manifest.obj_uid_vv/vh` 精确选择对象；再检查 `meta.mmsi` 等于 observations 的 MMSI。

字段：`objects[].meta.length_m/beam_m/mmsi/source/ais_time_utc/abs_time_delta_seconds`。本轮匹配的 1719 个极化视图全部 `source=raw_ais_review_only`，MMSI 不一致数为 0。不要读同对象的 `cxcywh_angle` 当船体真尺寸。

当前 cohort 有 79 个视图的旧 `source_object_index` 与实际 UID 所在索引不一致，故不能只按对象下标提取。本轮使用唯一 UID，未因此读错对象。

- 正数 JSON 长宽覆盖 fit 541 次/280 艘、held 329 次/169 艘。
- 其中 fit 540 次、held 329 次的 JSON 长宽对，能在同 MMSI、同 product 的已读原始 AIS 中找回，且同观测两个极化无尺寸冲突。
- 唯一未得到同产品原始佐证的 JSON 样本为 `obs_id=2189`、MMSI `236590000`、产品 `S1A_IW_GRDH_1SDV_20250316T062756_20250316T062821_058326_0735E5_8376`，JSON 为 102×18。此项是本轮未核实，不是已证伪。

逐视图值与本地 JSON 路径保存在 `object_dimension_audit.csv`。这条路径覆盖较少，不能只用 JSON 非空作为全训练集尺寸可用性的上限。

## 覆盖结果

每格为“观测数 / 至少一条满足条件的 MMSI 数”。同产品可用身份数并不表示该船所有观测都有同产品知识。

| 已验证条件 | fit：1296 / 525 | held：823 / 302 |
|---|---:|---:|
| 船级众数长宽均为正数 | 1257 / 509 | 789 / 290 |
| 已读原始 AIS 至少一个正数长宽对 | 1285 / 524 | 813 / 299 |
| 已读该 MMSI 原始记录只有一个正数长宽对 | 1244 / 509 | 782 / 288 |
| 同 MMSI、同产品至少一个正数长宽对 | 1141 / 510 | 730 / 296 |
| 同 MMSI、同产品只有一个正数长宽对 | 1140 / 509 | 730 / 296 |
| 同产品唯一且已读跨产品也唯一 | 1104 / 496 | 699 / 285 |
| UID 联结 JSON 有正数尺寸 | 541 / 280 | 329 / 169 |
| JSON 尺寸可在同产品 AIS 找回且极化一致 | 540 / 280 | 329 / 169 |
| 已读同 MMSI 有多个正数长宽对 | 41 / 15 | 31 / 11 |
| JSON 与该 MMSI 全时段众数不一致 | 24 / 10 | 25 / 10 |
| 返回 TIFF 头可联结投影采样间距 | 541 / 280 | 329 / 169 |
| 已有有效入射角 | 855 / 410 | 523 / 254 |
| 已有 relative_signed_deg | 103 / 77 | 73 / 56 |

全观测表的众数长宽正数覆盖 36942/37893 次、18087/18766 艘；可联结返回 TIFF 头覆盖 16642 次、9154 艘。未对全表进行原始尺寸逐条验证。

### 分类别覆盖

下表使用“已读同 MMSI 仅一个正数长宽对”，仍非准确性认证。

| 类别 | fit可用/总观测 | held可用/总观测 |
|---|---:|---:|
| 散货船 | 376/376 | 266/266 |
| 集装箱船 | 71/71 | 45/46 |
| 原油轮 | 39/40 | 27/27 |
| 渔船 | 177/197 | 126/136 |
| 普通货船 | 329/331 | 200/206 |
| 海工补给船 | 57/61 | 31/41 |
| 成品油/化学品船 | 136/141 | 35/35 |
| 拖轮 | 59/79 | 52/66 |

知识缺失/冲突不是各类均匀的。不能只在有知识样本上报告最终 SAR 分类结果；训练损失可遮罩，但评估仍保留原全 cohort。

## 必须保留的质量边界

1. 唯一记录也可能错误。训练船唯一长宽对中，有 MMSI `416002667` 的 5×24、`525300032` 的 50×50；最大报告船长达到 1022，最大长宽比 170.33。held 最大长度 888、最大比值 59.2。不能直接作为无噪声回归真值。
2. 存在实质冲突，不只是舍入。例如同 MMSI 众数与 JSON 出现 26×8 对 228×32、179×32 对 359×64、8×30 对 298×47。详细冲突见 `dimension_conflicts.json` 和联结 CSV。本轮不擅自选择较大值或相信 JSON 一定胜过原 AIS。
3. `dimension_review_flags.csv` 仅用 `L<=B`、`L>=500` 或 `L/B>30` 把明显需人工查看的记录列出，未据此删除数据或宣布其他记录准确；这些不是已验证的船型物理规则。
4. 众数表覆盖跨时段 AIS；不能在严格时间预测实验中无说明地用未来记录。首轮若以新船识别为目标，可明确尺寸作为源训练船静态附加标签；若要求严格时点可得性，使用同产品记录并另查时间窗口。
5. 不能用目标船类别、类别均值、类别名生成尺寸来冒充知识。类别均值只应作为一个明确的源训练对照；同一原始 AIS 尺寸生成的长宽比不是第三条独立知识。
6. 本轮未核实尺寸是否参与过历史标签人工细分或裁片选样，因此不能宣称这些尺寸完全独立于数据构建和类别标注。该点不阻止训练辅助监督，但限制“独立因果参照”的论证。

## GSD/尺度信息：有真头信息，不等于原生分辨率

`.../local_knowledge_sync_v1/private/server1_georef.jsonl` 与 `server2_georef.jsonl` 的 `remote_source_tiff/status/transform/epsg/width/height` 来自既有同步脚本对远端 GDAL 头的读取；本轮未连接远端。

联结键是 manifest 的 `source_tiff_vv/vh`。已查记录是 UTM EPSG 326xx/327xx，计算每像素两轴采样长度 `hypot(gt[1],gt[4])`、`hypot(gt[2],gt[5])`，所有已验证对均为 `(10,10)` 米。

- `manifest.width_*/height_*/target_long_side_*` 与检测、裁片有关，不是真实 AIS 船长宽。
- `crop_side_*` 是裁片边长像素；本轮没把它乘 GSD 当船长，也没把它直接说成神经网络输入的物理 GSD。
- `cxcywh_angle` 是检测/裁片框，已知旧对象大量使用 axis_aligned_for_chip_crop；不是真实船体姿态。
- legacy 行往往没有返回 `source_tiff_*` 映射；本轮没有读取 legacy 全量原图头。未联结不等于原始 GSD 不存在。
- 没有把 Sentinel-1 产品名中的 IW/GRDH 或 128/224 resize 尺寸转成逐样本真实分辨率。

## 对首轮设计的直接交付

可以将“真实 AIS 来源的有噪声尺寸监督”列入四组源小模型设计，**但应在正式启动前固定尺寸可用规则**。本审查推荐优先使用同产品原始联合长宽值，单独标出跨记录冲突和明显异常，不从类别补造真实目标；不要以众数表非空就全量使用。主代理可在已有授权设计中决定采用唯一身份对还是更严格同产品子集，并保持各 arm 原分类训练样本相同。

若最后选定一组有效尺寸对，长宽比直接由同一对 `L/B` 计算。sourcefit 的均值、标准化及类别均值对照只在 sourcefit 内拟合；held 尺寸不参与。缺尺寸样本可继续普通分类训练，不必删除。SAR-only 推理的接口不应要求 MMSI、AIS、真实长度或 GSD。

本轮不评估识别收益，不证明因果因素已经存在于该尺寸监督中，也不因缺少部分元数据而认定因果因素不存在。

## 产物与复现

所有文件仅写在本目录：

- `audit_knowledge.py`：本地只读审查脚本；一次运行约 8 秒，不训练。
- `audit_summary.json`：读取范围、来源路径、实际主键查询计划、对齐和来源摘要。
- `coverage.csv`、`class_coverage.csv`：分组/分类覆盖。
- `source_dimension_join.csv`：2119 条源样本的候选尺寸值、联结及冲突标记。
- `object_dimension_audit.csv`：1719 条实际 UID 联结对象及 JSON 路径。
- `dimension_conflicts.json`、`dimension_review_flags.csv`：冲突及需查看值，不自动清洗。
- `read_source_txt_paths.json`：实际读取的已有 AIS TXT 路径。

运行：`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe E:/临时会话/same_vessel_condition_study_20261010/knowledge_train_only_design_20261011/audit_knowledge.py`。
