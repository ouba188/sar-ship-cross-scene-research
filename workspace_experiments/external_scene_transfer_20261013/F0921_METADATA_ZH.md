# F0921 采集元数据与容量上限

日期：2026-10-13。仅元数据准备；没有抽样、拟合、训练、预测或特征提取。

## 结果与连接

原 manifest 共 123,489 行，按 sample_id 一对一连接，原顺序完整保留。全部 123,489 行具有格式有效船舶 MMSI，共 15,550 个不同身份。
唯一恢复采集 123,350 行，覆盖 5,646 次采集、18,107 个产品；139 行采集歧义保留空值。
输出字段：sample_id, class_id, class_name, identity, known_ship, product, acquisition, acquisition_known, scene, status。

## 状态语义

- acquisition = platform|absoluteorbit|datatake，保留轨道前导零；只解析实际唯一恢复产品。
- scene 原样保留日期/UTM集合名，不能当作采集或港口。统计中的“每景”以下均指 acquisition。
- status 原样保留角度恢复状态。outside_grid=389 与 missing_remote_grid=11 均有已知采集，角度缺失不等于采集未知。
- ambiguous_acquisition=139，均有两个候选；不使用 old_tile_product，不默认分配产品。
- known_ship 仅表示有效格式身份；不表示标签一致，也不是注册数据库核验。identity 仅接受 [2-7][0-9]{8}；本次两数据源的有效身份与项目旧 [1-9] 口径完全一致。
- status/method 与原恢复脚本逐项核对，组合计数见 JSON 的 status_method_semantics。

## 每类覆盖

| 类别 | 样本 | 不同船 | 已知采集样本 | 采集数 | 歧义样本 | 冲突身份样本 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 bulk_carrier | 28880 | 6251 | 28856 | 4010 | 24 | 180 |
| 1 fishing_vessel | 3800 | 289 | 3794 | 1345 | 6 | 12 |
| 2 general_cargo | 7564 | 1289 | 7555 | 2749 | 9 | 50 |
| 3 product_chemical_tanker | 23229 | 2412 | 23203 | 4212 | 26 | 88 |
| 4 container_ship | 30638 | 2648 | 30602 | 4026 | 36 | 160 |
| 5 crude_oil_tanker | 13947 | 1610 | 13929 | 3037 | 18 | 58 |
| 6 tug_towing | 13357 | 940 | 13348 | 3367 | 9 | 0 |
| 7 offshore_supply | 2074 | 162 | 2063 | 697 | 11 | 0 |

## 同景容量及 B16/B64

容量统计剔除 F0921 内部多类身份，保留 122,803 行、15,496 个不同船。原 CSV 不删行、不改标签。
每个采集的样本数、不同船数、类别船数、产品数、其他采集船容量与匹配上限，完整保存在 JSON acquisition_inventory；原 scene 集合的容量另存 original_scene_inventory。
每类匹配容量 = min(同采集不同船数, 其他采集不同船数)，再跨类求和；至少两类可用。不预留查询船时是最宽松上限；另报告预留至少5船，以及 max(5,ceil(30%船数)) 的上限。
这些仅为必要条件：尚未按实际查询身份从两侧同时排除，尚未分配查询类别、固定支持配额或检查特征可用性。因此不能把下表称为已确认可执行场景数。

| B | 未留查询的采集数上限 | 至少留5船上限 | 留30%且至少5船上限 |
| --- | ---: | ---: | ---: |
| B16 | 2460 | 1992 | 1847 |
| B64 | 415 | 333 | 105 |

去冲突后每采集不同船数：中位数 12，P90 57，P95 71，最大 153。完整分位数及逐景清单见 JSON。
每类船容量不可把同船多次观测相加；同船跨采集仍可重复出现，各景容量之和不是独立船总数。容量尚未排除与24港重叠身份，若协议要求排除，实际容量只能更小或相等。

## 身份标签冲突与24港交集

F0921 内部多类别 MMSI：51 船、548 行；24港内部：0 船、0 行。原始类别均保留。
24港清单 38,091 行、24 港、18,766 个已知船，198 行身份未知。两数据集共享 4,188 个已知 MMSI，对应 F0921 32,762 行、24港 7,464 行。
交集中类别集合不同 295 船，合并后具有多类别 295 船。完整身份、两侧标签/行数和港口存于 ports24_overlap.records；内部冲突清单存于 identity_conflicts。
声称跨数据集船只隔离时，主实验须从实际源拟合数据排除共享 MMSI；仅准备元数据不会执行排除，也不会消除历史编码器预训练暴露。

## 使用边界

F0921 已被历史开发使用，且有无标签编码器预训练暴露记录，不能称为全新独立确认；公开发布与许可状态在项目记录中未确立。nearest24 不能作为真实港口。
同一次采集不保证相同入射角、背景或港口；本次没有角度匹配或物理原因识别。这里只报告采集归属与数据容量限制，不将旧输入管线问题重述为领域研究原因。
JSON 保留所有统计口径、逐采集清单、身份冲突和交集。布尔列采用 True/False；读取 CSV 建议 identity/sample_id/acquisition 按字符串、keep_default_na=False，空身份不得填成伪船只。

## 来源与复现

- target_manifest: `E:\Hermes\paper_tracking\sensitivity\f0921_transfer_20261007\target_manifest.csv`
- incidence: `E:\Hermes\paper_tracking\sensitivity\f0921_incidence_recovery_20261010\f0921_incidence.csv`
- ports24_manifest: `E:\临时会话\visual_reliable_baseline\artifacts\eight_class_adaptive_20260916\manifest.csv`
- 语义来源：oof_reliability_check/external_knowledge_metadata.py 与 recover_f0921_incidence_online.py；读取既有恢复报告，未重新恢复或下载。
- 运行：现有 ClearSAR Python，参数 -X utf8 -B，执行本目录 f0921_metadata.py。
