# 客观元数据可用性与下一步解释边界

2026-10-13，独立只读支线核查。分母为本轮去重query，不累计支持抽样/context重复。未训练、未下载、未扫描图像或修改侧线文件。

| 范围 | 已核验可用性 |
|---|---|
| source FIT / HELD | 入射角152/242、269/418；同产品AIS尺寸按既有attribute_valid规则186/242、352/418 |
| 洛杉矶 / 巴生港 / 桑托斯 | 入射角286/318、2992/3118、1099/1357；按MMSI连接的历史AIS众数正长宽306/318、3052/3118、1306/1357 |
| F_0921 | len_m 123170/123170；恢复入射角122633/123170（99.56%） |
| OSS2 | XML入射角及人工SAR长度3440/3440；正AIS船长3039/3440、船宽2998/3440 |
| 既有图像描述 | 源FIT/HELD、三港、OSS2的原生宽高及对比度代理全部关联；OSS2占据率/伸长率3439/3440；此表不覆盖F |

F角度剩余缺失：388条越出网格，139条acquisition歧义，10条网格缺失。OSS2的length_m混合AIS与人工长度回填，不应把100%非零当作100%AIS真值。三港AIS众数是身份级历史摘要，不是逐次观测的尺寸真值。

## 连接约定

- query_ids是对应observations/manifest的行位置，不是target_manifest_row；源observations经manifest_row回接总manifest，instance_id一致。
- 源尺寸：`E:/临时会话/same_vessel_condition_study_20261010/knowledge_train_only_design_20261011/source_dimension_join.csv`，按obs_id连接；有效性用相邻研究`knowledge_train_only_20261011/attribute_mask.csv`。
- 三港角度：`E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv`，按instance_id+product+port连接，仅status=ok有效。
- 三港历史尺寸：`E:/Hermes/paper_tracking/port_ais_2024_2025_from_zip/dedup_tables/vessels_to_lookup_dedup.csv`，按MMSI连接。
- F角度：`E:/Hermes/paper_tracking/sensitivity/f0921_incidence_recovery_20261010/f0921_incidence.csv`，按sample_id连接。
- OSS2角度/产品/尺寸：`E:/临时会话/ports_root_cause_20261013/dataset_audit/public_metadata.csv`，按dataset+sample_id连接。
- 图像代理：`E:/临时会话/ports_public_data_study_20261012/audit/metadata.csv`，按dataset+sample_id连接，源sample_id对应instance_id。

## 已知差异，不等于根因

- AIS众数船长中位数：源FIT/HELD均172m，洛杉矶228m、巴生180m、桑托斯225m。图像对比度代理中位数依次为0.198/0.176、0.317、0.197、0.265，OSS2为0.041。整体差异受类别/数据来源混杂，代理不是物理散射量；本轮未完成类内尺度/对比度与选择错误关联。
- 已知入射角中位数：FIT40.85度、HELD40.20度、巴生42.77度、桑托斯39.24度；桑托斯P5–P95为34.10–39.93度，FIT为31.25–44.70度。更像范围内的分布重排，不是简单超出训练范围；源缺失约36%–37%，后续必须保留缺失组。
- OSS2 query有915/3440个SLC，全部接上`oss2_slc_native_grid.csv`，range压缩比中位0.249；源和三港为GRDH。本轮已进一步完成冻结产品分层及192对同船同时间同标签比较，见主报告和product_pairs.csv。不能将旧B64几何干预效果挪作本轮B16选择器证据。

源FIT/HELD这些整体尺度、角度和显示统计相近，不能单独解释contextual从FIT到HELD的退化。后续需要保持类别/船只等条件可比，避免仅凭边际分布不同宣布根因。
