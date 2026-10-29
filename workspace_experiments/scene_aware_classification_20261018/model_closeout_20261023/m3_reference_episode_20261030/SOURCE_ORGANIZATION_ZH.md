# M3源参考组织事实说明

2026-10-30，限界只读支线。已读 codex-brain/INDEX.md、04-workflows.md 的模块组合、数据层面原因与分组规则；核对下述源清单、既有结果及实现。未加载模型、使用GPU、读取目标标签选样、运行全量审计、新建聊天或操作Git；仅新增本文。训练、原图案例及实验实施由主代理负责。

路径约定：W=`E:/临时会话`；R=`W/scene_aware_classification_20261018/model_closeout_20261023`；P=`W/sar_m123_standalone/sar_m123`。自建源清单位于 `W/sar_m123_current_20261027/manifests`，OSS2源清单位于 `R/final_oss2_replication_20261029/manifests`。

## 结论与合法范围

- 对当前 **source_adapt 查询**，保持既有协议的参考候选池是该数据集的全部 **source_fit**：自建1205条，OSS2 478条，不限于原16/32条。两套 fit 与 adapt 的 **ID、MMSI、pass_group 三种交集均为0**；所以任取 fit 子集已自然满足与 adapt 查询异船、异过境。这里的自建是24港来源的既有八分类任务，不是F_0921；OSS2是既有四分类任务，不是八类或跨数据集迁移。
- `P/current/data.py:67` 的 `pair_mask` 只排除同ID和同MMSI，**不排除同pass_group/同港**。`P/selective_backbone.py:87` 再叠加查询与参考的极化可用性，`P/selective_matching.py:67` 再叠加双方槽位正质量。不能把“身份合法”写成“所有极化/槽位都有证据”。
- 分组划分与前向掩码是两层规则：既有 fit/adapt 分组已经隔开船与过境，但不能由此声称 `pair_mask` 实现了跨过境排除。如果以后改为 **fit内部查询/参考 episode**，同船仍必须排除；若该实验声明跨过境 episode，还需在组织阶段排除查询 pass_group。该附加条件对本次 adapt→fit 池没有额外影响。
- 当前 `probe.py:main` 仅从 fit 选参考，且断言参考ID属于 fit。源标签可以组织类均衡候选；不能用开发/目标查询真值决定该查询用哪个类别的参考。adapt标签用于既有监督训练合法，但把adapt并入参考池属于另一个训练组织方案，并非当前已验证的参考协议。validation/development/selection/evaluation也不应被顺手并入该池。

## 现有清单规模与字段

下表本次直接解析六份源JSON清单计数，未扫描原图或特征。船=MMSI，过境=pass_group；极化列为“VV+VH / 仅VV / 仅VH”。

| 数据与清单 | 观测 / 船 / 过境 | 极化 | 各类观测数，按类号递增 |
|---|---:|---:|---|
| 自建 source_fit | 1205 / 1004 / 88 | 1036 / 169 / 0 | 367,285,193,170,69,66,34,21 |
| 自建 source_adapt | 136 / 128 / 20 | 128 / 8 / 0 | 54,4,26,28,6,12,3,3 |
| 自建 references | 16 / 16 / 13 | 15 / 1 / 0 | 每类2 |
| OSS2 source_fit | 478 / 448 / 9 | 444 / 11 / 23 | 138,47,95,198 |
| OSS2 source_adapt | 241 / 235 / 4 | 227 / 5 / 9 | 83,26,46,86 |
| OSS2 references | 32 / 32 / 9 | 27 / 2 / 3 | 每类8 |

自建字段为 `id,mmsi,pass_group,port,label,split,vv,vh`。类号0..7依次为 bulk_carrier、fishing、general_cargo、chemical_tanker、container、crude_oil_tanker、tug、offshore。

OSS2字段为 `id,label,class_name,original_class_id,mmsi,pass_group,scene_id,role,split,input_scale,canvas_size,indices,source_paths,native_size,vv,vh`，无独立port字段。类号0..3为 bulk_carrier、general_cargo、product_chemical_tanker、container_ship，原八类号为0/2/3/4。`indices` 内含 `legacy_row_index,target_manifest_row,manifest_position,obs_id,table_index,row_index,up_index`，缓存定位使用 **table_index**，不能拿其他编号替代。

## Bank overlap是不是实际问题

**当前没有 adapt-query 与 source-bank 重合泄漏的证据。** 原库全部是 fit 子集：自建16/16个参考ID、OSS2 32/32个参考ID在fit中；原库与adapt的ID/MMSI/pass_group交集全部为0。源训练数据成为源参考本身合法，不是新的独立监督证据。

fit作为查询时确会遇到自身/同船参考，但前述 pair_mask 会屏蔽，不能把“银行属于fit”本身认定为漏洞。不同episode银行复用同一源船同样不构成查询泄漏，也不应算新增独立船。固定银行可能限制训练暴露的多样性，这是待检验的组织假设，不是已证失败根因。

确实存在的局部问题是 **probe的覆盖距离统计没有排除自身/同船**：`select/coverage` 中的 nearest distance 可含零距离，而实际前向会屏蔽这些配对。因此源描述覆盖下降不等于合法异船关系改善。`select` 对每类选中后禁选同MMSI，保留全部观测作为候选；这与先按船保留首行不同。

已完成 SOURCE_COVERAGE 的可复用事实：自建general_cargo原库VH仅1条；OSS2每类8参考的过境覆盖为3/7/3/7，源候选每类都有9过境。OSS2 fit合法同类参考每极化至少6个，不能解释成普遍没有同类参考。不要为本轮重复全池覆盖审计。

## 缓存接入边界

- `R/m3_reference_coverage_20261030/probe.py:PrivateCache` 调 `R/spatial_m12_refit_20261029/run.py:load`，只载fit/development/references。私有ID必须通过 `R/backbone_m2_pilot_20261026/table.csv` 的obs_id映射到CSV行位置。此次对照 `cache/complete.json.original_indices`：1365个已写行中，fit为1205/1205，refs为16/16，adapt为 **0/136**；adapt136条虽全部能映射原表，却不能直接读取旧精确prefix行。`R/m3_reference_delta_20261030/run.py:add_adapt` 已有原生输入计算prefix、追加GPU缓存并建立负索引映射的路径；本支线未调用。
- OSS2 `R/m3_reference_delta_20261030/public.py:Cache` 按传入groups收集记录，再用 `indices.table_index` 读取现成缓存。fit索引0..477，adapt706..946。`probe.py` 原调用仅传fit/validation/references，若要取adapt必须在实例化时包含adapt；原 `public.py:main` 已这样做。Cache本身会分配CUDA张量，不能当只读CPU清单工具导入运行。
- 缓存是冻结layer2 prefix及输入几何等，不是冻结的最终M3参考值。`P/current/model.py:reference_bank` 与 `P/selective_backbone.py:forward_from` 表明，学习后的参考路径仍在前向中计算。固定参考身份不等于值映射没有参与学习。

## 已做实验，避免重复

1. **冻结完整模型直接换银行，seed17已做。** `R/m3_reference_coverage_20261030/anchor_corrected/{private,oss2}/results_s17.json` 是修正锚点后的依据。原库、同量随机/coverage、双量随机/coverage；随机seed1030，coverage按源M3五描述流做MMSI倒数加权的贪心覆盖，各类选船不重复。自建原16条OA/BA=45.625/34.9281；random_2相同，coverage_2与coverage_4均45.000/34.7240，random_4为45.000/34.2788。OSS2原32条与random_8/coverage_8/random_16/coverage_16全部57.8947/50.1235，报告确认top1均未变。查询分别为开发160与源验证228。**这是静态换库，不是逐episode重采样训练。** 根目录private/oss2旧结果曾误用live layer3统计，应保留但不能替代anchor_corrected结论。
2. **原银行上的中心化和配套训练已做。** coverage轮包括直接中心化、同预算M3值路径续训和三顺序完整评价；完整范围相对control自建OA/BA -0.021680/-0.065070，OSS2 +0.264550/+0.184786个百分点，未形成双数据集净收益。`m3_center_joint_20261030` 又做了从原fit端点进入原adapt流程的LIVE+M3联合训练和路径移植，也没有双数据集净收益。
3. **固定原银行的其他修法已做。** `m3_reference_delta_20261030` 的差异/总参考补充、显式源参考类别证据及joint_low；`m3_source_replay_20261030` 的adapt-only对比fit+adapt查询混合；`m3_pairwise_repair_20261030` 的CE/rank/hybrid；`m3_query_interaction_20261030` 的q/m/q*m同容量补充。均有三顺序开发或相应冻结评价记录，均未达到共同目标。source_mix改变的是训练查询来源，并未做动态参考episode；pairwise的异类/异船/异过境条件约束的是两个训练查询，不是参考bank的额外掩码。

在本次核对的代码与收据中，没有看到“adapt查询固定、fit参考按训练episode重采样，并以固定原银行评价”的已完成实验。可把它与上述静态换库、查询混合明确区分；其是否有益仍需主代理实验，本文不预判收益。

证据入口：`R/m3_reference_coverage_20261030/{probe.py,SOURCE_COVERAGE_ZH.md,FINDINGS_ZH.md}`；以上各轮同目录 `FINDINGS_ZH.md`；`R/final_oss2_replication_20261029/{protocol.py,converter.py}`；本文列出的原缓存与pair_mask/前向实现。除六份源清单和私有表索引/缓存收据的有限核对外，既有实验结论均复用已存结果，没有重算模型输出。
