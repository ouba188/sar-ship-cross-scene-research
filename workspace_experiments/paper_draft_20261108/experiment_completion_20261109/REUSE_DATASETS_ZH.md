# OSS1、24港与跨数据集实验复用核查

## 2026-11-10执行状态更新

用户要求：跑过的不用重跑。以下原核查正文保留当时状态，后续执行以本节为准。

- 2026-11-10补充已完成：相邻 `dataset_completion_20261110/ROUND_FINDINGS_ZH.md` 中，当前存活算子以同一五阶段日程补OSS1四类和24港八类的Full/Plain/SE各3种子，共18次新训练。它们填补当前组合的数据集源端比较，不重复旧版实验。OSS1 Full OA/BA60.847/51.737，对Plain更高、对SE更低；24港Full46.298/38.475，对Plain和SE三个种子均更高。24港评价为3075条/1950船/81绝对轨道/18港，85是旧完整pass_group数量。当前M3同权重开关独立增量仍弱，未追加这些数据集的全套删除重训。

- 当前工作版OSS2到OSS1的921条冻结外推已经完成，不再列为待补、不重跑。实际结果见本目录 `oss1_transfer/FINDINGS_ZH.md`、预测与原图案例；排除与OSS2评价船重叠的850条为嵌套子集，亦已完成。
- 当前工作版上述外推的Full OA/BA为59.862/49.186；去M3后末阶段配套训练为59.537/48.620。它不是OSS1源端训练，也不是新分阶段训练候选的结果。
- OSS2共同起点的配套分阶段训练已在相邻 `staged_ablation_20261109/ROUND_FINDINGS_ZH.md` 中完成：Full、Plain、SE、noM1、noM2、noM3，三个训练种子，以及开发端选择。该轮无需重复。
- 24港历史完整模型、消融、调参及跨库历史结果继续复用。当前模块版本不同之处应明确标注，不将旧结果改名为新版结果，也不因版本不同就自动重跑全部历史实验。
- 后续只补论文确实需要、且当前版本尚无完成结果的关键数据集内验证。统计、逐类/场景分析与可视化优先复用已存预测；不能通过重新训练代替已有结果整理。

下文“尚未定位当前组合对此921条的预测”等句是后续完成之前的历史核查结论，不再代表当前缺项。

核查日期：2026-11-09。范围：本地报告、划分清单、summary、预测文件和版本清单的只读核查；不重训、不重跑推理、不修改正文、不推Git。本轮仅写此文件。当前OSS2代码和最关键方法对比由主代理负责，本文不重复审查。

## 结论

**已有实验并不只限于当前OSS2：OSS1四类单过境复现、OSS1/OSS2双向冻结迁移、24港来源的八类跨过境完整模型与重训消融、强简单对照和源端调参均已有真实结果与预测。主要缺口是版本一致性，不是这些实验类型从未做过。**

目前查到的上述外部结果均不能直接改名为当前CRFB/RGRP/RCEW完整模型的结果。可直接引用的是各自历史版本的结论；可用于新验证的是原图、标签映射、划分及配对查询清单。当前前端改变后，历史模型分数、特征缓存和M3残差不能直接冒充新模型输出。

本轮未定位到名为 `final_oss1_replication*` 的对应目录/报告；实际找到的是 `oss1_m123_replication_20261026`。已检查model_closeout目录（含隐藏目录）、论文目录、记忆索引及相关文件名。该结论仅表示本次范围内未定位到，**不等于断言从未运行过其他OSS1实验**。

## 1. 当前版本参照与兼容性口径

版本依据：`E:/临时会话/m1_guided_m2_20261106/current_version.json`，直接读取清单，没有分析当前OSS2实现代码。

- 当前为 `m1_guided_m2_joint_head_20261106`，`arm=class`、`freeze_head=false`，默认seed17，保留17/29/43；源719、四类、32源参考。
- 上游为 `E:/临时会话/m1_coverage_20261101/small_class_s17/final.pt`；当前末端训练180步，分类头学习率1e-5。
- M3为 `class_evidence_aggregate` / `MaskedCostLocalMatch` / `arm=aggregate`，权重 `E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/aggregate_17/last.pt`；`frequency=false`。位于layer3之后、layer4及深层区域读出之前；本轮组合采用已有固定缓存残差，并非与新M1/M2重新联合训练。
- 清单明确是OSS2四类浅层区域读出工作版，不是八类checkpoint转换。仅名称都叫M1/M2/M3，不足以认定历史实验与当前CRFB/RGRP/RCEW相同。

下表“当前组合”严格指以上**响应引导M2 + aggregate M3**配置；“否”不否认旧模块有价值，只说明不能当作此版本的直接实验。

## 2. 实验矩阵

| 组 | 已有实际评价规模 | 模型版本与划分 | 当前组合 | 可以复用 | 真正缺项/边界 |
|---|---|---|---|---|---|
| A OSS1四类复现，10-26 | 63观测/63船/1过境；全stack866 | 冻结ImageNet R50、native160/C4、A/D/T+分阶段M1/M2+旧both/B/空间/频域M3；五角色船/过境隔离 | 否 | 完整历史四臂表、累计前端结果、参考与查询清单、原图案例 | 尚未找到当前组合的OSS1复现；63条不能代表多采集或全OSS1 |
| B OSS2早期四类复现，10-26 | 378/369船/5过境；stack1519 | 与A同一历史算法族，源端重建；历史fold0 | 否 | 双向迁移的真实源模型出处、旧M3对比与同队列基线 | 不是10-29 final，更不是11-06当前版本；不得混成同一full |
| C OSS2 final复现，10-29 | 378/369船/5过境，4类 | 骨干内空间M1/M2+原空间—频域M3；full/plain各3顺序，fit160+adapt96步 | 否 | 同协议R50/full历史对比，真实预测、时间线、类别/过境表 | 可作历史R50参照；当前末端额外训练机会不同，不能直接当纯等预算模块归因 |
| D 24港来源 final消融，10-29 | 3075/1950船/85过境/18港，8类；开发160 | 空间M1/M2+原M3；plain/no_m1/no_m2/no_m3重训，full/control复用；fit408+adapt54 | 否 | 六臂表、完整流程删除消融、源拟合/适配分开分析、分组不确定性及案例 | 当前组合八类验证未在此处找到；不是24折LOPO，不能把旧no_m1/no_m2/no_m3改名为当前消融 |
| E 24港aligned与tuning，10-26 | 同一3075扩围，另485开发，8类 | 冻结R50/C4；raw_summary+PairMetric+配套极化读出；64源参考；源端选参 | 否 | 强简单MLP/ridge、固定槽余弦、未调/已调M3、负结果及适用边界 | 与D训练流程/参考数不同，不能拿数值差当单模块贡献；也不是当前aggregate |
| F OSS1→OSS2冻结迁移，10-26 | 1619/1347船/27过境，4类 | A的源模型，排除全部源监督角色的同船/过境；目标不训练；已有标准化修复完整重放 | 否 | 真正双向迁移之一；修复后四臂预测、目标清单和分组表 | 当前算法的正向迁移需先有当前算法OSS1源模型；旧缓存不兼容当前前端 |
| G OSS2→OSS1冻结迁移，10-26 | 921/742船/21过境，4类 | B的源模型；同样冻结与排除；修复后分数不变 | 否 | 可直接复用的外部查询协议与输入清单；历史负结果；当前OSS2外推的优先候选队列 | 尚未定位当前组合对此921条的预测；不能称现成的当前RCEW外测 |
| H 更早24港→F/OSS2，10-14 | F4654、OSS2 811；8类 | R50/R18/源域SSL R18+PCA/ridge，B0/16/64目标支持，五抽样 | 否 | 证明更早跨数据集与支持预算实验确已做过；历史协议/数据资产 | 架构和目标标签设置都不同，不应塞入当前无目标支持完整模型主表 |

四类顺序：散货、普通货、化学品/成品油轮、集装箱（旧八类ID `[0,2,3,4]` 映射为0至3）。八类顺序：散货、渔船、普通货、化学品/成品油轮、集装箱、原油轮、拖轮/拖带、海工补给。

## 3. A组：OSS1真正已有的复现

真实目录：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/`。

- 报告：该目录 `FINDINGS_ZH.md`、`DATA_PROTOCOL_ZH.md`。
- 划分：`STACK_SPLIT.json`、`stack_table.csv`、`stack_fit_train.csv`、`stack_fit_validation.csv`、`stack_utility_train.csv`、`stack_selection.csv`、`stack_evaluation.csv`、`stack_indices.npz`。
- 最终summary：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/m3_results_grid01/summary.csv`。
- 实际预测：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss1_m123_replication_20261026/m3_results_grid01/seed17/predictions.npz`，同级seed29/seed43也存在；本轮读取seed17数组确认 `old_both/B_fixed/spatial/frequency` 均为 `(866,4)`。

五角色观测数405/89/122/187/63，总866观测、829船。源fit405条/384船/9过境；源参考32，每类8船。评价仅2016-03-28历史过境 `S1A_010568_00FB64`，类别数15/7/10/31。**预测包的866不是测试数，必须按原evaluation索引取63行。**

可复用数值（OA/BA，%）：旧both64.5503/58.7250；B固定61.9048/56.5297；B+空间相同；B+空间—频域62.4339/57.3630。累计A/D/T54.4974/50.5338，+M1为56.6138/51.1111，+M2为58.2011/54.2729。累计结果不是配套从头删模块的消融。

有效用途：历史公开复现、旧后端取舍与一例频域门控净救回的边界；不能用于证明当前RGRP响应引导或当前aggregate RCEW有效。`m3_results/` 是误用0.05网格后中止的历史运行，最终口径为 `m3_results_grid01/`，不能挑旧目录更好结果。报告已记录早期目标暴露，不是新盲测。

## 4. B/C组：两代OSS2结果必须分开

### B：10-26分阶段算法族

真实目录：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss2_m123_replication_20261026/`。

报告 `FINDINGS_ZH.md`；汇总 `m3_results/summary.csv`；预测实路径为 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/oss2_m123_replication_20261026/m3_results/seed17/predictions.npz`，另有29/43。

stack1519观测/1468船，fit478、validation228、utility241、selection194、evaluation378；源参考32。正式evaluation四类132/26/83/137、5过境。旧both57.7601/53.3115；B固定59.6120/54.7284；空间相同；频域59.5238/54.6652。它是双向迁移的OSS2源模型来源，不是后面的final模型。

### C：10-29骨干内final模型

真实目录：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/`。

- 报告/协议：`FINDINGS_ZH.md`、`PROTOCOL_ZH.md`。
- 汇总：`analysis/summary.json`、`analysis/groups_summary.csv`、`analysis/RESULTS.md`。
- 实际预测：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_oss2_replication_20261029/predictions/evaluation_full_adapt_s17.npz`。同目录有 `evaluation_{full,plain}_{fit,adapt}_s{17,29,43}.npz`，12个端点；本轮读取full/adapt/17确认score `(378,4)`、ids `(378,)`。

源fit478/448船/9过境，adapt241/235船/4过境；32参考；评价378/369船/5过境，348双有效极化、12仅VV有效、18仅VH有效。这里“仅某极化”指模型可用性，不等同另一极化原始文件缺失。

历史full最终61.3757/57.6629，plain55.6437/53.5378；full源拟合62.169/57.293。full对plain最终+5.7319/+4.1251个百分点，但full自身适配OA下降；目标训练不同、容量不同，收益不能全部归给干预或单一结构。三顺序共享初始化，非三独立预训练。

兼容性：此full是当前上游谱系的一部分，但缺少后续响应引导M2与当前aggregate组合。plain可保留为同队列历史基础参照，是否需要匹配当前719样本末端训练机会，由主代理的当前版本审查决定；不以“已有plain”自动认定当前公平对比已补齐。

**划分复用修正：** 后续 `E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/expanded/cohorts.json` 与 `FINDINGS_ZH.md` 记录，selection194中69条与source共享平台—轨道 `S1A_010568`，严格额外selection为125；evaluation378仍按其既定独立角色使用。早期按更细pass_group写出的隔离表述，不能自动升级为194条都满足后来绝对轨道口径。该额外队列也不能与378合并冒充新独立测试。

## 5. D组：24港来源final模块消融已有完整证据

真实目录：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/`。

- 报告：`FINDINGS_ZH.md`、`PROTOCOL_ZH.md`、`CASE_NOTES_ZH.md`。
- summary/统计：`analysis/summary.json`、`analysis/main_after_metrics.csv`、`analysis/paired_metrics.csv`、`analysis/groups_summary.csv`、`analysis/timeline_summary.csv`。
- 删除臂实路径示例：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/predictions/expanded_no_m3_after_s17.npz`；同目录覆盖development/expanded、plain/no_m1/no_m2/no_m3、before/after、17/29/43共48个新评价档案。
- full实际来自 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/spatial_m12_full_20261029/predictions/expanded_after_spatial_s17.npz`，另有29/43及before。
- full_control实际来自 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/intervention_matched_control_20261029/predictions/expanded_control_s17.npz`，另有29/43。完整输入映射见上面的summary.json；72逻辑评价端点仅66个唯一预测档案，full/control的before共用，不能重复计数。
- 扩围清单：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/ports24_aligned_20261026/bundle/target_table.csv`。

源fit1205/1004船/88过境，adapt136/128船/20过境，16源参考。开发160（仅7个实际出现类别），扩围3075/1950船/85过境/18港、8类；类别数947/102/301/803/361/256/138/167，2876双有效极化、199仅VV有效。角色间ID、MMSI、过境隔离，港口共享。开发160不是E组开发485。

| 扩围终点 | OA% | BA% | 复用解释 |
|---|---:|---:|---|
| full | 47.317 | 40.402 | 历史空间M1/M2+原M3全流程 |
| plain R50 | 37.214 | 35.553 | 相同初始化、允许更新范围和步数；不等容量/目标 |
| no_m1 | 47.068 | 40.276 | M1平均小幅收益，BA过境区间跨0 |
| no_m2 | 41.127 | 36.673 | 删除整条骨干回写及区域读出，不是只删除一次池化 |
| no_m3 | 47.176 | 40.429 | full的OA微正、BA微负，不能写M3稳定提点 |
| full_control，仅CE+KL适配 | 45.127 | 41.039 | 后续目标控制；源拟合目标及各臂教师仍不同 |

本轮实际打开no_m3与full预测，均为3075×8，ids3075。可不重训直接复用历史消融表、逐类/逐港/过境统计、源拟合与适配时间线及12查询的既有原图案例记录。它能补充历史完整流程证据，**不能成为当前CRFB/RGRP/RCEW的删模块表**。尤其不得将旧空间—频域M3消融当作已关闭显式频域的aggregate消融。

## 6. E组：24港aligned、强简单对照和调参

真实目录分别为：

- `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/ports24_aligned_20261026/`
- `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/ports24_tuning_20261026/`

两目录均有 `FINDINGS_ZH.md`、`summary.csv`、`metrics.csv`。aligned的强简单预测位于 `baselines/pooled/external_predictions.npz` 和 `baselines/fit_only/external_predictions.npz`。完整结果实路径为 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/ports24_aligned_20261026/results/seed17/both.npz`；同级 `cosine_fixed.npz`、`frozen_metric.npz`，另有29/43。

tuning的源选配置为 `selected.json`，目标预测实路径 `E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/ports24_tuning_20261026/results/seed17/primary.npz`；同级runner_up/lambda_only及29/43齐全。

源角色1205 fit、296 validation、644 utility、612 selection；开发485；扩围仍为D组3075，8类、18港、85过境，不能算另一套独立测试。64源参考、每候选类检索2例，不是64个目标标注。与D的16源参考、源适配136、开发160和骨干训练方式不同。

已核实预测取数边界：aligned `training/seed17/predictions.npz` 为3242行，是源角色及开发阶段分数；正式 `results/seed17/both.npz` 为6317×8，含3242+3075。tuning `results/seed17/primary.npz` 的score同为6317×8。必须按obs_id/target清单选扩围3075，不能把整个数组都当测试。

可复用数值（OA/BA，%）：旧完整46.1030/39.8226；未调参aligned43.8374/38.1537；源选温度0.4+配套读出45.2466/39.5092。强简单合并源区域MLP45.9512/40.0751；同预算raw+summary MLP约45.63/41.36；ridge43.2520/42.4720。不同指标/配置之间有取舍，应保留负结果。

这里已有合理源端调参：12配置×3种子（3复用、33新拟合），644训练/612源选择28过境，源端冻结主候选后才外推；不是目标表中选最好。可复用为历史强对照和调参证据，但不是当前响应引导M2或当前aggregate已完成的敏感性实验。

## 7. F/G组：真正双向跨数据集结果及修复版本

真实根目录：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/public_m123_cross_dataset_20261026/`。

原报告 `FINDINGS_ZH.md` 保留未修复数值；修复报告及正式可复用版本位于 `normalization_repair_20261026/FINDINGS_ZH.md`。每方向各有三种子17/29/43，源reader、PCA、标尺、温度、匹配、门控及32源参考冻结，无目标训练/选模。

### F：OSS1→OSS2

- 查询清单/协议：根目录 `oss1-to-oss2/target_table.csv`、`target_eligibility.csv`、`protocol.json`。
- 修复后summary：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/public_m123_cross_dataset_20261026/normalization_repair_20261026/oss1-to-oss2/summary.csv`。
- 修复后预测：同方向 `seed17/predictions.npz`，29/43也存在；本轮读取四臂均为1619×4。
- 查询1619、1347船、27过境；类别455/147/354/663。排除源fit、fit_validation、utility、selection共803条监督/选择角色的重合MMSI和过境，不只是排除32参考。
- 修复后旧both49.0014/45.3139；B固定50.4632/46.8130；空间相同；空间—频域50.5044/46.8156。B对旧both有范围内收益；频域相对B仅+0.0412/+0.0027个百分点。

### G：OSS2→OSS1

- 查询/协议：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/public_m123_cross_dataset_20261026/oss2-to-oss1/target_table.csv`、同级 `target_eligibility.csv`、`protocol.json`。
- 修复后summary：根目录 `normalization_repair_20261026/oss2-to-oss1/summary.csv`。
- 修复后预测实路径：`E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/public_m123_cross_dataset_20261026/normalization_repair_20261026/oss2-to-oss1/seed17/predictions.npz`，另有29/43；本轮读取四臂均为921×4。
- 查询921、742船、21过境；类别675/71/72/103。排除OSS2上述四个源角色共1141条的重合船只与过境。
- 三种子四臂修复前后不变：旧both61.2378/52.0941；B固定/空间61.1654/50.6127；频域61.0206/50.7004。反向BA损失真实存在，不能用正向收益代替。

修复是在源fit严格恒零的D输入维度上屏蔽，完整重算下游；无重训。这是旧A/D/T算法族的数值修复，不是当前CRFB/RGRP/RCEW的新模块。使用修复版做历史表，并保留原结果作为修复前对照，不能混用两个版本的数值。

**最直接可复用的下一验证入口是G组921条清单。** `E:/临时会话/m3_frequency_increment_20261028/source_hard_task_20261101/oss2/internal_contribution/expanded/FINDINGS_ZH.md` 第105行附近已明确：OSS1 921条外推当时只读确认入口、尚未运行，并要求重提当前冻结前端缓存。该记录针对当时E/R候选，也不能当成后续响应引导M2组合已经运行。本次相关目录中未定位到当前组合对这921条的完成summary/预测。

若主代理后续安排该验证，可沿用现有类别映射、原图路径、目标eligibility与源参考清单，按当前实际监督/选模暴露核对排除口径，然后用当前冻结模型生成目标分数；这一方向原则上是推理验证，不必先重训。反向“当前算法OSS1→OSS2”则需要当前算法的OSS1源模型，不能把OSS2权重自测包装为双向迁移。本轮未执行这些工作。

## 8. H组：更早的跨库/支持预算证据，保留历史身份

报告：`E:/临时会话/scene_cross_dataset_redesign_20261014/cross_dataset/FINDINGS_ZH.md`。

同目录实际资产：`external_summary.csv`、`external_metrics.csv`、`external_predictions.csv`、`relative_draws.csv`、`relative_classes.csv`、`relative_bootstrap.csv`、`readout_protocol.json`。

合法24港源池28854；其中SSL训练14659、源验证5829，完整采集切分并排除跨角色同船。源PCA/头使用合法28854；目标图像不进入SSL。八类查询F4654张/船，类别1865/87/383/719/788/481/282/49；OSS2 811张/船，类别189/138/75/154/188/35/18/14。两目标B0、B16、B64；16/64明确是目标支持标签预算，每预算五支持抽样，不是五独立骨干训练。

G50+源SSL R18对G50：F B64 OA/BA+2.34/+0.58个百分点；OSS2 B16+1.87/+1.37、B64+2.20/+1.68。普通ImageNet R18在OSS2 B64的BA28.84，高于源SSL双路28.63。只能用来说明过去有真实跨库及预算研究；架构为双骨干PCA/ridge、设置含目标支持，与当前源32参考且无目标标签的模型不兼容。没有必要为“补实验数量”把这套旧算法纳入当前主表。

## 9. 数据规模不能互相替代

统计来源：`E:/临时会话/paper_draft_20261108/DATASET_PROFILE_ZH.md`、`dataset_24ports_by_port.csv`、`dataset_24ports_by_class.csv`。

- OSS1本地3030 GRDH记录，八类兼容1349/1110船；这不是A组四类评价63，也不是G组外推921。
- OSS2本地9120记录（6780 GRDH+2340 SLC）；当前/上述四类评价378不能写成全OSS2或八类复现。
- 自建总库38091/18766非空MMSI/24港/8类/841产品/720平台—绝对轨道组，198缺MMSI；原诊断4744/2973船/18港；沿用源身份排除后扩围3075/1950船/18港。**总库24港不代表实验做了24个留一港折。**
- D、E及其后续分析共享3075查询，不算几套独立外部验证。17/29/43和多次支持抽样也不能把同一查询变成更多独立船只。

## 10. 给主代理的补证优先级

1. **直接复用、无需重训：** 以历史版本清楚标注A、C、D、F/G的结果与预测出处；补足已有类别/过境分析和正负案例来源。D的重训删除表确已完成，不要重复宣称“完全没有消融”。是否纳入正文由主代理决定，本文件不写正文。
2. **当前版本外部证据最短缺口：** 当前OSS2冻结组合→OSS1 921条。现成清单、原图、类别映射和历史对照可复用，但当前预测缺完成证据；须生成当前前端/M3输出后才能称当前跨库结果，不能仅给旧summary换模块名称。
3. **当前算法OSS1源端复现：** 已有A组五角色划分与32源参考组织，当前算法在此上的完成结果本次未找到。若要真正双向当前算法迁移或第二公开集同算法复现，需要补相应源模型/评价，不能沿用旧A/D/T权重顶替。
4. **八类24港主张有条件补：** 若论文要声称当前三模块在八类自建集也成立，则缺对应八类实例及匹配实验；当前四类头与类别参考布局不能直接替换八类模型。若只作为历史设计诊断/数据现象支持，已有D/E即可按范围引用，不必为了覆盖所有旧版本自动重训。
5. **不列为自动新增工作：** 全24港LOPO、所有方向全重训、全参数扫参、旧双骨干支持预算重做。本轮没有发现这些是回答当前关键问题的必需前提。更早协议结果可归档引用，不能以数量堆叠替代版本一致证据。

核查强度：阅读上述完成报告、协议/版本清单和相关summary；定位三种子预测，选择主要seed17档案只读确认数组键、行数和类别轴，核对full/control的真实复用来源。没有重新计算全量指标、执行模型、重看全部原chip或声称独立复核历史视觉结论；既有原图回看状态按对应最终报告/CASE_NOTES引用。未找到的内容均限定为本次检索范围，不作全盘“从未做过”的判断。
