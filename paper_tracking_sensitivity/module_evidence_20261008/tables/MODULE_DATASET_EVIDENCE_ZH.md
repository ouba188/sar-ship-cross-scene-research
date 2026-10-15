# 现有模块与数据集证据表（2026-10-08）

仅整理已完成工件，无新实验。OA/BA 为百分数，差值为百分点。机器 JSON 和 CSV 保留未四舍五入值。

**此页为启动补测前的历史快照。** 同日补测的最新四臂数值见[当前匹配结果表](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/CORE_MATCHED_RESULTS.md)，最新模块覆盖见[module_coverage_completed.csv](E:/Hermes/paper_tracking/sensitivity/module_completion_20261008/module_coverage_completed.csv)。当前表保留各自参考轮次；M1/M4仍是原主线阶段差值，M2/M3/联合为新增匹配消融，不混算。

## 原主线结果与阶段差值

M1 表示 Base64−Source；M4 表示 Full5−Base64。这是阶段对比，不是 M1/M4 的 2×2 消融，也不等于对其内部全部机制的独立因果归因。

| 数据集/目标 | 目标/查询行 | 类 | Source OA/BA | Base64 OA/BA | Full5 OA/BA | M1 OA/BA差值 | M4 OA/BA差值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self24ports/23eligible_equal_port | 38027/36555 | 8 | 44.094334/43.268995 | 54.794330/48.540070 | 56.653819/50.382098 | 10.699996/5.271075 | 1.859489/1.842028 |
| correctedF0921/compatible_subset | 123489/123425 | 8 | 24.207575/20.314242 | 28.937087/24.901788 | 29.633543/25.105149 | 4.729512/4.587546 | 0.696455/0.203362 |
| publicOSS1/compatible_subset | 1349/1285 | 8 | 38.163424/21.233037 | 40.513619/31.510334 | 43.066148/32.330577 | 2.350195/10.277297 | 2.552529/0.820242 |
| publicOSS2/compatible_subset | 3745/3681 | 8 | 18.565607/19.457723 | 30.638413/29.295273 | 32.382505/29.938364 | 12.072806/9.837550 | 1.744091/0.643091 |
| PACS/art_painting | 2048/1984 | 7 | 72.106855/72.043209 | 78.608871/78.395563 | 80.655242/80.612386 | 6.502016/6.352354 | 2.046371/2.216823 |
| PACS/cartoon | 2344/2280 | 7 | 55.096491/61.274924 | 70.789474/74.436160 | 74.745614/77.864359 | 15.692982/13.161236 | 3.956140/3.428199 |
| PACS/photo | 1670/1606 | 7 | 96.612702/95.806211 | 97.721046/97.199079 | 95.105853/93.985480 | 1.108344/1.392868 | -2.615193/-3.213599 |
| PACS/sketch | 3929/3865 | 7 | 46.302717/40.267690 | 56.978008/62.917442 | 61.371281/67.013767 | 10.675291/22.649751 | 4.393273/4.096325 |
| PACS/equal_domain_mean | 9991/9735 | 7 | 67.529691/67.348009 | 76.024350/78.237061 | 77.969497/79.868998 | 8.494658/10.889052 | 1.945148/1.631937 |
| OfficeHome/Art | 2427/2363 | 65 | 65.890817/62.758376 | 66.559458/62.954531 | 66.347863/62.863962 | 0.668642/0.196154 | -0.211595/-0.090568 |
| OfficeHome/Clipart | 4365/4301 | 65 | 46.933271/47.428060 | 49.476866/49.630514 | 51.364799/51.342740 | 2.543595/2.202454 | 1.887933/1.712226 |
| OfficeHome/Product | 4439/4375 | 65 | 73.668571/72.046784 | 74.240000/72.438926 | 74.806857/72.893490 | 0.571429/0.392141 | 0.566857/0.454565 |
| OfficeHome/Real World | 4357/4293 | 65 | 75.588167/74.247931 | 75.956208/74.387229 | 76.622409/75.023750 | 0.368041/0.139298 | 0.666201/0.636521 |
| OfficeHome/equal_domain_mean | 15588/15332 | 65 | 65.520207/64.120288 | 66.558133/64.852800 | 67.285482/65.530986 | 1.037926/0.732512 | 0.727349/0.678186 |

## FUSAR 可选附表：7 个实际目标类

保留 8 个输出列，但目标存在 7 类；单图复制输入不构成双极化验证。不要与 8 类 SAR 表合成共同基准。

| 数据集/目标 | 目标/查询行 | 类 | Source OA/BA | Base64 OA/BA | Full5 OA/BA | M1 OA/BA差值 | M4 OA/BA差值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FUSAR/compatible_subset | 1158/1094 | 7 | 46.965265/22.463793 | 51.608775/28.446428 | 52.650823/28.894663 | 4.643510/5.982635 | 1.042048/0.448235 |

## 单模块历史候选：保留各自匹配参考

| 数据集 | 模块/轮次 | 完整 OA/BA | 参考 OA/BA | 对参考 ΔOA/ΔBA |
| --- | --- | --- | --- | --- |
| correctedF0921 | M2/standalone_projection_pilot | 29.828803/25.003274 | 29.633543/25.105149 | 0.195260/-0.101876 |
| publicOSS1 | M2/standalone_projection_pilot | 43.284047/31.970674 | 43.066148/32.330577 | 0.217899/-0.359902 |
| publicOSS2 | M2/standalone_projection_pilot | 32.442271/29.971866 | 32.382505/29.938364 | 0.059766/0.033502 |
| correctedF0921 | M3/standalone_aligned_pilot | 29.326474/25.435555 | 29.634353/25.106684 | -0.307879/0.328871 |

M2 表示补空间候选启用后的完整部署系统，M3 表示部署对齐候选启用后的完整部署系统；不是单独专家在所有查询上的诊断准确率。专家诊断值仅在机器表中另列。

## F0921 本轮匹配 M2×M3 2×2

本轮所有分支固定 M1、Base64、M4 规则、64 个支持标签和 6171/123425 次交接。M2 补空间与 M3 对齐训练是两个实验因子。

| 分支 | M2 | M3 | Full OA | Full BA | 相对本轮参考 ΔOA | ΔBA |
| --- | --- | --- | --- | --- | --- | --- |
| reference | 0 | 0 | 29.634353 | 25.106684 | 0.000000 | 0.000000 |
| complement_only | 1 | 0 | 29.829289 | 25.003473 | 0.194936 | -0.103212 |
| aligned_only | 0 | 1 | 29.326474 | 25.434966 | -0.307879 | 0.328281 |
| joint | 1 | 1 | 29.758639 | 25.431755 | 0.124286 | 0.325071 |

必须分开：生产校正参考 29.633543/25.105149；本轮匹配参考 29.634353/25.106684。原独立 M3 的 BA=25.435555，本轮匹配 M3 的 BA=25.434966，不能混拼；独立 M2 的 F0921 结果同样单独保留。

联合分支相对匹配参考 ΔOA=+0.124286、ΔBA=+0.325071；相对 aligned-only ΔOA=+0.432165、ΔBA=−0.003211；相对 complement-only ΔOA=−0.070650、ΔBA=+0.428282。仅为已使用的 F0921 开发结果。

## 逐数据集/域的模块完成矩阵

M1/M4 数字见原主线阶段差值表。下表采用本轮最新匹配 F0921 数字，旧单模块轮次见上表。null/not_run 不以诊断实验或其他模型结果填补。

| 数据集/目标 | 模块 | 状态 | OA/BA | 对匹配参考 ΔOA/ΔBA |
| --- | --- | --- | --- | --- |
| self24ports/23eligible_equal_port | M2 | not_run | null/null | null/null |
| self24ports/23eligible_equal_port | M3 | not_run | null/null | null/null |
| self24ports/23eligible_equal_port | joint_M2_M3 | not_run | null/null | null/null |
| correctedF0921/compatible_subset | M2 | completed | 29.829289/25.003473 | 0.194936/-0.103212 |
| correctedF0921/compatible_subset | M3 | completed | 29.326474/25.434966 | -0.307879/0.328281 |
| correctedF0921/compatible_subset | joint_M2_M3 | completed | 29.758639/25.431755 | 0.124286/0.325071 |
| publicOSS1/compatible_subset | M2 | completed | 43.284047/31.970674 | 0.217899/-0.359902 |
| publicOSS1/compatible_subset | M3 | not_run | null/null | null/null |
| publicOSS1/compatible_subset | joint_M2_M3 | not_run | null/null | null/null |
| publicOSS2/compatible_subset | M2 | completed | 32.442271/29.971866 | 0.059766/0.033502 |
| publicOSS2/compatible_subset | M3 | not_run | null/null | null/null |
| publicOSS2/compatible_subset | joint_M2_M3 | not_run | null/null | null/null |
| FUSAR/compatible_subset | M2 | not_run | null/null | null/null |
| FUSAR/compatible_subset | M3 | not_run | null/null | null/null |
| FUSAR/compatible_subset | joint_M2_M3 | not_run | null/null | null/null |
| PACS/art_painting | M2 | not_run | null/null | null/null |
| PACS/art_painting | M3 | not_run | null/null | null/null |
| PACS/art_painting | joint_M2_M3 | not_run | null/null | null/null |
| PACS/cartoon | M2 | not_run | null/null | null/null |
| PACS/cartoon | M3 | not_run | null/null | null/null |
| PACS/cartoon | joint_M2_M3 | not_run | null/null | null/null |
| PACS/photo | M2 | not_run | null/null | null/null |
| PACS/photo | M3 | not_run | null/null | null/null |
| PACS/photo | joint_M2_M3 | not_run | null/null | null/null |
| PACS/sketch | M2 | not_run | null/null | null/null |
| PACS/sketch | M3 | not_run | null/null | null/null |
| PACS/sketch | joint_M2_M3 | not_run | null/null | null/null |
| PACS/equal_domain_mean | M2 | not_run | null/null | null/null |
| PACS/equal_domain_mean | M3 | not_run | null/null | null/null |
| PACS/equal_domain_mean | joint_M2_M3 | not_run | null/null | null/null |
| OfficeHome/Art | M2 | not_run | null/null | null/null |
| OfficeHome/Art | M3 | not_run | null/null | null/null |
| OfficeHome/Art | joint_M2_M3 | not_run | null/null | null/null |
| OfficeHome/Clipart | M2 | not_run | null/null | null/null |
| OfficeHome/Clipart | M3 | not_run | null/null | null/null |
| OfficeHome/Clipart | joint_M2_M3 | not_run | null/null | null/null |
| OfficeHome/Product | M2 | not_run | null/null | null/null |
| OfficeHome/Product | M3 | not_run | null/null | null/null |
| OfficeHome/Product | joint_M2_M3 | not_run | null/null | null/null |
| OfficeHome/Real World | M2 | not_run | null/null | null/null |
| OfficeHome/Real World | M3 | not_run | null/null | null/null |
| OfficeHome/Real World | joint_M2_M3 | not_run | null/null | null/null |
| OfficeHome/equal_domain_mean | M2 | not_run | null/null | null/null |
| OfficeHome/equal_domain_mean | M3 | not_run | null/null | null/null |
| OfficeHome/equal_domain_mean | joint_M2_M3 | not_run | null/null | null/null |

## 行数、预算与聚合

自有库共 38091 行/24 港；当前主线纳入 38027 行/23 个合格目标港，每轮总查询 36555 行。每港独立 64 标签，共 1472 个支持标注/轮。先平均港内五次抽样，再等权平均 23 港；OA 也不是汇总样本 OA。具体港口行数和数值见 self23ports_detail.csv。

F0921、publicOSS1、publicOSS2 和可选 FUSAR 每个数据集每轮各用 64 个全局目标标签；两适配分支共享标签。PACS/OfficeHome 每个外层目标域各用 64 标签，四任务合计 256 标签/轮；四域均值等域加权，行数是四任务之和。

PACS 7 类、OfficeHome 65 类均保留。OfficeHome 每轮支持仅覆盖 34–46 类，未观察类不从查询评估中删除。负结果保留：PACS/photo 的 M4 BA 差值 −3.213599，OfficeHome/Art 为 −0.090568。

## 表征与证据边界

PACS/OfficeHome 使用各数据集自己的 RGB 特征：冻结 ImageNet R50 与按外层源域监督微调的 R18 编码同一图像。它们不是 SAR 双极化，不是港口知识验证，也没有验证本轮新增 M2/M3/联合模块。源域内部 head-OOF 以固定表征为条件，不是 encoder-OOF。

外部 SAR 为细标签兼容子集。F0921 继承 SSL 预训练语料暴露，各数据集均有历史开发使用；SAR 支持/查询是行级划分，非同船隔离。5% 是交接次数上限，不是特征计算上限，也不是相对源模型全部预测变化的上限。

## FMoW 与 Camelyon17：有界检索结论

FMoW：当前方法证据仅到特征清单，未发现当前 shared64/5% 残差交接或新联合模块的结果。但已核实旧实验 fullchain_fmowclip_result.json，协议为前 8 高频类、五地区、20% 预算；不能把旧实验填入本表，也不能说 FMoW 从未做过实验。

Camelyon17：已核实 archive.tar.gz 和 hf_cam17_feats/hf_cam17_pca512 下的 NPZ 文件存在；未加载大特征。当前方法 OA/BA、目标行数、类数均 null。未在检索范围发现当前结果不等于断言其他位置绝无实验。

检索范围和限制逐项写入 JSON 的 bounded_search。包含 sensitivity、xdom_generic、reports、internal_mainline、external_mainline，并补充 paper_tracking 中排除第三方仓库、运行日志、环境及备份的有限大小文本检索。

## 精确来源

以下均为已有工件绝对路径；JSON 每行还提供字段选择器。

- `E:/Hermes/paper_tracking/sensitivity/aligned_residual_pilot_20261008/residual_global/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/fusar_transfer_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/joint_residual_pilot_20261008/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/officehome_handoff_transfer_20261007/Art/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/officehome_handoff_transfer_20261007/Clipart/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/officehome_handoff_transfer_20261007/Product/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/officehome_handoff_transfer_20261007/Real_World/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/officehome_handoff_transfer_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/oof_base_adapt_results_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/pacs_handoff_transfer_20261007/art_painting/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/pacs_handoff_transfer_20261007/cartoon/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/pacs_handoff_transfer_20261007/photo/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/pacs_handoff_transfer_20261007/sketch/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/pacs_handoff_transfer_20261007/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/projection_residual_pilot_20261008/f0921/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/projection_residual_pilot_20261008/oss1/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/projection_residual_pilot_20261008/oss2/summary.json`

- `E:/Hermes/paper_tracking/sensitivity/joint_residual_pilot_20261008/JOINT_FINDINGS.md`

- `E:/Hermes/paper_tracking/sensitivity/CROSS_DATASET_READOUT_20261007.md`

- `E:/Hermes/paper_tracking/sensitivity/GENERIC_CROSS_DOMAIN_READOUT_20261007.md`

- `E:\Hermes\paper_tracking\xdom_generic\fullchain_fmowclip_result.json`

- `E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/AVAILABLE_FEATURES.md`

## 提取验证

从现有逐次 metrics.csv 核对汇总 OA/BA、港口行数和四域均值；不重做模型推理，不运行显著性检验。核对失败会停止生成并定位错误来源。检查记录见 extraction_checks.json。
