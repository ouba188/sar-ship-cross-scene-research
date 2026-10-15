# 既有替换证据与 sourcefit 兼容性

2026-10-13。本文件只整理旧证据及新几何替换的适用契约；新结果的独立审计另见本目录 `VERIFICATION_ZH.md`。

结论：旧 original/display 数据支持真实第二路相对指定重建器仍有净任务价值，但不回答 fresh source/display 配套重拟合后的 P/T 差距。旧 mapper 的输出坐标并非不兼容：它输出原始 R18 512 维特征，再由指定 bundle 转成部署坐标。可复用计算接口，不可迁移旧数值或声称 mapper 已配套重训。

## 已核实的旧数值

数据来源：`ports_input_controls_20261012/metrics.csv`，逐 fold/task/draw 配对重算，再先任务内平均、后任务等权；与该目录 `arm_effects.csv` 及 `ports_mechanism_evidence_20261012/replacement_totals.csv` 对账。`extract_existing.py` 只读执行，72 个汇总组合全部通过，逐效应最大误差 1.4322e-14。主表为 MLP、expert、scope=original，五组 B64；三港是 LA/Klang/Santos（15 个上下文-抽样），公开集各两折（10 个），不是独立训练种子。

每格为 original → display；BA 单位百分点，宏 Brier 增益为对照风险减候选风险，正数更好。这里 display **只改变目标支持与查询**。

| BA 差值 | 三港 | OSS1 | OSS2 |
|---|---:|---:|---:|
| P-R | +0.125062 → +0.338182 | +0.648470 → -0.111658 | +0.944484 → +0.212109 |
| E-R | +0.423352 → +0.582291 | +2.110702 → +1.115240 | +2.225160 → +0.666803 |
| PE-P | +0.406649 → +0.551147 | +1.849917 → +1.079800 | +1.619474 → +0.603573 |
| T-P | +0.640619 → +0.641803 | +2.502056 → +0.977117 | +1.529670 → +0.508328 |
| T-R | +0.765681 → +0.979985 | +3.150526 → +0.865460 | +2.474154 → +0.720437 |
| T-PE_scaled | +0.349048 → +0.304149 | +0.688118 → -0.021588 | +0.205270 → +0.083134 |

| 宏 Brier 增益 | 三港 | OSS1 | OSS2 |
|---|---:|---:|---:|
| P-R | .000816138 → .000978169 | .002793637 → .000222690 | .002588925 → .000981020 |
| E-R | .001085034 → .003092331 | .010073418 → .003367083 | .006140595 → .002477167 |
| PE-P | .000990587 → .002846688 | .008347328 → .003129726 | .005141357 → .002268214 |
| T-P | .001388577 → .003265767 | .009694990 → .003981612 | .004500814 → .001841375 |
| T-R | .002204714 → .004243936 | .012488627 → .004204302 | .007089738 → .002822395 |
| T-PE_scaled | .000613371 → .000939652 | .001965681 → .001850117 | .000592164 → .000775424 |

display 的 MLP T-P 正 BA 次数为三港 11/15、OSS1 9/10、OSS2 6/10。线性替换的 display T-P BA 为 +.624610/+.576802/+.531204；PE-P 为 +.189497/+.785737/+.689324，其宏 Brier 增益为 .002627294/.003242065/.002309031。两种有限替换都保留平均净差距，不能解释为所有单路方案无法恢复。OSS1 的 T-PE_scaled BA 已出现负值；真实 T 并不支配所有组合与指标。

## 替换定义及可复用代码

- `public_oss_comparison_20261012/fit_mappers.py:fit`：拼接 raw R50 2048 与 C4 pooled 2048，使用 mapper 自带 x/y 均值、标准差；输出原始 R18 512 特征。`dual_reconstruction_core.py:reconstruct` 明确在返回前撤销 y 标准化。
- `dual_cue_decomposition_20261012/run.py:ssl_map`：将真实及重建 R18 分别经同一 bundle 的 PCA、中心、尺度和保留/补空间映射；P 是重建后的部署坐标，T 是真实部署坐标，最后 E=T-P。不得在原始残差上重复中心化。
- 核为 R: K_R；P: K_R+PP'；E: K_R+EE'；PE: K_R+PP'+EE'；T: K_R+(P+E)(P+E)'。P/E 非正交，PE 缺少交叉项且改变能量。PE_scaled 每次只用支持特征 trace 解析配平，不搜索、不用查询定标。
- `public_oss_comparison_20261012/evaluate.py:predict`：上述六臂固定 R50 源先验；DD 使用 T 核加双路先验。每臂独立重求 ridge8 支持系数；compact C4 与 expert 不混用。E、PE、PE_scaled 都需要真实 R18，不是单路部署替代。
- `ports_input_controls_20261012/predict_inputs.py` 绑定旧 fold2/3 的 geometry 与 mapper；`ports_mechanism_evidence_20261012` 只是对其预测重统计，不能计为另一轮独立实验。
- `dual_cue_decomposition_20261012` 可复用核定义、残差与 trace 诊断逻辑；它的历史 23 港/F/OSS2 使用任务特定模型、完整 base 等契约，不能与本轮三港、冻结 e206、compact C4 数值拼接；该轮也没有 OSS1/display 证据。

## sourcefit 迁移边界

实际打开两折 mapper 与 archive/fresh geometry 核对：fold2/3 源行分别 31335/27417，archive、fresh original、fresh display 行数组一致；mapper 输入 4096、输出 512，fresh PCA 输入维度 2048/512 均兼容。source 行一致不等于源特征分布一致。输入提取和几何脚本复用同一 `extract_inputs.encoders/forward`，没有换 R18 特征语义。

因此，用新 geometry 重新映射旧 mapper 输出在接口上可行；必须同时用该 geometry 重建 R/T/P/E、源先验与支持系数。旧部署 P/E、核、系数和概率不能直接搬入新 geometry。旧 mapper 的标准化和拟合仍来自历史原图特征，即使 fresh original 也存在重提取数值漂移；fresh display 更不能称为同步重拟合 mapper 的匹配源替换。

`ports_root_cause_20261013/run_factorial.py` 的 fresh 系列仅含 R/T/DD，没有新 mapper。其 OO/DD 源/目标格与双路臂 DD 是不同概念。fresh 系列可回答匹配条件下第二路总净收益，不能单独回答 T 相对 P 的增量；也不能将其 T 与旧 P 相减。显示与 SLC 几何没有联合处理实验，不相加为组合收益。

## 已接受的最小新几何替换

采用主代理的限定设计：OSS2、archive bundle+同折旧 mapper、fold2/3、original/range_geometry/axis_sham、linear+MLP、七臂、原五组 B64，直接复用 `evaluate.predict`。输入已经缓存，无新编码器/mapper训练、源拟合或参数搜索。读取 original/geometry 数据会通过 `configure_evaluation` 重置模块全局量，故必须先物化 data，再绑定 OUT/fit_dir/mapper_path/target_arrays，随后同步调用 predict。已检查新 `predict_replacements.py` 符合此顺序。

这一设计回答“固定旧源系统经过几何干预后，真实第二路是否仍优于给定重建替代”，不回答 fresh source/display 配套 mapper 的公平上限。全 OSS2、SLC query、同 SAR 时间同标签且双方均为 query 的 paired_SLC 分开；正确轴需同时比较原图与等面积错轴。处理支持也会改变 GRDH 的适配预测，不能把全队列差异都归给被重编码的 SLC 查询。

新冻结回执 `REPLACEMENT_PREDICTIONS_FROZEN.json` 已记录六上下文、零新训练/源拟合，两折 original 概率最大回放误差均为 0，分类精确一致。该回执不替代本目录后续选定数值独立审计。全新匹配显示 P 对照仍未执行；本任务不建议为补齐该表启动重训。
