# 频带 specialist 冻结结果

本报告仅呈现结果与口径，不给出训练有效性或失效根因结论；结论等待主流程根因诊断。

全部 25 个匹配任务的 COMPLETE.json 存在后才读取标签。全部源域 SELECTION.json 在目标标签读取前载入，报告不重新选择策略；source_selected 已与源域 rule 对应预测逐项对齐，样本 ID、行索引、support 与标签沿用直接对齐核验。

old 为原始等权联合训练固定 checkpoint；continued、independent、specialist 为主流程生成的三个训练家族。learned 保留原 selective 主分支。四个家族分别比较 equal 与 LL/LH/HL/HH；具体训练目标以主流程记录为准。

## 主要对照 BA（百分数）

| dataset | mode | base | full | frozen_route |
| --- | --- | --- | --- | --- |
| f0921 | continued_equal | 26.1264 | 26.4324 | 26.4225 |
| f0921 | independent_equal | 26.1399 | 26.4456 | 26.4355 |
| f0921 | learned | 26.0348 | 26.3593 | 26.3593 |
| f0921 | old_equal | 26.1005 | 26.4069 | 26.3941 |
| f0921 | source_selected | 26.0277 | 26.3534 | 26.3442 |
| f0921 | specialist_equal | 26.1278 | 26.4347 | 26.4253 |
| oss2 | continued_equal | 31.1049 | 32.2042 | 32.1427 |
| oss2 | independent_equal | 31.1496 | 32.1821 | 32.1881 |
| oss2 | learned | 31.2783 | 32.3724 | 32.3724 |
| oss2 | old_equal | 30.9430 | 32.0829 | 31.9921 |
| oss2 | source_selected | 31.2458 | 32.3012 | 32.2941 |
| oss2 | specialist_equal | 31.1562 | 32.1910 | 32.1918 |
| self24ports | continued_equal | 52.3740 | 54.1437 | 54.0155 |
| self24ports | independent_equal | 52.3612 | 54.1313 | 54.0076 |
| self24ports | learned | 52.4994 | 54.2383 | 54.2383 |
| self24ports | old_equal | 52.3278 | 54.1356 | 53.9788 |
| self24ports | source_selected | 52.5264 | 54.2770 | 54.1670 |
| self24ports | specialist_equal | 52.3678 | 54.1327 | 54.0083 |

完整模式含各家族 support_static/support_local，均保存在 summary.csv；增益相对同阶段 learned。base 不交接专家，full 按各模式独立路由，frozen_route 统一使用 learned admission mask。

## 四频带互补上界

| dataset | stage | family | oracle_delta_ba | best_single_oracle_ba | union_minus_best_single_oracle_ba |
| --- | --- | --- | --- | --- | --- |
| f0921 | base | continued | 1.9277 | 27.2829 | 0.6796 |
| f0921 | base | independent | 1.9325 | 27.2939 | 0.6734 |
| f0921 | base | old | 1.6402 | 27.1885 | 0.4865 |
| f0921 | base | specialist | 1.9185 | 27.2784 | 0.6749 |
| f0921 | frozen_route | continued | 1.7808 | 27.5186 | 0.6216 |
| f0921 | frozen_route | independent | 1.7920 | 27.5316 | 0.6197 |
| f0921 | frozen_route | old | 1.5167 | 27.4283 | 0.4477 |
| f0921 | frozen_route | specialist | 1.7741 | 27.5137 | 0.6198 |
| oss2 | base | continued | 2.0527 | 32.6256 | 0.7054 |
| oss2 | base | independent | 1.9046 | 32.5086 | 0.6743 |
| oss2 | base | old | 1.3905 | 32.2769 | 0.3919 |
| oss2 | base | specialist | 2.0088 | 32.5524 | 0.7347 |
| oss2 | frozen_route | continued | 1.8586 | 33.5937 | 0.6373 |
| oss2 | frozen_route | independent | 1.7197 | 33.4812 | 0.6109 |
| oss2 | frozen_route | old | 1.2711 | 33.3097 | 0.3339 |
| oss2 | frozen_route | specialist | 1.7983 | 33.5076 | 0.6631 |
| self24ports | base | continued | 1.5129 | 53.6820 | 0.3303 |
| self24ports | base | independent | 1.4729 | 53.6692 | 0.3030 |
| self24ports | base | old | 1.3635 | 53.6467 | 0.2163 |
| self24ports | base | specialist | 1.4767 | 53.6712 | 0.3049 |
| self24ports | frozen_route | continued | 1.0630 | 55.0939 | 0.2074 |
| self24ports | frozen_route | independent | 1.0327 | 55.0891 | 0.1819 |
| self24ports | frozen_route | old | 0.9841 | 55.0908 | 0.1316 |
| self24ports | frozen_route | specialist | 1.0360 | 55.0906 | 0.1837 |

Oracle 保留 learned 正确项，只修复 learned 错误且至少一个频带正确的项；使用目标标签，仅为不可部署上界。best_single_oracle_ba 是每个 task/draw 内最佳 learned+单频带 oracle 的 BA；union_minus_best_single_oracle_ba 才是四频带相对该上界的额外增益。OA 对应字段也保存在 CSV。最佳单频带及 oracle 按 OA/BA 各自取最优后汇总，不是源域部署选择。仅对 base 与 frozen_route 构造 oracle，不合并各自重路由的 full 预测。

## 各频带独有救回（base）

| dataset | family | LL | LH | HL | HH |
| --- | --- | --- | --- | --- | --- |
| f0921 | continued | 2006 | 1107 | 1652 | 805 |
| f0921 | independent | 2268 | 987 | 1501 | 893 |
| f0921 | old | 393 | 938 | 2097 | 259 |
| f0921 | specialist | 1974 | 1089 | 1656 | 790 |
| oss2 | continued | 129 | 72 | 23 | 25 |
| oss2 | independent | 117 | 58 | 24 | 23 |
| oss2 | old | 23 | 63 | 9 | 13 |
| oss2 | specialist | 128 | 68 | 24 | 23 |
| self24ports | continued | 227 | 429 | 186 | 127 |
| self24ports | independent | 207 | 394 | 187 | 126 |
| self24ports | old | 109 | 376 | 190 | 101 |
| self24ports | specialist | 200 | 424 | 188 | 125 |

独有救回指 learned 错误、当前家族恰好一个频带正确；不表示跨家族独有。band_events.csv 同时包含 rescues、harms；rescue_overlaps.csv 记录每对频带救回集合的 intersection/union。两者均含 base/frozen_route 明细及汇总。预测不同本身不证明学到了不同且有用的证据。

## 冻结源域选择

| dataset | rule | tasks |
| --- | --- | --- |
| f0921 | independent_support_local | 1 |
| oss2 | continued_LH | 1 |
| self24ports | continued_HL | 1 |
| self24ports | continued_LH | 4 |
| self24ports | continued_LL | 1 |
| self24ports | continued_support_local | 1 |
| self24ports | continued_support_static | 2 |
| self24ports | independent_LH | 2 |
| self24ports | independent_equal | 1 |
| self24ports | learned | 2 |
| self24ports | old_LH | 1 |
| self24ports | old_support_local | 2 |
| self24ports | specialist_LH | 2 |
| self24ports | specialist_equal | 1 |
| self24ports | specialist_support_local | 3 |

source_selection.csv 保留 rule 与全部 source_delta_ba；目标指标不回流训练或选择。源域验证是否与表征训练独立，应由主流程训练/验证记录判定。

## 输出口径

OA/BA/macro_f1_8 为百分数，delta 为百分点；BA 使用评估集合出现的类别，macro F1 固定八类。summary 先平均任务内五个 draw，再等权平均任务；draw_metrics 在同一 draw 内等权平均任务。class_metrics 保留每任务/draw 的类别样本数、support 数、recall 与 delta_recall。所有救回、伤害和交并计数按 query-draw 累加，同一样本跨 draw 会重复，不代表独立样本数。

metrics.csv、task_means.csv、summary.csv、draw_metrics.csv、class_metrics.csv、events.csv 为三阶段指标；complementarity.csv 与 complementarity_summary.csv 为四家族 oracle 明细及等权汇总。本报告不生成图、不修改冻结预测或源域选择。
