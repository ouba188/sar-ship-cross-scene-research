# Fixed Reader Calibration 独立审阅

审阅日期：2026-10-22。范围仅限本轮协议、实现及与指定问题直接相关的既有缓存。只写本报告，不训练、不导入 torch、不操作 Git；原始 chip 分析由主代理完成。

## 当前结论与实际问题

已完成协议、本轮 `run.py`、`analyze.py` 和落盘缓存的独立审阅。**同一冻结 A、D 用于校准和测试、cal MMSI 排除、seen 逐类行数匹配、非 ensemble 重复评估、零 D 主表保留均通过。没有发现需要重训或修改门控训练实现的问题。**

发现一个统计展示问题，修正范围仅需汇总代码/表述，不需要新增实验：

**[P2] 旧 full-source 参照的 SD 由三组 seed 重复成九行后计算，不能作为三 seed 的 SD 展示。** `run.py:123–129` 将同一个 seed 的旧 full-source 预测在三个 inner 下重复记录，以便逐重复配对；这本身合理。但 `analyze.py:32` 对所有 arm 一律执行 `groupby('arm').agg(['mean','std'])`，把旧参照也按九行计算样本 SD。其三个 inner 没有不同模型、不同预测；使用 ddof=1 后，这个 SD 比真正三 seed SD 小 sqrt(3/4)，实际为：

| old_full_reference | 当前九行 SD（百分点） | 三个 seed SD（百分点） |
| --- | ---: | ---: |
| OA | 0.34844594 | 0.40235071 |
| BA | 0.35059054 | 0.40482709 |

建议保留逐 inner/seed 的配对行，但旧 full-source 参照的独立展示按 seed 去重后计算 SD，并注明该参照只有三 seed；其他 arm 的九次 SD 明确标为相关重复的描述性离散度，不是九个独立数据集的标准误。**均值与当前 paired MMSI bootstrap 不受这个展示问题影响**：后者已先按 observation 平均重复正确性，没有把重复旧参照当额外船只。若明确选择展示九次重复值的描述性 SD，则应单独标注该定义，避免把这列理解为旧模型的三 seed SD。

审阅版本：`run.py` 9,655 bytes，修改时间 2026-10-22 16:50:45；`analyze.py` 9,684 bytes，修改时间 16:54:00。读取 `TRAINING.json` 为 complete，记录 27 个复用上游读出器、0 个新上游 fit、108 个 gate fit。没有执行或导入这些研究模块。

## 协议判定

已读取本轮 `PROTOCOL.md`、上轮 `held_vessel_gate_20261022/run.py` 与 `review/OOF_REVIEW_ZH.md`，以及直接相关的上游标准化代码。协议符合本轮问题，未发现协议缺陷，实际执行核查见后文。

- 每个 outer fold / inner / seed 应使用同一份冻结 ridge、keep_refs、邻域残差及对应 normalization，在 calibration、seen、sourceval、test 生成 A、D。不能用上轮拼接 OOF 分数训练门控后换 full-source 模型测试，也不能校准后重训上游。
- calibration MMSI 应是上轮该 inner 的 held MMSI，与上游 fit 及 sourceval 均无交集。上轮代码的 ridge、keep_refs、邻域标准化均仅使用 fit_rows，监督拟合也仅使用 fit_rows；源验证选择未包含 held MMSI。复用外部冻结图像描述子不等于复用含 calibration 标签的监督头。
- seen 控制从该 inner 的 upstream fit 行逐类无放回抽取，匹配 calibration 的 class-row counts，再按 class/obs_id 排序。配对初始化及 minibatch 位置只保证名义预算与类别序列一致，不保证唯一船只数、船内重复或最终样本权重一致。协议已正确承认此边界。
- 每个 inner/seed 先合并三个互补 outer test fold，形成 4,744 个观测，再汇总九次重复；不能平均九组分数后 argmax 作为主结果，不能把九份同一观测当独立数据。MMSI bootstrap 应先平均每个观测的重复正确性/配对差值，再重抽船只，区间条件于已拟合模型。
- 全部零 D 模型须保留在主表；active-only 只能作为单列诊断。上轮实际 27 个内层 D 中有 12 个选零步，新轮复用它们时该边界仍存在，不能把零 D 的无变化解释为 gate 无法学习。

## 已知解释边界

本轮同一冻结 A、D 消除了上轮 OOF 到 full-source 的上游模型切换，但 seen/calibration 仍有身份暴露、样本组成、船内重复及权重差异，不能仅凭总体差值认定唯一因果机制。每个模型的 A+D 是首要参照；旧 full-source A+D 是训练数据成本不同的独立参照。原 sourceval 已参与上游及 gate 选步，既有测试集也用于此前研究，本轮是开发证据，不是新的独立数据确认。calibration 按身份留出，不额外声称 acquisition 隔离。

## 实现核查

以下行号对应本轮 `run.py`，另有说明除外。

| 要求 | 实现证据及判定 |
| --- | --- |
| 相同冻结 A、D | 58–73 行读取对应 fold/inner 的保存 normalization、ridge 系数，以及对应 seed 的两个上游 checkpoint，得到全体行的 a、d；74–78 行核对上轮该模型 held 分数。86 行转为固定分数张量，97 行验证、106 行训练、113 行测试均使用这同一组 a、d。92 行优化器仅接收 gate 参数，两个上游预测函数内部 no_grad，没有上游拟合或校准后重训。通过。 |
| cal 身份排除 | 47–53 行从旧 OOF 缓存恢复同一 assignment，fit/cal 是外训练行内的互补分区，并检查 cal 与 fit/val/test 的 MMSI 无交集。旧上游 ridge/keep_refs/D 监督仅使用 fit_rows，选步仅用 sourceval。本轮重算邻域 normalization 仍仅取 fit，并逐数组核对旧统计。通过。 |
| seen 逐类行匹配 | 24–35 行对每类从 fit 无放回取 cal 行数，双方按 class/obs_id 排序，断言类别序列完全一致。90–93 行逐 arm 重置同一初始化及 batch RNG，105 行抽相同位置；配对 constant 或 score 控制具有相同更新数、batch 和候选步。89 行分别按实际船只组成算权重，符合协议中“权重可不同”的边界。通过。 |
| checkpoint 选择 | 95–104 行候选为 0/50/100/150，仅以固定 sourceval BA、Brier 选择，严格大于才替换，平局保留较早步。测试标签只进入评价记录。通过。 |
| 无 ensemble | 127–129 行先对每个 fold/inner/seed/arm 独立 argmax 再记录 prediction。`analyze.py:19–34` 对 inner/seed/arm 合并互补 outer test 行，每组断言 4,744 唯一 obs，再算指标；没有跨 inner 或 seed 平均分数再 argmax。seed_means、split_means 也是指标均值。通过。 |
| 不扩大独立样本量 | `analyze.py:39–58` 先按 arm/obs_id 平均九次正确性，再按 class/MMSI 聚合配对差值和原始观测数，分层重抽船只；没有把 9×4,744 当独立观测，也没有用九次 SD/sqrt(9) 形成区间。区间标注 fixed fitted models。通过；旧参照展示 SD 问题见首节。 |
| 零 D 主表保留 | 79 行 active 仅是标记，87 行所有模型仍跑四 arm，115–129 行照常记录；`analyze.py:19–34` 主表未按 active 过滤。active 分组只出现在 group_results，图像案例另选 active 不改变主表。通过。 |
| 均值权重控制 | 120–122 行用该模型 calibration 集 w 均值替代动态 w，不看测试标签；seen_score 的 cal_mean 控制同样按协议取 cal 均值。通过。 |

## 独立只读缓存核查

为检出分区/标准化污染、模型替换、错误配对与漏掉零 D，只使用内联 Python 的标准库和 NumPy 读取 CSV/NPZ；未导入研究模块或反序列化 `.pt`，两次核查进程均确认 `torch_imported=false`。

- 九个 fold/inner 分区与上轮 OOF rows/assignment 一致；外 train/val/test 观测集合与旧 `neighborhood_level_split_20261021/partitions.csv` 一致，均排除 31237。cal MMSI 与 fit/val/test 交集均为零，seen 行唯一且全部来自对应 fit。
- 九个分区的 cal/seen 八类计数及排序后的类别序列完全一致。cal 行数为 1,421–1,554；cal 独立船只 880–901，seen 独立船只 1,082–1,187。每个分区 seen 船只都更多，实际并非等船只数对照；这是协议已声明的样本组织差异，不是错误匹配。
- 九份 normalization 的 ridge、keep_refs、raw/centered 描述子、level 均值和标准差，按实际 fit 行与极化有效行独立重算后均完全一致，最大误差 0。
- 27 份 `inputs_f*_k*_s*.npz` 的 A、D 在各自 held 行上与上轮 OOF 对应输出完全一致，最大误差 0。结合实现中单一全行前向路径，支持 cal/test 使用相同冻结读出器；本审阅未独立从 `.pt` 重建全行神经网络前向。
- 108 份测试缓存的 row 顺序正确，reference 等于本组 A+D，scores 等于本组 A+wD，最大误差 0；保存预测与各自 scores 的 argmax 一致。两个 score_cal_mean 控制的预测也与 calibration 平均权重代入结果一致。
- `predictions.csv` 有 384,264 行，即 9 个 arm × 9 次重复 × 4,744 个观测；每组均为同一 4,744 个唯一 obs、2,973 个 MMSI，类别和 MMSI 对应一致。81 个 summary_repetitions 的 OA/BA/rescue/harm 独立重算完全一致，最大差异 0。
- 108 组 validation_selection 均有四个预定候选；selected_steps 与各分区 metrics 中的 step 均符合 BA 优先、Brier 次优、平局较早步的规则。
- 从实际 D 数组确认 27 个读出器中 12 个全零，其余 15 个非零。零 D 组在每个 gate arm 的累计测试重复记录为 18,763 行，全部保留且 scores 与 reference 完全一致。这是重复记录量，不是 18,763 个独立观测或船只。

逐分区的实际规模如下；cal/seen 行数列同时适用于双方。

| fold / inner | 上游 fit 行数 | cal/seen 行数 | cal 船只 | seen 船只 |
| --- | ---: | ---: | ---: | ---: |
| 0 / 0 | 2891 | 1501 | 886 | 1128 |
| 0 / 1 | 2922 | 1470 | 883 | 1108 |
| 0 / 2 | 2971 | 1421 | 880 | 1082 |
| 1 / 0 | 2989 | 1430 | 896 | 1086 |
| 1 / 1 | 2865 | 1554 | 895 | 1187 |
| 1 / 2 | 2984 | 1435 | 893 | 1105 |
| 2 / 0 | 2879 | 1539 | 901 | 1184 |
| 2 / 1 | 2973 | 1445 | 901 | 1118 |
| 2 / 2 | 2984 | 1434 | 898 | 1098 |

零 D 的 `(fold, inner, seed)` 为 `(0,0,0)`, `(0,0,2)`, `(0,1,0)`, `(0,1,2)`, `(0,2,0)`, `(0,2,1)`, `(0,2,2)`, `(1,2,0)`, `(2,0,1)`, `(2,0,2)`, `(2,1,0)`, `(2,1,1)`。这些组的 gate 数据梯度无法经 D 改变分数；保留其零收益是正确的，不应据此声称 gate 在有效 D 下不可学习。

## 结果及证据边界

本轮固定读出器 A+D 的九次平均 OA/BA 为 39.5353% / 41.3143%，calibration_score 为 39.5072% / 41.2672%。calibration_score 相对其自身 A+D 的 OA/BA 差值为 -0.0281 / -0.0472 个百分点；相对 seen_score 为 -0.0304 / -0.0427 个百分点。逐 inner/seed 的净救回为 inner 0：0/0/-8，inner 1：0/-6/+2，inner 2：0/0/0。

读取已有条件 MMSI 区间：calibration_score − reference 的 OA 为 [-0.0694, +0.0118]、BA 为 [-0.0939, -0.0037] 个百分点；calibration_score − seen_score 的 OA 为 [-0.0738, +0.0164]、BA 为 [-0.0852, +0.0019]。本审阅检查了重抽单位及聚合代码，没有重复运行 bootstrap。这些结果未支持独立源船校准在本轮获得稳定净收益；不能据此声称方法等效、所有校准无效或单一训练失配根因已被证明。

旧 full-source 参照 OA/BA 为 41.5121% / 42.7074%。本轮固定模型 reference 已比它低 1.9768 / 1.3931 个百分点；calibration_score 与旧参照的总差距不能全归给 gate。反之，把本轮未涨点只归因于上游训练量减少也超出现有对照，模型选择、样本组成与 gate 可预测性并未被独立拆开。

`analyze.py:63–100` 与 `VERIFICATION.json` 记录主代理已从磁盘加载并回放 108 个 gate，最大 CPU 权重误差 1.7881393432617188e-7，测试预测一致，27 个 reader replay 完成。本审阅已检查代码与回执，但没有自行导入 torch 重放，不冒称为独立神经网络回放证据。

本次仅写此报告。无需扩大审计或重训；旧 full-source SD 的展示应修正或明确限定。原始 VV/VH 图像的事实分析与最终机制论述由主代理完成，本报告不替代该部分，也不把图像外观当成已验证的物理原因。

## 修复确认（2026-10-22，最终补记）

**首节 P2 的 SD 展示问题已修复并关闭。** 本补记更新前文的待修复状态；其余核心通过结论及解释边界不变。

已定向读取最新 `analyze.py:18–30` 的 `write_repeat_summary`、`summary_mean_std.csv` 和 `summary_counts.csv`：

- `old_full_reference` 的 OA、BA、macro F1、weight 均值和 SD 仅取 inner=0 的三个 seed；落盘 OA/BA SD 为 0.4023507111 / 0.4048270857 个百分点，与前次独立核查的三 seed 数值一致。
- rescue/harm 仍按全部九次配对汇总，因为参照的 reduced reference 随 inner/seed 改变，不能随旧 full-source 性能指标一并去重。落盘 rescue/harm 均值仍为 423.0 / 329.2222222222，SD 为 43.9573657081 / 26.5507899025，与原九次配对汇总一致。
- `summary_counts.csv` 明确记录旧 full-source 的 `performance_repetitions=3`、`paired_repetitions=9`；其他方法均为 9/9。这些计数描述性能重复与配对次数，不宣称独立数据集数量。
- `summary_repetitions` 的旧参照重复行保留用于配对，修复仅作用于展示汇总；性能均值无实质变化，原条件 MMSI 区间的聚合逻辑不受影响。本次未重跑 bootstrap 或训练，也未重复完整缓存核查。

修复范围恰当，无需进一步修改或扩展审计。此次仍仅修改本 review，未导入 torch、未操作 Git；审阅结束。
