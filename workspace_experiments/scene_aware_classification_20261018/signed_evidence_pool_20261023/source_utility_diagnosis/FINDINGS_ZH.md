# 源效用与选择目标诊断

2026-10-23。仅重放保存的 selected heads，未拟合、未重新选 checkpoint、未改父文件或 Git。自动发现父 42 头及 signed_only_retrain 24 头，共 66 头、198 个 head-role 比较；最终重放约 5.88 秒。

## 直接结论

**OA 损失不是到测试迁移时才出现。** 全部 66 个头的 selected 源验证 balanced NLL 比 epoch0 A 更低，但其中 47 个源验证 OA 更低，38 个源验证 BA 更低。按 arm/seed 合并六折后，全部 11 个运行的源验证 OA 都下降。选择程序确实执行了指定风险目标，不是误选了较差 balanced NLL。

46 个测试 OA 下降的头中，40 个在源验证已下降；另 6 个源验证 OA 不下降而测试下降。后者是实际观察到的 transfer discrepancy，但不能据此把全部损失归于域偏移，也不能把前者提升为“损失函数单独导致退化”的因果证明。没有重训不同目标，未比较所有 epoch 的 OA 最优点；本诊断只说明已选点与 epoch0 的效用取舍。

## 口径与来源

- 使用当前 `train_signed.CACHE/signed_features.npz`、现有 `descriptors/Correction` 和每个 checkpoint 保存的 mu/scale/state；按对应 held 索引重组 inner 矩与 OOF A，验证/测试使用 outer 矩与原 A。未重新标准化、重拟合或抽图。
- source OOF train 为 **18525 个 fold-sample 事件 / 5520 个独立 ID**，不是 18525 独立船图；基础读出 OOF，但修正头在这些训练事件上拟合过，不是修正头的泛化评价。
- source validation 为 5121 个独立 ID；test 为 5695 个独立 ID。各折内隔离沿用原协议，但跨角色/折复用同一队列，不能把 train/validation/test 汇总当三套独立外部数据。
- pooled 指标按事件汇总；多 seed 表为指标算术平均，不是预测集成。逐折选点实际按各折 balanced NLL，而非下面的 pooled balanced NLL 重新选点。
- saved 源验证选中 loss 重放最大误差 5.55e-17；测试 score 对原 CSV 最大误差 8.88e-16。此前已审阅的类别轴/隔离不重新全量审计。没有新实验臂。

## 三个角色的效用

A 的 OA/BA/NLL/balNLL 分别为：train OOF 81.473684/79.741364/0.532202/0.560870，validation 82.132396/80.795853/0.522488/0.546300，test 81.650571/80.265736/0.526008/0.548927。

下表均为 selected 减 A；OA/BA 单位 pp，风险负值表示改善。绝对值、各 seed 与各折完整保存在 CSV。

| arm | role | ΔOA | ΔBA | ΔNLL | Δbalanced NLL |
|---|---|---:|---:|---:|---:|
| linear_distribution | train OOF | -0.5506 | +0.0198 | -0.01371 | -0.01903 |
| | validation | -0.8397 | -0.3096 | -0.01631 | -0.02405 |
| | test | -0.5092 | +0.0438 | -0.01144 | -0.01947 |
| conditional_std | train OOF | -0.2159 | +0.1506 | -0.02538 | -0.02804 |
| | validation | -0.3189 | +0.0494 | -0.02603 | -0.03172 |
| | test | -0.3161 | -0.1122 | -0.02244 | -0.02473 |
| conditional_signed | train OOF | -0.1134 | +0.5898 | -0.02852 | -0.04201 |
| | validation | -0.6314 | +0.0505 | -0.02210 | -0.03835 |
| | test | -0.3161 | +0.3240 | -0.01668 | -0.03137 |
| signed_only | train OOF | -0.0288 | +0.3730 | -0.02021 | -0.02661 |
| | validation | -0.5923 | -0.1563 | -0.01362 | -0.02421 |
| | test | -0.2400 | +0.3343 | -0.00980 | -0.02109 |
| scalar_signed | train OOF | -0.5290 | -0.0540 | +0.00969 | -0.00114 |
| | validation | -0.7030 | -0.2447 | +0.00928 | -0.00212 |
| | test | -0.4917 | -0.0610 | +0.01003 | -0.00132 |

scalar_signed 最直接展示类均衡风险与按样本风险的分离：三个角色 balanced NLL 均略改善，普通 NLL 却均变差。conditional_std 在验证 BA 平均略升、测试略降；conditional_signed 与 signed_only 测试 BA 反而比验证增幅更大。因此不是所有效用一起发生单向 transfer collapse。

| arm | 头数 | val 风险改善但 OA 下降 | val 风险改善但 BA 下降 | val OA 不降、test OA 下降 |
|---|---:|---:|---:|---:|
| linear_distribution | 6 | 6 | 4 | 0 |
| conditional_std | 18 | 10 | 10 | 3 |
| conditional_signed | 18 | 13 | 9 | 2 |
| signed_only | 18 | 12 | 10 | 1 |
| scalar_signed | 6 | 6 | 5 | 0 |

所有 selected epoch 均大于 0。signed_only 的测试净变化 -3/-5/-33，不能用父模型事后删除 std 路径的正结果代替重训结果。

## 类别与边界解释

类顺序为 bulk/fishing/general。validation 类数 2876/1019/1226，test 为 3170/1147/1378，组成比例相近但并非同样本/同读出。下表为每 seed 平均净救回向量，不是独立事件平均置信区间。

| arm | validation 净变化 (bulk, fishing, general) | test 净变化 (bulk, fishing, general) |
|---|---|---|
| linear_distribution | (-59, -11, +27) | (-57, -7, +35) |
| conditional_std | (-34, -6.67, +24.33) | (-29.33, -16, +27.33) |
| conditional_signed | (-62.67, -8.67, +39) | (-59.33, -10.67, +52) |
| signed_only | (-45.33, -7, +22) | (-49.33, -2, +37.67) |
| scalar_signed | (-46, +3, +7) | (-44, +3, +13) |

多数臂“散货损失、普通货船获益”的方向已在源验证出现。conditional_signed 的 validation/test 各类 NLL 变化分别为 (+0.01309,-0.03034,-0.09779) / (+0.01651,-0.02210,-0.08851)，普通货船风险改善可抵消散货损失。conditional_std 三类 NLL 都改善，但散货与渔船 recall 仍下降，说明不仅是类别加权，还存在概率风险与 argmax 不等价。

边界按 **A 真类分数减最强错类分数的绝对值** 预先固定为 ≤0.25、(0.25,1]、>1，不按结果调阈值。再按 A 对/错分组；使用标签仅作事后诊断，不进入模型。近边界 validation 有 151 个 A 错、211 个 A 对；test 有 184 个 A 错、220 个 A 对。

| arm | validation 近边界救回 / 误伤 | test 近边界救回 / 误伤 |
|---|---|---|
| linear_distribution | 26 / 64 | 38 / 66 |
| conditional_std | 33.33 / 48.33 | 31 / 45.67 |
| conditional_signed | 54.33 / 81.33 | 60.33 / 76.67 |
| signed_only | 40.67 / 66.33 | 54 / 68.67 |
| scalar_signed | 17 / 53 | 19 / 47 |

以 conditional_signed 为例：validation 原本正确且 margin>1 的 3143 例无误伤，其 NLL 平均降低 0.06300；原本错误且 margin<-1 的 388 例平均仅救回 0.33，NLL 却增加 0.35510。test 对应 3563/443 例分别为 NLL -0.06645/+0.42477，均无翻转。正确高余量组对总体 balanced NLL 的贡献为 validation -0.04550、test -0.04849；错误高余量组为 +0.02696/+0.03396。其余组贡献后合成 -0.03835/-0.03137。

因此连续风险改善可以主要来自大量不改变预测的概率调整，同时边界误伤超过救回。这里计算的是组内平均和加权贡献，不表示每一例概率变化同向，也不证明网络有意“牺牲”某组。test 错误高余量风险惩罚比 validation 大，是描述性差异，不能独立识别其由观测条件、reader 变化还是样本组成造成。

## 复用原图与停止边界

本次实际复看父目录四张既有 VV/VH 面板，不重新抽图、不新增挑图搜索；固定 conditional_signed seed17。四例均为散货，不能代替类别统计。当前 checkpoint 的重放值保存于 `existing_case_replay.csv`，包含其余既有案例与 signed-only/scalar。

- `pair_0031218` 救回：VV64/VH72，中心团状高亮；margin -0.107590 -> +0.110199，NLL 0.814028 -> 0.658400。
- `pair_0031199` 误伤：VV64/VH88，长亮结构、框外下方延伸和明显周围纹理；margin +0.156864 -> -0.080524，NLL 0.826896 -> 0.905490。
- `pair_0030906` CC：VV72/VH80，近纵向长亮结构，VV 周围纹理较亮；margin +1.772555 -> +2.216690，NLL 0.227792 -> 0.115412。是概率改善而 top1 不增加的具体例子。
- `pair_0030918` WW：VV64/VH72，斜长亮结构与横向重复亮响应；margin -0.152923 -> -0.470854，NLL 0.813088 -> 0.967347。不把重复亮响应直接命名为导致错误的某种散射机制。

最有证据的解释是：**指定 balanced NLL 下的已选解，在源端已形成与 OA/BA 不一致的概率和类别取舍；测试有额外差异，但不是唯一解释。** 不能由此宣称有符号信息无效、balanced NLL 本身错误或换成 OA 选择必然有效。未保存所有 epoch 权重，本诊断不能反事实比较其他 epoch 的 OA/BA，也不追加该实验。

交付索引：`diagnose.py`；`inventory.csv`（66 头及真实 cache）；`head_role_metrics.csv`（198 比较）；`role_pooled.csv`/`arm_role_summary.csv`；`selection_vs_transfer.csv`/`selection_summary.csv`；`class_role_metrics.csv`；`boundary_role_metrics.csv`（含可加和的 balanced NLL 贡献）；`existing_case_replay.csv`；`replay_checks.csv`/`summary.json`。所有生成文件只在本目录。
