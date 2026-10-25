# 归一化修复独立审阅

日期：2026-10-26。范围：只读审阅 `run_repair.py`、`transfer.py` 及实际训练/推理依赖；唯一写入为本报告。已读本地 AGENTS、Codex Brain 相关规则，采用 karpathy-guidelines。没有训练、Git 操作、重新计算指标或制作图像。

## 发现

**[已解决，原 P1] 源回放基线缓存缺少 axis/A/U。** 初次审阅版本仅加载 `frontend/scores_seed{seed}.npz`，随后访问其中不存在的 `old_f['axis']`，会触发 `KeyError: 'axis'`。实际 NPZ 字段核查证据保留如下；该问题已不适用于当前实现。

修复复核：已查看当前 `run_repair.py:61` 起的代码，`source_checks()` 使用默认、未传 mask 的 `tr.frontend_predict(raw, sums, models, seed)` 重算原 baseline 的所有 frontend 阶段，因此 `old_f` 包含 axis/A/U；再用 `tr.compare_replay()` 将 frontend_base/m1/s12 与既有 scores 缓存核对数值及分类。这一修法成立，无需改为读取 base 缓存，也没有重新拟合或放宽不变性检查。

缓存键证据：

- `scores_seed17.npz`：obs_id, sample_id, role, base, B, m1, s1, s12, delta_m1, delta_m2, class_ids, seed。
- `base_seed17.npz`：obs_id, axis, A, D, U, B, base, class_ids。

当前无未解决的审阅问题。用户反馈源全表及两个迁移方向均已实际跑完，session 61797 exit 0；此运行状态依据用户反馈记录，本次仅复核修复行，不重新运行或扩展审阅。本报告不修改共享代码。

## 完整重放依赖

| 对象 | D/base 改变后的要求 | 代码依据 |
| --- | --- | --- |
| axis、A | 不受该 mask 影响 | `transfer.py:156` 的 axis/five 路径 |
| D、U、校准 T、frontend_base | 重算，T 的输入是新 U 的中心化值 | `transfer.py:176` 起 |
| M1、M2 及各自残差 | 重算；M1 输入含新 frontend_base，M2 输入含新 m1 | `transfer.py:181` 起；`train_frontend.py` 的 regional_raw |
| 图像响应摘要、tau、PCA、图像 token、32 源参考和检索 | 不依赖 D，可复用冻结来源；不能重新拟合 | `extract_frontend.py:208`、`transfer.py` 的 project_raw/source_bank/retrieve |
| M3 metric/transport 的 local | 输入是 token、参考和冻结 metric，不依赖 base；本实现仍实际调用 matcher 重算 | `train_m3.py:168`；`transfer.py:227` |
| M3 joint、VV、centered VH 残差 | 必须重算；local_raw 的后三维含中心化 base、最强竞争类和分差，两个 PolarHead 都取这三维 | `train_m3.py:68`；`sar_m123/m3.py:47` |
| spatial/frequency gate 与最终四臂 | 必须重算；gate_raw 同时含 base、新 dv/dh、竞争分差及 softmax | `train_m3.py:75`；`transfer.py:248` 起 |

`run_repair.py:34` 的 frozen_prediction 先调用带 mask 的完整 frontend，再把新 s12 转 float32 传入完整 m3_predict。后者重新构造 local_raw，执行 joint/polar heads，重新构造 gate_raw 并执行两种 gate。旧 predictions 只用于比较，没有作为残差或门控输入。因此“只替换 base、沿用旧 gate/极化残差”的问题在当前代码中不存在。

## mask 与冻结参数

`source_mask()` 仅取 `fit_train & usable`，并核对 scaler 的 fit_obs_id；`ptp == 0` 严格识别源常量通道，不按目标数据或截断后的 std 调阈值。训练时 `train_frontend.py` 对相同源可用行计算均值/std，再对 std 施加 1e-5 下限。只看该下限不能一般性地证明方差为零，当前直接检查源范围的做法正确。

既有 `normalization_diagnostic/source_floor_channels.csv` 记录 OSS1 VV10/VH8、OSS2 VV6/VH6 个通道的源 min=max=std=0；因此当前严格规则与已定位的恒零证据一致。先标准化、再屏蔽 centered 的对应通道，保留 level 和 usable，与原诊断干预一致。该证据支持修复已定位通道，不等于证明剩余所有低方差通道都稳定。

源恒零输入列不能给 Neighborhood 第一层对应权重提供数据梯度；目标激活后除以 1e-5，会通过已有权重放大。mask 将这些输入恢复为源拟合时的零值，不必更换网络宽度或模型参数。校准/M1/M2/M3 的历史 scaler、checkpoint、VH null、alpha 和 PCA 都应保持原值；当前调用链符合这一点。

源 fit 不变不自动代表 validation/utility/selection/evaluation 不变，这些集合可能出现源 fit 未激活的通道。脚本计划对其分别报告且不重新选模型，符合冻结修复的边界。该实验是历史下游参数上的定向反事实重放，不是重新训练整套模型，也不是独立盲测。

## 验证边界

初次完成静态依赖/调用审阅及实际 NPZ 字段核查；本次补充核对 source_checks 的修复并关闭原 P1。审阅者未启动完整回放，未独立重算指标。目标读取使用去标签表，三种子预测保存后才调用 evaluate_saved；修复输出路径位于独立目录，原正式预测只读。完整运行完成情况按上述用户反馈记录。
