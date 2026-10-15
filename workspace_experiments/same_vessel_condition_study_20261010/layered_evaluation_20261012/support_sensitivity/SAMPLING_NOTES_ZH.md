# 支持集抽样独立核查

2026-10-12。仅阅读代码、记录及现有 NPZ；未运行训练、模型重拟合或原实验入口。唯一新增文件为本笔记。

## 1. 源域：3 组真实 draw，跨 context 与评估 seed 标签共享

- 原始生成位置为 [run_incidence_transport.py:37](E:/临时会话/same_vessel_condition_study_20261010/run_incidence_transport.py:37)：先以 `default_rng(91011+j)` 生成三个 held episode，再进入第 71 行的模型 seed 循环；第 115-119 行将同一组索引存为不同 seed 文件。因此 `31415/27182` 的评估支持集标签不代表六次独立抽样。
- 实际算法见 [explicit_candidate_projection.py:51](E:/临时会话/same_vessel_condition_study_20261010/explicit_candidate_projection.py:51)：按 MMSI 分组，均匀、无放回选择 64 艘船，每船均匀取一条观测；query 是其余船的全部观测。没有类别分层或每类配额；这是船均匀抽样，不是观测行均匀抽样。`groupby('mmsi')` 默认不包含缺失 MMSI 组。
- [case_relation_headroom.py:174](E:/临时会话/same_vessel_condition_study_20261010/case_relation_headroom.py:174) 直接读上述缓存，并在第 180-181 行要求 support/query 均属于 held cohort、MMSI 完全不交叠；三个 context 使用相同路径。LA/Klang/Santos 在这里是同一源域 cohort 的三种几何版本。
- [case_relation_student.py:76](E:/临时会话/same_vessel_condition_study_20261010/case_relation_student.py:76) 对每个模型复用相同评估索引；第 212-217 行的两个训练 seed 则是真实不同训练。另须区分训练 episode：第 117-126 行在 vessel fold 内按行无放回抽 support64，再排除支持船后抽 query64；训练支持不保证 64 艘不同船，也未类别分层。
- [case_support_completion.py:158](E:/临时会话/same_vessel_condition_study_20261010/case_support_completion.py:158) 复用 headroom 缓存，第 181 行记录唯一 split 数；它自己的 source-OOF 选剂量 episode 在第 119-122 行，使用 `101200+fold*10+draw` 调用同一 `identity_episode`，同 fold/draw 跨 context 共享规则，与最终 held 三组不可混算。
- 本次只读比对三个 context 的全部六个 headroom 标签，support/query 索引均与对应三组源缓存完全一致。按 class_id 0..7 的支持计数分别为 `[19,6,21,2,3,4,4,5]`、`[16,7,18,5,4,5,6,3]`、`[24,6,15,2,6,3,4,4]`；query 数为 663、627、645。不能将三 context 乘两标签写成 18 次独立支持抽样。

## 2. 目标域：每港 5 组行均匀随机 64，仅排除支持行

- [prepare_learned_frequency.py:40](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/prepare_learned_frequency.py:40) 读取历史 policies；第 51-56 行将 `support_rows` 映射为本港局部索引，保留原 `eval_masks`。上游 [analyze_c4_conditions.py:95](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/analyze_c4_conditions.py:95)、第 118-120 行原样传递 mask/support；历史生成规则在 [run_ports_projection_completion.py:482](E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/run_ports_projection_completion.py:482) 明确校对为 `default_rng(draw).choice(q,64,replace=False)`。
- 本次直接读取三个当前 `target.npz`：每个 `support` 均为 `(5,64)`，逐 draw 精确等于 `default_rng(draw).choice(n,64,replace=False)`，draw 为 0..4；每个 mask 精确等于全 True 后仅将当组 support 行置 False。因此是港内行均匀、单次无放回、跨 draw 可重叠，非类别分层，也非船均匀。三港复用 RNG 编号不代表相同支持案例或训练 seed。
- [support_completion_target.py:19](E:/临时会话/same_vessel_condition_study_20261010/support_completion_target.py:19) 至第 31 行对齐历史 rows/support/masks，第 46-48 行仅检查支持行不入评价；第 98-109 行按该 mask 评价并统计同船交叠。已有 [scope.csv:2](E:/临时会话/same_vessel_condition_study_20261010/support_completion_target_20261012/scope.csv:2) 显示每组只有 55-64 个已知支持 MMSI，query 中 69-244 行与支持共享已知 MMSI；部分支持/查询身份缺失。64 标签不能写成 64 艘独立船，也不能称目标 query 船身份隔离。
- **base 已使用同组 64 例。** [full_teacher_run.py:169](E:/临时会话/same_vessel_condition_study_20261010/full_teacher_run.py:169) 至第 177 行逐组用 `support`、`support_labels[draw]` 构建 base kernel 并调用 `score`。completion 第 25、49-68 行保持该 draw 的 base 不变，仅更新 expert 后融合。“base 不变”只指新旧方法间固定，不能解释成 base 不依赖支持抽样。

## 3. 随机 64 绝不等于每类 8 例

上述源、目标采样函数都没有类别配额。目标缓存实例如下，计数顺序均为 class_id 0..7：

| 港 / draw | 实际支持类别计数 |
| --- | --- |
| Los Angeles / 0 | 10, 3, 3, 10, 7, 23, 5, 3 |
| Port Klang / 0 | 19, 0, 3, 17, 14, 5, 4, 2 |
| Port Klang / 3 | 23, 0, 6, 14, 12, 5, 4, 0 |
| Santos / 1 | 39, 0, 2, 4, 9, 2, 7, 1 |

支持缺类是现有数据中真实发生的情况；不能用训练/读出中的类别平衡权重或 BA 的宏平均定义，倒推出支持集经过类别分层。

## 4. 对当前缓存敏感性评价的约束

- 64 例的组成与具体案例会改变支持求解及 OA/BA，应纳入评价。旧协议每 draw 的 query 也改变，直接比较其 OA/BA 波动同时混入支持变化与被评价样本变化。
- 主分析按 context 固定跨 draw 共同 query，逐 draw 配对新旧方法，再报告 OA/BA、差值与支持组成/波动，能隔离掉 query 集合变化。目标共同 query 为 LA 318、Klang 3118、Santos 1357 行；源域必须用实际 query ID 交集，保留原有整船排除，不能仅扣掉 192 条支持行。
- 源域重复评估 seed 标签去重，学生真实训练 seed 单独保留；target base/expert/fusion 都按同组 draw 配对。三组/五组波动仅描述现有支持抽样，不能当作独立船级置信区间；组成与效果的关联也不足以单独证明某类数量是根因。
- 历史逐 draw query 结果与共同 query 结果应分别标明。当前分析沿用已定义采样协议，不把随机64改成8-way 8-shot，也不将目标行排除擅自改成整船排除。
