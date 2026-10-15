# 支持残差漏项勘误：两版修正结果

2026-10-11。本次仅修正原支持调制增量的目标编码，未重训响应模型、未改数据、超参数、调制系数或选择规则，未评价目标港。两个修正版均已运行完成。

## 结论

**实现确有错误，但不是此前分类负结果的原因。** 修正后两版全部 OA/BA、逐样本类别、验证 BA 与增益选择不变；全部源验证选择仍为 gain=0。原始分数及 MSE 确实受影响，应引用本目录与增量版修正目录的新 MSE。

## 错误与修正

`condition_support_matching.py:82` 原增量残差是 `(2*onehot-prior_s)*present`，漏了目标编码中的 `-1`。baseline 调用的原岭回归本来就是正确的，错误仅存在于调制增量及相应残差返回值。

修正为 `R=(2*onehot-1-prior_s)*present`，分数为 `baseline + gain*(op*(multiplier-1))@R`。op 是有符号岭回归支持算子，不是概率。

对同一组调制，旧分数减新分数严格等于：

`gain * sum_s(op_qs*(multiplier_qs-1)) * present_class`

本次三个验证支持集和六个源留出 episode 的支持集均覆盖八类，所以该差异是每个查询的公共类别分数偏移：不改 argmax、类间 margin 或 softmax，但会改未中心化的分数 MSE。两版最大原始分数改变量分别为 0.379748821851、0.069078498292；类间偏移差异最大仅 4.44e-16，实测与上式误差均小于 3.5e-16。

缺类时它只偏移被支持覆盖的类别，因此可能改预测和分类损失。审查发现 native-fold 每种子 18/400 个训练 episode 缺类，那是新配对模块可复用的训练资源，**不是本次旧桥接两版评价出现了 18 个缺类 episode**。修正不能省略，也不能据此倒推旧负结果由它造成。

## 旧新数值

均为两种子、各三组源留出 episode 的等权均值；OA/BA 单位为百分比，MSE 为按查询类别平衡的八类原始分数误差。forced1 为固定诊断，不是所选部署结果。OA/BA 两列在旧新版本完全相同，故合并显示。

| 版本/读出 | 旧=新 OA | 旧=新 BA | 旧 MSE | 修正 MSE |
|---|---:|---:|---:|---:|
| 两版 baseline / 全部 selected | 42.619123 | 39.379981 | 0.384264267 | 0.384264267 |
| 原匹配 fixed forced1 | 42.723966 | 39.369158 | 0.389833181 | 0.388202844 |
| 原匹配 anchor forced1 | 42.695941 | 39.355677 | 0.388749793 | 0.387505920 |
| 原匹配 capacity forced1 | 42.722523 | 39.365940 | 0.388376877 | 0.387216897 |
| 原匹配 condition forced1 | 42.695199 | 39.356862 | 0.387720229 | 0.386763548 |
| 原匹配 sham forced1 | 42.695199 | 39.356862 | 0.387838640 | 0.386849772 |
| 原匹配 gated forced1 | 42.695941 | 39.355677 | 0.388682978 | 0.387456039 |
| 增量 fixed forced1 | 42.645705 | 39.396089 | 0.384931136 | 0.384880252 |
| 增量 anchor forced1 | 42.619123 | 39.379981 | 0.384264267 | 0.384264267 |
| 增量 capacity forced1 | 42.619123 | 39.379981 | 0.384027071 | 0.384020302 |
| 增量 condition forced1 | 42.568847 | 39.352357 | 0.383680180 | 0.383648053 |
| 增量 sham forced1 | 42.568847 | 39.352357 | 0.383745768 | 0.383719941 |
| 增量 gated forced1 | 42.619123 | 39.379981 | 0.384220516 | 0.384220488 |

原匹配 condition 仍比基线 OA +0.076076 个百分点、BA -0.023119，7 次救回/4 次误伤；移除有益/有害支持的平均 margin 量仍为 0.005149/0.002307，净值 -0.002842。增量 condition 仍为 OA -0.050277、BA -0.027623，0 次救回/2 次误伤；平均 margin +0.000731，但没有转换成准确率提升。曝光次数不是独立船数。两版 condition 与 sham 的 OA/BA 均相同，因此仍不能认定角度带来独特分类增益。

## 原始档与重放

- 原始目录 `../condition_support_matching_20261011/` 和 `../condition_support_increment_20261011/` 的 scores、CSV、PROTOCOL、REPLAY、报告均未覆盖或改写。旧 MSE 是旧错误公式的真实结果，不删除。
- 修正目录为本目录，以及 `../condition_support_increment_20261011_residual_fixed/`。新 PROTOCOL 的 residual_formula 为 `signed_target_minus_prior_v2`。
- 旧报告的“精确重放”仅证明旧实现可复现，不证明残差语义正确。现在 replay 依据协议识别旧档，明确打印 `LEGACY` 和 `legacy_error_reproduced=true`，按旧错误公式重放，但不覆盖旧 REPLAY.json。新档使用修正公式并写带版本的新 REPLAY.json。
- 重放均在 NumPy 导入前设置 OMP/OPENBLAS/MKL=1；未读查询标签、未拟合。四个目录各 78 个分数数组、36 个 multiplier 数组全部精确重放；其中修正版合计 156/72，旧档合计 156/72。
- `old_vs_fixed_summary.csv`、`old_vs_fixed_validation.csv`、`old_vs_fixed_scores.csv` 与 `ERRATUM_COMPARISON.json` 保存每版本的完整对照；baseline、selected scores、multiplier 和选择结果均精确一致。

## 测试与入口

新增缺类、类别不均衡、非单位 multiplier 的回归测试，用逐查询原始岭回归求解作独立参照。旧实现确实失败，最大偏差 0.02677986；修正后全部 7 项测试通过。旧测试只对单位 multiplier 检查 baseline，增量恒为零，故漏掉了此错误。

运行入口：`run_condition_support_residual_fixed.py`。两个目录以 exist_ok=False 创建，已存在时不会覆盖结果。

重放入口：`replay_condition_support_matching.py --out <目录>`，既支持原匹配/增量旧档，也支持两版修正目录；旧增量 replay 入口仍可使用且明确报告 legacy。没有修改主代理的 task_pair_relation 文件、memory 或 skill，也没有追加实验。
