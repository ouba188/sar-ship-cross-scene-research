# 增量版残差勘误

2026-10-11。完整两版说明与数值表见 `../condition_support_matching_20261011_residual_fixed/ERRATUM_ZH.md`。

本目录是旧 `condition_support_increment_20261011` 的独立修正版，修正 `(2*onehot-prior_s)*present` 中漏掉的 `-1`。原始档全部保留，未覆盖。

所有 selected 仍为 gain=0；baseline 与 selected OA/BA/MSE 为 42.619123/39.379981/0.384264267。condition forced1 的 OA/BA 仍为 42.568847/39.352357，MSE 从 0.383680180 修正为 0.383648053；sham 同 OA/BA，修正 MSE 为 0.383719941。逐样本分类不变，仍 0 次救回、2 次误伤，原负结论未被推翻。

本轮支持集全八类覆盖，漏项只造成公共类别分数偏移；缺类情况下不保证这一性质，已新增能在旧实现上失败的缺类非零调制回归测试，修正后全 7 项通过。

本目录 78 个分数数组、36 个 multiplier 数组精确重放，REPLAY.json 明确记录 `signed_target_minus_prior_v2`。旧档重放仅代表错误公式可复现，不是公式正确性证明；新 replay 自动标注 legacy 且不改写旧档重放记录。

完整旧新对照在本目录 `old_vs_fixed_summary.csv`、`old_vs_fixed_validation.csv`、`old_vs_fixed_scores.csv`、`ERRATUM_COMPARISON.json`。全部运行已结束，无主模型修改、无目标港评价、无扩展实验。
