# 频带互补性结果

25 个任务、125 个 support draw 的预测全部冻结后读取目标标签；样本 ID、行索引、support 标签均已对应核验。

这是冻结特征及预测层面的描述性比较，不构成物理频率或因果解释。Oracle 使用目标真值，只是不可部署的上界；没有用它选择部署策略。

选择器在源域按 episode 港口留出评估，但表征训练共享源港口，不是独立表征留港验证。动态策略是硬候选选择；选择多样性或权重变化不能证明学到了有效注意力。

## 部署策略与静态对照（base）

| dataset | mode | oa | ba | delta_oa | delta_ba |
| --- | --- | --- | --- | --- | --- |
| f0921 | dynamic_scores | 30.4433 | 26.0667 | -0.0831 | 0.0319 |
| f0921 | dynamic_support | 30.5001 | 26.1104 | -0.0263 | 0.0757 |
| f0921 | fixed_equal | 30.4945 | 26.1005 | -0.0319 | 0.0657 |
| f0921 | learned | 30.5264 | 26.0348 | 0.0000 | 0.0000 |
| f0921 | source_selected | 30.5078 | 26.1442 | -0.0186 | 0.1094 |
| f0921 | source_static | 30.5078 | 26.1442 | -0.0186 | 0.1094 |
| oss2 | dynamic_scores | 32.8987 | 31.0156 | -0.1576 | -0.2627 |
| oss2 | dynamic_support | 33.0725 | 31.0773 | 0.0163 | -0.2010 |
| oss2 | fixed_equal | 32.6922 | 30.9430 | -0.3640 | -0.3353 |
| oss2 | learned | 33.0562 | 31.2783 | 0.0000 | 0.0000 |
| oss2 | source_selected | 33.0834 | 31.2643 | 0.0272 | -0.0140 |
| oss2 | source_static | 33.0834 | 31.2643 | 0.0272 | -0.0140 |
| self24ports | dynamic_scores | 61.2632 | 52.4976 | 0.0677 | -0.0018 |
| self24ports | dynamic_support | 61.3293 | 52.4505 | 0.1338 | -0.0489 |
| self24ports | fixed_equal | 61.0049 | 52.3278 | -0.1907 | -0.1716 |
| self24ports | learned | 61.1956 | 52.4994 | 0.0000 | 0.0000 |
| self24ports | source_selected | 61.3028 | 52.5125 | 0.1073 | 0.0130 |
| self24ports | source_static | 61.2952 | 52.4919 | 0.0997 | -0.0075 |

## Oracle 互补上界

Oracle 保留 learned 的所有正确预测，只在 learned 错误且候选正确时修复。最佳单频带及最佳 learned+单频带 oracle 均在每个 task/draw 上用标签描述性选择，OA 和 BA 分别取最大值；全四频带相对最佳单频带 oracle 的增益才衡量额外多频带多样性。

| dataset | stage | family | oracle_delta_oa | oracle_delta_ba | union_minus_best_single_oracle_oa | union_minus_best_single_oracle_ba |
| --- | --- | --- | --- | --- | --- | --- |
| f0921 | base | fixed | 1.6308 | 1.6402 | 0.4903 | 0.4865 |
| f0921 | base | selective | 1.1176 | 1.1810 | 0.0898 | 0.0821 |
| f0921 | frozen_route | fixed | 1.4909 | 1.5167 | 0.4513 | 0.4477 |
| f0921 | frozen_route | selective | 1.0238 | 1.0947 | 0.0813 | 0.0752 |
| oss2 | base | fixed | 1.5213 | 1.3905 | 0.5325 | 0.3919 |
| oss2 | base | selective | 1.0541 | 1.0928 | 0.0380 | 0.0269 |
| oss2 | frozen_route | fixed | 1.3366 | 1.2711 | 0.4292 | 0.3339 |
| oss2 | frozen_route | selective | 0.9617 | 1.0255 | 0.0326 | 0.0217 |
| self24ports | base | fixed | 1.4378 | 1.3635 | 0.2740 | 0.2163 |
| self24ports | base | selective | 0.9234 | 0.9206 | 0.0333 | 0.0221 |
| self24ports | frozen_route | fixed | 1.0595 | 0.9841 | 0.1872 | 0.1316 |
| self24ports | frozen_route | selective | 0.7081 | 0.6849 | 0.0217 | 0.0133 |

只计算 base 与固定 learned admission mask 的 frozen_route oracle；不拼接独立重路由的 full 预测，避免突破 5% 专家预算。

## 机会捕获与代价（fixed 家族，base）

| dataset | mode | opportunities | captured | harms | capture_proportion |
| --- | --- | --- | --- | --- | --- |
| f0921 | dynamic_scores | 10064 | 7213 | 7726 | 0.7167 |
| f0921 | dynamic_support | 10064 | 6926 | 7088 | 0.6882 |
| f0921 | fixed_equal | 10064 | 4500 | 4711 | 0.4471 |
| f0921 | source_selected | 10064 | 6942 | 7057 | 0.6898 |
| f0921 | source_static | 10064 | 6942 | 7057 | 0.6898 |
| oss2 | dynamic_scores | 280 | 156 | 185 | 0.5571 |
| oss2 | dynamic_support | 280 | 190 | 187 | 0.6786 |
| oss2 | fixed_equal | 280 | 133 | 201 | 0.4750 |
| oss2 | source_selected | 280 | 127 | 122 | 0.4536 |
| oss2 | source_static | 280 | 127 | 122 | 0.4536 |
| self24ports | dynamic_scores | 2419 | 1242 | 1334 | 0.5134 |
| self24ports | dynamic_support | 2419 | 1374 | 1329 | 0.5680 |
| self24ports | fixed_equal | 2419 | 1218 | 1662 | 0.5035 |
| self24ports | source_selected | 2419 | 464 | 415 | 0.1918 |
| self24ports | source_static | 2419 | 372 | 363 | 0.1538 |

capture_proportion = 汇总 captured / 汇总 opportunities，不是各任务比例的平均。harms 是 learned 正确而策略错误的数量；捕获机会不等于净收益。selective 家族的捕获仅表示机会集合重叠，动态策略实际候选仍是 fixed 家族。

## 源域选择

| dataset | rule | tasks |
| --- | --- | --- |
| f0921 | fixed_HL | 1 |
| oss2 | fixed_LH | 1 |
| self24ports | dynamic_scores | 4 |
| self24ports | dynamic_support | 5 |
| self24ports | fixed_HL | 1 |
| self24ports | fixed_LH | 7 |
| self24ports | learned | 6 |

## 输出口径

OA/BA/F1 为百分数，增益为百分点。summary 先平均任务内五个 draw，再等权平均任务；计数按 query-draw 累加，同一样本跨 draw 会重复，不能视为独立样本。BA 使用当前评估集合出现的类别，macro F1 固定八类。

metrics.csv、task_means.csv、summary.csv、draw_metrics.csv、class_metrics.csv、events.csv 给出三种路由阶段；complementarity.csv 给出每个任务/draw 的 oracle；band_events.csv 含独有救回；rescue_overlaps.csv 含两两交并计数；capture_summary.csv 用总计数计算捕获率。

各 target 下 report_opportunities.npz 保存 rows、eval_masks、choices、每 query 的两个频带家族救回频带数及两个动态策略捕获标记。家族轴为 fixed/selective，动态轴为 dynamic_scores/dynamic_support；非评估行必须用 eval_masks 排除。这些均为报告产物，不得回流选择器。
