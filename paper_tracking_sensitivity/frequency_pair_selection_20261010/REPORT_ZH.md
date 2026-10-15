# 频率类别/类别对选择器冻结报告

全部25个匹配任务完成冻结后才读取目标标签，按原始manifest样本ID、rows、support和eval_masks对齐。目标标签仅用于事后评估和诊断，不参与阈值或策略选择。根因与研究结论由主线另写。

## Source-selected full

| dataset | tasks | ba | draws | delta_ba_vs_previous_selected | positive_draws_ba_vs_previous_selected | delta_ba_vs_smooth_parent | positive_draws_ba_vs_smooth_parent | delta_ba_vs_learned | positive_draws_ba_vs_learned |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | 1 | 26.4777 | 5 | -0.0110 | 1 | -0.0110 | 1 | 0.1184 | 3 |
| oss2 | 1 | 32.4695 | 5 | 0.0000 | 0 | 0.0000 | 0 | 0.0971 | 3 |
| self24ports | 23 | 54.2258 | 5 | -0.0003 | 2 | 0.0120 | 3 | -0.0125 | 3 |

previous_selected精确重放PREV source_selected；smooth_parent仅按PREV源域验证均值从smooth/smooth02/smooth05选择，继承固定阈值。class_local与pair_local继承该阈值。当前策略按相同源域episode base BA均值从previous_selected/class_local/pair_local选择，同分旧策略优先。

## 口径

OA、BA、固定八类macro F1单位为百分数，差值为百分点。summary先平均每任务五次draw，再在各数据集内等权平均任务，self24ports为task23；不混合all25。draw_metrics固定draw后等权平均任务。positive_draws统计五个均值中严格正差次数；positive_task_draws累计任务/draw正差次数。五次support抽样不是独立训练重复。

base不交接；full使用自身路由；frozen_previous_route全部模式使用PREV source_selected admission mask。各阶段均与同阶段previous_selected、smooth_parent、learned比较。

## Capture

| dataset | stage | mode | capture_rate | opportunities | captured | harms |
| --- | --- | --- | --- | --- | --- | --- |
| f0921 | base | class_local | 0.6300 | 12259 | 7849 | 7420 |
| f0921 | base | learned | 0.0000 | 12259 | 0 | 0 |
| f0921 | base | pair_local | 0.6327 | 12259 | 7857 | 7330 |
| f0921 | base | previous_selected | 0.6422 | 12259 | 8012 | 7465 |
| f0921 | base | smooth_parent | 0.6422 | 12259 | 8012 | 7465 |
| f0921 | base | source_selected | 0.6300 | 12259 | 7849 | 7420 |
| oss2 | base | class_local | 0.2214 | 392 | 89 | 97 |
| oss2 | base | learned | 0.0000 | 392 | 0 | 0 |
| oss2 | base | pair_local | 0.2065 | 392 | 83 | 89 |
| oss2 | base | previous_selected | 0.2286 | 392 | 92 | 96 |
| oss2 | base | smooth_parent | 0.2286 | 392 | 92 | 96 |
| oss2 | base | source_selected | 0.2286 | 392 | 92 | 96 |
| self24ports | base | class_local | 0.3173 | 2617 | 787 | 782 |
| self24ports | base | learned | 0.0000 | 2617 | 0 | 0 |
| self24ports | base | pair_local | 0.3252 | 2617 | 805 | 806 |
| self24ports | base | previous_selected | 0.3208 | 2617 | 789 | 727 |
| self24ports | base | smooth_parent | 0.3199 | 2617 | 791 | 798 |
| self24ports | base | source_selected | 0.3214 | 2617 | 791 | 727 |

机会为learned base错误且至少一个preserved频带正确的并集；captured为其中被当前base救回的数量，harms为learned正确却被改错的数量。计数累计query-draw事件，重复样本不是独立样本。

## 精确冻结路由分解

| dataset | mode | fixed_previous_route_delta_ba | changed_route_delta_ba | full_delta_ba |
| --- | --- | --- | --- | --- |
| f0921 | class_local | -0.0144 | 0.0034 | -0.0110 |
| f0921 | learned | -0.1639 | 0.0346 | -0.1294 |
| f0921 | pair_local | -0.0087 | 0.0068 | -0.0019 |
| f0921 | previous_selected | 0.0000 | 0.0000 | 0.0000 |
| f0921 | smooth_parent | 0.0000 | 0.0000 | 0.0000 |
| f0921 | source_selected | -0.0144 | 0.0034 | -0.0110 |
| oss2 | class_local | -0.0188 | -0.0031 | -0.0219 |
| oss2 | learned | -0.0968 | -0.0003 | -0.0971 |
| oss2 | pair_local | 0.0017 | 0.0085 | 0.0101 |
| oss2 | previous_selected | 0.0000 | 0.0000 | 0.0000 |
| oss2 | smooth_parent | 0.0000 | 0.0000 | 0.0000 |
| oss2 | source_selected | 0.0000 | 0.0000 | 0.0000 |
| self24ports | class_local | -0.0252 | 0.0204 | -0.0048 |
| self24ports | learned | -0.0826 | 0.0948 | 0.0122 |
| self24ports | pair_local | -0.0066 | 0.0279 | 0.0213 |
| self24ports | previous_selected | 0.0000 | 0.0000 | 0.0000 |
| self24ports | smooth_parent | -0.0308 | 0.0185 | -0.0123 |
| self24ports | source_selected | -0.0001 | -0.0003 | -0.0003 |

逐任务/draw计算当前frozen减previous full，以及当前full减当前frozen，两项之和等于full差值；这是预测算术分解，不能直接解释为根因。

## Utility与coverage

| dataset | utility | auc | rescue_events | harm_events | events | auc_defined |
| --- | --- | --- | --- | --- | --- | --- |
| f0921 | class | 0.5230 | 22302 | 22470 | 44772 | 20 |
| f0921 | pair | 0.5321 | 22302 | 22470 | 44772 | 20 |
| f0921 | smooth | 0.5301 | 22302 | 22470 | 44772 | 20 |
| oss2 | class | 0.5164 | 689 | 916 | 1605 | 20 |
| oss2 | pair | 0.5208 | 689 | 916 | 1605 | 20 |
| oss2 | smooth | 0.5199 | 689 | 916 | 1605 | 20 |
| self24ports | class | 0.5663 | 5640 | 7455 | 13095 | 404 |
| self24ports | pair | 0.5658 | 5640 | 7455 | 13095 | 404 |
| self24ports | smooth | 0.5694 | 5640 | 7455 | 13095 | 404 |

| dataset | band | coverage_fraction | mean_rho | n | covered |
| --- | --- | --- | --- | --- | --- |
| f0921 | HH | 0.0259 | 0.0101 | 617125 | 15974 |
| f0921 | HL | 0.0301 | 0.0118 | 617125 | 18606 |
| f0921 | LH | 0.0065 | 0.0025 | 617125 | 3991 |
| f0921 | LL | 0.0315 | 0.0122 | 617125 | 19427 |
| oss2 | HH | 0.0221 | 0.0082 | 18405 | 406 |
| oss2 | HL | 0.0223 | 0.0086 | 18405 | 410 |
| oss2 | LH | 0.0108 | 0.0043 | 18405 | 199 |
| oss2 | LL | 0.0261 | 0.0104 | 18405 | 480 |
| self24ports | HH | 0.0183 | 0.0081 | 182775 | 3703 |
| self24ports | HL | 0.0179 | 0.0080 | 182775 | 3638 |
| self24ports | LH | 0.0069 | 0.0029 | 182775 | 962 |
| self24ports | LL | 0.0192 | 0.0086 | 182775 | 3643 |

AUC仅在各频带相对learned base的rescue/harm事件上计算，rescue为正类；缺少任一类留空。先平均任务内有效band/draw，再等权平均有效任务，auc_defined记录有效项数。coverage在全部评估query上计算rho>0比例和rho均值，逐任务/draw/band保存在coverage_probe.csv。

rho=min(neff_a,neff_b)/(min(neff_a,neff_b)+2)，相同8近邻中任一query预测类缺失时为0，singleton不超过1/3。local_contrasts.csv直接保存逐任务/draw/stage的class_local减smooth_parent与pair_local减class_local。

source_target_comparison连接冻结源域选择与目标三阶段任务均值，仅用于描述。metrics/class_metrics/task_means/summary/draw_metrics/events/draw_events保存指标与事件；capture_probe/route_probe/utility_probe及逐频带表保存诊断。无图。
