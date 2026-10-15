# 连续 margin 选择器冻结报告

本报告描述冻结预测与事后诊断；机制解释和结论边界见 FINDINGS_ZH.md。全部25个匹配任务的 COMPLETE.json 及全部源域规则在读取目标标签前检查；样本ID、rows、support、eval_masks、support标签直接对齐。

## 主要比较：source_selected 对 hard（full）

| dataset | ba | delta_ba_vs_hard | positive_draws_ba_vs_hard | delta_ba_vs_standalone_local | positive_draws_ba_vs_standalone_local | draws |
| --- | --- | --- | --- | --- | --- | --- |
| all25 | 52.2463 | -0.0543 | 1 | -0.0675 | 2 | 5 |
| f0921 | 26.4887 | 0.1257 | 4 | 0.1238 | 4 | 5 |
| oss2 | 32.4695 | 0.0741 | 3 | 0.0095 | 3 | 5 |
| self24ports | 54.2261 | -0.0678 | 1 | -0.0792 | 1 | 5 |

source_selected 仅从 hard、smooth、smooth02、smooth05 中按源域验证 episode 平均 base BA 选择，同分按固定规则顺序。source_delta_ba 相对原 learned；报告只核对冻结规则，不重选策略。源域获胜不意味着目标增益，目标差值仅来自冻结预测评估。源域端口与表征训练共享，不能称为独立表征验证；目标数据属于此前使用过的开发数据。

## 指标口径

base 无专家交接；full 使用每模式自身路由；frozen_hard_route 对全部模式统一使用 PREV preserved_support_local admission mask。learned、standalone_local、hard 的 base/full/selected 逐项核对旧基线；新 frozen 阶段统一重放 hard 路由，不与旧 learned 路由阶段混用。

OA、BA、macro_f1_8 为百分数，delta 为百分点。BA 使用评估集合实际出现的类别，F1固定八类。所有三阶段指标提供 vs_hard、vs_learned、vs_standalone_local。summary 先平均任务内五次draw，再等权平均任务；all25 同样等权平均25任务。draw_metrics 在固定draw内等权平均任务。positive_draws 统计五个任务等权draw均值中严格正差的次数；positive_task_draws 统计任务/draw正差次数。五次draw是support抽样，不是训练随机种子或独立显著性重复。

## Capture

| dataset | stage | mode | capture_rate | added_capture_rate | opportunities | captured | harms | standalone_opportunities | added_opportunities | captured_added | added_keep_main | added_wrong_band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all25 | base | hard | 0.1262 | 0.0569 | 15268 | 2448 | 2327 | 14886 | 686 | 64 | 617 | 5 |
| all25 | base | smooth | 0.6524 | 0.5951 | 15268 | 9884 | 9422 | 14886 | 686 | 392 | 277 | 17 |
| all25 | base | smooth02 | 0.1634 | 0.0670 | 15268 | 3045 | 2279 | 14886 | 686 | 78 | 603 | 5 |
| all25 | base | smooth05 | 0.0444 | 0.0097 | 15268 | 740 | 347 | 14886 | 686 | 14 | 672 | 0 |
| all25 | base | source_selected | 0.3300 | 0.2522 | 15268 | 8893 | 8288 | 14886 | 686 | 332 | 338 | 16 |
| f0921 | base | hard | 0.1585 | 0.0979 | 12259 | 2051 | 2011 | 12008 | 508 | 54 | 449 | 5 |
| f0921 | base | smooth | 0.6422 | 0.5625 | 12259 | 8012 | 7465 | 12008 | 508 | 282 | 210 | 16 |
| f0921 | base | smooth02 | 0.1920 | 0.1132 | 12259 | 2482 | 1756 | 12008 | 508 | 57 | 446 | 5 |
| f0921 | base | smooth05 | 0.0429 | 0.0239 | 12259 | 561 | 204 | 12008 | 508 | 12 | 496 | 0 |
| f0921 | base | source_selected | 0.6422 | 0.5625 | 12259 | 8012 | 7465 | 12008 | 508 | 282 | 210 | 16 |
| oss2 | base | hard | 0.0810 | 0.1016 | 392 | 30 | 36 | 359 | 37 | 4 | 33 | 0 |
| oss2 | base | smooth | 0.7090 | 0.5794 | 392 | 281 | 305 | 359 | 37 | 23 | 13 | 1 |
| oss2 | base | smooth02 | 0.2286 | 0.1844 | 392 | 92 | 96 | 359 | 37 | 8 | 29 | 0 |
| oss2 | base | smooth05 | 0.0237 | 0.0000 | 392 | 8 | 10 | 359 | 37 | 0 | 37 | 0 |
| oss2 | base | source_selected | 0.2286 | 0.1844 | 392 | 92 | 96 | 359 | 37 | 8 | 29 | 0 |
| self24ports | base | hard | 0.1268 | 0.0530 | 2617 | 367 | 280 | 2519 | 141 | 6 | 135 | 0 |
| self24ports | base | smooth | 0.6504 | 0.5973 | 2617 | 1591 | 1652 | 2519 | 141 | 87 | 54 | 0 |
| self24ports | base | smooth02 | 0.1593 | 0.0595 | 2617 | 471 | 427 | 2519 | 141 | 13 | 128 | 0 |
| self24ports | base | smooth05 | 0.0453 | 0.0095 | 2617 | 171 | 133 | 2519 | 141 | 2 | 139 | 0 |
| self24ports | base | source_selected | 0.3208 | 0.2412 | 2617 | 789 | 727 | 2519 | 141 | 42 | 99 | 0 |

capture_probe 只针对 base，以 learned base 错误且至少一个 preserved 频带正确为机会；added 是这些机会中 PREV standalone 四频带均无法救回的部分。captured_added、added_keep_main 与 added_wrong_band 分别表示救回、保留主分支、选错频带。harms 为 learned正确却被选择器改错。所有计数累加 query-draw 事件，同样本跨draw重复，不表示独立样本数。capture率先任务内均值再任务等权均值。

## 精确路由分解

| dataset | mode | fixed_hard_route_delta_ba | changed_route_delta_ba | full_delta_ba |
| --- | --- | --- | --- | --- |
| all25 | hard | 0.0000 | 0.0000 | 0.0000 |
| all25 | smooth | -0.2096 | 0.1767 | -0.0329 |
| all25 | smooth02 | -0.0545 | 0.0609 | 0.0063 |
| all25 | smooth05 | -0.0637 | 0.0460 | -0.0177 |
| all25 | source_selected | -0.1641 | 0.1097 | -0.0543 |
| f0921 | hard | 0.0000 | 0.0000 | 0.0000 |
| f0921 | smooth | 0.1042 | 0.0215 | 0.1257 |
| f0921 | smooth02 | -0.0054 | 0.0143 | 0.0089 |
| f0921 | smooth05 | -0.0009 | 0.0144 | 0.0135 |
| f0921 | source_selected | 0.1042 | 0.0215 | 0.1257 |
| oss2 | hard | 0.0000 | 0.0000 | 0.0000 |
| oss2 | smooth | -0.5472 | 0.0952 | -0.4520 |
| oss2 | smooth02 | 0.0216 | 0.0525 | 0.0741 |
| oss2 | smooth05 | -0.0587 | 0.0297 | -0.0289 |
| oss2 | source_selected | 0.0216 | 0.0525 | 0.0741 |
| self24ports | hard | 0.0000 | 0.0000 | 0.0000 |
| self24ports | smooth | -0.2086 | 0.1870 | -0.0216 |
| self24ports | smooth02 | -0.0600 | 0.0633 | 0.0033 |
| self24ports | smooth05 | -0.0667 | 0.0481 | -0.0186 |
| self24ports | source_selected | -0.1838 | 0.1160 | -0.0678 |

每任务/draw依次比较 hard full、当前模式 frozen_hard_route、当前模式 full；固定路由变化加改变路由变化严格分解 full 差值。此为冻结预测算术分解，不是根因结论。

## Utility 区分度

| dataset | utility | auc | nonzero | rescue_events | harm_events | events | nonzero_events | auc_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all25 | hard | 0.5190 | 0.1781 | 28631 | 30841 | 59472 | 9979 | 444 |
| all25 | smooth | 0.5658 | 1.0000 | 28631 | 30841 | 59472 | 59472 | 444 |
| f0921 | hard | 0.4985 | 0.1181 | 22302 | 22470 | 44772 | 6098 | 20 |
| f0921 | smooth | 0.5301 | 1.0000 | 22302 | 22470 | 44772 | 44772 | 20 |
| oss2 | hard | 0.5184 | 0.2067 | 689 | 916 | 1605 | 372 | 20 |
| oss2 | smooth | 0.5199 | 1.0000 | 689 | 916 | 1605 | 1605 | 20 |
| self24ports | hard | 0.5199 | 0.1795 | 5640 | 7455 | 13095 | 3509 | 404 |
| self24ports | smooth | 0.5694 | 1.0000 | 5640 | 7455 | 13095 | 13095 | 404 |

utility_probe 仅在每频带相对 learned base 的 rescue 或 harm 事件上计算AUC；rescue为正类，utility越大越倾向rescue。缺少任一类别时AUC为空，不填0.5。nonzero 是相同事件中utility非零比例。先在每任务内平均有效band/draw值，再等权平均有效任务；auc_defined给出有效band/draw数。事件计数是query-band-draw，同query可能在多个频带重复。utility_band_probe另存逐频带汇总。smooth02和smooth05共享smooth utility，因此不重复AUC。

metrics、class_metrics、task_means、summary、draw_metrics、events 为三阶段CSV；source_selection保留冻结源域规则和全部源域差值；source_target_comparison按任务连接源域差值和目标三阶段差值，只用于描述，不用于选择。capture_probe、route_probe、utility_probe及其summary保存诊断。无图。
