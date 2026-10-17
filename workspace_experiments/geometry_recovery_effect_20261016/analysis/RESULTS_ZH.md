# 恢复几何后的配对贡献汇总

固定五留港研发任务，S32、3种子、5支持抽样。全体任务是主口径，不把有效子集BA冒充总体BA。

| origin | arm | oa | ba | macro_nll | macro_brier |
|---|---|---|---|---|---|
| cross | B | 62.862909 | 57.155429 | 1.503288 | 0.666728 |
| cross | optional_input | 62.867933 | 57.182249 | 1.503154 | 0.666693 |
| cross | optional_new | 62.867933 | 57.182249 | 1.503154 | 0.666693 |
| cross | optional_old | 62.867933 | 57.182249 | 1.503152 | 0.666692 |
| cross | strict_input | 62.836297 | 57.157470 | 1.503222 | 0.666710 |
| cross | strict_new | 62.836297 | 57.157470 | 1.503222 | 0.666710 |
| cross | strict_old | 62.821371 | 57.133660 | 1.503207 | 0.666704 |
| same | B | 66.552230 | 59.918523 | 1.462056 | 0.649033 |
| same | optional_input | 66.498377 | 59.829049 | 1.462089 | 0.649038 |
| same | optional_new | 66.498377 | 59.829049 | 1.462089 | 0.649038 |
| same | optional_old | 66.488427 | 59.819411 | 1.462090 | 0.649039 |
| same | strict_input | 66.562617 | 59.897303 | 1.461978 | 0.649018 |
| same | strict_new | 66.562617 | 59.897303 | 1.461978 | 0.649018 |
| same | strict_old | 66.547692 | 59.874910 | 1.462039 | 0.649039 |

## 相对自身旧版本的独立改判样本与重复事件

| family | origin | changed_samples | changed_ships | rescue_events | harm_events |
|---|---|---|---|---|---|
| optional | same | 2 | 2 | 2 | 0 |
| strict | cross | 2 | 2 | 6 | 3 |
| strict | same | 5 | 5 | 6 | 3 |

事件包含种子/支持抽样重复，不能当独立船数。详细变化见 changed_cases.csv。

## 配对差值

| origin | comparison | delta_oa | delta_ba | delta_macro_nll | delta_macro_brier |
|---|---|---|---|---|---|
| cross | optional_input - old | 0.000000 | 0.000000 | 0.000002 | 0.000000 |
| cross | optional_new - input | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| cross | optional_new - old | 0.000000 | 0.000000 | 0.000002 | 0.000000 |
| cross | strict_input - old | 0.014925 | 0.023810 | 0.000015 | 0.000006 |
| cross | strict_new - input | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| cross | strict_new - old | 0.014925 | 0.023810 | 0.000015 | 0.000006 |
| same | optional_input - old | 0.009950 | 0.009637 | -0.000001 | -0.000001 |
| same | optional_new - input | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| same | optional_new - old | 0.009950 | 0.009637 | -0.000001 | -0.000001 |
| same | strict_input - old | 0.014925 | 0.022392 | -0.000061 | -0.000021 |
| same | strict_new - input | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| same | strict_new - old | 0.014925 | 0.022392 | -0.000061 | -0.000021 |

class_contributions.csv 和 group_contributions.csv 保持全任务的类别与样本分母，所有类别/组贡献之和已与主表差值核对。

新增数据作用范围、参数一致性与固定新覆盖的角度屏蔽诊断见 ../FINDINGS_ZH.md 和 ../diagnostics/。

## 检查

{"tasks_per_arm_seed": 190, "prediction_rows": 176400, "historical_replays": 45, "max_replay_error": 9.71445146547012e-17, "full_task_class_group_accounting": true, "new_input_equals_refit": true}
