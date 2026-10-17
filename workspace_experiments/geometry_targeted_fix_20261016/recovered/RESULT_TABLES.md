# recovered: fixed training result

S32; three training seeds; five paired support draws; development cohort.

| arm | oa | ba | macro_nll | macro_brier |
|---|---|---|---|---|
| B | 62.862909 | 57.155429 | 1.503288 | 0.666728 |
| conditional_rank | 62.867933 | 57.182249 | 1.503152 | 0.666692 |
| geometry_pair | 62.839638 | 57.159038 | 1.503246 | 0.666718 |
| geometry_rank | 62.851868 | 57.153420 | 1.503246 | 0.666718 |
| geometry_single | 62.841199 | 57.159454 | 1.503217 | 0.666709 |

## Paired Differences

| scope | arm | reference | delta_oa | delta_ba | delta_macro_nll | delta_macro_brier |
|---|---|---|---|---|---|---|
| full | conditional_rank | B | 0.005024 | 0.026820 | -0.000136 | -0.000036 |
| full | geometry_pair | B | -0.023272 | 0.003609 | -0.000042 | -0.000010 |
| full | geometry_pair | geometry_single | -0.001561 | -0.000416 | 0.000030 | 0.000010 |
| full | geometry_rank | B | -0.011041 | -0.002009 | -0.000042 | -0.000011 |
| full | geometry_rank | conditional_rank | -0.016065 | -0.028830 | 0.000094 | 0.000025 |
| full | geometry_rank | geometry_pair | 0.012230 | -0.005619 | -0.000000 | -0.000001 |
| full | geometry_single | B | -0.021711 | 0.004025 | -0.000071 | -0.000020 |
| geometry | conditional_rank | B | 0.202822 | 0.487654 | -0.001146 | -0.000276 |
| geometry | geometry_pair | B | -0.066138 | 0.312169 | 0.000038 | 0.000109 |
| geometry | geometry_pair | geometry_single | 0.008818 | 0.000000 | 0.000163 | 0.000086 |
| geometry | geometry_rank | B | 0.101411 | 0.045855 | -0.000455 | -0.000126 |
| geometry | geometry_rank | conditional_rank | -0.101411 | -0.441799 | 0.000691 | 0.000150 |
| geometry | geometry_rank | geometry_pair | 0.167549 | -0.266314 | -0.000493 | -0.000235 |
| geometry | geometry_single | B | -0.074956 | 0.312169 | -0.000124 | 0.000023 |

## Checks

{"stage": "recovered", "trained": 60, "fallback": 0, "tasks_per_arm_seed": 190, "probability_rows": 126000, "baseline_max_error": 0.0, "fallback_exact": true, "class_accounting": true, "sampling_traces": 60}
