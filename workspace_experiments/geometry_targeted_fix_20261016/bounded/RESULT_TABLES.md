# bounded: fixed training result

S32; three training seeds; five paired support draws; development cohort.

| arm | oa | ba | macro_nll | macro_brier |
|---|---|---|---|---|
| B | 62.862909 | 57.155429 | 1.503288 | 0.666728 |
| conditional | 62.792210 | 57.116830 | 1.503308 | 0.666733 |
| geometry | 62.821371 | 57.133660 | 1.503207 | 0.666704 |
| spatial | 62.811122 | 57.123884 | 1.503314 | 0.666735 |

## Paired Differences

| scope | arm | reference | delta_oa | delta_ba | delta_macro_nll | delta_macro_brier |
|---|---|---|---|---|---|---|
| full | conditional | B | -0.070700 | -0.038599 | 0.000020 | 0.000005 |
| full | conditional | spatial | -0.018913 | -0.007055 | -0.000006 | -0.000002 |
| full | geometry | B | -0.041538 | -0.021769 | -0.000081 | -0.000024 |
| full | geometry | conditional | 0.029162 | 0.016830 | -0.000101 | -0.000029 |
| full | spatial | B | -0.051787 | -0.031544 | 0.000026 | 0.000007 |
| geometry | conditional | B | -1.199295 | -0.761905 | 0.000246 | 0.000258 |
| geometry | conditional | spatial | -0.317460 | -0.148148 | -0.000103 | -0.000040 |
| geometry | geometry | B | -0.705467 | -0.439153 | 0.000076 | 0.000046 |
| geometry | geometry | conditional | 0.493827 | 0.322751 | -0.000170 | -0.000213 |
| geometry | spatial | B | -0.881834 | -0.613757 | 0.000350 | 0.000298 |

## Checks

{"stage": "bounded", "trained": 36, "fallback": 9, "tasks_per_arm_seed": 190, "probability_rows": 100800, "baseline_max_error": 0.0, "fallback_exact": true, "class_accounting": true, "sampling_traces": 36}
