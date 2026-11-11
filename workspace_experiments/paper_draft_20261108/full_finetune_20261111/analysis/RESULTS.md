# Full-Tuned frozen-prediction analysis

Computed from completed runs. All 12 panels reviewed on 2026-11-12; see ../FINDINGS_ZH.md.

Selection: scope=full, backbone LR=0.0001; seeds 17/29/43.
Full-Tuned retains M1/M2/M3 and continues each seed's historical Full checkpoint.
R50-Tuned is a separate reference-free baseline, not this complete method.

| Method | Test OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |
|---|---:|---:|---:|
| Full-Tuned | 64.727 +/- 2.702 | 59.396 +/- 3.831 | 57.693 +/- 3.187 |
| Full | 61.023 +/- 2.610 | 55.630 +/- 3.498 | 54.020 +/- 3.267 |
| Plain | 55.908 +/- 1.332 | 52.176 +/- 1.506 | 50.173 +/- 1.359 |
| R50-Tuned | 63.051 +/- 2.138 | 57.631 +/- 2.625 | 55.929 +/- 1.862 |

| Full-Tuned minus baseline | Metric | Mean +/- SD (pp) | Paired MMSI 95% CI | Positive seeds |
|---|---|---:|---|---:|
| Full | oa | +3.704 +/- 5.311 | [-0.087, +7.339] | 2/3 |
| Full | ba | +3.766 +/- 7.232 | [-1.242, +8.671] | 2/3 |
| Full | macro_f1 | +3.673 +/- 6.417 | [-0.311, +7.689] | 2/3 |
| R50-Tuned | oa | +1.675 +/- 0.666 | [-1.243, +4.674] | 3/3 |
| R50-Tuned | ba | +1.765 +/- 4.236 | [-1.558, +5.177] | 2/3 |
| R50-Tuned | macro_f1 | +1.764 +/- 2.452 | [-1.030, +4.698] | 2/3 |
| Plain | oa | +8.818 +/- 3.927 | [+4.509, +13.351] | 3/3 |
| Plain | ba | +7.220 +/- 5.293 | [+1.654, +13.073] | 3/3 |
| Plain | macro_f1 | +7.520 +/- 4.489 | [+3.034, +12.313] | 3/3 |

| Seed | Executed epochs | Best epoch | Steps | Fit seconds |
|---|---:|---:|---:|---:|
| 17 | 23 | 15 | 690 | 628.31 |
| 29 | 19 | 11 | 570 | 521.11 |
| 43 | 22 | 14 | 660 | 621.62 |

New search/confirmation: 8 unique fits, 162 epochs, 4860 steps, 116478 repeated query exposures, 5492.80 fit seconds; receipt peak 4.095 GiB.
Inherited Full: 2208 steps and 294.51 fit seconds across three seeds.
Epoch 0 may be best, but all attempted optimization is counted. Seed17 scout reuse is counted once.
See training_budget.json and training_budget_by_run.csv for search versus selected and inherited costs.

| Method | Class | N | Test recall mean +/- SD |
|---|---|---:|---:|
| Full-Tuned | Bulk carrier | 132 | 66.162 +/- 11.245 |
| Full-Tuned | General cargo | 26 | 39.744 +/- 11.103 |
| Full-Tuned | Chemical/product tanker | 83 | 61.847 +/- 3.873 |
| Full-Tuned | Container ship | 137 | 69.830 +/- 3.746 |
| Full | Bulk carrier | 132 | 53.535 +/- 4.566 |
| Full | General cargo | 26 | 35.897 +/- 8.006 |
| Full | Chemical/product tanker | 83 | 58.635 +/- 2.508 |
| Full | Container ship | 137 | 74.453 +/- 2.632 |
| Plain | Bulk carrier | 132 | 50.000 +/- 1.312 |
| Plain | General cargo | 26 | 39.744 +/- 2.221 |
| Plain | Chemical/product tanker | 83 | 51.807 +/- 2.087 |
| Plain | Container ship | 137 | 67.153 +/- 1.264 |
| R50-Tuned | Bulk carrier | 132 | 63.131 +/- 8.780 |
| R50-Tuned | General cargo | 26 | 35.897 +/- 14.561 |
| R50-Tuned | Chemical/product tanker | 83 | 63.855 +/- 2.410 |
| R50-Tuned | Container ship | 137 | 67.640 +/- 2.563 |

| Method | Polarization | N | OA | BA | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Full-Tuned | both | 348 | 66.762 | 60.522 | 58.803 |
| Full | both | 348 | 62.739 | 56.571 | 54.918 |
| Plain | both | 348 | 58.716 | 55.470 | 52.776 |
| R50-Tuned | both | 348 | 66.092 | 60.218 | 58.286 |
| Full-Tuned | single | 30 | 41.111 | 44.381 | 39.108 |
| Full | single | 30 | 41.111 | 45.707 | 40.251 |
| Plain | single | 30 | 23.333 | 20.896 | 20.049 |
| R50-Tuned | single | 30 | 27.778 | 27.146 | 23.500 |
| Full-Tuned | vv_only | 12 | 36.111 | 50.000 | 36.706 |
| Full | vv_only | 12 | 27.778 | 29.167 | 20.357 |
| Plain | vv_only | 12 | 25.000 | 16.667 | 18.585 |
| R50-Tuned | vv_only | 12 | 22.222 | 26.389 | 24.603 |
| Full-Tuned | vh_only | 18 | 44.444 | 42.870 | 38.591 |
| Full | vh_only | 18 | 50.000 | 61.852 | 52.038 |
| Plain | vh_only | 18 | 22.222 | 28.519 | 18.318 |
| R50-Tuned | vh_only | 18 | 31.481 | 28.889 | 25.662 |

Metrics are percentages; gains are percentage points. SD uses ddof=1 across three fixed seeds.
BA averages truth-present classes. Macro-F1 uses all four labels, undefined F1 set to zero.
MMSI bootstrap resamples ships jointly for both methods and all fixed seeds, retaining all observations.
BA draws missing an originally present class are excluded; valid counts are in paired_ship_ci.csv.
These intervals condition on observed scenes and trained runs, not new ports or model-selection uncertainty.
Per-scene, pass-group, platform-orbit and complete-dual strata include denominators; empty strata are omitted.
Scene and polarization associations do not prove missing-view or physical causes.
Full-Tuned and R50-Tuned have different initialization, auxiliary losses and training budgets.
This is a system/continuation comparison, not a matched pure module or unfreezing ablation.
The historically used evaluation cohort is not a fresh blind test. No test-based model selection is performed.

Native review: 12 panels in index.html; cases_local.json includes IDs, source paths, scores,
true-vs-best-rival and true-vs-fixed-rival margins, seeds, actual 32-reference bank and polarity eligibility.
Stable-wrong includes wrong-to-wrong flips, counted separately in paired_by_seed.csv.
Absent rescue/harm categories are reported honestly, including when epoch 0 retains historical decisions.
Native VV/VH and displayed legal source references have been inspected; interpretations are in ../FINDINGS_ZH.md.
No physical mechanism or highest-contribution reference is inferred from these panels.
No training, inference, cache construction, manuscript edits or Git actions are performed.
