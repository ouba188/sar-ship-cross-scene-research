# R50-Tuned frozen-prediction analysis

Status: computed from completed runs; 10 of 16 native panels visually reviewed; see ../FINDINGS_ZH.md.

Selection: scope=full, backbone LR=0.0001; seeds=17/29/43.
Only the selected three R50 runs supply test predictions. Historical predictions are reused.

| Method | Test OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |
|---|---:|---:|---:|
| R50-Tuned | 63.051 +/- 2.138 | 57.631 +/- 2.625 | 55.929 +/- 1.862 |
| Full | 61.023 +/- 2.610 | 55.630 +/- 3.498 | 54.020 +/- 3.267 |
| Plain | 55.908 +/- 1.332 | 52.176 +/- 1.506 | 50.173 +/- 1.359 |
| SE | 54.233 +/- 0.954 | 53.497 +/- 1.802 | 49.862 +/- 1.323 |
| CPINet | 56.261 +/- 1.986 | 56.988 +/- 0.960 | 52.413 +/- 1.482 |

SD is the sample SD of three trained seeds, not three independent scenes.

| R50 minus baseline | Metric | Delta (pp) | Paired ship 95% CI | Positive seeds |
|---|---|---:|---|---:|
| Full | oa | +2.028 | [-2.133, +6.191] | 2/3 |
| Full | ba | +2.001 | [-3.513, +7.510] | 1/3 |
| Full | macro_f1 | +1.909 | [-2.612, +6.210] | 2/3 |
| Plain | oa | +7.143 | [+2.469, +11.994] | 3/3 |
| Plain | ba | +5.455 | [-0.246, +11.329] | 3/3 |
| Plain | macro_f1 | +5.756 | [+0.972, +10.616] | 3/3 |

| Method | Polarization subset | N | OA mean | BA mean | Macro-F1 mean |
|---|---|---:|---:|---:|---:|
| R50-Tuned | both | 348 | 66.092 | 60.218 | 58.286 |
| Full | both | 348 | 62.739 | 56.571 | 54.918 |
| Plain | both | 348 | 58.716 | 55.470 | 52.776 |
| SE | both | 348 | 56.226 | 55.756 | 51.625 |
| CPINet | both | 348 | 57.375 | 57.369 | 52.860 |
| R50-Tuned | single | 30 | 27.778 | 27.146 | 23.500 |
| Full | single | 30 | 41.111 | 45.707 | 40.251 |
| Plain | single | 30 | 23.333 | 20.896 | 20.049 |
| SE | single | 30 | 31.111 | 38.763 | 31.820 |
| CPINet | single | 30 | 43.333 | 55.177 | 43.840 |

| Method | Class | N | Recall mean +/- SD |
|---|---|---:|---:|
| R50-Tuned | Bulk carrier | 132 | 63.131 +/- 8.780 |
| R50-Tuned | General cargo | 26 | 35.897 +/- 14.561 |
| R50-Tuned | Chemical/product tanker | 83 | 63.855 +/- 2.410 |
| R50-Tuned | Container ship | 137 | 67.640 +/- 2.563 |
| Full | Bulk carrier | 132 | 53.535 +/- 4.566 |
| Full | General cargo | 26 | 35.897 +/- 8.006 |
| Full | Chemical/product tanker | 83 | 58.635 +/- 2.508 |
| Full | Container ship | 137 | 74.453 +/- 2.632 |
| Plain | Bulk carrier | 132 | 50.000 +/- 1.312 |
| Plain | General cargo | 26 | 39.744 +/- 2.221 |
| Plain | Chemical/product tanker | 83 | 51.807 +/- 2.087 |
| Plain | Container ship | 137 | 67.153 +/- 1.264 |
| SE | Bulk carrier | 132 | 51.768 +/- 1.577 |
| SE | General cargo | 26 | 53.846 +/- 3.846 |
| SE | Chemical/product tanker | 83 | 47.791 +/- 3.681 |
| SE | Container ship | 137 | 60.584 +/- 1.931 |
| CPINet | Bulk carrier | 132 | 55.808 +/- 3.813 |
| CPINet | General cargo | 26 | 62.821 +/- 8.006 |
| CPINet | Chemical/product tanker | 83 | 50.201 +/- 7.746 |
| CPINet | Container ship | 137 | 59.124 +/- 0.730 |

Metrics are percentages; differences are percentage points. BA averages truth-present classes;
macro-F1 always uses four classes with zero-division set to zero. Subset class counts are explicit.
Ship bootstrap retains all observations per MMSI and pairs methods and fixed seeds in each draw.
It conditions on the existing scenes and trained runs, and does not estimate new-port or selection uncertainty.
BA replicates missing an originally present class are excluded; valid replicate counts are recorded.
Scene/pass-group tables include complete-dual subsets; inspect denominators before interpreting effects.

R50-Tuned versus historical Plain/Full is a system comparison with different training recipes.
Only the R50 scope comparisons share the new recipe; these results cannot isolate unfreezing alone.
This historically used evaluation cohort is not a fresh blind test.

Native review: 16 panels in index.html; cases_local.json records real paths, margins and all seed states.
Rescue/harm direction is baseline to R50. Stable-wrong includes wrong-to-wrong flips, counted separately.
Representative cases do not estimate prevalence; no physical explanation is generated automatically.
Main agent inspected 10 native panels, including all eight Full panels and their legal source references.
Local paths, predictions and raw panels are local artifacts; this script performs no Git actions.
