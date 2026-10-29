# Full adaptation CPU numeric results

Full source-selected adaptation: M1/M2 explicit frequency branches removed in spatial; M3 retained. Each before endpoint is its own architecture/order-seed source fit; after is that fit plus its own source136 intervention selection and adaptation (planned 6 epochs/54 steps). Same source-fit1205 data; source-adapt136 is separate from fit. Fixed 16 references. Only layer3.5.conv1/2/3 are trained in adaptation; all remaining parameters including M1/M2/M3 heads are frozen but participate in differentiable computation. Each teacher independently selects interventions, so the final comparison includes selection differences and does NOT isolate their contribution from architecture or adaptation. Seeds 17/29/43 vary BOTH source-fit and adaptation data order with the same initialization, NOT independent initializations. Development160 is reused development, NOT a new blind test; expanded3075 is also a historical cohort. No target-based parameter selection. OA/BA are percentages and deltas percentage points. BA averages present classes; absent classes are blank, not zero. NLL uses natural logs and Brier sums over all eight classes. SD is sample SD across the three order seeds. Repeated seed events, ships, passes and reference exposures are not independent samples. Paired outcome changes and selected image cases do not establish a physical mechanism.

## development

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| legacy_before | 41.666667 +/- 2.953635 | 33.222892 +/- 1.943356 | 2.647204 +/- 0.282654 | 0.887192 +/- 0.063569 |
| legacy_after | 45.000000 +/- 0.625000 | 35.504071 +/- 1.449416 | 2.426006 +/- 0.157551 | 0.838792 +/- 0.016081 |
| spatial_before | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |
| spatial_after | 45.000000 +/- 1.082532 | 35.733663 +/- 0.740113 | 2.430989 +/- 0.162197 | 0.840963 +/- 0.018059 |

| Comparison | Metric | Paired delta mean | SD |
|---|---|---:|---:|
| spatial_after_minus_legacy_after | OA | 0.000000 | 1.250000 |
| spatial_after_minus_legacy_after | BA | 0.229592 | 0.778375 |
| spatial_after_minus_legacy_after | NLL | 0.004983 | 0.007248 |
| spatial_after_minus_legacy_after | Brier | 0.002171 | 0.001978 |
| legacy_after_minus_before | OA | 3.333333 | 2.525907 |
| legacy_after_minus_before | BA | 2.281179 | 1.508452 |
| legacy_after_minus_before | NLL | -0.221198 | 0.131796 |
| legacy_after_minus_before | Brier | -0.048400 | 0.048826 |
| spatial_after_minus_before | OA | 3.958333 | 1.909407 |
| spatial_after_minus_before | BA | 2.534477 | 1.603763 |
| spatial_after_minus_before | NLL | -0.233579 | 0.130913 |
| spatial_after_minus_before | Brier | -0.052289 | 0.049398 |

| Comparison | State | Seed events | Unique queries |
|---|---|---:|---:|
| spatial_after_minus_legacy_after | rescued | 3 | 3 |
| spatial_after_minus_legacy_after | harmed | 3 | 3 |
| spatial_after_minus_legacy_after | CC | 213 | 87 |
| spatial_after_minus_legacy_after | WW | 261 | 101 |
| legacy_after_minus_before | rescued | 24 | 22 |
| legacy_after_minus_before | harmed | 8 | 8 |
| legacy_after_minus_before | CC | 192 | 82 |
| legacy_after_minus_before | WW | 256 | 98 |
| spatial_after_minus_before | rescued | 28 | 24 |
| spatial_after_minus_before | harmed | 9 | 8 |
| spatial_after_minus_before | CC | 188 | 81 |
| spatial_after_minus_before | WW | 255 | 99 |
## expanded

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| legacy_before | 46.168022 +/- 1.353929 | 38.597257 +/- 1.966963 | 2.344990 +/- 0.072632 | 0.816400 +/- 0.003730 |
| legacy_after | 47.306233 +/- 1.466062 | 40.391397 +/- 1.042846 | 2.139581 +/- 0.044009 | 0.786748 +/- 0.005272 |
| spatial_before | 46.352304 +/- 1.122933 | 38.682940 +/- 1.856018 | 2.344226 +/- 0.076481 | 0.815594 +/- 0.001003 |
| spatial_after | 47.317073 +/- 1.125128 | 40.402194 +/- 1.067667 | 2.137044 +/- 0.047298 | 0.786188 +/- 0.004741 |

| Comparison | Metric | Paired delta mean | SD |
|---|---|---:|---:|
| spatial_after_minus_legacy_after | OA | 0.010840 | 0.358216 |
| spatial_after_minus_legacy_after | BA | 0.010797 | 0.403075 |
| spatial_after_minus_legacy_after | NLL | -0.002537 | 0.003424 |
| spatial_after_minus_legacy_after | Brier | -0.000560 | 0.002055 |
| legacy_after_minus_before | OA | 1.138211 | 0.203089 |
| legacy_after_minus_before | BA | 1.794140 | 1.277403 |
| legacy_after_minus_before | NLL | -0.205408 | 0.038767 |
| legacy_after_minus_before | Brier | -0.029652 | 0.004440 |
| spatial_after_minus_before | OA | 0.964770 | 0.190552 |
| spatial_after_minus_before | BA | 1.719254 | 0.879254 |
| spatial_after_minus_before | NLL | -0.207182 | 0.044143 |
| spatial_after_minus_before | Brier | -0.029405 | 0.005738 |

| Comparison | State | Seed events | Unique queries |
|---|---|---:|---:|
| spatial_after_minus_legacy_after | rescued | 53 | 53 |
| spatial_after_minus_legacy_after | harmed | 52 | 51 |
| spatial_after_minus_legacy_after | CC | 4312 | 1720 |
| spatial_after_minus_legacy_after | WW | 4808 | 1894 |
| legacy_after_minus_before | rescued | 356 | 299 |
| legacy_after_minus_before | harmed | 251 | 221 |
| legacy_after_minus_before | CC | 4008 | 1671 |
| legacy_after_minus_before | WW | 4610 | 1877 |
| spatial_after_minus_before | rescued | 360 | 298 |
| spatial_after_minus_before | harmed | 271 | 234 |
| spatial_after_minus_before | CC | 4005 | 1666 |
| spatial_after_minus_before | WW | 4589 | 1854 |

Pass-cluster 95% percentile intervals below condition on these fixed fitted models; they are not independent-training confidence intervals.

| Comparison | Metric | Low | High |
|---|---|---:|---:|
| spatial_after_minus_legacy_after | OA | -0.245572 | 0.276623 |
| spatial_after_minus_legacy_after | BA | -0.236845 | 0.293297 |
| spatial_after_minus_legacy_after | NLL | -0.004683 | -0.000348 |
| spatial_after_minus_legacy_after | Brier | -0.001420 | 0.000287 |
| legacy_after_minus_before | OA | 0.374257 | 1.811126 |
| legacy_after_minus_before | BA | 1.149274 | 2.449294 |
| legacy_after_minus_before | NLL | -0.241069 | -0.165939 |
| legacy_after_minus_before | Brier | -0.038551 | -0.019758 |
| spatial_after_minus_before | OA | 0.111206 | 1.774400 |
| spatial_after_minus_before | BA | 0.887891 | 2.590519 |
| spatial_after_minus_before | NLL | -0.244556 | -0.165451 |
| spatial_after_minus_before | Brier | -0.038716 | -0.019004 |

Class/port/pass/actual-available groups and paired flips are in groups.csv and groups_summary.csv. selection_summary.json contains final-useful counts and architecture intersections; selection_groups.csv distinguishes final useful from the earlier choice screen. Raw training receipts are embedded in summary.json; the CPU analysis does not certify frozen weights or internal reference usage.

Fixed rivals come from mean legacy_before nontruth logits. Full per-case logits and all four condition outcomes are in cases/cases.json. [Native gallery](cases/index.html): eight unique expanded queries, sixteen separate VV/VH panels, actual eligible true/rival refs. Generated pixels have NOT been reviewed. Main-agent visual review and mechanism interpretation remain pending.
