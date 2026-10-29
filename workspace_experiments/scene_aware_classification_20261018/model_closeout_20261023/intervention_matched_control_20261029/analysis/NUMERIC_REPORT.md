# Matched intervention numerical results

Matched spatial M1/M2 without explicit frequency branches, with M3 retained; native160 and fixed reference16. Primary attribution: guided minus control. Auxiliary comparisons: each after minus its shared before. Control is CE + 0.1 KL, intervention coefficient 0 and useful all False: no intervention-selection loss. Its teacher_original reuses the existing original scores from the same source-fitted teacher. Guided adds the original intervention-selected margin_floor at coefficient 0.3683066666126251. This estimates the incremental selected-guidance training procedure, not selection alone or a physical mechanism. Both continuations use the matched source fit and order, planned 6 epochs/54 steps on adapt136, fit1205 separate. Seeds 17/29/43 vary training order with the SAME initialization, NOT independent initializations. Development160 and expanded3075 are historical cohorts, NOT blind tests. OA/BA are percentages; differences are percentage points. BA averages present classes only. NLL uses natural logs; Brier sums over eight classes; SD is sample SD across the three order seeds. Pass bootstrap conditions on fixed fitted models; repeated seeds, vessels and references are not independent evidence. Protocol statements describe the supplied design; raw training receipts are retained, not inferred from logits. Generated native cases remain not_reviewed until the main agent inspects them.

## development

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| before | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |
| control | 40.208333 +/- 2.009094 | 33.283189 +/- 1.814038 | 2.231615 +/- 0.072852 | 0.855070 +/- 0.006755 |
| guided | 45.000000 +/- 1.082532 | 35.733663 +/- 0.740113 | 2.430989 +/- 0.162197 | 0.840963 +/- 0.018059 |

| Comparison | Metric | Paired delta mean | SD | Pass bootstrap low | High |
|---|---|---:|---:|---:|---:|
| guided_minus_control | OA | 4.791667 | 1.572882 | 3.174603 | 6.763285 |
| guided_minus_control | BA | 2.450474 | 1.266750 | 1.875902 | 3.246753 |
| guided_minus_control | NLL | 0.199374 | 0.090589 | 0.000482 | 0.350547 |
| guided_minus_control | Brier | -0.014107 | 0.013616 | -0.070953 | 0.030271 |

guided_minus_control: 3 passes; 0/2000 resamples omit an originally present class.

| control_minus_before | OA | -0.833333 | 1.301041 | -4.347826 | 2.645503 |
| control_minus_before | BA | 0.084003 | 0.987046 | -1.621315 | 1.659452 |
| control_minus_before | NLL | -0.432953 | 0.217249 | -0.653229 | -0.209715 |
| control_minus_before | Brier | -0.038182 | 0.060402 | -0.112443 | 0.036422 |

control_minus_before: 3 passes; 0/2000 resamples omit an originally present class.

| guided_minus_before | OA | 3.958333 | 1.909407 | 2.415459 | 5.820106 |
| guided_minus_before | BA | 2.534477 | 1.603763 | 0.283447 | 3.535354 |
| guided_minus_before | NLL | -0.233579 | 0.130913 | -0.302682 | -0.138089 |
| guided_minus_before | Brier | -0.052289 | 0.049398 | -0.082172 | -0.028814 |

guided_minus_before: 3 passes; 0/2000 resamples omit an originally present class.

| Comparison | State | Seed events | Unique queries |
|---|---|---:|---:|
| guided_minus_control | rescued | 30 | 22 |
| guided_minus_control | harmed | 7 | 6 |
| guided_minus_control | CC | 186 | 77 |
| guided_minus_control | WW | 257 | 99 |
| control_minus_before | rescued | 23 | 19 |
| control_minus_before | harmed | 27 | 20 |
| control_minus_before | CC | 170 | 75 |
| control_minus_before | WW | 260 | 100 |
| guided_minus_before | rescued | 28 | 24 |
| guided_minus_before | harmed | 9 | 8 |
| guided_minus_before | CC | 188 | 81 |
| guided_minus_before | WW | 255 | 99 |
## expanded

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| before | 46.352304 +/- 1.122933 | 38.682940 +/- 1.856018 | 2.344226 +/- 0.076481 | 0.815594 +/- 0.001003 |
| control | 45.127371 +/- 1.561427 | 41.039015 +/- 0.600579 | 1.939647 +/- 0.023102 | 0.783313 +/- 0.010681 |
| guided | 47.317073 +/- 1.125128 | 40.402194 +/- 1.067667 | 2.137044 +/- 0.047298 | 0.786188 +/- 0.004741 |

| Comparison | Metric | Paired delta mean | SD | Pass bootstrap low | High |
|---|---|---:|---:|---:|---:|
| guided_minus_control | OA | 2.189702 | 0.469390 | 1.183688 | 3.249427 |
| guided_minus_control | BA | -0.636821 | 0.477128 | -1.606833 | 0.366585 |
| guided_minus_control | NLL | 0.197397 | 0.051522 | 0.141813 | 0.252697 |
| guided_minus_control | Brier | 0.002875 | 0.009243 | -0.010469 | 0.015555 |

guided_minus_control: 85 passes; 0/2000 resamples omit an originally present class.

| control_minus_before | OA | -1.224932 | 0.441527 | -2.788434 | 0.174625 |
| control_minus_before | BA | 2.356075 | 1.323391 | 1.190828 | 3.438428 |
| control_minus_before | NLL | -0.404579 | 0.089953 | -0.494560 | -0.307961 |
| control_minus_before | Brier | -0.032281 | 0.011120 | -0.052128 | -0.009968 |

control_minus_before: 85 passes; 0/2000 resamples omit an originally present class.

| guided_minus_before | OA | 0.964770 | 0.190552 | 0.111206 | 1.774400 |
| guided_minus_before | BA | 1.719254 | 0.879254 | 0.887891 | 2.590519 |
| guided_minus_before | NLL | -0.207182 | 0.044143 | -0.244556 | -0.165451 |
| guided_minus_before | Brier | -0.029405 | 0.005738 | -0.038716 | -0.019004 |

guided_minus_before: 85 passes; 0/2000 resamples omit an originally present class.

| Comparison | State | Seed events | Unique queries |
|---|---|---:|---:|
| guided_minus_control | rescued | 525 | 376 |
| guided_minus_control | harmed | 323 | 246 |
| guided_minus_control | CC | 3840 | 1559 |
| guided_minus_control | WW | 4537 | 1803 |
| control_minus_before | rescued | 592 | 417 |
| control_minus_before | harmed | 705 | 486 |
| control_minus_before | CC | 3571 | 1517 |
| control_minus_before | WW | 4357 | 1793 |
| guided_minus_before | rescued | 360 | 298 |
| guided_minus_before | harmed | 271 | 234 |
| guided_minus_before | CC | 4005 | 1666 |
| guided_minus_before | WW | 4589 | 1854 |

Primary attribution is guided-control; guided-before alone includes ordinary continuation. All paired differences, class/port/pass/actual-polarization groups and three-condition predictions/margins are in CSVs. Bootstrap resamples whole passes with all paired observations, averages fixed-seed differences, and uses present-class BA. The development cohort has few passes, so its conditional interval has limited resolution. Raw training receipts and input provenance are retained in summary.json.

[Native gallery](cases/index.html): 8 unique expanded queries, 16 separate VV/VH panels. Cases include actual eligible same-class and rival references for all displayed seeds and conditions. Visual status: not_reviewed. Main-agent image review, visible facts and mechanism interpretation remain pending.
