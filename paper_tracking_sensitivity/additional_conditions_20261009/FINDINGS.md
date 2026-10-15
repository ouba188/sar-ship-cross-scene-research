# Cached-pass source response probe

Only source response models were fitted; no classifier changes or target-query outcomes were used.
These are conditional association replications in existing source-fitted Qs/Qd and retained128 geometry, not nested OOF representation tests or causal effects.

Pass: 24423/38091 rows, 523/841 products. Complete angle/position/pass: 19542 rows, 328 products, 18 ports, 377 port-MMSI groups with both passes.
Exact-product cache only; no representative SAFE metadata, mid-swath angle substitution, or missing-value imputation.

## Fixed comparison

Same-port, same-MMSI adjacent observations; duplicate times averaged. Source-label-conflicting identities excluded. The original angle-complete identity split (seed20261009; one third held) is assigned BEFORE filtering pass. Months3/6/9/12 held. All six arms use identical observations/pairs/weights within each split.
Angle inputs are differences of t=(angle-38)/5 and t^2. Pass is Ascending=1, Descending=0; interaction adds differences of p*t and p*t^2, not products of differences. Position is existing east/north displacement divided by10km. Zero-intercept weighted ridge=.01; no tuning.
Weights balance class, port within class, ship within class-port, and pairs within ship. Pass arms require at least10 training switch pairs from5 ships; held switch sufficiency is reported by the same count rule. This is a coverage rule, not a significance test.
skill=1-weighted residual energy/weighted observed change energy. Increment is a difference of skills, not classification accuracy. Source folds overlap heavily;23 positive folds are not23 independent confirmations.

## Held source results

| Space | Split | Arm | Mean skill (%) | Positive folds |
|---|---|---|---:|---:|
| retained128 | new_month | angle | -0.5075 | 2/23 |
| retained128 | new_month | angle_pass | -1.3043 | 0/23 |
| retained128 | new_month | angle_pass_interaction | -1.5303 | 0/23 |
| retained128 | new_month | angle_position | -1.0554 | 0/23 |
| retained128 | new_month | angle_position_pass | -1.8860 | 0/23 |
| retained128 | new_month | angle_position_pass_interaction | -2.2353 | 0/23 |
| retained128 | new_ship | angle | -1.0117 | 0/23 |
| retained128 | new_ship | angle_pass | -1.4503 | 0/23 |
| retained128 | new_ship | angle_pass_interaction | -2.1112 | 0/23 |
| retained128 | new_ship | angle_position | -1.6864 | 0/23 |
| retained128 | new_ship | angle_position_pass | -2.1419 | 0/23 |
| retained128 | new_ship | angle_position_pass_interaction | -2.7772 | 0/23 |
| retained128 | new_ship_month | angle | -0.5611 | 4/23 |
| retained128 | new_ship_month | angle_pass | -1.1929 | 2/23 |
| retained128 | new_ship_month | angle_pass_interaction | -1.5558 | 1/23 |
| retained128 | new_ship_month | angle_position | -1.1206 | 0/23 |
| retained128 | new_ship_month | angle_position_pass | -1.8091 | 1/23 |
| retained128 | new_ship_month | angle_position_pass_interaction | -2.2943 | 1/23 |
| stable_drift | new_month | angle | 0.8351 | 21/23 |
| stable_drift | new_month | angle_pass | -0.8313 | 7/23 |
| stable_drift | new_month | angle_pass_interaction | -0.7517 | 5/23 |
| stable_drift | new_month | angle_position | 0.4633 | 17/23 |
| stable_drift | new_month | angle_position_pass | -1.1918 | 6/23 |
| stable_drift | new_month | angle_position_pass_interaction | -1.3866 | 6/23 |
| stable_drift | new_ship | angle | -0.4591 | 8/23 |
| stable_drift | new_ship | angle_pass | -0.6659 | 6/23 |
| stable_drift | new_ship | angle_pass_interaction | -1.2424 | 6/23 |
| stable_drift | new_ship | angle_position | -1.2882 | 4/23 |
| stable_drift | new_ship | angle_position_pass | -1.5295 | 4/23 |
| stable_drift | new_ship | angle_position_pass_interaction | -1.8084 | 5/23 |
| stable_drift | new_ship_month | angle | 0.9874 | 18/23 |
| stable_drift | new_ship_month | angle_pass | -0.4910 | 11/23 |
| stable_drift | new_ship_month | angle_pass_interaction | -0.5365 | 11/23 |
| stable_drift | new_ship_month | angle_position | 0.2708 | 14/23 |
| stable_drift | new_ship_month | angle_position_pass | -1.2936 | 7/23 |
| stable_drift | new_ship_month | angle_position_pass_interaction | -1.8829 | 3/23 |

## Added information on matched cases

| Space | Split | Comparison | Mean increment (percentage points) | Positive/negative folds |
|---|---|---|---:|---:|
| retained128 | new_month | interaction_over_angle | -1.0228 | 1/22 |
| retained128 | new_month | interaction_over_angle_position | -1.1799 | 2/21 |
| retained128 | new_month | interaction_over_pass | -0.2260 | 3/20 |
| retained128 | new_month | interaction_over_pass_position | -0.3492 | 2/21 |
| retained128 | new_month | pass_over_angle | -0.7968 | 2/21 |
| retained128 | new_month | pass_over_angle_position | -0.8307 | 2/21 |
| retained128 | new_ship | interaction_over_angle | -1.0995 | 0/23 |
| retained128 | new_ship | interaction_over_angle_position | -1.0908 | 0/23 |
| retained128 | new_ship | interaction_over_pass | -0.6609 | 0/23 |
| retained128 | new_ship | interaction_over_pass_position | -0.6353 | 1/22 |
| retained128 | new_ship | pass_over_angle | -0.4386 | 1/22 |
| retained128 | new_ship | pass_over_angle_position | -0.4556 | 1/22 |
| retained128 | new_ship_month | interaction_over_angle | -0.9947 | 1/22 |
| retained128 | new_ship_month | interaction_over_angle_position | -1.1736 | 2/21 |
| retained128 | new_ship_month | interaction_over_pass | -0.3629 | 1/22 |
| retained128 | new_ship_month | interaction_over_pass_position | -0.4851 | 2/21 |
| retained128 | new_ship_month | pass_over_angle | -0.6318 | 1/22 |
| retained128 | new_ship_month | pass_over_angle_position | -0.6885 | 1/22 |
| stable_drift | new_month | interaction_over_angle | -1.5868 | 1/22 |
| stable_drift | new_month | interaction_over_angle_position | -1.8499 | 0/23 |
| stable_drift | new_month | interaction_over_pass | 0.0796 | 15/8 |
| stable_drift | new_month | interaction_over_pass_position | -0.1948 | 12/11 |
| stable_drift | new_month | pass_over_angle | -1.6664 | 0/23 |
| stable_drift | new_month | pass_over_angle_position | -1.6551 | 0/23 |
| stable_drift | new_ship | interaction_over_angle | -0.7833 | 7/16 |
| stable_drift | new_ship | interaction_over_angle_position | -0.5202 | 8/15 |
| stable_drift | new_ship | interaction_over_pass | -0.5765 | 5/18 |
| stable_drift | new_ship | interaction_over_pass_position | -0.2789 | 9/14 |
| stable_drift | new_ship | pass_over_angle | -0.2068 | 8/15 |
| stable_drift | new_ship | pass_over_angle_position | -0.2413 | 8/15 |
| stable_drift | new_ship_month | interaction_over_angle | -1.5239 | 2/21 |
| stable_drift | new_ship_month | interaction_over_angle_position | -2.1538 | 1/22 |
| stable_drift | new_ship_month | interaction_over_pass | -0.0455 | 11/12 |
| stable_drift | new_ship_month | interaction_over_pass_position | -0.5893 | 8/15 |
| stable_drift | new_ship_month | pass_over_angle | -1.4784 | 2/21 |
| stable_drift | new_ship_month | pass_over_angle_position | -1.5645 | 1/22 |

## Coverage and limits

| Split | Pairs per fold | Switch pairs | Switch ships | Adequate switch coverage |
|---|---:|---:|---:|---:|
| fit | 1295-1988 | 67-184 | 49-134 | 23/23 |
| new_month | 527-829 | 27-85 | 24-69 | 23/23 |
| new_ship | 589-1062 | 42-100 | 24-73 | 23/23 |
| new_ship_month | 236-444 | 19-47 | 17-39 | 23/23 |

Port/class row coverage, observation split assignments, pair tables, per-class/per-source-port and switch/non-switch metrics retain all negative results. Missing metadata is structured by port and complete-case findings do not describe uncovered ports.
Pass can co-vary with orbit, acquisition time, location, ship state and image content. First-order position control cannot remove these ambiguities. Chip angles include accepted pair-reference locations; VV/VH co-registration is not established here.

## Existing additional inventory

Manifest crop geometry covers25,656 rows; the existing pooled VV/VH image-proxy cache has3,718 rows. Exact numeric column coverage is in existing_clue_inventory.csv. These clues can contain vessel/class information; no nuisance correction was fitted from them.
Product acquisition times are parsed by the existing helper convention; month_inventory.csv records observed month/port/class coverage. Product prefixes S1A32184/S1C5907 are inventoried only; the main branch owns the platform probe. All38091 rows have IW and1SDV tokens, so these mode tokens have no variation. Paired VV/VH image evidence remains. Relative orbit was not derived without a retained verified formula.
The previously inspected AIS export index has503 exact-product files (693.55MiB). No MMSI-motion join was run and no heading/course/speed field was accepted. This is unfinished metadata validation, not evidence that AIS motion is absent. No further AIS search or download was performed.
