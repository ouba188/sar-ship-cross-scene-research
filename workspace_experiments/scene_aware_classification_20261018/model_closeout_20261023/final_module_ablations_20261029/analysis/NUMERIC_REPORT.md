# Private retrained ablation numerical results

Private retrained final-module ablations: source-fit408 then adapt54, native160, fixed source reference16. full is the reused guided spatial M1/M2 + M3 endpoint; full_control is its matched CE+0.1KL continuation. no_m1/no_m2/no_m3 each use their OWN source-fit teacher, intervention selection and guided adaptation; plain uses its OWN teacher and CE+0.1KL, with no reference contribution. full-minus-plain includes architecture AND training-objective differences, not pure architecture. full_control-minus-plain matches the subsequent CE+KL objective, while source-fit objectives, architecture and own teachers still differ. Module deletion contrasts compare entire retrained procedures, not fixed-teacher causal effects. Before/after uses each arm's own source fit; only full/full_control share their before archive. Matched samples, initialization, permitted update range and step counts do NOT match parameter counts or FLOPs. The pre-save checks establish deletion of the disabled module's dedicated paths and losses; logits cannot establish this. Seeds17/29/43 vary training order with shared initialization, NOT independent initializations; CUDA is not strict deterministic. Receipt trainable counts and seconds are descriptive only: cache and machine-memory conditions differ, so no formal speed claim. Development160 and expanded3075 are historical, NOT blind tests; shared ports are not strict held-port evaluation. OA/BA are percent, paired deltas percentage points; BA uses present classes, natural-log NLL, sum-over-eight-class Brier. Sample SD is across order seeds. Pass bootstrap is conditional on these fixed models, not independent-training uncertainty. Repeated query/vessel/seed/reference exposure is not independent evidence. Intervals are descriptive, not multiplicity-adjusted. Native case selection is outcome-based, not prevalence; image-visible facts and physical mechanisms remain unreviewed.

## development

### Main final endpoints

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_after | 45.000000 +/- 1.082532 | 35.733663 +/- 0.740113 | 2.430989 +/- 0.162197 | 0.840963 +/- 0.018059 |
| plain_after | 36.250000 +/- 2.864110 | 30.889610 +/- 2.930816 | 2.198189 +/- 0.097070 | 0.871550 +/- 0.023957 |
| no_m1_after | 44.583333 +/- 1.572882 | 34.233199 +/- 1.695513 | 2.415391 +/- 0.148488 | 0.839267 +/- 0.013111 |
| no_m2_after | 41.458333 +/- 2.009094 | 34.377860 +/- 1.985490 | 2.337484 +/- 0.118211 | 0.828531 +/- 0.042836 |
| no_m3_after | 45.208333 +/- 1.909407 | 35.515512 +/- 2.134342 | 2.399672 +/- 0.134228 | 0.835392 +/- 0.011601 |

### Matched subsequent objective

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_control_after | 40.208333 +/- 2.009094 | 33.283189 +/- 1.814038 | 2.231615 +/- 0.072852 | 0.855070 +/- 0.006755 |
| plain_after | 36.250000 +/- 2.864110 | 30.889610 +/- 2.930816 | 2.198189 +/- 0.097070 | 0.871550 +/- 0.023957 |

### Supplementary source-fit endpoints

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_before | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |
| plain_before | 39.583333 +/- 3.207251 | 32.348485 +/- 4.434825 | 2.489041 +/- 0.081594 | 0.864514 +/- 0.059707 |
| no_m1_before | 41.250000 +/- 3.125000 | 33.293187 +/- 1.830369 | 2.644953 +/- 0.275164 | 0.887236 +/- 0.064463 |
| no_m2_before | 41.250000 +/- 5.728220 | 34.512678 +/- 1.759513 | 2.530393 +/- 0.150334 | 0.868753 +/- 0.080275 |
| no_m3_before | 41.875000 +/- 3.479853 | 32.205318 +/- 1.849661 | 2.590954 +/- 0.243552 | 0.869410 +/- 0.059931 |


| Contrast | Metric | Paired delta | SD | Pass CI low | High |
|---|---|---:|---:|---:|---:|
| full_minus_plain_after | OA | 8.750000 | 2.864110 | 6.878307 | 10.714286 |
| full_minus_plain_after | BA | 4.844053 | 2.988079 | 0.277778 | 7.590066 |
| full_minus_plain_after | NLL | 0.232799 | 0.122205 | 0.084738 | 0.383172 |
| full_minus_plain_after | Brier | -0.030588 | 0.014459 | -0.059775 | 0.012643 |
| full_minus_no_m1_after | OA | 0.416667 | 1.804220 | -0.966184 | 1.587302 |
| full_minus_no_m1_after | BA | 1.500464 | 1.082473 | 0.432900 | 2.380952 |
| full_minus_no_m1_after | NLL | 0.015598 | 0.026578 | 0.005303 | 0.035280 |
| full_minus_no_m1_after | Brier | 0.001695 | 0.005550 | -0.000934 | 0.007652 |
| full_minus_no_m2_after | OA | 3.541667 | 1.572882 | -0.483092 | 7.142857 |
| full_minus_no_m2_after | BA | 1.355803 | 2.084927 | -5.334467 | 7.575758 |
| full_minus_no_m2_after | NLL | 0.093504 | 0.180462 | -0.021403 | 0.205778 |
| full_minus_no_m2_after | Brier | 0.012432 | 0.040055 | -0.031060 | 0.041720 |
| full_minus_no_m3_after | OA | -0.208333 | 2.818281 | -2.898551 | 2.380952 |
| full_minus_no_m3_after | BA | 0.218151 | 1.614409 | -0.889851 | 0.797289 |
| full_minus_no_m3_after | NLL | 0.031317 | 0.031789 | 0.021304 | 0.050286 |
| full_minus_no_m3_after | Brier | 0.005570 | 0.007033 | -0.003541 | 0.015477 |
| full_control_minus_plain_after | OA | 3.958333 | 4.434124 | 2.898551 | 7.142857 |
| full_control_minus_plain_after | BA | 2.393579 | 4.165113 | -2.380952 | 5.389247 |
| full_control_minus_plain_after | NLL | 0.033425 | 0.083281 | -0.022417 | 0.084256 |
| full_control_minus_plain_after | Brier | -0.016481 | 0.023406 | -0.085903 | 0.012739 |
| full_minus_plain_before | OA | 1.458333 | 3.442232 | 1.058201 | 1.932367 |
| full_minus_plain_before | BA | 0.850701 | 5.193113 | -4.930255 | 4.184704 |
| full_minus_plain_before | NLL | 0.175527 | 0.279710 | 0.112479 | 0.244262 |
| full_minus_plain_before | Brier | 0.028738 | 0.079931 | -0.025661 | 0.060457 |
| full_after_minus_before | OA | 3.958333 | 1.909407 | 2.415459 | 5.820106 |
| full_after_minus_before | BA | 2.534477 | 1.603763 | 0.283447 | 3.535354 |
| full_after_minus_before | NLL | -0.233579 | 0.130913 | -0.302682 | -0.138089 |
| full_after_minus_before | Brier | -0.052289 | 0.049398 | -0.082172 | -0.028814 |
| full_control_after_minus_before | OA | -0.833333 | 1.301041 | -4.347826 | 2.645503 |
| full_control_after_minus_before | BA | 0.084003 | 0.987046 | -1.621315 | 1.659452 |
| full_control_after_minus_before | NLL | -0.432953 | 0.217249 | -0.653229 | -0.209715 |
| full_control_after_minus_before | Brier | -0.038182 | 0.060402 | -0.112443 | 0.036422 |
| plain_after_minus_before | OA | -3.333333 | 1.909407 | -5.952381 | 0.000000 |
| plain_after_minus_before | BA | -1.458874 | 3.261696 | -2.751693 | 1.082251 |
| plain_after_minus_before | NLL | -0.290851 | 0.067370 | -0.408782 | -0.155618 |
| plain_after_minus_before | Brier | 0.007037 | 0.039414 | -0.034358 | 0.045536 |
| no_m1_after_minus_before | OA | 3.333333 | 2.525907 | 1.190476 | 4.232804 |
| no_m1_after_minus_before | BA | 0.940012 | 2.486501 | 0.290433 | 2.477152 |
| no_m1_after_minus_before | NLL | -0.229562 | 0.139700 | -0.299952 | -0.133976 |
| no_m1_after_minus_before | Brier | -0.047968 | 0.051622 | -0.079511 | -0.026579 |
| no_m2_after_minus_before | OA | 0.208333 | 3.767321 | -3.571429 | 1.932367 |
| no_m2_after_minus_before | BA | -0.134818 | 1.736634 | -7.142857 | 2.851474 |
| no_m2_after_minus_before | NLL | -0.192909 | 0.032979 | -0.260114 | -0.069404 |
| no_m2_after_minus_before | Brier | -0.040222 | 0.037797 | -0.050184 | -0.012543 |
| no_m3_after_minus_before | OA | 3.333333 | 4.254287 | 1.190476 | 3.864734 |
| no_m3_after_minus_before | BA | 3.310194 | 2.535755 | 2.976221 | 8.032708 |
| no_m3_after_minus_before | NLL | -0.191281 | 0.115987 | -0.235316 | -0.143557 |
| no_m3_after_minus_before | Brier | -0.034018 | 0.048435 | -0.058903 | -0.016958 |

| Contrast | State | Seed events | Unique queries | Unique ships |
|---|---|---:|---:|---:|
| full_minus_plain_after | rescued | 76 | 41 | 41 |
| full_minus_plain_after | harmed | 34 | 21 | 21 |
| full_minus_plain_after | CC | 140 | 62 | 62 |
| full_minus_plain_after | WW | 230 | 89 | 88 |
| full_minus_no_m1_after | rescued | 6 | 5 | 5 |
| full_minus_no_m1_after | harmed | 4 | 3 | 3 |
| full_minus_no_m1_after | CC | 210 | 87 | 87 |
| full_minus_no_m1_after | WW | 260 | 101 | 100 |
| full_minus_no_m2_after | rescued | 56 | 38 | 38 |
| full_minus_no_m2_after | harmed | 39 | 25 | 25 |
| full_minus_no_m2_after | CC | 160 | 69 | 69 |
| full_minus_no_m2_after | WW | 225 | 89 | 88 |
| full_minus_no_m3_after | rescued | 9 | 9 | 9 |
| full_minus_no_m3_after | harmed | 10 | 9 | 9 |
| full_minus_no_m3_after | CC | 207 | 86 | 86 |
| full_minus_no_m3_after | WW | 254 | 96 | 95 |
| full_control_minus_plain_after | rescued | 68 | 37 | 37 |
| full_control_minus_plain_after | harmed | 49 | 30 | 30 |
| full_control_minus_plain_after | CC | 125 | 55 | 55 |
| full_control_minus_plain_after | WW | 238 | 91 | 90 |
| full_minus_plain_before | rescued | 51 | 36 | 36 |
| full_minus_plain_before | harmed | 44 | 31 | 31 |
| full_minus_plain_before | CC | 146 | 65 | 65 |
| full_minus_plain_before | WW | 239 | 100 | 99 |
| full_after_minus_before | rescued | 28 | 24 | 24 |
| full_after_minus_before | harmed | 9 | 8 | 8 |
| full_after_minus_before | CC | 188 | 81 | 81 |
| full_after_minus_before | WW | 255 | 99 | 98 |
| full_control_after_minus_before | rescued | 23 | 19 | 19 |
| full_control_after_minus_before | harmed | 27 | 20 | 20 |
| full_control_after_minus_before | CC | 170 | 75 | 75 |
| full_control_after_minus_before | WW | 260 | 100 | 99 |
| plain_after_minus_before | rescued | 25 | 18 | 18 |
| plain_after_minus_before | harmed | 41 | 27 | 27 |
| plain_after_minus_before | CC | 149 | 68 | 68 |
| plain_after_minus_before | WW | 265 | 102 | 101 |
| no_m1_after_minus_before | rescued | 25 | 22 | 22 |
| no_m1_after_minus_before | harmed | 9 | 7 | 7 |
| no_m1_after_minus_before | CC | 189 | 83 | 83 |
| no_m1_after_minus_before | WW | 257 | 98 | 97 |
| no_m2_after_minus_before | rescued | 20 | 16 | 16 |
| no_m2_after_minus_before | harmed | 19 | 17 | 17 |
| no_m2_after_minus_before | CC | 179 | 79 | 79 |
| no_m2_after_minus_before | WW | 262 | 105 | 104 |
| no_m3_after_minus_before | rescued | 25 | 21 | 21 |
| no_m3_after_minus_before | harmed | 9 | 9 | 9 |
| no_m3_after_minus_before | CC | 192 | 82 | 82 |
| no_m3_after_minus_before | WW | 254 | 97 | 96 |

| Contrast | Passes | Resamples missing a present class | Total resamples |
|---|---:|---:|---:|
| full_minus_plain_after | 3 | 0 | 2000 |
| full_minus_no_m1_after | 3 | 0 | 2000 |
| full_minus_no_m2_after | 3 | 0 | 2000 |
| full_minus_no_m3_after | 3 | 0 | 2000 |
| full_control_minus_plain_after | 3 | 0 | 2000 |
| full_minus_plain_before | 3 | 0 | 2000 |
| full_after_minus_before | 3 | 0 | 2000 |
| full_control_after_minus_before | 3 | 0 | 2000 |
| plain_after_minus_before | 3 | 0 | 2000 |
| no_m1_after_minus_before | 3 | 0 | 2000 |
| no_m2_after_minus_before | 3 | 0 | 2000 |
| no_m3_after_minus_before | 3 | 0 | 2000 |
## expanded

### Main final endpoints

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_after | 47.317073 +/- 1.125128 | 40.402194 +/- 1.067667 | 2.137044 +/- 0.047298 | 0.786188 +/- 0.004741 |
| plain_after | 37.214092 +/- 0.123120 | 35.552536 +/- 0.886773 | 1.962057 +/- 0.049032 | 0.834224 +/- 0.016731 |
| no_m1_after | 47.067751 +/- 1.255021 | 40.275660 +/- 1.078867 | 2.129804 +/- 0.034206 | 0.786423 +/- 0.005758 |
| no_m2_after | 41.127371 +/- 0.891990 | 36.672560 +/- 0.543512 | 2.177252 +/- 0.049900 | 0.834183 +/- 0.008330 |
| no_m3_after | 47.176152 +/- 1.024433 | 40.429019 +/- 1.121971 | 2.107877 +/- 0.052508 | 0.782152 +/- 0.006085 |

### Matched subsequent objective

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_control_after | 45.127371 +/- 1.561427 | 41.039015 +/- 0.600579 | 1.939647 +/- 0.023102 | 0.783313 +/- 0.010681 |
| plain_after | 37.214092 +/- 0.123120 | 35.552536 +/- 0.886773 | 1.962057 +/- 0.049032 | 0.834224 +/- 0.016731 |

### Supplementary source-fit endpoints

| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| full_before | 46.352304 +/- 1.122933 | 38.682940 +/- 1.856018 | 2.344226 +/- 0.076481 | 0.815594 +/- 0.001003 |
| plain_before | 41.452575 +/- 0.196023 | 35.358035 +/- 1.003572 | 2.276706 +/- 0.011150 | 0.841827 +/- 0.003966 |
| no_m1_before | 46.124661 +/- 1.352366 | 38.443885 +/- 1.942835 | 2.339095 +/- 0.061038 | 0.816707 +/- 0.004676 |
| no_m2_before | 40.888889 +/- 0.049676 | 35.495461 +/- 1.301246 | 2.329889 +/- 0.033579 | 0.851597 +/- 0.005278 |
| no_m3_before | 46.959350 +/- 0.567011 | 38.791189 +/- 1.381278 | 2.281743 +/- 0.082909 | 0.801701 +/- 0.006930 |


| Contrast | Metric | Paired delta | SD | Pass CI low | High |
|---|---|---:|---:|---:|---:|
| full_minus_plain_after | OA | 10.102981 | 1.236343 | 8.519701 | 11.674835 |
| full_minus_plain_after | BA | 4.849658 | 0.725856 | 3.260488 | 6.328906 |
| full_minus_plain_after | NLL | 0.174987 | 0.048529 | 0.102024 | 0.249336 |
| full_minus_plain_after | Brier | -0.048035 | 0.020388 | -0.069244 | -0.028285 |
| full_minus_no_m1_after | OA | 0.249322 | 0.284127 | -0.000000 | 0.516053 |
| full_minus_no_m1_after | BA | 0.126534 | 0.027402 | -0.158113 | 0.474761 |
| full_minus_no_m1_after | NLL | 0.007241 | 0.017279 | 0.002855 | 0.011376 |
| full_minus_no_m1_after | Brier | -0.000235 | 0.003823 | -0.001827 | 0.001280 |
| full_minus_no_m2_after | OA | 6.189702 | 0.487082 | 5.027700 | 7.431891 |
| full_minus_no_m2_after | BA | 3.729634 | 0.870915 | 2.344438 | 4.897566 |
| full_minus_no_m2_after | NLL | -0.040207 | 0.079588 | -0.087314 | 0.004710 |
| full_minus_no_m2_after | Brier | -0.047994 | 0.012821 | -0.064170 | -0.033505 |
| full_minus_no_m3_after | OA | 0.140921 | 0.592846 | -0.262254 | 0.552181 |
| full_minus_no_m3_after | BA | -0.026825 | 0.217137 | -0.415156 | 0.371196 |
| full_minus_no_m3_after | NLL | 0.029168 | 0.005300 | 0.019199 | 0.037807 |
| full_minus_no_m3_after | Brier | 0.004036 | 0.002798 | 0.000112 | 0.007653 |
| full_control_minus_plain_after | OA | 7.913279 | 1.664689 | 6.483761 | 9.365787 |
| full_control_minus_plain_after | BA | 5.486478 | 0.527829 | 3.812846 | 7.081425 |
| full_control_minus_plain_after | NLL | -0.022410 | 0.069619 | -0.061505 | 0.018085 |
| full_control_minus_plain_after | Brier | -0.050911 | 0.027181 | -0.066885 | -0.034667 |
| full_minus_plain_before | OA | 4.899729 | 1.012492 | 3.641298 | 6.284457 |
| full_minus_plain_before | BA | 3.324905 | 1.736030 | 1.942247 | 4.526071 |
| full_minus_plain_before | NLL | 0.067520 | 0.087125 | 0.006608 | 0.125163 |
| full_minus_plain_before | Brier | -0.026233 | 0.003158 | -0.045204 | -0.009117 |
| full_after_minus_before | OA | 0.964770 | 0.190552 | 0.111206 | 1.774400 |
| full_after_minus_before | BA | 1.719254 | 0.879254 | 0.887891 | 2.590519 |
| full_after_minus_before | NLL | -0.207182 | 0.044143 | -0.244556 | -0.165451 |
| full_after_minus_before | Brier | -0.029405 | 0.005738 | -0.038716 | -0.019004 |
| full_control_after_minus_before | OA | -1.224932 | 0.441527 | -2.788434 | 0.174625 |
| full_control_after_minus_before | BA | 2.356075 | 1.323391 | 1.190828 | 3.438428 |
| full_control_after_minus_before | NLL | -0.404579 | 0.089953 | -0.494560 | -0.307961 |
| full_control_after_minus_before | Brier | -0.032281 | 0.011120 | -0.052128 | -0.009968 |
| plain_after_minus_before | OA | -4.238482 | 0.235258 | -5.407149 | -3.004653 |
| plain_after_minus_before | BA | 0.194502 | 0.189108 | -1.008551 | 1.331632 |
| plain_after_minus_before | NLL | -0.314648 | 0.057216 | -0.393817 | -0.228180 |
| plain_after_minus_before | Brier | -0.007603 | 0.013055 | -0.022104 | 0.007368 |
| no_m1_after_minus_before | OA | 0.943089 | 0.130081 | 0.228719 | 1.595896 |
| no_m1_after_minus_before | BA | 1.831776 | 0.946668 | 1.199405 | 2.485285 |
| no_m1_after_minus_before | NLL | -0.209291 | 0.041881 | -0.244774 | -0.168543 |
| no_m1_after_minus_before | Brier | -0.030284 | 0.005396 | -0.039511 | -0.019869 |
| no_m2_after_minus_before | OA | 0.238482 | 0.855682 | -0.442055 | 0.861496 |
| no_m2_after_minus_before | BA | 1.177099 | 0.793224 | 0.358184 | 1.896148 |
| no_m2_after_minus_before | NLL | -0.152637 | 0.016895 | -0.186297 | -0.116315 |
| no_m2_after_minus_before | Brier | -0.017415 | 0.004810 | -0.024257 | -0.010596 |
| no_m3_after_minus_before | OA | 0.216802 | 0.460290 | -0.583709 | 0.959198 |
| no_m3_after_minus_before | BA | 1.637830 | 0.867185 | 0.953478 | 2.344038 |
| no_m3_after_minus_before | NLL | -0.173866 | 0.036490 | -0.209148 | -0.132733 |
| no_m3_after_minus_before | Brier | -0.019548 | 0.005188 | -0.028360 | -0.009001 |

| Contrast | State | Seed events | Unique queries | Unique ships |
|---|---|---:|---:|---:|
| full_minus_plain_after | rescued | 1545 | 838 | 693 |
| full_minus_plain_after | harmed | 613 | 394 | 367 |
| full_minus_plain_after | CC | 2820 | 1177 | 982 |
| full_minus_plain_after | WW | 4247 | 1686 | 1187 |
| full_minus_no_m1_after | rescued | 83 | 80 | 78 |
| full_minus_no_m1_after | harmed | 60 | 59 | 58 |
| full_minus_no_m1_after | CC | 4282 | 1714 | 1299 |
| full_minus_no_m1_after | WW | 4800 | 1889 | 1307 |
| full_minus_no_m2_after | rescued | 1220 | 723 | 620 |
| full_minus_no_m2_after | harmed | 649 | 441 | 404 |
| full_minus_no_m2_after | CC | 3145 | 1315 | 1063 |
| full_minus_no_m2_after | WW | 4211 | 1677 | 1180 |
| full_minus_no_m3_after | rescued | 179 | 171 | 167 |
| full_minus_no_m3_after | harmed | 166 | 154 | 148 |
| full_minus_no_m3_after | CC | 4186 | 1670 | 1279 |
| full_minus_no_m3_after | WW | 4694 | 1838 | 1275 |
| full_control_minus_plain_after | rescued | 1467 | 803 | 678 |
| full_control_minus_plain_after | harmed | 737 | 459 | 423 |
| full_control_minus_plain_after | CC | 2696 | 1136 | 960 |
| full_control_minus_plain_after | WW | 4325 | 1711 | 1217 |
| full_minus_plain_before | rescued | 1149 | 752 | 638 |
| full_minus_plain_before | harmed | 697 | 515 | 448 |
| full_minus_plain_before | CC | 3127 | 1365 | 1094 |
| full_minus_plain_before | WW | 4252 | 1747 | 1203 |
| full_after_minus_before | rescued | 360 | 298 | 268 |
| full_after_minus_before | harmed | 271 | 234 | 225 |
| full_after_minus_before | CC | 4005 | 1666 | 1270 |
| full_after_minus_before | WW | 4589 | 1854 | 1283 |
| full_control_after_minus_before | rescued | 592 | 417 | 365 |
| full_control_after_minus_before | harmed | 705 | 486 | 427 |
| full_control_after_minus_before | CC | 3571 | 1517 | 1191 |
| full_control_after_minus_before | WW | 4357 | 1793 | 1258 |
| plain_after_minus_before | rescued | 503 | 343 | 314 |
| plain_after_minus_before | harmed | 894 | 567 | 484 |
| plain_after_minus_before | CC | 2930 | 1245 | 1035 |
| plain_after_minus_before | WW | 4898 | 1932 | 1331 |
| no_m1_after_minus_before | rescued | 355 | 306 | 281 |
| no_m1_after_minus_before | harmed | 268 | 230 | 219 |
| no_m1_after_minus_before | CC | 3987 | 1665 | 1272 |
| no_m1_after_minus_before | WW | 4615 | 1877 | 1300 |
| no_m2_after_minus_before | rescued | 377 | 317 | 291 |
| no_m2_after_minus_before | harmed | 355 | 303 | 274 |
| no_m2_after_minus_before | CC | 3417 | 1467 | 1149 |
| no_m2_after_minus_before | WW | 5076 | 2017 | 1373 |
| no_m3_after_minus_before | rescued | 296 | 243 | 224 |
| no_m3_after_minus_before | harmed | 276 | 232 | 213 |
| no_m3_after_minus_before | CC | 4056 | 1653 | 1270 |
| no_m3_after_minus_before | WW | 4597 | 1821 | 1266 |

| Contrast | Passes | Resamples missing a present class | Total resamples |
|---|---:|---:|---:|
| full_minus_plain_after | 85 | 0 | 2000 |
| full_minus_no_m1_after | 85 | 0 | 2000 |
| full_minus_no_m2_after | 85 | 0 | 2000 |
| full_minus_no_m3_after | 85 | 0 | 2000 |
| full_control_minus_plain_after | 85 | 0 | 2000 |
| full_minus_plain_before | 85 | 0 | 2000 |
| full_after_minus_before | 85 | 0 | 2000 |
| full_control_after_minus_before | 85 | 0 | 2000 |
| plain_after_minus_before | 85 | 0 | 2000 |
| no_m1_after_minus_before | 85 | 0 | 2000 |
| no_m2_after_minus_before | 85 | 0 | 2000 |
| no_m3_after_minus_before | 85 | 0 | 2000 |

All deltas are alternative minus reference. Positive OA/BA favors the alternative; negative NLL/Brier favors it. Whole-pass resampling retains paired queries and all fixed seeds; development has only three passes. Groups CSVs preserve class/port/pass/actual-polarization differences and prediction transitions, including WW changes. Training/check receipts are retained verbatim in summary.json; logits alone do not certify optimization or mechanism.

[Native gallery](cases/index.html): 12 distinct queries, 24 VV/VH panels, not_reviewed. Main-agent image inspection, visible facts, interpretation and study report remain pending.
