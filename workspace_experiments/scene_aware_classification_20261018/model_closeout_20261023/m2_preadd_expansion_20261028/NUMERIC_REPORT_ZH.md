# Frozen C/preadd expansion numeric report

Frozen historical 3075 observations, 1950 ships, 85 passes, 18 ports, 8 classes. No fitting, inference, tuning or model selection in this CPU analysis. Baseline/E reuse six historical scores after main-owned matched replay; only C has three new inference runs. Metrics are computed per seed 17/29/43 then averaged; no probability ensemble. OA/BA are percentages and deltas percentage points; BA averages classes present in the group. Natural-log NLL and summed eight-class Brier are lower-is-better. SD uses ddof=1. Historical targets are not a new blind test; seeds are not independent datasets.

## Three-seed means and sample SD

| Arm | OA | BA | NLL | Brier |
|---|---:|---:|---:|---:|
| baseline | 47.349593 +/- 0.117254 | 39.372503 +/- 0.103603 | 2.163407 +/- 0.020969 | 0.789796 +/- 0.002200 |
| stage | 47.067751 +/- 0.123120 | 39.852226 +/- 0.112115 | 2.182185 +/- 0.016582 | 0.794291 +/- 0.001931 |
| preadd | 47.382114 +/- 0.065041 | 39.650674 +/- 0.072522 | 2.183971 +/- 0.017135 | 0.792983 +/- 0.002181 |

## Paired differences

| Comparison | Metric | Mean +/- SD | Improved seeds | 95% pass CI |
|---|---|---:|---:|---:|
| preadd_vs_baseline | OA | +0.032520 +/- 0.168981 | 1/3 | [-0.601247, +0.672006] |
| preadd_vs_baseline | BA | +0.278172 +/- 0.099210 | 3/3 | [-0.252122, +0.862805] |
| preadd_vs_baseline | NLL | +0.020564 +/- 0.004799 | 0/3 | [+0.005512, +0.036408] |
| preadd_vs_baseline | Brier | +0.003187 +/- 0.000220 | 0/3 | [-0.001173, +0.007862] |
| preadd_vs_stage | OA | +0.314363 +/- 0.179108 | 3/3 | [-0.054262, +0.718974] |
| preadd_vs_stage | BA | -0.201551 +/- 0.140143 | 0/3 | [-0.461978, +0.081104] |
| preadd_vs_stage | NLL | +0.001785 +/- 0.002787 | 1/3 | [-0.004634, +0.008421] |
| preadd_vs_stage | Brier | -0.001308 +/- 0.000408 | 3/3 | [-0.004087, +0.001442] |
| stage_vs_baseline | OA | -0.281843 +/- 0.018776 | 0/3 | [-0.913303, +0.311092] |
| stage_vs_baseline | BA | +0.479723 +/- 0.213135 | 3/3 | [+0.007247, +0.988231] |
| stage_vs_baseline | NLL | +0.018778 +/- 0.007307 | 0/3 | [+0.003006, +0.035045] |
| stage_vs_baseline | Brier | +0.004495 +/- 0.000293 | 0/3 | [-0.000727, +0.010513] |

2000 paired percentile bootstrap draws of 85 whole passes with replacement, RNG seed 20261028. Identical draws apply to all arms and seeds. Conditional on nine fixed models and the fixed reference bank; not retraining uncertainty, independent ships, or multiplicity-adjusted confirmation. A ship can recur across passes. BA averages present classes in each draw; all draws retained.

## Rescue/harm and persistent outcomes

| Comparison | State | Seed events | Unique IDs | Unique ships |
|---|---|---:|---:|---:|
| preadd_vs_baseline | rescue | 206 | 97 | 95 |
| preadd_vs_baseline | harm | 203 | 102 | 101 |
| preadd_vs_baseline | CC | 4165 | 1424 | 1125 |
| preadd_vs_baseline | WW | 4651 | 1584 | 1117 |
| preadd_vs_stage | rescue | 104 | 75 | 74 |
| preadd_vs_stage | harm | 75 | 54 | 54 |
| preadd_vs_stage | CC | 4267 | 1464 | 1151 |
| preadd_vs_stage | WW | 4779 | 1636 | 1150 |
| stage_vs_baseline | rescue | 197 | 98 | 96 |
| stage_vs_baseline | harm | 223 | 115 | 112 |
| stage_vs_baseline | CC | 4145 | 1420 | 1126 |
| stage_vs_baseline | WW | 4660 | 1593 | 1124 |

preadd_vs_baseline: both rescued and harmed across seeds: 1 IDs / 4 ships.


preadd_vs_stage: both rescued and harmed across seeds: 2 IDs / 3 ships.


stage_vs_baseline: both rescued and harmed across seeds: 5 IDs / 5 ships.

Repeated seed events are not independent observations. Counts by one/two/three seeds are in summary.json.

## Group differences

Class-group OA/BA both equal recall. Old three ports are Jebel Ali, Port Klang and Singapore. Group mean/SD and unique-ID/ship outcomes are in paired_summary.csv; complete per-seed results include all 85 passes.

| Comparison | Dimension | Group | n | Delta OA | Delta BA | Delta NLL | Delta Brier |
|---|---|---|---:|---:|---:|---:|---:|
| preadd_vs_baseline | class | 0:bulk_carrier | 947 | -0.527983 | -0.527983 | +0.034119 | +0.014283 |
| preadd_vs_stage | class | 0:bulk_carrier | 947 | +0.844773 | +0.844773 | -0.022528 | -0.009497 |
| stage_vs_baseline | class | 0:bulk_carrier | 947 | -1.372756 | -1.372756 | +0.056648 | +0.023780 |
| preadd_vs_baseline | class | 1:fishing | 102 | +0.000000 | +0.000000 | +0.051473 | +0.010769 |
| preadd_vs_stage | class | 1:fishing | 102 | +0.000000 | +0.000000 | +0.024372 | +0.003164 |
| stage_vs_baseline | class | 1:fishing | 102 | +0.000000 | +0.000000 | +0.027101 | +0.007605 |
| preadd_vs_baseline | class | 2:general_cargo | 301 | +1.328904 | +1.328904 | -0.094519 | -0.037551 |
| preadd_vs_stage | class | 2:general_cargo | 301 | -1.882614 | -1.882614 | +0.063032 | +0.023586 |
| stage_vs_baseline | class | 2:general_cargo | 301 | +3.211517 | +3.211517 | -0.157551 | -0.061137 |
| preadd_vs_baseline | class | 3:chemical_tanker | 803 | -0.415110 | -0.415110 | -0.000597 | +0.004384 |
| preadd_vs_stage | class | 3:chemical_tanker | 803 | +1.494396 | +1.494396 | -0.053673 | -0.021081 |
| stage_vs_baseline | class | 3:chemical_tanker | 803 | -1.909506 | -1.909506 | +0.053076 | +0.025465 |
| preadd_vs_baseline | class | 4:container | 361 | -0.184672 | -0.184672 | +0.213162 | +0.033145 |
| preadd_vs_stage | class | 4:container | 361 | +0.092336 | +0.092336 | +0.079822 | +0.025477 |
| stage_vs_baseline | class | 4:container | 361 | -0.277008 | -0.277008 | +0.133340 | +0.007667 |
| preadd_vs_baseline | class | 5:crude_oil_tanker | 256 | +2.864583 | +2.864583 | -0.205200 | -0.059381 |
| preadd_vs_stage | class | 5:crude_oil_tanker | 256 | -1.562500 | -1.562500 | +0.034993 | +0.014446 |
| stage_vs_baseline | class | 5:crude_oil_tanker | 256 | +4.427083 | +4.427083 | -0.240192 | -0.073827 |
| preadd_vs_baseline | class | 6:tug | 138 | -0.241546 | -0.241546 | -0.008634 | +0.011090 |
| preadd_vs_stage | class | 6:tug | 138 | +0.000000 | +0.000000 | +0.044144 | +0.005698 |
| stage_vs_baseline | class | 6:tug | 138 | -0.241546 | -0.241546 | -0.052778 | +0.005393 |
| preadd_vs_baseline | class | 7:offshore | 167 | -0.598802 | -0.598802 | +0.187862 | +0.027923 |
| preadd_vs_stage | class | 7:offshore | 167 | -0.598802 | -0.598802 | +0.027542 | +0.004756 |
| stage_vs_baseline | class | 7:offshore | 167 | -0.000000 | -0.000000 | +0.160320 | +0.023167 |
| preadd_vs_baseline | port | Busan | 34 | +0.000000 | -0.540123 | -0.006895 | -0.002798 |
| preadd_vs_stage | port | Busan | 34 | +0.000000 | +0.000000 | +0.000526 | +0.001142 |
| stage_vs_baseline | port | Busan | 34 | +0.000000 | -0.540123 | -0.007421 | -0.003940 |
| preadd_vs_baseline | port | Callao | 8 | +0.000000 | +0.000000 | -0.007395 | +0.027651 |
| preadd_vs_stage | port | Callao | 8 | -4.166667 | -8.333333 | +0.070470 | +0.038981 |
| stage_vs_baseline | port | Callao | 8 | +4.166667 | +8.333333 | -0.077865 | -0.011330 |
| preadd_vs_baseline | port | Fujairah | 110 | +3.939394 | +5.902031 | +0.069322 | +0.007271 |
| preadd_vs_stage | port | Fujairah | 110 | +1.212121 | -0.143369 | +0.001600 | -0.001152 |
| stage_vs_baseline | port | Fujairah | 110 | +2.727273 | +6.045400 | +0.067722 | +0.008422 |
| preadd_vs_baseline | port | Houston | 7 | +0.000000 | +0.000000 | +0.196926 | +0.073583 |
| preadd_vs_stage | port | Houston | 7 | +4.761905 | +4.166667 | +0.000589 | +0.012164 |
| stage_vs_baseline | port | Houston | 7 | -4.761905 | -4.166667 | +0.196336 | +0.061419 |
| preadd_vs_baseline | port | Jebel Ali | 628 | +0.053079 | +0.048253 | +0.056274 | +0.007716 |
| preadd_vs_stage | port | Jebel Ali | 628 | +0.690021 | +0.007882 | +0.008126 | -0.002928 |
| stage_vs_baseline | port | Jebel Ali | 628 | -0.636943 | +0.040371 | +0.048147 | +0.010643 |
| preadd_vs_baseline | port | Los Angeles | 103 | +3.559871 | +0.738080 | -0.071850 | -0.027875 |
| preadd_vs_stage | port | Los Angeles | 103 | +0.000000 | -1.428571 | +0.031088 | +0.001955 |
| stage_vs_baseline | port | Los Angeles | 103 | +3.559871 | +2.166651 | -0.102939 | -0.029830 |
| preadd_vs_baseline | port | Mombasa | 7 | -14.285714 | -18.518519 | +0.036188 | +0.059791 |
| preadd_vs_stage | port | Mombasa | 7 | -4.761905 | -11.111111 | +0.025823 | +0.002541 |
| stage_vs_baseline | port | Mombasa | 7 | -9.523810 | -7.407407 | +0.010364 | +0.057249 |
| preadd_vs_baseline | port | Newcastle | 20 | +8.333333 | +4.629630 | -0.023903 | -0.007140 |
| preadd_vs_stage | port | Newcastle | 20 | +5.000000 | +2.777778 | -0.087493 | -0.041022 |
| stage_vs_baseline | port | Newcastle | 20 | +3.333333 | +1.851852 | +0.063590 | +0.033882 |
| preadd_vs_baseline | port | Ningbo-Zhoushan | 195 | -0.341880 | -0.673401 | +0.005119 | +0.001780 |
| preadd_vs_stage | port | Ningbo-Zhoushan | 195 | +0.000000 | -0.829106 | +0.016147 | +0.000022 |
| stage_vs_baseline | port | Ningbo-Zhoushan | 195 | -0.341880 | +0.155705 | -0.011028 | +0.001758 |
| preadd_vs_baseline | port | Port Hedland | 40 | -14.166667 | -3.238095 | +0.092134 | +0.055555 |
| preadd_vs_stage | port | Port Hedland | 40 | +0.000000 | +0.000000 | -0.068286 | -0.039857 |
| stage_vs_baseline | port | Port Hedland | 40 | -14.166667 | -3.238095 | +0.160421 | +0.095412 |
| preadd_vs_baseline | port | Port Klang | 798 | +0.751880 | +1.144557 | -0.008101 | -0.004766 |
| preadd_vs_stage | port | Port Klang | 798 | +0.375940 | -0.046554 | +0.003696 | +0.000264 |
| stage_vs_baseline | port | Port Klang | 798 | +0.375940 | +1.191111 | -0.011798 | -0.005030 |
| preadd_vs_baseline | port | Port Said | 96 | +1.388889 | +1.830517 | -0.068666 | -0.011599 |
| preadd_vs_stage | port | Port Said | 96 | +2.430556 | +1.826250 | -0.024837 | -0.010498 |
| stage_vs_baseline | port | Port Said | 96 | -1.041667 | +0.004267 | -0.043828 | -0.001101 |
| preadd_vs_baseline | port | Richards Bay | 195 | -0.512821 | +1.908180 | +0.060558 | +0.024158 |
| preadd_vs_stage | port | Richards Bay | 195 | +1.196581 | +1.444900 | -0.034999 | -0.013650 |
| stage_vs_baseline | port | Richards Bay | 195 | -1.709402 | +0.463279 | +0.095557 | +0.037808 |
| preadd_vs_baseline | port | Rotterdam | 30 | +0.000000 | +0.000000 | -0.186007 | -0.064571 |
| preadd_vs_stage | port | Rotterdam | 30 | -1.111111 | -0.833333 | +0.048266 | +0.016230 |
| stage_vs_baseline | port | Rotterdam | 30 | +1.111111 | +0.833333 | -0.234273 | -0.080801 |
| preadd_vs_baseline | port | Santos | 239 | -1.673640 | +0.046356 | +0.073083 | +0.022280 |
| preadd_vs_stage | port | Santos | 239 | -0.139470 | -0.340136 | +0.015781 | +0.000695 |
| stage_vs_baseline | port | Santos | 239 | -1.534170 | +0.386492 | +0.057302 | +0.021585 |
| preadd_vs_baseline | port | Singapore | 498 | -0.602410 | +0.062845 | +0.024638 | -0.000081 |
| preadd_vs_stage | port | Singapore | 498 | -0.267738 | -0.611953 | -0.004083 | +0.004026 |
| stage_vs_baseline | port | Singapore | 498 | -0.334672 | +0.674798 | +0.028720 | -0.004107 |
| preadd_vs_baseline | port | Sydney Botany | 7 | +0.000000 | +0.000000 | +0.138210 | +0.055237 |
| preadd_vs_stage | port | Sydney Botany | 7 | +0.000000 | +0.000000 | +0.039688 | +0.002007 |
| stage_vs_baseline | port | Sydney Botany | 7 | +0.000000 | +0.000000 | +0.098522 | +0.053231 |
| preadd_vs_baseline | port | Tanger Med | 60 | -1.666667 | -0.657081 | -0.029177 | +0.000176 |
| preadd_vs_stage | port | Tanger Med | 60 | -3.888889 | -2.253401 | +0.006162 | +0.008074 |
| stage_vs_baseline | port | Tanger Med | 60 | +2.222222 | +1.596320 | -0.035339 | -0.007899 |
| preadd_vs_baseline | polarization | VV | 199 | -0.167504 | +0.605907 | +0.035260 | +0.008048 |
| preadd_vs_stage | polarization | VV | 199 | +0.167504 | +0.476190 | -0.000493 | +0.003264 |
| stage_vs_baseline | polarization | VV | 199 | -0.335008 | +0.129717 | +0.035752 | +0.004783 |
| preadd_vs_baseline | polarization | VV+VH | 2876 | +0.046361 | +0.400946 | +0.019547 | +0.002850 |
| preadd_vs_stage | polarization | VV+VH | 2876 | +0.324525 | -0.220030 | +0.001943 | -0.001625 |
| stage_vs_baseline | polarization | VV+VH | 2876 | -0.278164 | +0.620975 | +0.017604 | +0.004475 |
| preadd_vs_baseline | port_scope | old_three_ports | 1924 | +0.173250 | +0.413247 | +0.021385 | +0.000521 |
| preadd_vs_stage | port_scope | old_three_ports | 1924 | +0.311850 | -0.222059 | +0.003129 | +0.000196 |
| stage_vs_baseline | port_scope | old_three_ports | 1924 | -0.138600 | +0.635305 | +0.018256 | +0.000325 |
| preadd_vs_baseline | port_scope | other_fifteen_ports | 1151 | -0.202722 | +0.461620 | +0.019191 | +0.007643 |
| preadd_vs_stage | port_scope | other_fifteen_ports | 1151 | +0.318564 | +0.078038 | -0.000460 | -0.003823 |
| stage_vs_baseline | port_scope | other_fifteen_ports | 1151 | -0.521286 | +0.383582 | +0.019651 | +0.011466 |

## Native cases

Two deterministic rounds of rescue/harm/CC/WW per comparison, using prior expansion selection. C-vs-E receives its own cases. Main diagnostic IDs are appended when available. Cases are posthoc illustrations, not prevalence estimates or causal proof.

Each case has comparison/arm/reference, representative_seed, fixed_rival and per_seed baseline_logits/stage_logits/preadd_logits. Actual same-class and confusion-class references plus the entire 16-reference bank are rendered separately for VV/VH, with source paths and preprocessing in native_sources.json.

Generated panels do not establish visual review. Eight C-comparison queries have now been reviewed in VV/VH with actual references; see CASE_NOTES_ZH.md. Other panels remain generated illustrations. All case_paths interventions are on C/preadd, regardless of the comparison used to select the case.

[Native gallery](analysis/cases/index.html)

- preadd_vs_baseline / 34733 / rescue / bulk_carrier / seed 29
- preadd_vs_baseline / 31166 / harm / container / seed 29
- preadd_vs_baseline / 26109 / CC / container / seed 29
- preadd_vs_baseline / 32844 / WW / crude_oil_tanker / seed 29
- preadd_vs_baseline / 30134 / rescue / container / seed 17
- preadd_vs_baseline / 25929 / harm / bulk_carrier / seed 29
- preadd_vs_baseline / 9724 / CC / bulk_carrier / seed 17
- preadd_vs_baseline / 36034 / WW / container / seed 29
- preadd_vs_stage / 7872 / rescue / bulk_carrier / seed 29
- preadd_vs_stage / 28401 / harm / chemical_tanker / seed 17
- preadd_vs_stage / 35487 / CC / bulk_carrier / seed 29
- preadd_vs_stage / 26914 / WW / bulk_carrier / seed 17
- preadd_vs_stage / 8647 / rescue / chemical_tanker / seed 29
- preadd_vs_stage / 2490 / harm / crude_oil_tanker / seed 29
- preadd_vs_stage / 4248 / CC / container / seed 29
- preadd_vs_stage / 23004 / WW / offshore / seed 43
- stage_vs_baseline / 34733 / rescue / bulk_carrier / seed 29
- stage_vs_baseline / 4007 / harm / container / seed 43
- stage_vs_baseline / 36543 / CC / crude_oil_tanker / seed 43
- stage_vs_baseline / 32844 / WW / crude_oil_tanker / seed 29
- stage_vs_baseline / 30134 / rescue / container / seed 29
- stage_vs_baseline / 27591 / harm / chemical_tanker / seed 29
- stage_vs_baseline / 9724 / CC / bulk_carrier / seed 43
- stage_vs_baseline / 23004 / WW / offshore / seed 43
