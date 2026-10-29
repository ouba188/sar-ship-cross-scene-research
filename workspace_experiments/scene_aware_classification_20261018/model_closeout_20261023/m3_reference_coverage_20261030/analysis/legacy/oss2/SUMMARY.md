# development / legacy / oss2

Complete three-order pair: True. Queries: 228.

raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training.

| Arm | Orders | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| raw_source | 3 | 58.479532 | 49.636699 | 1.381566 | 0.618582 |
| direct_center | 3 | 58.333333 | 49.540913 | 1.375738 | 0.616770 |
| control | 3 | 57.894737 | 48.863958 | 1.383957 | 0.616579 |
| trained_center | 3 | 58.187135 | 49.051223 | 1.378238 | 0.615151 |
| bank_original | 1 | 57.894737 | 50.123534 | 1.397182 | 0.622810 |
| bank_random_8 | 1 | 57.894737 | 50.123534 | 1.397555 | 0.622857 |
| bank_coverage_8 | 1 | 57.894737 | 50.123534 | 1.397660 | 0.622836 |
| bank_random_16 | 1 | 57.894737 | 50.123534 | 1.397628 | 0.622936 |
| bank_coverage_16 | 1 | 57.894737 | 50.123534 | 1.397310 | 0.622730 |

| Arm - Reference | Orders | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue/Harm events |
|---|---|---|---|---|---|---|
| direct_center - raw_source | 3 | -0.146199 | -0.095785 | -0.005828 | -0.001811 | 1/2 |
| control - raw_source | 3 | -0.584795 | -0.772741 | +0.002392 | -0.002003 | 5/9 |
| trained_center - raw_source | 3 | -0.292398 | -0.585475 | -0.003328 | -0.003430 | 7/9 |
| bank_original - raw_source | 1 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/0 |
| bank_random_8 - raw_source | 1 | +0.000000 | +0.000000 | +0.000374 | +0.000048 | 0/0 |
| bank_coverage_8 - raw_source | 1 | +0.000000 | +0.000000 | +0.000478 | +0.000026 | 0/0 |
| bank_random_16 - raw_source | 1 | +0.000000 | +0.000000 | +0.000447 | +0.000126 | 0/0 |
| bank_coverage_16 - raw_source | 1 | +0.000000 | +0.000000 | +0.000129 | -0.000080 | 0/0 |
| trained_center - control | 3 | +0.292398 | +0.187266 | -0.005719 | -0.001427 | 2/0 |
| trained_center - direct_center | 3 | -0.146199 | -0.489690 | +0.002500 | -0.001619 | 7/8 |

raw_source is the saved baseline logits from center NPZ; no separate raw_source file exists.
BA averages present classes only. Seeds are minibatch orders, not independent initialization or data.
Relation rows are direct-center snapshots, not a post-training remeasurement. same_mass averages all 20 slots including zero-mass slots; entropy is in nats.
Coverage is class-macro of within-class ship-weighted nearest source distance; self/same-ship distances were not masked.

Interpret only this named scope; do not mix development and final outcomes.
