# development / anchor_corrected / oss2

Complete three-order pair: True. Queries: 228.

raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training.

| Arm | Orders | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| raw_source | 3 | 58.479532 | 49.636699 | 1.381566 | 0.618582 |
| direct_center | 3 | 58.333333 | 49.540913 | 1.375499 | 0.616642 |
| control | 3 | 57.894737 | 48.863958 | 1.383959 | 0.616579 |
| trained_center | 3 | 58.187135 | 49.051223 | 1.378076 | 0.615117 |
| bank_original | 1 | 57.894737 | 50.123534 | 1.397182 | 0.622810 |
| bank_random_8 | 1 | 57.894737 | 50.123534 | 1.397555 | 0.622857 |
| bank_coverage_8 | 1 | 57.894737 | 50.123534 | 1.398169 | 0.622983 |
| bank_random_16 | 1 | 57.894737 | 50.123534 | 1.397628 | 0.622936 |
| bank_coverage_16 | 1 | 57.894737 | 50.123534 | 1.397357 | 0.622767 |

| Arm - Reference | Orders | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue/Harm events |
|---|---|---|---|---|---|---|
| direct_center - raw_source | 3 | -0.146199 | -0.095785 | -0.006067 | -0.001940 | 1/2 |
| control - raw_source | 3 | -0.584795 | -0.772741 | +0.002393 | -0.002003 | 5/9 |
| trained_center - raw_source | 3 | -0.292398 | -0.585475 | -0.003490 | -0.003465 | 7/9 |
| bank_original - raw_source | 1 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/0 |
| bank_random_8 - raw_source | 1 | +0.000000 | +0.000000 | +0.000374 | +0.000048 | 0/0 |
| bank_coverage_8 - raw_source | 1 | +0.000000 | +0.000000 | +0.000987 | +0.000173 | 0/0 |
| bank_random_16 - raw_source | 1 | +0.000000 | +0.000000 | +0.000447 | +0.000126 | 0/0 |
| bank_coverage_16 - raw_source | 1 | +0.000000 | +0.000000 | +0.000176 | -0.000043 | 0/0 |
| trained_center - control | 3 | +0.292398 | +0.187266 | -0.005883 | -0.001462 | 2/0 |
| trained_center - direct_center | 3 | -0.146199 | -0.489690 | +0.002577 | -0.001524 | 7/8 |

raw_source is the saved baseline logits from center NPZ; no separate raw_source file exists.
BA averages present classes only. Seeds are minibatch orders, not independent initialization or data.
Relation rows are direct-center snapshots, not a post-training remeasurement. same_mass averages all 20 slots including zero-mass slots; entropy is in nats.
Coverage is class-macro of within-class ship-weighted nearest source distance; self/same-ship distances were not masked.

Interpret only this named scope; do not mix development and final outcomes.
