# development / legacy / private

Complete three-order pair: True. Queries: 160.

raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training.

| Arm | Orders | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| raw_source | 3 | 45.000000 | 35.733663 | 2.430989 | 0.840963 |
| direct_center | 3 | 44.375000 | 34.705525 | 2.415956 | 0.837282 |
| control | 3 | 44.791667 | 35.895228 | 2.398730 | 0.839229 |
| trained_center | 3 | 45.000000 | 35.963255 | 2.395961 | 0.839130 |
| bank_original | 1 | 45.625000 | 34.928108 | 2.532032 | 0.835259 |
| bank_random_2 | 1 | 45.625000 | 34.928108 | 2.531815 | 0.835195 |
| bank_coverage_2 | 1 | 45.000000 | 34.724026 | 2.531812 | 0.835641 |
| bank_random_4 | 1 | 45.000000 | 34.278757 | 2.530922 | 0.834580 |
| bank_coverage_4 | 1 | 45.625000 | 34.928108 | 2.531448 | 0.835329 |

| Arm - Reference | Orders | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue/Harm events |
|---|---|---|---|---|---|---|
| direct_center - raw_source | 3 | -0.625000 | -1.028139 | -0.015032 | -0.003681 | 0/3 |
| control - raw_source | 3 | -0.208333 | +0.161565 | -0.032259 | -0.001733 | 3/4 |
| trained_center - raw_source | 3 | +0.000000 | +0.229592 | -0.035028 | -0.001833 | 3/3 |
| bank_original - raw_source | 1 | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0/0 |
| bank_random_2 - raw_source | 1 | +0.000000 | +0.000000 | -0.000217 | -0.000063 | 0/0 |
| bank_coverage_2 - raw_source | 1 | -0.625000 | -0.204082 | -0.000220 | +0.000382 | 0/1 |
| bank_random_4 - raw_source | 1 | -0.625000 | -0.649351 | -0.001110 | -0.000679 | 0/1 |
| bank_coverage_4 - raw_source | 1 | +0.000000 | +0.000000 | -0.000584 | +0.000070 | 0/0 |
| trained_center - control | 3 | +0.208333 | +0.068027 | -0.002769 | -0.000099 | 1/0 |
| trained_center - direct_center | 3 | +0.625000 | +1.257730 | -0.019996 | +0.001848 | 6/3 |

raw_source is the saved baseline logits from center NPZ; no separate raw_source file exists.
BA averages present classes only. Seeds are minibatch orders, not independent initialization or data.
Relation rows are direct-center snapshots, not a post-training remeasurement. same_mass averages all 20 slots including zero-mass slots; entropy is in nats.
Coverage is class-macro of within-class ship-weighted nearest source distance; self/same-ship distances were not masked.

Interpret only this named scope; do not mix development and final outcomes.
