# final / anchor_corrected / private

Complete three-order pair: True. Queries: 3075.

raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training.

| Arm | Orders | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| raw_source | 3 | 47.317073 | 40.402194 | 2.137044 | 0.786188 |
| trained_center | 3 | 46.601626 | 40.705488 | 2.115207 | 0.790493 |
| control | 3 | 46.623306 | 40.770558 | 2.117489 | 0.790349 |

| Arm - Reference | Orders | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue/Harm events |
|---|---|---|---|---|---|---|
| trained_center - raw_source | 3 | -0.715447 | +0.303294 | -0.021837 | +0.004305 | 81/147 |
| control - raw_source | 3 | -0.693767 | +0.368364 | -0.019555 | +0.004161 | 85/149 |
| trained_center - control | 3 | -0.021680 | -0.065070 | -0.002282 | +0.000144 | 42/44 |

raw_source = historical saved original-model logits; control = freshly evaluated matched value-path training; trained_center = the frozen corrected candidate.
No final relation traces were saved; development relation measurements are not presented as final measurements.
Candidate and hyperparameters were frozen before this final evaluation. Cohorts have historical exposure and are not newly blind tests.
BA averages present classes; three orders share initialization and queries. Rescue/harm events across orders are not independent samples.

Interpret only this named scope; do not mix development and final outcomes.
