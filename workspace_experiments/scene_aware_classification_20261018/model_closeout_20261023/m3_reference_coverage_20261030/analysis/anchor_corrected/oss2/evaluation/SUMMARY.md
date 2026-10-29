# final / anchor_corrected / oss2

Complete three-order pair: True. Queries: 378.

raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training.

| Arm | Orders | OA | BA | NLL | Brier |
|---|---|---|---|---|---|
| raw_source | 3 | 61.375661 | 57.662916 | 1.274771 | 0.574993 |
| trained_center | 3 | 61.111111 | 58.347054 | 1.281784 | 0.578540 |
| control | 3 | 60.846561 | 58.162269 | 1.288939 | 0.580595 |

| Arm - Reference | Orders | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue/Harm events |
|---|---|---|---|---|---|---|
| trained_center - raw_source | 3 | -0.264550 | +0.684139 | +0.007012 | +0.003547 | 14/17 |
| control - raw_source | 3 | -0.529101 | +0.499353 | +0.014168 | +0.005603 | 11/17 |
| trained_center - control | 3 | +0.264550 | +0.184786 | -0.007155 | -0.002055 | 4/1 |

raw_source = historical saved original-model logits; control = freshly evaluated matched value-path training; trained_center = the frozen corrected candidate.
No final relation traces were saved; development relation measurements are not presented as final measurements.
Candidate and hyperparameters were frozen before this final evaluation. Cohorts have historical exposure and are not newly blind tests.
BA averages present classes; three orders share initialization and queries. Rescue/harm events across orders are not independent samples.

Interpret only this named scope; do not mix development and final outcomes.
