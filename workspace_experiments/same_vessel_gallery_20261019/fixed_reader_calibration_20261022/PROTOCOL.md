# Fixed Reader, Independent Source Calibration

Frozen 2026-10-22 before this round's results.

Question: with the same A and D at gate training and evaluation, does learning
from independent source vessels improve residual-strength selection?

Reuse all 3 inner partitions and 3 seeds from held_vessel_gate_20261022, without
selecting a favorable split. Each inner model was fitted on 2/3 outer source
vessels, with its own normalization/ridge/keep_refs/neighborhood head. Its remaining
1/3 source vessels were excluded from every upstream fit and validation selection.
Those held vessels now form its calibration set. All upstream parameters remain
frozen; do not retrain on calibration vessels after fitting the gate.

The existing outer source validation set chooses 0/50/100/150 gate steps, BA then
Brier. It has already selected upstream checkpoints, so this remains development
evidence. Inner source calibration is not acquisition-isolated from source fitting;
outer test identity/date/product separation stays unchanged. No test-label selection.

Keep A+wD, w=2sigmoid, class-aligned A8,D8 inputs, 16->8->1 gate (145 parameters),
or scalar constant (1 parameter). AdamW lr/wd .001, 150 updates, batch128, original
class/vessel weighted MSE. Initial w=1. Same preprocessing, native crop/pad, no resize.

Four matched arms: seen_constant, calibration_constant, seen_score,
calibration_score. For the seen control, sample without replacement from upstream
fit observations to match calibration observation counts for each class. Report
unique vessel counts separately; matching rows is not matching identities.
Sort both pools by class and observation ID; use paired initialization and minibatch
positions. The gates have identical nominal row counts, class counts and update
budgets; their within-vessel repetition and resulting weights can still differ.

Each arm tests the exact model that generated its training scores. Compare first
to that model's no-gate A+D reference and constant control. Report old full-source
A+D as a separate total-data-cost reference, not as if the upstream model were
unchanged. Calibration consumes source labels and reduces upstream fit volume.

All 3 calibration splits x 3 seeds are repeated evaluations, NOT a 3-model inference
ensemble, and NOT nine independent datasets. Pool the three outer test folds per
split/seed (4,744 observations), then summarize repetitions. Also show seed means
and split means. Bootstrap class-stratified MMSI after averaging repeated outcomes,
conditional on fitted models. Keep zero-D models in the main table; report active
model diagnostics separately without selecting a new favorable main result.

Controls: replace dynamic weights by that model's calibration-set mean weight;
inspect training/validation/test residual rescue/harm and weight variation.
Save native VV/VH rescue/harm/CC/WW and previous anchors, actual score evidence.
No target support adaptation, new image backbone, geometry inputs or new data.

Stop after one bounded round. Retain small scoped gains if supported; do not call
training mismatch the sole root cause or zero-output groups evidence of unlearnability.
