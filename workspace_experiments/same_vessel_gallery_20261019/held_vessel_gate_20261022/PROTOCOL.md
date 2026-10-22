# Source Held-Vessel Gate Probe

Frozen before results on 2026-10-22.

## Question

Does training a residual-strength gate on predictions for unseen source vessels
transfer better than training it on in-sample predictions?

The previous scalar family had optimistic label-informed rescue space, but the
learned geometry gate realized little of it. This does not establish learnability.
The new experiment tests training exposure, not a new geometry or image module.

## Data And Scope

- Reuse the existing 24-port-derived cohort, 3 outer identity/date splits and
  the same 4,744 evaluation observations. Exclude observation 31237.
- Native 160 crop/pad, frozen image descriptors, no resize or new image inference.
- Whole-chip baseline A and neighborhood residual D remain unchanged at evaluation.
- Existing source validation is reused for checkpoint selection. This is development
  evidence, not a new independent acquisition test or full target-support pipeline.
- Inner folds hold out MMSI within outer source training; they do not hold out
  source acquisitions. Outer validation/test identity and product isolation stays.

## Entire-Stack Cross-Fitting

Within each outer training partition, stratify vessels by class into 3 inner folds.
For each inner fold, refit normalization and the ridge classifier, then the
keep_refs residual, then the separated-neighborhood residual, excluding the held
vessels throughout. Original source validation selects ridge alpha and checkpoints;
it contains no inner held-vessel identities. Predict held source vessels once each.
Frozen externally pretrained descriptors are reused; no fold-specific supervised
head or normalization trained on a held vessel is reused for its OOF prediction.

Ridge: original alpha grid 0.1/1/10. Each neural head: original 150 steps, batch 128,
AdamW lr=0.001, weight_decay=0.001, class/vessel weighted MSE, 3 paired seeds.
Source validation selects 0/50/100/150 steps by BA then Brier.

## Matched Gate Comparison

Use class-aligned score coordinates [A (8), D (8)], not latent hidden coordinates
whose bases differ across refitted networks. No true class, MMSI, port, angle or
target labels are gate inputs. Score inputs are not normalized with target data.

Output A + w D, w=2 sigmoid(g), initially w=1.

Four arms: in-sample constant, OOF constant, in-sample score gate, OOF score gate.
Score gate is 16->8 GELU->1 (145 parameters); constant is 1 parameter.
Both exposure conditions use identical source rows, labels, weights, minibatch
indices, initializations, training steps, objective and source validation.
Only gate-training predictions differ. Evaluation always uses the same existing
full-source A,D and fixed test queries. OOF models see about 2/3 source vessels;
report this train-size shift and do not attribute all effects solely to leakage.

## Evidence And Stop Rule

Report OA, BA, macro-F1, Brier, class/port outcomes, rescue/harm by seed, gate
weights and source in-sample versus OOF error distributions. Paired MMSI bootstrap
is conditional on fixed fitted models, not an independent training replicate.
Compare each score gate to its constant control, and OOF to in-sample score gate.
Inspect native VV/VH rescue/harm/persistent cases and prior anchors 10461,30237.
Audit held identities across all learned stages and replay saved selected gates.

One bounded round; no target-guided search or extra steps after seeing results.
Positive scope may be retained without universal wins. If not improved, distinguish
unseen-error training exposure from residual-direction limits and objective limits;
do not claim causal/physical evidence separation.
