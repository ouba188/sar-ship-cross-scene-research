# Error-focused expert follow-up

Declared 2026-10-07 AFTER the complementary-score experiment's negative
readout, but BEFORE this follow-up's query readout. Preserve that failed
primary and report both studies, without treating this as its rescue arm.
This is the only follow-up in this round; no weight or regularization sweep.

## Intervention

Keep the original E0-anchored expert residual target,192D source-standardized
features/sqrt192 plus penalized intercept, lambda8, and unshrunk adapted B64
fallback. Do NOT add the just-tested complementary score decomposition.
Focus expert fitting on the adapted base's errors through sample weights:
  difficulty_i = 2 if argmax(Bminus_i) != y_i, else 1
  w_i = (64 / K_present) * difficulty_i /
        sum(difficulty_j for j with y_j=y_i).
Thus every observed class keeps EXACTLY its old64/K weight mass and total
weight remains64. Only relative weights WITHIN a class change; singleton
classes cannot gain extra total weight. Fixed two-to-one emphasis is a
declared single design choice, not an optimized parameter or novel principle.

Bminus_i uses the existing verified63-label base LOO fit, with fold-local
class balancing. Add B0 to its cached correction; never use selected alpha.
The omitted y_i is intentionally used to define the expert training weight,
not to fit its own base LOO prediction. Expert fitting uses all64 labels;
this is not held-out expert validation or a zero-target-label method.

## Fixed Controls And Evaluation

Experts: original uniform-class-balanced E64, LOO-error-weighted E64 (primary),
and in-sample-error-weighted E64 using B64 instead of Bminus_i. The latter
isolates whether the held-out mistake signal matters. All share sourceprior,
features, labels and regularization. No filtering or source facility inputs.

Policies: unchanged base_visual, visual_handoff, bias_handoff, and the two
new experts under the unchanged source confidence gate. Actual base/receiver
features and disagreement are recomputed; observed-expert-class guard,
positive score, global row ties, floor(.05*N_eval) cap all retained.
Same23 ports/five64-label supports. Freeze ALL policies before own query
readout. Primary: loo_error_handoff minus visual_handoff. Report in-sample
comparison, base-only incremental value and bias-handoff comparison, whether
positive or negative. Full-query experts are separate diagnostics, not5%
results. Report rescues of both-old-wrong and losses of old-expert-correct.

## Scope

5% means expert admissions relative to adapted base, not all source-base
prediction changes or compute savings. All expert scores accessed before
selection. Source gate is not newly calibrated for the changed score
distribution. All targets are historically development-exposed and support/
query split is row-only. Novelty and independent confirmation are unproved.
Independent augmented-SVD checks catch incorrect class-mass normalization
or the wrong weighting signal; reconstruct policies/metrics/controls. Reuse
already-verified LOO inputs with exact row alignment, not a redundant full
rerun. No per-port winner selection and no silent main-model replacement.
