# Support-only residual shrinkage

Written 2026-10-07 before this candidate's query readout. Development study,
not an untouched confirmation. User-approved setting: both heads share the
same 64 target labels; 5% limits expert handoff, not changes from source base.

## Fixed Intervention

Preserve the 23 ports, five original support draws, source-only geometry,
base VV128 and expert VV128+VH64 features, original priors, lambda=8 residual
ridge, penalized intercept, and frozen four-feature source gate. No new data,
source/gate fitting, per-port rules, or parameter search.

For each branch and support draw, perform 64 exact leave-one-out residual
fits. Each uses only 63 labels, recomputing class-balanced training weights
from those 63 labels. A held-out singleton's class is absent from its fold.
Predict the residual at the omitted row. Select a single scalar alpha in
[0,1] minimizing class-balanced squared error over these 64 held-out rows:
alpha = clip(sum(w * desired_residual * loo_residual) /
             sum(w * loo_residual**2), 0, 1).
Zero denominator gives alpha=0. Validation weights use all 64 validation
labels; these labels intentionally select alpha, but do not train their own
LOO prediction. This loss is a fitting surrogate, NOT calibrated risk, a
balanced-accuracy guarantee, or an unbiased estimate of selected-model risk.
Scale the existing all64-fitted correction by alpha for deployment. There
are only 64 unique target labels, not extra validation labels. Support/query
exclusion remains row-only, not vessel/product-disjoint.

## Arms And Readout

Unchanged controls: base_visual, visual_handoff, bias_handoff.
Candidates: shrink_base (no handoff), shrink_base_handoff (only base scaled),
shrink_expert_handoff (only expert scaled), shrink_both_handoff.
Primary is shrink_base_handoff minus visual_handoff; other arms are fixed
secondary diagnostics, not replacements for a failed primary. Also compare
shrink_base_handoff to shrink_base and shrink_both_handoff to bias_handoff.
Preserve all arms even when negative. Report port/draw differences, accuracy,
class recall, actual handoff counts, original-base changes and both alphas.

Apply the unchanged frozen gate to the actual two branch scores of each
policy. Eligibility remains disagreement, expert predicted class observed
in support, gate>0, evaluation row. Rank with global row tie break and cap
floor(.05*N_eval). Scores for all experts are accessed; no execution-saving
claim. Save all port policies before reading query labels for this readout.

This is an ordinary cross-validated shrinkage candidate, not established
novelty. Existing poor-port readouts motivated this study. No hardcoded
Mombasa/Newcastle exceptions, query-label tuning or selective port reports.
Verification checks omitted-label independence, exact fold weights, original
control preservation, actual receiver masks/caps and metric reconstruction.
