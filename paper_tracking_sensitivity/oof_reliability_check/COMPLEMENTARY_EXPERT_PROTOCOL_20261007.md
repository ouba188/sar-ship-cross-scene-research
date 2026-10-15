# Cross-fitted correction of the adapted base

Written 2026-10-07 before this study's query readout. Development experiment
after support-scalar shrinkage failed; not an independent confirmation.
Same user-approved64 labels shared by base and expert.5% caps actual expert
handoffs from the adapted base, not changes from the original source model.

## Fixed Mechanism

Keep all23 ports and five original64-label support draws. Source base B0,
source expert E0, source-standardized192D features/sqrt192 plus intercept,
class-balanced support weights, penalized-intercept ridge lambda8, and the
original source gate stay fixed. No backbone/source-head/gate retraining,
new inputs, shrinkage selection, parameter sweep or special port rule.

B64 is the unchanged globally adapted VV-only base. For each support row i,
Bminus_i is its genuine63-label adapted-base prediction; training weights
were recomputed within each63-row fit. Reuse these stored LOO residuals from
the verified support-shrinkage run, WITHOUT reusing its selected alphas.
Let D=E0-B0, the existing source-trained expert/base score difference.
It carries inherited source-domain learned information, not newly acquired
port facility, geography or AIS information. Both heads use the same8 score
columns. No claim that D isolates purely VH or is an optimal transfer prior.

Primary complementary expert:
  C = ridge(phi_support, (+/-1 targets) - [Bminus_i + D_support])
  Enew(x) = B64(x) + D(x) + phi(x) C.

The expert fits the remaining score error relative to an honest held-out
base prediction, instead of adapting independently of the deployed base.
This is score-residual specialization, not training only on base mistakes.
The final expert itself trains on all64 labels; its support predictions are
NOT held-out expert validation and must not be reported as such. The63/64
base training difference is an acknowledged deployment mismatch. There is
no theorem or established novelty claim attached to this construction.

## Isolating Controls

Keep three unchanged policies: base_visual, visual_handoff, bias_handoff.
Three candidate experts and their5% handoff policies:
- crossfit: primary recipe above;
- insample: use B64 at support instead of Bminus_i, all else identical;
- no_source: use the held-out base recipe with D identically zero.
All experts use the SAME192D features,64 labels,lambda8 and class weights.
No source-contrast-free claim for no_source: source features and B0 remain;
it only removes the explicit expert-minus-base score contrast.

Gate features are recomputed from the actual unchanged B64 and each actual
receiving expert. Eligibility: base/expert disagreement, expert predicted
class observed among64 labels, gate>0, evaluation row. Rank by gate with
global row tie break; cap floor(.05*N_eval). Existing gate coefficients and
threshold remain unchanged, so adapted-score calibration remains unproved.
Full expert scores are accessed before admission; no compute-saving claim.

Primary contrast: crossfit_handoff minus visual_handoff. Secondary contrasts
against insample_handoff, no_source_handoff, bias_handoff and base_visual
show whether each mechanism contributes; a failed primary is not rescued by
selecting another arm after readout. Report all negatives and actual counts.

Also report full-query expert quality separately from5% policies: accuracy,
BA, class recalls, rescues/harms relative to adapted base, rescues where both
base and old expert were wrong, and losses on old-expert-correct rows. These
are diagnostics, not5% deployment results or label-informed routing rules.
Save all23 policies before reading query labels for this study's evaluation.

## Validation And Limits

Independent augmented-SVD fits check the correction, exact old controls,
actual receiver gate/masks and budget counts; reconstruct policy and expert
metrics separately. Reuse previously verified LOO caches with exact support
alignment; no need to repeat14720 completed fits. Focused tests cover source
base reduction to the old expert, deployment/training separation and exact
zero-residual behavior. A fixed63-row subset replay may check cache usage.

Ports, not the115 overlapping support draws, are the statistical units.
All targets have development exposure; support/query is row-only, not
vessel/product-disjoint. Inherited source/head/encoder provenance remains
unchanged. Never pick a different winner per port or describe this as a
proven complementary-information or journal-level innovation before evidence.
