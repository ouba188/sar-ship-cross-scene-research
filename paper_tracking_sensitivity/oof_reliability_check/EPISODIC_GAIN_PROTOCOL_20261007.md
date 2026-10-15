# Source-episodic reliability for an adapted receiving expert

Fixed before candidate target readout, 2026-10-07. Purpose: improve budgeted
BA, not add nominal modules. Reuse genuine nested source OOF head predictions
and the strong target64 bias-adapted expert. No backbone, PCA or original
classification-head fitting. No new target labels, parameter sweep or facility.

## Matched source episodes

For each outer target port, load its verified genuine OOF source cache. Each
inner-held source port's old base/expert predictions came from heads trained
without that inner port and outer target, with the inherited identity purges.
Within that held port, sample64 source labels using seeds0..4 and adapt the
expert with EXACTLY the existing bias-only residual recipe: +/-1 class targets,
class-balanced support weights summing64, fixed strength8, penalized intercept.
Score all remaining rows, never including those64 adaptation rows in gate
training for that episode. Held ports with <=64 available rows cannot form a
64-support/query episode and are skipped by size alone, with counts recorded.
Target support/query and source episode adaptation/query isolation remain ROW
ONLY, not vessel/product-disjoint. Head-training purge is a separate boundary.

Use only the original four confidence features: adapted expert/base top-two
score gap and entropy. Source feature center/scale is fit on all episode query
rows, with weights equal across held ports, five support draws and present
true classes within each query. Weights sum1 before eligibility filtering.
Deployment eligibility is base!=adapted_expert and support_observed[expert].
Fit gate models only on such source episode query rows. Keep source port,
episode support/query indices, features, weights, predictions and true utility.

## Two fixed fitted models

Both use sklearn LogisticRegression(C=1, lbfgs, max_iter=2000, tol=1e-9), same
four standardized inputs, sample weights N_all_episode_queries*w.

- matched_binary: discard zero-utility rows, fit recovery(+1) versus harm(-1).
  Rank by fitted log odds, require positive score. This isolates training on
  appropriately adapted source experts instead of the old unadapted experts.
- gain_3way (candidate): fit harm(-1), neutral(0), recovery(+1), retaining
  BOTH-WRONG disagreement cases. Rank by P(recovery)-P(harm), require >0.
  This targets expected correction benefit rather than conditional win odds.
  Outcomes and true-class weights are source training information only.

No new features or target coefficient adaptation. Four-target-feature scaling
is source-only. Neither gate is claimed perfectly calibrated under domain shift.
Fit each outer gate once, then apply it to all five original target supports.

## Targets and controls

Same23 target ports, seeds0..4,64 labels/draw and strict floor(B*(Nquery-64))
admission cap. B=.05 primary, .10/.20 secondary. Base fallback unchanged.
New gates receive the SAME previously verified bias-adapted expert predictions
as the frozen receiver-consistent bias baseline. Keep both frozen bias and
frozen VISUAL pipelines as full-policy controls on identical evaluation rows.
Tie-break by ascending globalrow; do not force nonpositive-score admissions.
Count admissions and changes separately; no compute-saving claim because
query expert scores are available to the gate before selection.

Save all source models, episode banks, target scores, masks and predictions
before target query readout. Primary development screen for gain_3way:
positive mean BA differences versus BOTH frozen_visual and matched_binary
at5%. Report gain_3way versus frozen_bias as well, and matched_binary versus
both frozen baselines. A better binary result remains useful pipeline evidence,
but does not establish value from the neutral-outcome modeling. No per-port
winner mixing, best-budget rescue or post-readout regularization adjustments.

Equal-port BA after five-draw means, accuracy, recoveries/harms relative to
base, admission rates, per-class recall, positive ports, draw deltas and paired
port-bootstrap descriptive intervals. Preserve negative ports and all fixed
arms. These historically exposed data provide development evidence only.
