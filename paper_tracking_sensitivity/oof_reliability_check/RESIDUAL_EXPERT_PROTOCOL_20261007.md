# Source-anchored visual residual expert: fixed development screen

Date 2026-10-07. Written before candidate readout. This tests a changed
classifier, not another gate feature or an established novel algorithm.

## Hypothesis

The frozen source expert can be improved using the same 64 target labels to
learn a regularized visual correction. Unlike an independent target classifier
blended on the old selected 5%, this changes the eight-class decision function
while retaining the frozen source expert as its prior. It can correct examples
where both frozen predictors are wrong. Bias-only adaptation is the matched
control for gains that need no sample-dependent visual correction.

## Fixed inputs and split

23 active outer ports, seeds 0..4, original uniform 64-row supports, same query
rows excluding support. Source PCA128+PCA64 feature cache pred_cache2, aligned
to original manifest and pred_cache. Reproduce each source expert head on the
cached features before use; source-only scaling and fitting, no target labels.
No encoder/PCA retraining, no extra labels, no facility or identity inputs.
Outer source purge is inherited. Target support/query is row-disjoint ONLY.
This is exposed development data, not independent confirmation.

Pre-readout numerical correction: first run stopped after three frozen ports
because float32 cache reconstruction changed one Fujairah SOURCE row's argmax.
Its original top gap was 9.24e-7, row error 3.40e-6, whole-fold max 1.10e-5.
Reconstruction now requires max score error <1e-4 and every argmax difference
to have original top gap <=2*that row's error. Preserve counts. The original
source scores remain the actual prior, never replaced by reconstructed scores.
No candidate metrics were read; failed artifacts retained, rerun in new v2 dir.

## Models

Inputs x: concatenate Xs(128) and Z(64), standardize using all outer source
rows as the original fusion head does, then divide by sqrt(192). Append one
constant coordinate. Class targets are +1 for true class, -1 otherwise,
matching RidgeClassifier's eight-class source score convention. Target class
weights are 64/(number_of_observed_classes * class_count), sum 64.

For target support A and fixed lambda=8, solve
  delta = argmin_D sum_i w_i ||Y_i - prior_i - phi_i D||^2 + 8 ||D||_F^2.
Predicted scores are prior(x)+phi(x)delta. Intercept correction is penalized
equally. Lambda 8 reuses the existing adaptation strength; it is not tuned.
Source scaling and feature normalization are identical for all matched arms.

- fusion: unchanged original source expert.
- oof_lr0: unchanged archived best original gate (same support/evaluation).
- legacy_target64: reproduce existing balanced Ridge(alpha=1), weights sum1,
  one-hot targets, unnormalized [Xs,Z], free intercept, on the SAME supports.
- target64: matched visual ridge from zero prior, same phi/weights/lambda.
- anchor_bias64: frozen expert prior + penalized eight-class bias correction.
- anchor_visual64: frozen expert prior + penalized visual/bias correction.

No observed-class guard is imposed on the new classifier: absent target
classes retain the source prior subject to the regularized multiclass update.
This is not an assertion that absent classes are impossible. No label-based
early stopping, parameter grid or portwise winner selection.

Save all source scalers, support rows, corrections, score arrays and policies
for all ports before candidate outcome readout. Primary contrasts are
anchor_visual64 minus fusion and anchor_visual64 minus anchor_bias64. Also
report comparisons against oof_lr0 and both target-only baselines. Screen
passes only if BOTH primary equal-port mean BA differences are positive.
No significance threshold or requirement to win every port. Positive means
only justify further development, not top-journal readiness.

## Measurements and follow-through

Report BA and accuracy on identical query rows, average five draws within
each port then equal across ports. Include per-class recall, recovery/harm
relative to fusion, and balanced recovery on both-frozen-models-wrong rows.
Report descriptive paired port-bootstrap intervals and support-draw deltas.
Do not confuse increased pairwise oracle headroom with deployable gain.

This stage evaluates expert quality on all queries, so it is NOT a 5% compute
or replacement-budget result. Only an effective expert will justify a later
budgeted handoff test and proper source episode/OOF gate training. Failure
stops this fixed branch without a lambda sweep or universal negative claim.
