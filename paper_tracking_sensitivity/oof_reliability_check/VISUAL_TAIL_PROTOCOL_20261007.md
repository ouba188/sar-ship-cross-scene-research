# Recover omitted visual information: fixed expert experiment

Written2026-10-07 before candidate query readout. Change the expert's INPUT
information, not the gate, kernel shape or original source classifier. The
source-metric/local-kernel branch remains closed. This is not a new residual
learning principle or an established top-journal contribution.

## Inputs and isolation

Reuse exactly the original frozen VV2048 pooled_features.npy and VH512
b2_ssl_feats_38091.npz/F, all38091 rows in the original manifest order.
Encoder access/provenance is inherited, not newly proved. Retain the original
outer source rows, source PCA128/64 caches, source prior logits, five target
support draws of64 labels, query exclusion, and frozen four-feature gate.
No new target labels, feature extractor training, image filtering or port
selection. All feature fitting below uses outer SOURCE rows only, no labels.

For each polarization separately, let P be its existing source-standardized
PCA features divided bysqrt192. Center P and raw features F on source rows.
Fit the unregularized least-squares decoder D from P_source to F_source.
Its residual T=F-mean_F-(P-mean_P)D is the omitted linear feature component.
Normalize T so its mean SOURCE squared norm equals that polarization's P
norm. Do not coordinate-whiten the tail. Retain each polarization's original
128:64 energy allocation, rather than weighting512/2048 by raw dimension.

Save decoder/centers/scales and source orthogonality/energy diagnostics.
The residual is orthogonal to retained features in SOURCE sample space;
it is not claimed pointwise orthogonal to an exact PCA basis. Before any
experiment run, the draft inverse-decoder reconstruction check was removed:
randomized approximate PCA need not span a covariance-invariant subspace.
Instead compare the raw files/manifest to existing OOF input receipts once,
reusing the builder's established checks to skip expensive PCA/head replays.
Abort on mismatch, do not mint a new provenance claim. Normalization does
not consume target feature statistics.

## Only two new experts

- visual_tail: phi=[P_VV,P_VH,T_VV,T_VH,1]. P remains exactly unchanged.
- energy_control: phi=[sqrt(2)*P_VV,sqrt(2)*P_VH,1]. Same total source visual
  energy as visual_tail, but no extra information. This is a regularization/
  scaling control, not an attempt to match every effective eigenvalue.

Use existing residual_expert.fit_correction: source expert scores are the
prior; support targets are2*one_hot(y)-1; class-balanced support weights
sum64; strength8 also penalizes the intercept. Only those64 support labels
fit the correction. No widths, strengths, mixtures, seeds or budgets swept.
The two kernels share the constant bias contribution. Low+tail energy is
twice the archived linear visual energy; the control is essential to avoid
crediting stronger correction to new information.

Controls: unchanged linear residual and bias-only experts, and the earlier
action-advantage visual whole-policy reference at5% (not a new candidate gate).
Retain ALL arms regardless of outcomes; no per-port winner mixing.

## Endpoints

Primary admission cap5%, floor(.05*N_eval), same positive gate score,
observed-expert-class guard, actual disagreement and global-row tie break.
Recompute frozen gate confidence inputs on each actual receiver. Primary
mean-only development screen requires visual_tail to beat linear,
energy_control AND advantage_reference at5%. Report all three contrasts,
plus energy_control versus linear and visual_tail versus bias.

Full-query expert BA/accuracy are secondary, not5% results. Report rescues
of cases both base and archived linear expert get wrong, and harms against
that FULL linear expert; budget versions of the latter include fallback
opportunity loss, not incremental matched-budget harm. Metrics use target
query labels only AFTER all23 ports' policies have been saved. Statistical
unit is each port's mean of five overlapping support draws; intervals are
descriptive port-bootstrap intervals, not independent confirmation.

## Limits and stopping

Tail dimension is2752 total visual coordinates (192 retained +2560 residual
coordinates), with redundancy. It restores information already in the frozen
embeddings; it does not add new images, native scale, facilities or AIS. More
coordinates do not guarantee useful signal, and target support/query remains
ROW ONLY, not product/vessel-disjoint. Tail memory may exploit acquisition or
identity similarity; this experiment alone cannot identify that mechanism.
All data are historically exposed development data. Every receiver is scored
before admission, so5% is NOT compute savings. Do not promote if it loses to
the strong controls, and do not rescue a failed screen by target-based tuning.
