# Bounded facility OOF pilot, 2026-10-07

Fixed before candidate readout. This is a development information-value test,
not a validated novel architecture or independent confirmation.

Reuse 23 active outer ports, five original support draws, 64 labels per draw,
and the original genuine source OOF predictions. No head/backbone training.
Keep all target queries, excluding only each draw's support rows from scoring.
The inherited target split is row-only, not vessel/product-disjoint. Selection
is uncapped; this is NOT the historical 5% replacement experiment.

Inputs: immutable ../port_context_audit_20261007/facility_features.npz,
ONLY bounded values/known, 60 columns, current manifest order. No historical
mask, class labels, ship identity, port centroid or trajectory as features.
Recorded coordinate proxies and unresolved 2025/2026 map/SAR temporal alignment
remain limitations. A supported layer is not complete ground truth absence.

Four source-only logistic gates (C=1, lbfgs, tol=1e-9, max_iter=2000):
- oof_lr0: reproduce original four-confidence-feature reference on all source.
- covered_conf: same four features, fit only covered informative source rows.
- covered_mask: same rows plus 60 binary known masks.
- covered_values: same rows plus masks plus 60 masked standardized values.

Informative means disagreement with one expert correct. The fixed source
port/class weights are computed on ALL source rows before filtering, retaining
the original N*w weighting; no rebalancing by outcome or availability. All
three covered arms use exactly the same training rows and weights. Confidence
scaling is the original full-source scaling. Facility mean/std uses only known
source entries, weighted by the same source weights, without query data.
Missing standardized values are zero, distinguished by masks. Features never
observed in source are disabled in target; coverage means at least one usable
known feature. If covered training lacks either sign, use the original gate.
There is no target coefficient update or hyperparameter selection.

All arms retain the original observed-expert-class guard from the same 64
target labels. Entirely unsupported queries fall back EXACTLY to oof_lr0.
The suffix 0 does not mean zero target labels. Baseline predictions simple,
s2, base, fusion reuse the already verified enriched-run policies.

Save all 23 ports' scores, coefficients and policies BEFORE readout. Primary
contrasts: covered_values minus covered_mask, and covered_values minus oof_lr0.
Diagnostic contrasts: covered_mask minus covered_conf; covered_conf minus
oof_lr0; covered_values minus simple. Report equal-port mean BA, first averaging
the five overlapping draws; port bootstrap descriptive intervals, positive
ports, full-query results, and supported-subset diagnostics on IDENTICAL rows.
No global permutation (deferred by user), tuning sweep or best-port selection.
Promotion requires positive mean for BOTH primary contrasts, not significance;
failure stops this fixed branch, not all possible context methods.
