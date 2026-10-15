# Strong baseline first: adapt the base with the same64 labels

Written2026-10-07 after the negative visual-tail readout, before THIS study's
candidate query readout. This is a separately declared strong-baseline study,
NOT a rescue or parameter change to the failed visual-tail module.

## Important change of setting

User explicitly accepted this setting on2026-10-07 after initial experiment
readout: same64 labels may globally adapt the base;5% limits expert handoffs.
This approval does not retroactively make the study independent validation
or change the earlier fixed-base results.

Earlier5% experiments held the original base prediction fixed and admitted
at most5% to an expert. This study permits target-support adaptation of the
BASE itself. The5% cap now means expert handoff relative to the ADAPTED base,
NOT at most5% changes relative to the original source base. Report changes
relative to the original base separately. Do not attribute the entire gain
over an old fixed-base pipeline to an innovative gate or an equal overall
prediction-change budget. This is authorized64-label adaptation, not a
label-free cross-port method. No compute savings are established.

## Fixed recipes

Keep23 ports, five original64-label supports, original row-only support/query
exclusion and all source/PCA/encoder caches. Base uses VV128 only. Its prior
is the original source-base eight-class scores. Fit two familiar residuals
using residual_expert.fit_correction, class-balanced support weights sum64,
targets +/-1, strength8 including intercept:
- base_bias: intercept-only correction;
- base_visual: source-standardized VV128/sqrt128 plus intercept correction.

Independent sklearn weighted Ridge solves must reproduce both
coefficients before deployment; failure aborts, never relax to obtain a gain.
Post-run terminology correction only: sklearn cholesky switches to a dual
solve when features exceed samples, so the original draft's blanket word
"primal" was inaccurate. No fitting recipe, tolerance or result changed.
The same64 labels are shared by both base and existing visual expert; do NOT
count them as128 labels. Expert scores remain the archived anchor_visual64,
not refit or selected between receiver variants. No new feature/kernel/gate
training, tuning grid, support draw or class filtering beyond existing guards.

## Seven arms

original_base, base_bias, base_visual (no expert handoffs);
bias_handoff, visual_handoff (each adapted base plus the unchanged expert);
frozen_handoff and advantage_reference (unchanged old5% whole policies).

For the two new handoff arms, apply the SAME frozen4D source gate parameters
to the actual adapted-base and receiving-expert margins/entropies. Positive
score, base!=expert, support-observed expert class, floor(.05*N_eval) cap,
global-row tie break. Gate distribution compatibility remains unproved; no
newly calibrated-reliability claim is made. Retain the adapted bases alone so
their gains cannot be hidden inside an alleged expert-routing contribution.

Primary mean-only screen for useful updated workflow:
visual_handoff > base_visual AND visual_handoff > advantage_reference.
Separately test base_visual > original_base and base_bias, bias_handoff >
base_bias, and visual_handoff > bias_handoff. Report every negative contrast.
If adaptation helps but handoff fails, retain only baseline evidence and do
not claim the expert is necessary. Report BA, accuracy, admission/actual
handoff counts, changes from original base, class recalls and port/draw deltas.
Save all23port policies before new query-label readout. Port is the statistical
unit; five overlapping supports are not115 independent datasets.

## Limits

Residual adaptation is an existing principle, not new method novelty. All
targets are development-exposed, and support/query is row-only, not vessel
or product-disjoint. Source fitting preserves inherited purges. The pipeline
reads all expert scores before admission and cannot claim sparse expert
execution. Do not rewrite manuscripts or substitute this changed setting for
the old fixed-base results without an explicit methodological distinction.
