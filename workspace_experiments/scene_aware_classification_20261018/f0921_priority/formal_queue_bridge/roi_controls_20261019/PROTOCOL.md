# Matched ROI readout controls

Frozen 2026-10-19 before evaluating new arms. Reuse the parent 269-row seven-class
cohort, all seven date/budget tasks, five nested support draws, fixed queries,
source priors, frozen backbones, masks and final 50/50 probability fusion.
Do not tune weights or pick a budget using the new outcomes. Retain missing
target masks. The task remains held-date few-shot adaptation, not independent
confirmation or guaranteed different-acquisition query/support retrieval.

Image basis: the already reviewed row80/236 rescues show different target-local
and surrounding responses, but that does not establish the need for a 2x2
target layout rather than one ordinary target pooled vector. Test that narrower
question, without additional geometry, attention, backbones or detector calls.

Recover whole-target and whole-nontarget averages by area-weighted merging of
the cached prepool region means and masses. This is exactly the linear masked
pooling operation on the same layer1 map, up to cached float32 rounding. It is
NOT a new crop resized and independently encoded, nor rotated RoIAlign. The
nontarget region includes other candidates and remaining pixels, not pure sea.

All supplemental arms use K=.5*K_global+.25*K_grid2+.25*K_supplement.
Token descriptors use support-only centering, valid-block normalization and
per-polarization total normalization, followed by the existing mean VV/VH kernel.
There are no learned parameters in these readouts. One/two/12 token dimensions
differ; do not portray the simple arms as dimension-matched.

- roi_global: one whole-target mean per polarization, zero if target missing.
- fg_bg_global: target and nontarget means, equal normalized token treatment.
- roi_repeat_target: keep 12 postpool slots and their original availability;
  replace the four target slots with the same area-merged target mean, then
  center each slot on its actual valid supports. Other slots stay unchanged.
  This matches slot count, availability and context while removing within-target
  spatial content differences. Position-dependent availability/centering remains;
  do not call it perfectly position-free or a separate trained model.
- target_only: reproduce the parent's four prepool target slots + other
  postpool slots. This is the frozen current candidate, not retuned here.
- global_only, global_grid, postpool: replay the parent baselines.

Refit BOTH support-support and query-support pathways via existing residual
ridge. Replay every archived probability for the four existing arms. Test weighted
merging against direct masked pooling on an independent small nonuniform map;
test missing targets and support-only centering. These checks detect wrong
area weighting, invented missing-region vectors and query-dependent transforms.

Report equal-date/query-weighted OA/BA, Brier/NLL/MAP, paired draw differences,
rescue/harm, per-class/date and no-target conditions. Repeated draws reuse queries.
Use current vs ordinary ROI and current vs matched repeated-target controls for
native VV/VH + actual-support review, including persistent correct/wrong and
prior row80/236 cases. The parent daily-mosaic pixel-provenance caveat remains.
Archive bounded code/results/review panels, no inputs/features/weights.

Posthoc diagnostic added after seeing the first paired results: fg_bg_global
reduces the target token's normalized share versus roi_global. dose_check.py
retains its exact foreground kernel and removes only the background kernel;
it writes separate dose_* outputs and replays fg_bg_global. This distinguishes
background-term contribution from foreground attenuation alone. It does not
identify physical background semantics or select a final model on held labels.
