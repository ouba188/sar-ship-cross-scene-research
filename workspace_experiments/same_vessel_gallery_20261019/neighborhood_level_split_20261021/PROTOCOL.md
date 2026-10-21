# Neighborhood level/content split: frozen-baseline pilot

Status: protocol fixed before training on 2026-10-21.

## Question

Does separating neighborhood display level from its local variation improve the
existing native five-point classifier? This is an input/readout experiment, not
a new convolution, physical backscatter recovery, or full support-adaptation run.

## Data and baseline

- Reuse all three identity/date-disjoint splits of the native experiments.
- Exclude observation 31237; no new identity or label corrections.
- Freeze the complete `sampling_preservation_20261020/mse_aligned/keep_refs`
  predictor for each of three training seeds. Replay against stored predictions.
- Native 160px crop/pad; preserve target pixels, never resize. VV/VH are separate.
- Supplemental neighborhood R is the native valid footprint minus target OBB
  and other accepted detection OBBs. It is not guaranteed pure water.
- Supplemental views require a target mask and nonempty R. Disable conflicting
  VH exactly as the preceding input-control experiment. Missing views are zero
  features plus a shared availability indicator; baseline predictions remain.

## Four arms

A: frozen original predictor, no extra trainable channel.
B: A + raw-R descriptor.
C: A + mean-centered-R descriptor.
D: A + mean-centered-R descriptor + two separate VV/VH R means.

Let x be stored PNG values divided by 255 and mu the mean inside R. Both B and
C/D use the SAME fixed signed-compatible affine encoding: outside R is 0.5;
inside R is 0.5+x/2 for B, 0.5+(x-mu)/2 for C/D. No clipping or per-image contrast
normalization. Thus C removes additive R level, not local contrast amplitude.
This encoding changes the supplemental input relative to a conventional PNG
reader, but B/C are matched; the old predictor is unchanged.

Read out frozen R50 stem features with masked mean/max pooling in R, averaged
over four right-angle rotations. 128 values per polarization. C and D share the
identical descriptor cache. Four extra slots are two means and two availability
flags; B/C mean slots are zero. All B/C/D heads are 260->32 GELU->8 (8616 params),
zero-initialized last layer, same batches/seeds/optimizer. There are no class,
port, date, angle, or target identity inputs. Mask geometry is still observable.

Only source-training data determine normalization. Three seeds x three folds x
three supplemental arms =27 fits. AdamW lr=.001, weight_decay=.001, batch128,
150 updates; weighted MSE to +/-1 class targets, reusing class/ship weighting.
Source validation selects step 0/50/100/150 by BA then Brier. Step 0 is the exact
original model. Report terminal train/validation as well as selected test scores.
No test-derived adjustments or parameter searches this round.

## Evidence

- Primary: BA, macro-F1, class recalls; OA, NLL/Brier and paired net flips also.
- Report each seed and fold, not nine independent test datasets. Ship bootstrap
  on seed-averaged correctness gives conditional intervals, not retraining CIs.
- D fixed-head source-mean replacement tests use of its explicit mean slots;
  they do not prove physical causality or full-model brightness invariance.
- Source-constant residual distinguishes image-specific corrections from class
  bias. It is a diagnostic, not a newly selected deployment arm.
- Checks: old-model replay, native target preservation, R-only input isolation,
  centered additive-shift invariance, C4 consistency, C/D shared cache,
  chosen-head score replay, source-only split membership.
- Inspect original VV/VH: requested examples 4002/22543 plus current strong-
  baseline rescue, harm, persistent correct and persistent wrong. Previous
  diagnostic-head rescue/harm labels must NOT be reused for this model.

The corpus has already been used for development. This is broader matched
development evidence, not independent confirmation or proof of general benefit.
