# Deployment-aligned residual expert training

Directory date follows the machine's execution timestamp, 2026-10-08.
This continues neural_complement_pilot_20261007 without overwriting it.

## Decision

Adopt residual_global as a usable BA-oriented development candidate on
corrected F0921. BA rises from frozen replay25.106684% to25.435555%
(+0.328871pp,4/5support draws positive), while OA falls from29.634353%
to29.326474%(-0.307879pp,all5draws negative). The user explicitly accepts
primary-metric improvements with honestly reported tradeoffs.

The previous utility-tail candidate remains recorded: BA25.319728%,
OA29.565971%. New aligned-global is +0.115827ppBA/-0.239498ppOA relative
to that candidate. This is a useful BA-focused configuration, not uniform
dominance or evidence of top-journal novelty by itself.

Do not enable the additional handoff CE term by default. In the matched
comparison it lowers BA0.067969pp and OA0.114401pp relative to aligned-global.
It remains a completed negative branch, not an extra positive paper module.

## The Head Mismatch Fix

Old pilot: train a neural classification/utility head, discard it, refit
PCA/ridge on the resulting features, then deploy the adapted ridge expert.

New training forward pass:

1. Sample one source-port episode with its existing64support rows and a
   disjoint query minibatch. Its fixed prior head was fitted on other source
   ports, with the existing MMSI/product exclusions.
2. Encode support AND query through the frozen e206 prefix and trainable R18
   layer4, retaining the original VH RGB/255 recipe and frozen BN statistics.
3. Apply the frozen source PCA64, concatenate standardized R50 PCA128,
   standardize with source geometry, and apply the fixed source ridge prior.
4. Construct the same193coordinate residual feature, including its bias.
   Solve the support-balanced64x64 dual ridge with strength8 and +/-1 targets.
5. Evaluate the adapted expert on source query rows. Backpropagate class-
   balanced CE through query features, support features, support prior scores,
   and the ridge solution into layer4. No auxiliary classification/utility
   head is called or optimized.

Deployment uses the same operator and the PRE-TRAINING global source PCA,
normalization, ridge prior, and gate. There is NO post-training head/PCA/gate
refit. Source episodic folds and the global deployment model naturally have
different source-fitted coefficients; both use the same retained-head recipe.
The base R50 branch is unchanged. Its64label adaptation and final5%handoff
remain the original deployment rules.

The prior issue of optimizing a discarded neural classifier is removed.
This does not claim literal differentiable optimization of hard top-k accuracy:
CE is a smooth surrogate, admission indices are detached, and the training
5%ranking population is each minibatch, not the entire external target pool.

## Fixed Run Settings

- Same123489target rows,64global target labels,123425query rows,6171admissions
  in each of five unchanged draws. Adapted-base predictions match exactly.
- Source episodes:20ports,28402query rows, same64supports per eligible port
  as the previous pilot. Three tiny source ports have no64+query episode;
  their data remain in applicable source prior fits.
- Initialize from original e206, NOT the previous trained candidate. Six
  epochs,128query rows plus64support rows per batch, AdamW lr1e-5,
  weight_decay1e-4,clip5,seed20261008,noaugmentation,finalepochonly.
- All20source episode geometries and the global deployment model are frozen
  before encoder training. The global gate is NOT retrained on new scores.
- Two matched arms share initialization, episode/query orders and optimizer.
  residual_global uses adapted-query CE. residual_handoff adds CE on actual
  admissions under the fixed gate, disagreement/observed-class guards and
  floor5%cap. No admissions means zero added loss, not forced routing.
- Both trained arms and all five policies finish before target readout.
- Relative to the previous pilot, LR, episodic batching and retained head
  policy differ. Do not attribute the full cross-run BA difference solely
  to one isolated change. No additional ablation grid was run.

## Full Model Results

Values are percentages; deltas are percentage points. All5draws share one
historically used target dataset and one neural training seed.

| Configuration | OA | BA | BA gain over Base64 |
| --- | ---: | ---: | ---: |
| Unchanged Base64 | 28.937087 | 24.901788 | 0 |
| Frozen replay Full5 | 29.634353 | 25.106684 | +0.204897 |
| Previous utility-tail Full5 | 29.565971 | 25.319728 | +0.417940 |
| Aligned-global Full5 | 29.326474 | 25.435555 | +0.533768 |
| Aligned-handoff Full5 | 29.212072 | 25.367586 | +0.465798 |

Both aligned arms improve BA over Base64 in5/5draws and over frozen replay
in4/5draws. Do not conflate these two reference comparisons. Neither aligned
arm improves OA over frozen replay in any draw, although both improve OA
over Base64 on average.

## Fixed-Prediction Diagnosis

If the aligned-global expert were used on every query, OA would be29.802714%
and BA27.036078%, versus frozen expert31.055945%/25.982723%. This exceeds
the5%deployment cap and is diagnostic only. It suggests a class-balanced
recognition benefit with an ordinary-accuracy tradeoff, not a uniformly
stronger expert or a claim that routing can resolve all its errors.

Actual aligned-global handoff averages1685.2recoveries and1204.6harms,
versus frozen1798.2/937.6. Weighted recovery/harm are1.373489/0.839722ppBA,
versus1.320424/1.115527. BA improves despite fewer unweighted net recoveries.

## Verification And Scope

- 22focused CPU/CUDA tests passed: NumPy deployment-solve equivalence,
  imbalanced/absent support classes, float precision, fixed projection, and
  finite-difference gradients through support features and priors.
- Real source64support/32query parity against the unchanged NumPy solver:
  max absolute score difference6.8709578e-7. Both training runs and evaluation
  completed successfully. Selected recovery/harm counts reconcile with OA.
- ClearSAR sklearn1.8 performs fitting and final deployment evaluation.
  GPU env sklearn1.7.2 emitted pickle version warnings when loading the saved
  container solely to obtain the gate's NumPy arrays; no sklearn estimator
  prediction or fitting is called in GPU training. Geometry uses saved arrays.
- Historical/transductive F0921 exposure and row-level support/query remain.
  Encoder learns source episode labels; no encoder-level OOF claim. No target
  labels train the encoder, and no extra target annotation budget is spent.
- 5%means expert admissions, not feature-extraction or training compute.
- Fixed-head drift and source-to-target differences remain possible; solving
  the discarded-head mismatch does not establish universal transfer success.

## Files

Implementation under E:/Hermes/paper_tracking/sensitivity/oof_reliability_check:
prepare_aligned_residual.py, aligned_residual_ops.py, train_aligned_residual.py,
evaluate_aligned_residual.py, test_aligned_residual_ops.py. Fixed diagnostics
reuse read_neural_complement.py. Preparation/evaluation use ClearSAR; training
uses the existing CUDA environment. Pass --out with a new run directory.

Saved protocol, source episode arrays, retained source model, layer4 checkpoints,
training histories, full features, policies, summaries and diagnostics make
the observed positive and negative branches available for later use.
