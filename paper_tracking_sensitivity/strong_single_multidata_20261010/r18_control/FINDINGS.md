# Trained R18 Direct Readout Control

Completed prediction tasks: 25/25; coverage: {'f0921': 1, 'oss2': 1, 'self24ports': 23}.

Fixed CPU protocol: unit-feature sqrt32 + constant for each single; concat of two unit blocks times sqrt16 + one constant;
three separately fitted balanced source ridge heads (alpha1, unpenalized bias), balanced64-support residual lambda8.
Fourth arm is exactly the .5/.5 mixture of the two adapted single softmax outputs, without fitting a fusion head.
No target choice of arm, weights, epoch or hyperparameters. Source fitting uses only legal source rows.

| Dataset | Fixed configuration | OA % | BA % |
| --- | --- | ---: | ---: |
| f0921 | concat_shared_head | 31.72129 | 25.27965 |
| f0921 | current_dual_reference | 35.05335 | 27.72335 |
| f0921 | equal_single_softmax | 31.53786 | 24.98280 |
| f0921 | r18_task_trained | 28.21130 | 23.42676 |
| f0921 | r50_frozen_c4 | 28.05801 | 22.78462 |
| oss2 | concat_shared_head | 38.73947 | 30.91920 |
| oss2 | current_dual_reference | 42.59712 | 34.67442 |
| oss2 | equal_single_softmax | 39.09264 | 31.60881 |
| oss2 | r18_task_trained | 36.07172 | 29.67983 |
| oss2 | r50_frozen_c4 | 34.59386 | 29.40225 |
| self24ports | concat_shared_head | 64.21081 | 54.97794 |
| self24ports | current_dual_reference | 65.57430 | 55.98728 |
| self24ports | equal_single_softmax | 64.43951 | 55.32786 |
| self24ports | r18_task_trained | 60.59915 | 53.78602 |
| self24ports | r50_frozen_c4 | 59.98567 | 50.75928 |

## Paired Error Flow

Positive deltas favor candidate. Counts are overlapping query/draw exposures, not independent samples.

| Dataset | Candidate | Reference | OA pp | BA pp | Rescue | Harm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| f0921 | concat_shared_head | current_dual_reference | -3.33206 | -2.44371 | 61895 | 82458 |
| f0921 | concat_shared_head | r18_task_trained | +3.50999 | +1.85288 | 77590 | 55929 |
| f0921 | concat_shared_head | r50_frozen_c4 | +3.66328 | +2.49502 | 69315 | 46708 |
| f0921 | equal_single_softmax | current_dual_reference | -3.51550 | -2.74055 | 56909 | 78604 |
| f0921 | equal_single_softmax | r18_task_trained | +3.32655 | +1.55604 | 64625 | 44096 |
| f0921 | equal_single_softmax | r50_frozen_c4 | +3.47985 | +2.19818 | 61213 | 39738 |
| f0921 | r18_task_trained | current_dual_reference | -6.84205 | -4.29659 | 61816 | 104040 |
| f0921 | r18_task_trained | r50_frozen_c4 | +0.15329 | +0.64214 | 98281 | 97335 |
| f0921 | r50_frozen_c4 | current_dual_reference | -6.99534 | -4.93873 | 58600 | 101770 |
| oss2 | concat_shared_head | current_dual_reference | -3.85765 | -3.75522 | 1617 | 2327 |
| oss2 | concat_shared_head | r18_task_trained | +2.66775 | +1.23937 | 1903 | 1412 |
| oss2 | concat_shared_head | r50_frozen_c4 | +4.14561 | +1.51695 | 2602 | 1839 |
| oss2 | equal_single_softmax | current_dual_reference | -3.50448 | -3.06561 | 1509 | 2154 |
| oss2 | equal_single_softmax | r18_task_trained | +3.02092 | +1.92898 | 1615 | 1059 |
| oss2 | equal_single_softmax | r50_frozen_c4 | +4.49878 | +2.20656 | 2357 | 1529 |
| oss2 | r18_task_trained | current_dual_reference | -6.52540 | -4.99459 | 1617 | 2818 |
| oss2 | r18_task_trained | r50_frozen_c4 | +1.47786 | +0.27758 | 3182 | 2910 |
| oss2 | r50_frozen_c4 | current_dual_reference | -8.00326 | -5.27217 | 1826 | 3299 |
| self24ports | concat_shared_head | current_dual_reference | -1.36349 | -1.00934 | 10620 | 12574 |
| self24ports | concat_shared_head | r18_task_trained | +3.61166 | +1.19192 | 15678 | 9630 |
| self24ports | concat_shared_head | r50_frozen_c4 | +4.22514 | +4.21865 | 15444 | 8895 |
| self24ports | equal_single_softmax | current_dual_reference | -1.13479 | -0.65943 | 9813 | 11543 |
| self24ports | equal_single_softmax | r18_task_trained | +3.84036 | +1.54184 | 13972 | 7700 |
| self24ports | equal_single_softmax | r50_frozen_c4 | +4.45384 | +4.56857 | 14841 | 8068 |
| self24ports | r18_task_trained | current_dual_reference | -4.97515 | -2.20126 | 10866 | 18868 |
| self24ports | r18_task_trained | r50_frozen_c4 | +0.61348 | +3.02674 | 21372 | 20871 |
| self24ports | r50_frozen_c4 | current_dual_reference | -5.58863 | -5.22800 | 11280 | 19783 |

## Port Direction

| Candidate | Reference | LOPO BA positive / zero / negative |
| --- | --- | --- |
| concat_shared_head | current_dual_reference | 7 / 0 / 16 |
| concat_shared_head | r18_task_trained | 16 / 0 / 7 |
| concat_shared_head | r50_frozen_c4 | 23 / 0 / 0 |
| equal_single_softmax | current_dual_reference | 12 / 0 / 11 |
| equal_single_softmax | r18_task_trained | 21 / 0 / 2 |
| equal_single_softmax | r50_frozen_c4 | 23 / 0 / 0 |
| r18_task_trained | current_dual_reference | 5 / 0 / 18 |
| r18_task_trained | r50_frozen_c4 | 19 / 0 / 4 |
| r50_frozen_c4 | current_dual_reference | 0 / 0 / 23 |

## Interpretation Boundaries

- This fills the direct trained-R18-alone readout gap. R18 was trained in the original dual-task context, not optimized independently as the best single backbone.
- The R18 outer-task representation has seen inner-source labels. No new independent inner validation or hyperparameter selection is claimed.
- R50 and R18 differ in pretraining, input and task training; these effects are not isolated. F has historical unlabeled SSL exposure.
- The main agent audited code and weights in this round: current R50 and the historical extract_full_six_spatial_features.py use meanRGB repeat2*1.5 conv1 weights, identical for VV and VH. The task-R50 prefix is frozen.
  Audit artifact: E:/Hermes/paper_tracking/sensitivity/strong_single_polarization_20261010/OBSERVABILITY.json. The float64 audit records old antisymmetric response max 1.0246394606700237e-14, new antisymmetric response RMS 1.7404581770394036, common-mode response difference max 6.375879291198316e-7, and unchanged total parameters 23521288. These supersede the earlier float32 exploratory response of 3.6e-7.
  At the encoder input this collapses the two channels to their sum, whereas R18 receives VH. This establishes an observability limitation, not its performance contribution or a requirement for two backbones; input-information complementarity remains a candidate explanation.
  The main-line difference-preserving single-input control uses wVV=wsum+.5*(wR-wG), wVH=wsum-.5*(wR-wG), with unchanged parameter/forward counts and source/target re-extraction plus matched readout refitting. Its performance result is pending; no such experiment was added here and original readout results are unchanged.
- The two new combined arms isolate fixed readout choices on these cached feature functions. Current full dual is a reference with additional modules and a different readout, not an architecture-only intervention.
- This R50 arm reuses historical pooled_c4.npy. The task-R50 frozen control re-extracts pixels through the current bf16 network.
  A prior source-initialization probe measured about .004 relative geometry difference; this is not a full-cohort estimate or a proven rounding-only cause.
  These cached readout arms are internally matched, but are not the identical task-script frozen control. Predictions are never forced to match its reported numbers.
- The main line fixed the measured-single common-hard set to include R18-only before reading the final metrics. Only r18_task_trained is eligible; concat_shared_head and equal_single_softmax both use two backbones and must not enter that single set.
- Five overlapping support draws are not five neural seeds; current R18 weights and full-dual reference are each fixed per task.
- Both benefit and harm are retained. No failed arm is removed or target-best configuration promoted.

- Increased concatenated dimension is not itself evidence of complementarity; the B=A duplicate-branch identity is checked on synthetic data.

## Numerical And Artifact Contract

- Float32 L2 normalization matches the original geometry normalization. Subsequent scale arithmetic, source statistics and support kernels are float64.
- In unit coordinates, source alpha1 on scaled features equals penalties 1/32 for singles or 1/16 for concatenation, with zero penalty on source bias.
- Support kernels are 32 Ur Us^T+1, 32 Vr Vs^T+1, and 16(Ur Us^T+Vr Vs^T)+1. The support intercept is regularized as in the original residual operator.
- Source sufficient statistics and target scores are streamed in blocks of512. No full F feature matrix is copied to float64.
- targets/{dataset}/{task}/predictions.npz stores modes, predictions, margins, source_rows, rows, ids, support, support_labels, eval_masks and fixed current_dual_predictions.
- scores.npy is mmap-able float64 [5,4,N,8]: first three modes logits, fourth probabilities. priors.npy is [3,N,8].
- source_heads.npz stores coefficients in unit coordinates, including the final bias row; support_coefficients.npz stores all fixed residual coefficients.
- class_recall.csv and class_rescue_harm.csv retain task/draw/class outcomes and support coverage. Counts and macro BA contributions are not interchangeable.
- SELF_TEST.json, SHAPE_CHECKS.json, PROTOCOL.json and per-task INPUTS/PREDICTED receipts document verification and provenance.
