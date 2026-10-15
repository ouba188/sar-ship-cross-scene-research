# Frozen cross-dataset transfer, 2026-10-07

## Question and fixed comparison

Does the current same-64 adapted-base + 5% expert-handoff method transfer from
the existing 24-port source to public OpenSARShip2? Exactly three policies:
original source base, base_visual64, visual_handoff5. No new module or target
hyperparameter search. Existing main results remain unchanged.

## Data and labels

- Source: eight_class_adaptive_20260916/manifest.csv, 38,091 rows, eight classes.
- Target: public OpenSARShip2 native manifest, 9,120 rows. Explicit fine-label
  bridge retains 3,745 rows across all eight source classes. Generic cargo or
  tanker labels are not converted into unsupported fine classes. No quality,
  image size, interference, or SAR product-type filtering.
- Remove source MMSI/product matches against ALL native target rows before
  fitting source PCA, classifiers, or the source OOF gate. Existing encoders
  are frozen, not retrained under this purge.
- Draw 64 target rows uniformly without replacement, seeds 0..4. Both branches
  receive the SAME 64 labels. Evaluate the remaining 3,681 rows. This follows
  the main row-level protocol, not vessel-disjoint target support/query splits.
- Class ontology is defined for the cohort, but fitting receives only support
  labels. Query labels are used for readout after all five policies are saved.

## Actual representation and method

The base is NOT VV-only. It uses a two-input-channel VV/VH ImageNet ResNet50 V2,
with 2x2 spatial pooling then mean pooling, producing 2,048 coordinates.
The extra expert representation is VH RGB through the frozen domain-SSL
ResNet18 e206 checkpoint, producing 512 coordinates. The SSL pretraining pool
was E:/F_0921_pool_v1; this is not a new independent encoder.

Reuse original base input normalization and bfloat16/spatial-float16 pipeline;
SSL uses Resize(128), ToTensor and float32 without ImageNet normalization.
Non-square SSL inputs are grouped by transformed shape, preserving aspect ratio.
Eight fixed source rows test that checkpoint/preprocessing are compatible with
the existing feature caches before target extraction. Save actual numerical
differences; do not claim bitwise equivalence across hardware/library paths.

Fit source PCA128/PCA64 and balanced ridge(alpha=1). Build source leave-one-port-
out scores with the existing vessel/scene purge inside each fold. Fit the existing
four-feature logistic gate (C=1), equal-port/equal-within-port-class weighting.
Adapt both branches through the existing source-normalized residual ridge,
strength=8, using only the same 64 support labels. Freeze the source gate.

Admit only non-support rows where adapted branches disagree, the expert's
predicted class is observed in support, and the gate score is positive. Rank
by the fixed score, break ties by manifest row, cap at floor(0.05*N_eval)=184.
The budget is expert admission, not feature-extraction cost or the proportion
of changes relative to the original unadapted base.

## Readout and interpretation

Report BA, accuracy, class recall, admission count, paired handoff increment,
and all five support draws. Five overlapping draws are not five independent
datasets; do not manufacture a dataset-level confidence interval from them.
Public OpenSARShip2 has previous development experiments, including e181 and
external_os2_fusar. This run is a new frozen cross-dataset transfer protocol,
NOT untouched external confirmation. Do not tune against its query results.

## Existing feature cache decision

E:/Hermes/paper_tracking/external_os2_fusar/feats_os2.npz contains R50(5006,2048)
and SSL(5006,512). Its producer uses a standard three-channel R50 on VV RGB,
not the current paired two-channel R50. Its cohort is interference0 GRDH with
three coarse labels, and its SSL images are forced square before Resize.
The cache is preserved, but is not a drop-in replacement for this protocol.
