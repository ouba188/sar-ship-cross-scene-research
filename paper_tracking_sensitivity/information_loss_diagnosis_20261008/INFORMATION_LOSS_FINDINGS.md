# Where Cross-Domain Discrimination Is Lost

Date: 2026-10-08. Completed development diagnosis; no production model changed.

## Decision

Include all four requested datasets: 24 ports, corrected F0921, public
OpenSARShip 1.0 and OpenSARShip 2.0. The evidence does NOT yet identify
foreground, physical ship scale, or attention as the main missing mechanism.
It identifies a demonstrated input-construction loss in F0921 and a useful
raw-feature/readout direction across the external datasets. Investigate that
direction before another backbone or spatial-attention module.

The next bounded method experiment should preserve information outside the
existing source PCA and test whether the SAME 64 support labels can use it.
Keep the existing base, source prior and 5% handoff protocol as the reference.
Start with a matched raw-feature expert comparison, not a newly named module.
Only develop a support-conditioned residual representation if that experiment
shows usable expert information. This is a research decision, not a demonstrated
64-label method improvement.

## Completed Protocol

- Frozen current paired VV/VH R50 (2048D) and VH SSL R18 (512D); no GPU training.
- Fixed balanced +/-1 ridge, alpha=8, train-only standardization, explicit
  penalized bias and division by sqrt(feature dimension). No parameter search.
- One prespecified seed, 20261008. Nested 64/512 labeled training rows; also
  4096 for the two 24-port tasks and F0921. Identical query rows across views.
- Within-dataset splits hold out 25% of recorded vessel groups. An additional
  24-port task holds out six entire ports and purges their vessels/products.
- Source PCA is fitted only on the 24-port diagnostic training pool. External
  tasks retain the previously fitted source PCA. Encoders are historical.
- Probes fit a new within-dataset classifier, NOT the production source-prior
  residual adaptation rule. Labels above 64 are diagnostic supervision only.
- Total feature-probe fitting time recorded by the script: 25.594 seconds.

Population: 24 ports 38,091 rows; F0921 123,489; OSS1 1,349; OSS2 3,745.
Public datasets use the existing explicit eight-class fine-compatible cohorts,
not their full native taxonomy. OSS1 is the current GRDH subset.

## Matched Feature Readability

All values below are percentages. PCA joint = R50 128 + SSL 64 dimensions;
raw joint = R50 2048 + SSL 512 dimensions. Each table row compares identical
training/query samples and the same fixed classifier recipe.

| Diagnostic task | Labels | Joint PCA OA / BA | Joint raw OA / BA |
| --- | ---: | ---: | ---: |
| 24-port within-dataset | 512 | 46.522 / 40.805 | 47.193 / 38.959 |
| 24-port six held-out ports | 512 | 44.816 / 37.694 | 44.758 / 37.772 |
| F0921 | 512 | 33.630 / 35.400 | 35.602 / 36.296 |
| Public OpenSARShip 1.0 | 512 | 54.094 / 41.220 | 54.678 / 42.573 |
| Public OpenSARShip 2.0 | 512 | 39.695 / 35.569 | 45.279 / 37.106 |
| 24-port within-dataset | 4096 | 52.629 / 49.129 | 53.027 / 47.559 |
| 24-port six held-out ports | 4096 | 53.311 / 46.194 | 52.790 / 43.784 |
| F0921 | 4096 | 41.041 / 44.144 | 45.430 / 47.547 |

Thus raw joint features are a promising external-data direction, NOT uniformly
better. Compression is beneficial regularization on some 24-port comparisons.
Raw versus PCA changes the function space, feature scaling and effective ridge
geometry; these experiments do not isolate irrecoverable information deletion.

At 64 labels, raw joint BA changes relative to PCA joint are approximately
+1.48 pp on F0921, +1.42 pp on OSS1, but -0.16 pp on OSS2. In particular, the
larger-label gains do not establish an across-dataset 64-label solution.

On F0921, fixed raw joint features yield OA/BA of 26.563/24.564 at 64 labels,
35.602/36.296 at 512, and 45.430/47.547 at 4096. Some class information is
readable without changing the encoder, and the small-support readout is a
material constraint. The curve does not measure a theoretical accuracy ceiling
or distinguish sample coverage from all other classifier-estimation effects.

## Demonstrated Input Loss In F0921

The historical and corrected pipelines have matched labels, support draws,
query masks, source head and handoff caps. The input intervention combines
association, centering, spatial extent and chipwise radiometric statistics.

| Fixed method arm | Old OA | Corrected OA | Delta OA pp | Delta BA pp |
| --- | ---: | ---: | ---: | ---: |
| Source-only base | 17.920356 | 24.207575 | +6.287219 | +1.467595 |
| Base adapted with 64 labels | 21.433259 | 28.937087 | +7.503828 | +4.459808 |
| Full 5% handoff | 21.834150 | 29.633543 | +7.799392 | +4.265153 |

The source-only fallback predictions change on ZERO rows. All source-only
gain is on the changed inputs, providing direct evidence of an input-pipeline
contribution. On changed rows alone, OA improves 10.140 pp but their own BA
falls 1.593 pp: the recovery is class-selective, not a universal correction.
For example, bulk and general cargo improve while crude-oil tanker recall
declines under the fixed source classifier.

Changed inputs contribute +8.105975 pp OA / +4.477808 pp fixed-eight BA to the
full model. Unchanged fallback inputs contribute -0.306583 / -0.212655 pp via
changed support adaptation and handoff competition, not via recovered pixels.
Source raw-to-8bit scaling remains unresolved. These results do not separately
identify foreground alignment, physical scale or radiometry as the cause.

## Geometry Confound: Do Not Promote The Apparent Large Gain

The five 24-port geometry fields are present for ALL 25,656 rows in classes
0/1/2 and missing for ALL 12,435 rows in classes 3/4/5/6/7. Availability therefore
encodes a class-family/data-construction shortcut. Appending these fields must
not be described as a clean scale/shape contribution.

| 24-port 4096-label probe | Within OA / BA | Held-port OA / BA |
| --- | ---: | ---: |
| Raw R50 | 49.487 / 44.483 | 50.261 / 43.669 |
| Raw R50 + missingness flags only | 65.975 / 55.857 | 67.252 / 54.346 |
| Raw R50 + geometry and flags | 68.207 / 57.051 | 72.871 / 55.940 |

Most of the apparent increment is reproducible without geometry values.
A separate complete-case comparison restricts BOTH training and query rows
to classes 0/1/2 with observed geometry; it is explicitly a different,
three-class task. Adding geometry changes BA 78.437 -> 78.784 within-dataset
and 81.904 -> 81.889 across held ports. This is not strong evidence for a
generalizable geometric mechanism, and cannot be compared to eight-class BA.

On F0921 at 4096 labels, raw R50 BA is 43.530; adding current detector/crop
geometry gives 43.551, only +0.021 pp. Adding privileged AIS ship length gives
45.506 (+1.976 pp), which supports an additional-metadata hypothesis, not a
claim that those meters are recoverable from normalized image pixels. F0921
and source geometry descriptors differ and are not a shared scale module.

## Remaining Classification Difficulty

Confusions remain even with more diagnostic supervision. At the largest
budget and raw joint features, F0921 recalls include fishing 72.43%, container
61.40%, general cargo 26.39%, chemical/product tanker 36.15% and bulk 38.85%.
OSS2 recalls include fishing 85.96%, general cargo 15.28%, crude tanker 23.53%
and tug 15.00%. This argues against treating SAR as uniformly uninformative;
the observed readout has class-specific limitations. It does not establish
whether the remaining errors arise from semantic labels, pixels or encoders.

## Limits And Files

One diagnostic split, historical development data, no independent confirmation
or significance claim. OSS1 query has only 2 crude tankers, 3 tugs and 4 offshore
ships; its BA can change materially with one prediction. Some 64-label fits
miss a class. Within-dataset vessel isolation is not scene isolation. There
are 198 source rows with missing vessel identity, treated as distinct unknown
groups, so true-identity isolation cannot be guaranteed for those rows.
The SSL encoder previously saw unlabeled F0921 data. Source PCA energy capture
is not the fraction of class information preserved. None of these tables
replace equal-port or global-64-label production results in the manuscript.

- `feature_probes/protocol.json`, `results.csv`, `summary.json`: full protocol,
  every view/budget, populations, per-class support/recall and runtime.
- `feature_probes/<task>/split.npz`, `predictions.npz`: exact training/query
  rows and predictions for reproducible matched comparisons.
- `input_strata/INPUT_FINDINGS.md`: input intervention decomposition and
  consistency checks. 51 recorded MMSIs have multiple labels (548 rows), not
  an estimated label-error rate.
- `geometry_followup/results.json`, `availability_by_class.csv`: missingness
  and complete-case decomposition, with matched row arrays in NPZ files.
- Scripts: `oof_reliability_check/diagnose_feature_readability.py`,
  `diagnose_f0921_input_strata.py`, `diagnose_geometry_confound.py`.

Verification performed: exact feature/manifest row alignment, disjoint recorded
vessel groups, matched historical support/query/label arrays, reproduction of
historical scores, and additive contribution reconciliation. No main-model
weights, images, manuscript result tables or production predictions changed.
