# Source-only response diagnostic

Status: eight full/fold bundles evaluated; 80 response-only LOPO ridge fits
completed. No neural fit or target readout. See FINDINGS.md for actual results.

## Prior findings that affect interpretation

The old incidence_factor_chain estimator used ship weights, no context, and old
raw512/stable128/condition32 spaces. Its scores are background, not numerical
baselines for the new whitened32 predictor with class/ship/pair weights.

The old sourcefit pair table has 384 unordered pairs / 88 ships / 8 classes;
heldraw has 189 / 43 / 8. At absolute angle gap >=2 degrees, these become
59 / 30 / 7 and 53 / 18 / 6. Reverse directions double pair rows, not evidence.
The sourcefit offshore class has no >=2-degree pairs. Heldraw class IDs 4 and 5
each have only one paired ship and no >=2-degree pair.

In heldraw, both products are seen in the original factor-fit products for
149 pairs / 40 ships / 8 classes; intersecting >=2 degrees leaves 37 / 16 / 6.
This distinguishes vessel transfer within familiar scenes from scene transfer.
Fold scene flags must use that fold's fitting products, not full-fit products.

Old tables contain relative-angle measurements for only 19 fit and 10 held
pairs. No held pair meets the old <=10-degree relative-angle criterion.
Do not introduce a strict-angle matched result with no usable observations.
Median pair time separation is about 72 days in fit and held. Measured time
gap and absolute incidence gap have unweighted pair Spearman correlations
0.1082 and 0.2648 respectively, descriptive only. Port/platform/pass matching
does not control intervening motion, background, time, or loading changes.

## Concrete diagnostic blocks

1. Compare actual raw feature delta with each main-core prediction and zero.
   Evaluate sourcefit and identity-held raw pairs separately, for full and folds.
2. Use class -> ship -> pair weights on each slice. Retain directed/undirected
   counts, unique observations, ships, classes and ports alongside each score.
3. Report all, >=2-degree, <2-degree, both-scene-seen, and seen-plus->=2-degree
   subsets. Report per-class and per-port rows using identical arm masks.
4. Split signed directional projection and prediction amplitude from total
   squared error. Keep the held-data best amplitude ratio descriptive only.
5. Contrast conditional against shared, no-condition and within-cell shuffled
   arms. A no-condition model can predict contextual mean changes; it is not
   equivalent to zero. Evaluate the shuffled-fit arm using real held angles.
6. Leave-held-port-out response estimation calls the main core on sourcefit
   rows excluding the held port, then predicts that held port's raw pairs.
   Reuse frozen base/context; label this response-only transfer, NOT base LOPO.

No bootstrap, coefficient search, extra neural training, or target query access.

## Integration boundary

`diagnose_predictions(table, actual, predictions, cohort=..., model=..., seed=...)`
accepts an already aligned pair table and delta arrays, returns a DataFrame.
Required table columns: a, b, mmsi, class_id, port, delta_angle, both_scenes_seen.
`predictions` is an arm-name -> (n_pairs, 32) array mapping. Each prediction
must come from the main core using only anchor context and real evaluation
angles. The diagnostic module does not infer saved-array keys or refit logic.

`write_reports(results, port_transfer=None, reference=...)` writes only this
mechanism directory. The runner now loads confirmed source exports and uses
the main core's matched_pairs, fit_response, predict_delta and shuffled_angles.
Run with the existing ClearSAR Python: diagnose_incidence_transport.py.
Input exports are immediately subset to original source fit/held IDs; prior,
kernel, and target prediction arrays are not loaded or evaluated.

LOPO results are also pooled from their cached per-port predictions and
reweighted class -> ship -> pair across the full held cohort. They are not
an unweighted average of per-port metrics. The endpoint angle-range check
uses paired sourcefit endpoints, not the deployment gate's all-fit range.

The angle_sham rows in pair_prediction_metrics.csv evaluate real angles,
including sourcefit rows. Actual shuffled-angle training-objective metrics
are separately saved in sham_training_objective.csv.

## Verification

ClearSAR Python analytic checks passed for class/ship weighting, directed vs
undirected counts, perfect/zero/oversized/reversed predictions, parallel plus
orthogonal error accounting, duplication of all pair rows of one ship, matched
arm comparisons, subgroup alignment, and empty slices. All 128 comparable
source fit/held, all/gap2 arm metrics match the main response_results.csv;
maximum explained-percent difference is zero. Summed feature error divided
by 32 matches main MSE. Full legacy unordered pair cohorts match exactly.
