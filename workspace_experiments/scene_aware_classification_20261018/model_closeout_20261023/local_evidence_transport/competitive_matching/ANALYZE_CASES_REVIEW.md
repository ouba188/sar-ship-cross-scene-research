# Case Analysis Preparation Review

Status: prepared only. AST syntax parsing passed. No analysis execution, raw-image
rendering, training, solver replay, label changes, or upload was performed.
The expected metrics/prediction/score outputs were absent at preparation time.
Only analyze_cases.py and this review were added; run.py and solver.py were not edited.

## Input Contract

- Default inputs: competitive_matching/source_metrics.csv,
  competitive_matching/predictions_17.npz, and competitive_matching/scores.npz.
  If predictions are saved in results/, pass --predictions results/predictions_17.npz
  from this directory. --scores and --metrics also accept explicit paths.
- Predictions: separate full, fixed, uncoupled, coupled keys, each (3387, 8).
- Local score cache: separate fixed, uncoupled, coupled keys, each (1741, 8, 2, 3).
  full is used as the final classification baseline, not presumed to have local terms.
- idx maps 1741 cache rows to full-table positional indices. refs contains full-table
  positional indices, shape (1741, 8, 2); it is never interpreted as cache positions.
- coupled_weights: (1741, 8, 2, 3, 8); baseline_mass: (1741, 2, 3, 8).
  Region order is body, side, end. End has four real slots and four zero padding slots.
- Metadata: ../regional_utility_pilot/upstream/table.csv relative to P.
  Existing P/tokens.npz supplies identity alignment, geometry, validity and token mass.
- Imports geo.native and slots from the existing extract_tokens module, following
  semantic_projection/analysis/diagnose_source.py and relational_kernel/analyze_cases.py.
  This imports existing dependencies but does not invoke feature extraction or a model.

## Selection And Evidence

Selection is confined to role=evaluation (485 rows). For rescued, harmed, CC and WW,
choose the minimum numeric obs_id independently. An empty category is recorded with
zero cases, never filled using another split or a favorable substitute.

The shown rival is the fixed actual error for rescued cases and the coupled actual
error for harmed/WW cases. For CC it is coupled's highest-scoring nontrue class,
explicitly labeled as a competitor rather than an erroneous prediction.
The same shown rival is used for every arm's comparison margin; each arm's own
highest-scoring nontrue class and margin are recorded separately.

Each case displays VV and VH separately, with columns for the query, both actual
true-class references, and both actual rival-class references. No reference reselection
or substitution occurs. Native160 crop/pad and fixed 0..255 display follow the existing
pipeline. Region contours come from actual footprint masks and detection metadata.
Invalid cached local inputs remain visible and explicitly marked invalid.

Weight figures show baseline query mass and candidate-specific coupled query weights
for true/rival classes. End padding is excluded. They are not transport plans and do
not imply per-reference contributions or additive final-logit attribution. No automatic
physical interpretation is attached to body/side/end or to changed query weights.

## Validation And Remaining Limits

Static review covered evaluation-only selection, full/cache index separation, actual
two-reference lookup, fixed-rival comparisons, polarization axes, and end padding.
The only executed check was AST parsing, which detects syntax errors without executing
the analysis or importing its image/model dependencies.

When later run, explicit checks stop on incompatible array shapes, missing evaluation
indices, nonfinite evaluation/case evidence, nonzero end padding, token/table identity
mismatch, incorrect reference class/bank role, or rendered geometry/mass inconsistent
with the cached tokens. These failures would invalidate case identity or visual evidence;
the input contract or mask reconstruction must then be corrected before interpretation.
Missing required result files cause an error before the analysis output directory is made.

Real-cache compatibility and final figure layout are not runtime-verified yet. The
source_metrics CSV is read for provenance; metrics are not independently recomputed.
No head or solver replay is claimed. All eight cached local candidate scores and query
weights are exported for each selected case. Actual visual facts and mechanistic
interpretations remain for the subsequent raw-chip review.

## Outputs On Future Execution

analysis/case_selection.csv, case_margins.csv, local_scores.csv, query_weights.csv,
image_records.json, analysis_receipt.json, CASE_REVIEW.md, plus one 2x5 image panel
and one 2x3 query-weight panel per available outcome (at most four cases).
Image records preserve observation/instance/vessel IDs, class, role, port, acquisition,
original image path and dimensions, crop/pad placement, geometry and slot masses.

No generated evidence outputs exist as part of this preparation-only delivery.
