# Single R50 Functional Crossover

Completed fixed-lambda8 diagnosis. No selection or training uses this report.

## Results

| Prior / Kernel | OA mean | Neural-seed SD | BA mean | Neural-seed SD |
| --- | ---: | ---: | ---: | ---: |
| frozen_prior_frozen_kernel | 78.77902 | 0.00000 | 73.28351 | 0.00000 |
| trained_prior_frozen_kernel | 78.91086 | 0.63053 | 73.00380 | 0.76003 |
| frozen_prior_trained_kernel | 79.16022 | 0.08172 | 73.88790 | 0.05659 |
| trained_prior_trained_kernel | 79.99427 | 0.70115 | 74.52966 | 0.28697 |

Neural seeds: [20261010, 20261011, 20261012]. First average five support draws within each seed;
then compute mean and sample SD (ddof1) over neural-seed means.
Support draws are not independent neural training runs.

Existing matched dual reference, five-draw mean: OA 81.16939, BA 75.63792.
This is one existing trained reference, reused across comparisons, NOT three dual retrainings.

## Direction Across Seeds

- Trained full classifier versus frozen_single, OA: mean +1.21525pp; positive/zero/negative seed means: 3/0/0.
- Trained full classifier versus frozen_single, BA: mean +1.24615pp; positive/zero/negative seed means: 3/0/0.
- Trained full classifier versus current_matched_dual, OA: mean -1.17512pp; positive/zero/negative seed means: 0/0/3.
- Trained full classifier versus current_matched_dual, BA: mean -1.10826pp; positive/zero/negative seed means: 0/0/3.

## What The Crossover Means

- Frozen/trained prior means each source head evaluated on its OWN corresponding embedding.
  It is the complete prior function, not a head-weight-only change.
- Frozen/trained kernel means the corresponding feature geometry used to relate query and supports.
- All four cells re-solve the same balanced64-label residual system, with lambda8 unchanged.
  The two original cells exactly reproduce stored predictions using original float32 torch CPU operations.
- Prior-only and kernel-only deltas locate functional changes. Their interaction is reported;
  nonlinear accuracy effects need not add. Crossed cells are diagnostics, not target-selected deployment choices.
- The analysis does not identify distinct physical mechanisms, prove causality, or equate pretraining histories.
  The current dual comparison is a practical reference, not a matched architecture-only intervention.

## Error Accounting

`rescue_harm_by_seed_draw.csv` counts candidate-correct/reference-wrong rescues and the reverse harms.
OA difference and class-balanced net rescue differences are checked against exact classifier metrics.
`per_class_signs.csv` includes both references, class recalls, support coverage, net counts and their signs.
Counts are within-draw exposures; overlapping queries across five draws are not independent samples.
Use actual seed curves, CACHE_ALIGNMENT.json and prior/tail update norms to interpret a negative result;
the crossover alone does not prove a training bug or that a research direction is impossible.

## Files

- `metrics_by_seed_draw.csv`: four complete eight-class classifiers, each seed and support draw.
- `neural_seed_means.csv` / `neural_summary.csv`: five-draw means then neural-seed mean and SD.
- `support_draw_means.csv`: support-draw results averaged across neural seeds, kept separately.
- `interactions_by_seed_draw.csv`: prior/kernel conditional effects and nonadditivity.
- `net_effect_summary.csv`: paired metric differences versus frozen single and fixed matched dual.
- `seed_*_predictions.npz`: persisted fixed crossover predictions and exact common row/support metadata.
