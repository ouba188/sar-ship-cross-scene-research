# Explanation Results

Analyzed tasks only: {'f0921': 1, 'self24ports': 23}; neural seeds: [20261010, 20261011, 20261012].
One fixed current dual reference per task. No GPU, training, target selection or new crossover.

## Paired Gaps

| Dataset | Candidate - reference | OA pp | BA pp | BA neural-seed SD |
| --- | --- | ---: | ---: | ---: |
| f0921 | dual_vs_trained | +7.94858 | +5.07727 | 0.24740 |
| f0921 | trained_vs_frozen | -0.95826 | -0.15053 | 0.24740 |
| self24ports | dual_vs_trained | +6.98986 | +4.33365 | 0.19907 |
| self24ports | trained_vs_frozen | -1.39193 | +0.89319 | 0.19907 |

## Exact Configuration Bridge

D-T = (D-R) + (R-F) + (F-T). D: current dual; R: current full R50-only;
F: frozen compact; T: task-trained compact. These are configuration differences, NOT independent causal effects.
D-R is the existing matched second-route increment; R-F includes readout/geometry and other configuration differences.
F-T is the negative of the observed task-training change. Do not attribute the whole gap to backbone count.

| Dataset | Metric | D-R | R-F | F-T | Total D-T |
| --- | --- | ---: | ---: | ---: | ---: |
| f0921 | BA | +1.75866 | +3.16809 | +0.15053 | +5.07727 |
| f0921 | OA | +3.63054 | +3.35977 | +0.95826 | +7.94858 |
| self24ports | BA | +3.79948 | +1.42736 | -0.89319 | +4.33365 |
| self24ports | OA | +5.46586 | +0.13207 | +1.39193 | +6.98986 |

## Neural-Seed Stability

Counts below are query/draw exposures counted once across seeds, not independent samples.
| Dataset | Contrast | Query/draw exposures | All-seed rescue | All-seed harm |
| --- | --- | ---: | ---: | ---: |
| f0921 | dual_vs_trained | 617125 | 64065 | 16834 |
| f0921 | trained_vs_frozen | 617125 | 14487 | 28970 |
| self24ports | dual_vs_trained | 182775 | 11056 | 4698 |
| self24ports | trained_vs_frozen | 182775 | 4090 | 5128 |

Sample IDs, seed votes and deduplicated all-eligible-draw events are exported in tasks/*/*/*.csv.gz.
Fixed-dual correctness prevents rescue and harm from co-occurring across seeds for the same query/draw;
across different support draws, dual correctness can change. Unanimity across seeds does not imply support stability.

## Fixed Tested Single Set

Includes all requested trained/frozen compact seeds, current full R50-only and the existing MLP reconstruction.
The current CLI additionally requires the task-trained R18-only control; each task receipt records the exact fixed set.
Repeated frozen predictions do not provide extra independent evidence; common-hard is a logical intersection.
| Dataset | Exposures | All tested singles wrong | Dual rescues those | All including dual wrong | Dual wrong, any single correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| f0921 | 617125 | 227451 | 7925 | 219526 | 181276 |
| self24ports | 182775 | 34585 | 675 | 33910 | 30167 |

These are common failures of the measured configurations; they do not prove impossibility for a single backbone.

## Original-Score Gap Strata

Thresholds fixed at .05 and .20. Raw logit gaps are NOT historical softmax probability gaps or calibrated confidence.
| Dataset | Contrast | Units | Stratum | Exposures | Rescue | Harm | Net OA pp |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| f0921 | dual_vs_trained | original_logit_gap | gap_005_020 | 599660 | 119221 | 60278 | +9.82940 |
| f0921 | dual_vs_trained | original_logit_gap | gap_ge_020 | 994732 | 134628 | 76500 | +5.84358 |
| f0921 | dual_vs_trained | original_logit_gap | gap_lt_005 | 256983 | 57135 | 27048 | +11.70778 |
| f0921 | trained_vs_frozen | original_logit_gap | gap_005_020 | 599660 | 65703 | 78701 | -2.16756 |
| f0921 | trained_vs_frozen | original_logit_gap | gap_ge_020 | 994732 | 72450 | 67506 | +0.49702 |
| f0921 | trained_vs_frozen | original_logit_gap | gap_lt_005 | 256983 | 30413 | 40100 | -3.76951 |
| self24ports | dual_vs_trained | original_logit_gap | gap_005_020 | 131623 | 28309 | 13760 | +11.05354 |
| self24ports | dual_vs_trained | original_logit_gap | gap_ge_020 | 359648 | 22365 | 14304 | +2.24136 |
| self24ports | dual_vs_trained | original_logit_gap | gap_lt_005 | 57054 | 14757 | 6730 | +14.06913 |
| self24ports | trained_vs_frozen | original_logit_gap | gap_005_020 | 131623 | 15396 | 19446 | -3.07697 |
| self24ports | trained_vs_frozen | original_logit_gap | gap_ge_020 | 359648 | 13618 | 11498 | +0.58947 |
| self24ports | trained_vs_frozen | original_logit_gap | gap_lt_005 | 57054 | 7669 | 10882 | -5.63151 |

Stratum exposure sums repeat queries across draws/seeds; dataset OA/BA instead use task macro means.

## Boundaries And Next Intervention

- Historical expert parts explain DD-RR logits, conditioned on present dual-versus-trained events. They do NOT sum to D-T.
- Where original phi/prior replay is available, trained-minus-frozen prior and support-correction deltas are exact score accounting.
  The support-correction delta includes both changed relations and changed coefficients; no isolated kernel cause is claimed.
- Saved four-cell results are reused only after original diagonal alignment. Missing crossover/decomposition is reported as unavailable.
- No physical specialization, architecture necessity, equalized pretraining or untouched confirmation is established. F had historical unlabeled SSL exposure.
- All prespecified 23 LOPO + F comparisons are complete with three neural seeds. Source-selected controls and target estimates remain separate; do not generalize a dataset-specific path effect to every dataset.
- If source-versus-target kernel reversal persists, one controlled follow-up is source-only relation-preservation regularization during the same single-backbone task training.
  Compare against the existing objective with the same seeds/budget and source-only selection; examine prior, kernel and class harm, not just mean BA.
- If the full-readout bridge term dominates, first compare a single R50 with that complete readout under matched source fitting; do not add a second backbone to explain this term.
