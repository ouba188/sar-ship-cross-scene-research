# Explanation Results

Analyzed tasks only: {'oss2': 1}; neural seeds: [20261010, 20261011, 20261012].
One fixed current dual reference per task. No GPU, training, target selection or new crossover.

## Paired Gaps

| Dataset | Candidate - reference | OA pp | BA pp | BA neural-seed SD |
| --- | --- | ---: | ---: | ---: |
| oss2 | dual_vs_trained | +9.17323 | +4.25216 | 0.86697 |
| oss2 | trained_vs_frozen | -1.19171 | +1.07944 | 0.86697 |

## Exact Configuration Bridge

D-T = (D-R) + (R-F) + (F-T). D: current dual; R: current full R50-only;
F: frozen compact; T: task-trained compact. These are configuration differences, NOT independent causal effects.
D-R is the existing matched second-route increment; R-F includes readout/geometry and other configuration differences.
F-T is the negative of the observed task-training change. Do not attribute the whole gap to backbone count.

| Dataset | Metric | D-R | R-F | F-T | Total D-T |
| --- | --- | ---: | ---: | ---: | ---: |
| oss2 | BA | +4.42276 | +0.90884 | -1.07944 | +4.25216 |
| oss2 | OA | +10.06792 | -2.08639 | +1.19171 | +9.17323 |

## Neural-Seed Stability

Counts below are query/draw exposures counted once across seeds, not independent samples.
| Dataset | Contrast | Query/draw exposures | All-seed rescue | All-seed harm |
| --- | --- | ---: | ---: | ---: |
| oss2 | dual_vs_trained | 18405 | 2009 | 622 |
| oss2 | trained_vs_frozen | 18405 | 574 | 969 |

Sample IDs, seed votes and deduplicated all-eligible-draw events are exported in tasks/*/*/*.csv.gz.
Fixed-dual correctness prevents rescue and harm from co-occurring across seeds for the same query/draw;
across different support draws, dual correctness can change. Unanimity across seeds does not imply support stability.

## Fixed Tested Single Set

Includes all requested trained/frozen compact seeds, current full R50-only and the existing MLP reconstruction.
Repeated frozen predictions do not provide extra independent evidence; common-hard is a logical intersection.
| Dataset | Exposures | All tested singles wrong | Dual rescues those | All including dual wrong | Dual wrong, any single correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| oss2 | 18405 | 7184 | 888 | 6296 | 4269 |

These are common failures of the measured configurations; they do not prove impossibility for a single backbone.

## Original-Score Gap Strata

Thresholds fixed at .05 and .20. Raw logit gaps are NOT historical softmax probability gaps or calibrated confidence.
| Dataset | Contrast | Units | Stratum | Exposures | Rescue | Harm | Net OA pp |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| oss2 | dual_vs_trained | original_logit_gap | gap_005_020 | 20912 | 4552 | 2324 | +10.65417 |
| oss2 | dual_vs_trained | original_logit_gap | gap_ge_020 | 24752 | 3406 | 1728 | +6.77925 |
| oss2 | dual_vs_trained | original_logit_gap | gap_lt_005 | 9551 | 2267 | 1108 | +12.13485 |
| oss2 | trained_vs_frozen | original_logit_gap | gap_005_020 | 20912 | 2618 | 2909 | -1.39155 |
| oss2 | trained_vs_frozen | original_logit_gap | gap_ge_020 | 24752 | 1699 | 1724 | -0.10100 |
| oss2 | trained_vs_frozen | original_logit_gap | gap_lt_005 | 9551 | 1283 | 1625 | -3.58078 |

Stratum exposure sums repeat queries across draws/seeds; dataset OA/BA instead use task macro means.

## Boundaries And Next Intervention

- Historical expert parts explain DD-RR logits, conditioned on present dual-versus-trained events. They do NOT sum to D-T.
- Where original phi/prior replay is available, trained-minus-frozen prior and support-correction deltas are exact score accounting.
  The support-correction delta includes both changed relations and changed coefficients; no isolated kernel cause is claimed.
- Saved four-cell results are reused only after original diagonal alignment. Missing crossover/decomposition is reported as unavailable.
- No physical specialization, architecture necessity, equalized pretraining or untouched confirmation is established. F had historical unlabeled SSL exposure.
- Finish the prespecified 23 LOPO + F comparisons before extending OSS2 conclusions. Source-selected controls and target estimates remain separate.
- If source-versus-target kernel reversal persists, one controlled follow-up is source-only relation-preservation regularization during the same single-backbone task training.
  Compare against the existing objective with the same seeds/budget and source-only selection; examine prior, kernel and class harm, not just mean BA.
- If the full-readout bridge term dominates, first compare a single R50 with that complete readout under matched source fitting; do not add a second backbone to explain this term.
