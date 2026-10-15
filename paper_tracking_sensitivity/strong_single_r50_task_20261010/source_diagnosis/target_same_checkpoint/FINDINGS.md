# Same-Checkpoint Source To OSS2 Diagnostic

The exact selected inner checkpoints and inner-fitted initial prior now appear on both domains.
No source refit, additional training, hyperparameter selection or main-result replacement.

| Prior / kernel | Source OA | Source BA | OSS2 OA | OSS2 BA |
| --- | ---: | ---: | ---: | ---: |
| frozen_prior_frozen_kernel | 63.281250 | 59.340697 | 34.132029 | 28.586809 |
| trained_prior_frozen_kernel | 62.923177 | 59.705761 | 37.424613 | 30.183347 |
| frozen_prior_trained_kernel | 60.156250 | 60.280062 | 28.803767 | 25.886019 |
| trained_prior_trained_kernel | 61.067708 | 60.761534 | 32.556371 | 28.759255 |

## Kernel Effects With Training Phase Controlled

| Metric | Prior | Source delta | OSS2 delta |
| --- | --- | ---: | ---: |
| BA | frozen_prior | +0.939365pp | -2.700790pp |
| BA | trained_prior | +1.055773pp | -1.424092pp |
| OA | frozen_prior | -3.125000pp | -5.328262pp |
| OA | trained_prior | -1.855469pp | -4.868242pp |

## Discriminating Conclusion

The BA sign reversal persists under BOTH prior contexts for the SAME inner checkpoints. It cannot be explained solely by all-source refitting. The learned support geometry has held-source BA benefit but fails to transfer that benefit to OSS2 in this protocol.

This localizes the problem to empirical cross-domain transfer of the learned relation, not a demonstrated physical causal factor or a universal impossibility of single-backbone learning.
Held-source OA/BA tradeoffs remain visible. Source checkpoint selection used these episodes, so positive source BA is selection-conditioned development evidence, not independent confirmation.
The crossed cells require both frozen and trained feature functions during diagnosis; they are not automatically deployable as a single-backbone, single-forward model.

## Seed Directions

- frozen_prior, source BA kernel effect: 3 positive / 0 zero / 0 negative neural seeds.
- frozen_prior, target BA kernel effect: 0 positive / 0 zero / 3 negative neural seeds.
- trained_prior, source BA kernel effect: 2 positive / 0 zero / 1 negative neural seeds.
- trained_prior, target BA kernel effect: 0 positive / 0 zero / 3 negative neural seeds.

## Metadata And Boundaries

Selected inner epochs: {20261010: 5, 20261011: 5, 20261012: 6}. Each seed averages five fixed64-label target draws; neural-seed SD is computed after these averages.
Source frozen cache contains 1445 unique row IDs across 1536 row exposures. Four rotations are separate forwards, not unique samples.
Only source diagnosis count metadata was corrected. All-source-refit training/results and original target predictions remain unchanged. Diagnostic target predictions were persisted before query truth.
Three inner-model target extractions only; initial frozen target geometry was reused.
