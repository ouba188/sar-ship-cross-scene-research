# Source-Held Prior/Kernel Crossover

One bounded diagnostic using the original eight held-source episodes per neural seed.
No new training, target-query labels, or target-driven parameter selection.

| Prior / kernel | Source OA | Source BA | Target OA | Target BA |
| --- | ---: | ---: | ---: | ---: |
| frozen_prior_frozen_kernel | 63.281250 | 59.340697 | 34.615594 | 29.342822 |
| trained_prior_frozen_kernel | 62.923177 | 59.705761 | 38.384497 | 31.157628 |
| frozen_prior_trained_kernel | 60.156250 | 60.280062 | 28.713212 | 26.804446 |
| trained_prior_trained_kernel | 61.067708 | 60.761534 | 33.423888 | 30.422258 |

## Kernel Effects

| Metric | Prior context | Source kernel change | Target kernel change |
| --- | --- | ---: | ---: |
| OA | frozen | -3.125000pp | -5.902382pp |
| OA | trained | -1.855469pp | -4.960609pp |
| BA | frozen | +0.939365pp | -2.538377pp |
| BA | trained | +1.055773pp | -0.735370pp |

## Interpretation

The trained kernel helps mean held-source BA under BOTH prior contexts but hurts target BA. The direction is consistent with transfer-specific failure, not a kernel that is generally worse already on held-source validation.

OA already declines on held source, while mean BA increases. Therefore there is also a source-side OA/BA tradeoff; positive source BA is not uniform improvement. The additional BA sign reversal on target is a separate observation.
Source kernel BA effects under frozen prior:3 positive,0 zero,0 negative neural seeds. Under trained prior:2 positive,0 zero,1 negative neural seed.

Kernel/prior swaps are functional interventions, not physical causal identification.
The source run uses selected INNER checkpoints and an INNER-training fitted frozen prior.
The target run uses ALL-allowed-source refits. This deliberate phase difference means a sign reversal could also involve refit behavior; the comparison cannot isolate domain shift from refitting without further evidence.
Held-source episodes selected the checkpoint, so their results are development evidence, not independent confirmation.

## Replay And Scope

Original OA/BA diagonal replay exact (tolerance1e-9): True; maximum absolute OA/BA difference: 0pp.
The original curve stored metrics, not per-sample validation predictions. Metric equality is NOT a claim of bitwise equality of unavailable original predictions.
Encoding repeats the original192-row episode, microbatch16, four rotations, bf16 forward and float32 GPU prior/residual operations. No cache/dtype algebra substitution is made.
The frozen eight episode embeddings/priors are reused across seeds after checking the original saved episodes match. The source prior was fitted once on inner training only.
Each seed first averages eight episodes. Means/SD then aggregate neural seeds; the shared frozen reference is not three independent frozen trainings.

## Files

- `by_seed_episode_mode.csv`: every source OA/BA.
- `per_class.csv`: recalls, support coverage and net rescue/harm signs versus frozen single.
- `neural_seed_means.csv`, `neural_summary.csv`: episode averages then neural mean/SD.
- `effects_by_seed.csv`: prior-only/kernel-only/joint deltas and conditional effects.
- `DIAGONAL_CHECKS.json`: original versus replay OA/BA and every discrepancy.
- `seed_*_predictions.npz`: query predictions, support/query rows and source episode metadata.
