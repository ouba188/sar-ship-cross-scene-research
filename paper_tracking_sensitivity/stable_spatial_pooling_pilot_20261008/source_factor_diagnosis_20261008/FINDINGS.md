# Source-only spatial factor diagnosis

Fixed outer Antwerp-Bruges; held source Jebel Ali, Port Klang and Singapore. Original purged train rows, original source64 supports, frozen geometry and unchanged settings. No Antwerp target predictions or metrics were read, and no Antwerp target labels were used. Original negative pilot outputs are preserved.

The large loss localizes to the auxiliary prior factor in all three episodes for both pooling arms. The kernel-only loss is much smaller; adding the kernel to the auxiliary prior slightly moderates its loss. This separates the dominant integration factor, not a unique underlying statistical cause. No clear implementation bug was found, so no corrected model or alternate parameter setting was run.

| Pool | Factor | Expert OA % | Expert BA % | Balanced score MSE |
|---|---|---:|---:|---:|
| ordinary_pool | original | 46.6523 | 44.8760 | 0.572504 |
| ordinary_pool | prior_only | 41.0526 | 39.3418 | 0.671897 |
| ordinary_pool | kernel_only | 46.5000 | 44.6369 | 0.574506 |
| ordinary_pool | full | 41.5037 | 39.6309 | 0.661036 |
| stable_pool | original | 46.6523 | 44.8760 | 0.572504 |
| stable_pool | prior_only | 41.6235 | 39.0388 | 0.678379 |
| stable_pool | kernel_only | 46.4594 | 44.6592 | 0.574993 |
| stable_pool | full | 42.1539 | 39.3727 | 0.666889 |

Equal weight across the three held source episodes; support rows excluded from held metrics.

## Factor separation

- ordinary_pool: prior-only -5.5342 pp BA; kernel-only -0.2391 pp; full -5.2451 pp; factor interaction +0.5283 pp.
- stable_pool: prior-only -5.8372 pp BA; kernel-only -0.2167 pp; full -5.5033 pp; factor interaction +0.5507 pp.

| Pool | Split | Original prior MSE | Augmented prior MSE | Original prior BA % | Augmented prior BA % |
|---|---|---:|---:|---:|---:|
| ordinary_pool | train_fit | 1.125269 | 0.830766 | 48.0816 | 47.7817 |
| ordinary_pool | held_query | 1.313826 | 1.036356 | 37.9061 | 35.9265 |
| stable_pool | train_fit | 1.125269 | 0.847042 | 48.0816 | 47.8331 |
| stable_pool | held_query | 1.313826 | 1.060444 | 37.9061 | 36.0756 |

The auxiliary head improves its class-balanced score MSE on training rows AND held-source rows, but held-prior BA falls from 37.9061% to 35.9265% (ordinary) / 36.0756% (stable). After the coordinated64 correction, both MSE and BA are worse than the original corrected expert. Thus the measured failure is not simply failure to reduce the residual regression objective, nor does it require a target-only distribution shift. Score-regression improvement does not preserve class decisions or the original expert benefit from support adaptation.

## Attention and units

| Pool | Split | Entropy nats (uniform=1.3863) | Max weight | Base top2 contains truth % | Feature RMS |
|---|---|---:|---:|---:|---:|
| ordinary_pool | train_fit | 1.2412 | 0.4254 | 72.203 | 1.0005 |
| ordinary_pool | held_query | 1.2448 | 0.4140 | 54.158 | 1.0094 |
| stable_pool | train_fit | 1.1935 | 0.4614 | 72.203 | 1.0161 |
| stable_pool | held_query | 1.2289 | 0.4287 | 54.158 | 0.9275 |

Truth-in-top2 is diagnostic only and never selects attention or examples. All MSEs average score-coordinate squared errors, then give each present true class equal weight. Per-class recalls, class-balanced mean score errors, centered-score MSE and per-episode values are saved.

The .25 coefficient multiplies ONLY the added adaptation kernel z z_support.T / 128. It does not attenuate the learned prior z W, whose fixed multiplier is 1. Every factor refits the support correction jointly against its own prior and kernel. The original retained/complement kernel and its bias remain intact. The residual readout fits source-RMS-normalized 128-dimensional z without an added intercept; its ridge is 1. The 64-support correction ridge is 8.

- ordinary_pool: held auxiliary score RMS 0.50686; prior-only corrected-score delta RMS 0.42422; mean survival ratio 0.8371; spatial/original support-kernel diagonal ratio 0.0138.
- stable_pool: held auxiliary score RMS 0.48193; prior-only corrected-score delta RMS 0.41439; mean survival ratio 0.8596; spatial/original support-kernel diagonal ratio 0.0109.

## Implementation checks

Saved original classifier and installed sklearn 1.8.0 use LabelBinarizer(neg_label=-1, pos_label=1) with columns 0..7. This exactly matches 2*onehot-1. The new residual head is checked against sklearn Ridge(alpha=1, fit_intercept=False) using the same original expert prior and exact balanced class weights. The train-prior helper is replayed on held rows against the held-view prior; episode expert coefficients/intercepts equal their frozen geometry counterparts. Maximum residual-head coefficient disagreement with sklearn is 5.61e-15. Original corrected scores replay exactly; all six full-arm BA values reproduce the original pilot. The train/held prior helper difference is at most 2.63e-6 from floating-point evaluation order. No alternative head, intercept or strength is selected.

See VERIFICATION.json for numerical errors, archived original/full replay, source exclusion and label-encoding checks. Directions, PCA, RMS moments and frozen encoder still use all outer-source data: not fully inner-OOF. Only these three fixed source episodes were diagnosed; no independent confirmation or unique root-cause claim follows. The missing auxiliary intercept is an intentional difference from the original classifier, not an established bug. Row-centered score MSE is essentially identical to uncentered MSE here, so common score offsets do not explain the loss; class-specific calibration and representation effects remain unresolved. An intercept change, joint retraining or changed regularization would be new experiments and were not performed.
