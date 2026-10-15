# Enriched OOF reliability: completed fixed development comparison

Date: 2026-10-07. Main run completed with exit 0 in 127.71 seconds.
Run: `E:/Hermes/paper_tracking/sensitivity/oof_enriched_results_20261007`.
Protocol: `ENRICHED_PROTOCOL_20261007.md`, saved before candidate readout.

## Decision

The fixed richer ridge and small nonlinear baseline do not improve the
previous four-confidence-feature models or simple gate. Retain the original
baselines; do not promote this implementation, sweep regularization/trees,
or claim that all class-conditioned reliability is impossible.

The new meaningful result is narrower: adding complete relative logits and
ordered class-pair indicators, under this fixed OOF regression recipe, did
not recover extra target BA. The tree has nearly the same average score as
the richer linear model. A spatial/physical context branch would introduce
different information and remains a distinct hypothesis, not refuted here.

## Results

BA is mean over the five support draws within each port, then equal over
23 ports. Five overlapping support draws are not five independent models or
datasets. `0` means no coefficient adaptation; the same 64 labels still define
the observed-expert-class guard in ALL arms.

| Model | BA, % | Minus simple, percentage points |
|---|---:|---:|
| Original base | 43.2690 | -4.6344 |
| Original Fusion | 47.0938 | -0.8096 |
| Simple gate | 47.9034 | 0 |
| s2 | 47.9064 | +0.0030 |
| Previous OOF LR0 | 48.0408 | +0.1374 |
| Previous/conf ridge0 | 47.9869 | +0.0835 |
| Previous/conf ridge64 | 47.4594 | -0.4440 |
| Rich ridge0 | 47.5247 | -0.3787 |
| Rich ridge64 | 47.2232 | -0.6802 |
| Rich HGB0 | 47.5292 | -0.3742 |
| Rich HGB64 | 47.2460 | -0.6574 |

Primary feature contrast, rich_ridge64 - conf_ridge64: -0.236151 pp,
10/23 ports positive, descriptive port bootstrap 95% interval
[-0.594656, +0.118679] pp, Wilcoxon p=0.3001.

Primary expression contrast, rich_hgb64 - rich_ridge64: +0.022754 pp,
9/23 ports positive, interval [-0.110888, +0.161321] pp, p=0.9643.
This does not establish equivalence or that nonlinear methods cannot help.

All four new rich arms have negative equal-port means versus simple in each
of the five support draws. These are correlated descriptive readouts; they
are not five independent replications.

## Implementation and checks

- 69 new small gate fits; zero encoder/PCA/classification-head retraining.
- 115 contexts, six saved gate-score arrays and ten saved prediction arms per
  context. Two previous LR metrics are inherited, explicitly marked in CSV.
- Source models, target residual coefficients, exact support/query indices,
  evaluation masks and scores are archived. All port policies were saved
  before query-label readout. Historical outcome exposure remains disclosed.
- Six tests passed: feature identities, independent weighted ridge solve,
  equality to the original 4D adaptation, no-disagreement identity, support
  guard and three-valued reward.
- Parent controls reproduce with maximum error 0. Old Sinkhorn warnings use
  the unchanged inherited fallback; no solver thresholds were changed.
- Independent confusion-matrix replay verified 6,325 metrics and aggregate
  means. Maximum error 2.22e-16; no head-training replay was performed.

## Scope carried forward

This is uncapped model selection, NOT the earlier 5% E/G replacement study.
The 64 support rows are excluded from evaluation, but same-vessel/product
rows are not purged from the target evaluation batch. Source-head OOF does
purge recorded outer/inner port, MMSI and product identities. Encoder access
is a user declaration, not a newly verified pretraining lineage.

Do not repeat the earlier broad claims of information-theoretic
unidentifiability, universal harm from target labels, or leakage quantified
solely by an in-sample/OOF performance difference. The original numerical
outputs remain useful bounded development evidence.

## Next dependency

Audit real facility/relative-location joins against the CURRENT 38,091-row
manifest. The old three-class 7.59% coverage is not a current coverage result.
Do not manufacture port-centroid features, feed AIS class labels to a model,
or silently drop knowledge-missing query rows. A knowledge experiment needs
its own fixed protocol after this availability check.
