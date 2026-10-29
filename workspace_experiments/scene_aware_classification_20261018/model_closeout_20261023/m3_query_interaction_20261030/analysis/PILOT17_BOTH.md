# Seed17 screening: private development160 and OSS2 validation228

Both seed17 datasets are complete. Orders29/43 are still pending for this report; their outputs have not been read. These are screening results under the fixed protocol, not a three-order conclusion or a replacement recommendation. Source coverage/schedule provenance is reused in SOURCE_CONTEXT.md.

| Dataset | Interaction OA/BA vs original (pp) | Interaction OA/BA vs CE (pp) | Interaction OA/BA vs query_only or ref_only (pp) |
| --- | --- | --- | --- |
| private | -0.625 / +0.484694 | +0.625 / +0.938776 | 0 / +0.367347 |
| OSS2 | -0.877193 / -0.924538 | 0 / +0.587628 | 0 / +0.581170 |

Interaction versus either simpler arm exchanges one rescue and one harm in each dataset. Relative to original, private interaction has 2 rescues/3 harms/70 CC/85 WW; OSS2 has 2 rescues/4 harms/128 CC/94 WW. Interaction versus CE has private2/1/70/87 and OSS2 2/2/128/96. Full event and class-tradeoff tables remain in each dataset directory.

OSS2 interaction improves NLL/Brier over query_only by -0.001450/-0.000548 and over ref_only by -0.004555/-0.001300. Private interaction instead worsens those risk metrics versus both simpler arms. The relative BA/risk signal supports the main worker's already chosen fixed-order expansion; OA below original in both datasets does not establish repair success.

## Common versus query-specific increment

All quantities below concern the new-minus-original logit increment, centered across classes within each query. Common is its across-query mean; residual is the remaining query-specific increment. Energies are sums of squared centered logits, not causal-information fractions. Compare magnitude within each dataset; dimensions and query counts differ between datasets.

| Dataset/arm | Total energy | Common energy | Residual energy | Common fraction | Full/common-only prediction agreement | Actual original-relative flips reproduced by common-only |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| private query_only | 479.990552 | 451.660995 | 28.329556 | 94.0979% | 155/160 | 13/14 |
| private ref_only | 486.792510 | 459.507776 | 27.284734 | 94.3950% | 155/160 | 12/13 |
| private interaction | 399.172863 | 369.749236 | 29.423627 | 92.6289% | 154/160 | 12/15 |
| OSS2 query_only | 95.796142 | 91.667317 | 4.128824 | 95.6900% | 226/228 | 12/12 |
| OSS2 ref_only | 103.898806 | 100.211425 | 3.687381 | 96.4510% | 225/228 | 11/12 |
| OSS2 interaction | 94.743383 | 90.029828 | 4.713555 | 95.0249% | 224/228 | 10/10 |

Interaction increases absolute residual energy over both simpler arms in both datasets, so the fraction increase is not solely a denominator effect. However, the common component still dominates. Query-specific residuals affect six private decisions and four public decisions relative to common-only; this does not by itself imply useful conditional matching.

| Interaction endpoint | private OA / BA | OSS2 OA / BA |
| --- | --- | --- |
| Original | 45.625 / 34.928108 | 57.894737 / 50.123534 |
| Full | 45.000 / 35.412801 | 57.017544 / 49.198996 |
| Common-only | 44.375 / 34.474026 | 57.894737 / 49.985681 |
| Residual-only | 46.250 / 35.499536 | 57.456140 / 49.842635 |

Adding the residual to common-only helps private by OA+0.625/BA+0.938776pp but hurts OSS2 by OA-0.877193/BA-0.786685pp. For OSS2, common-only reproduces all ten actual full-model flips and makes four additional flips; omitting those four in the full model costs two net correct predictions. Thus the current evidence shows some query-specific decision changes with inconsistent utility, not a broad shift from common offsets to useful query-specific classification. The decomposition is computed from development logits and is descriptive; residual-only is not a trained or deployable correction.

## Added-writer dependence controls

All six off controls across both datasets recover original logits exactly. All six uniform controls preserve their normal arm's top-1 predictions. Query_only uniform is exactly identical in logits. Uniform maximum logit differences for ref_only/interaction are private0.043771/0.042155 and OSS2 0.032970/0.034407. Uniform slightly improves NLL/Brier in both reference-using arms on both datasets.

This pilot does not demonstrate top-1 utility from learned reference weights in the added writer. The control preserves reference content, transport and original M3, so it does not establish reference independence or removal of all query dependence.

## Main visual review

Only four queries/eight native VV/VH panels per dataset were generated, including actual legal same-class and rival-class references. No exhaustive gallery audit was performed. Selection is interaction versus CE, one distinct query per state, sorted by absolute fixed-rival margin change within this single order.

- private: rescued33658, harmed35728, CC8343, WW31141. Query33658 VH is unavailable.
- OSS2: rescued003623, harmed000354, CC001056, WW003764 (full IDs have prefix pub2_opensarship2_). Query003764 VV is unavailable.

Gallery paths: private/interaction_vs_ce/index.html and oss2/interaction_vs_ce/index.html. Native source paths, available polarizations, reference IDs and score margins are recorded alongside each gallery. Main will view; these files do not claim completed visual review or physical causes.

CPU analysis completed without missing seed17 inputs, torch imports, GPU use, source prediction access, final-test access or Git operations. Next comparison is the already authorized fixed29/43 repetition; do not infer its result from this seed17 report.
