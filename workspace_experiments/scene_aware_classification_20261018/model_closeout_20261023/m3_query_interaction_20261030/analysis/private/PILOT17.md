# Private17 bounded CPU findings

Completed private development160 only: three new arms, reused original and prior CE endpoints, six supplied controls. Public results are not included. One training order is screening evidence, not three-order confirmation.

| New arm | OA vs original (pp) | BA vs original (pp) | OA vs CE (pp) | BA vs CE (pp) | Rescue/harm vs original |
| --- | ---: | ---: | ---: | ---: | --- |
| query_only | -0.625 | +0.117347 | +0.625 | +0.571429 | 2/3 |
| ref_only | -0.625 | +0.117347 | +0.625 | +0.571429 | 2/3 |
| interaction | -0.625 | +0.484694 | +0.625 | +0.938776 | 2/3 |

Interaction versus CE has 2 rescues, 1 harm, 70 persistently correct and 87 persistently wrong. Versus query_only or ref_only, interaction exchanges one rescue for one harm: OA unchanged, BA +0.367347pp. Its NLL/Brier are worse than either simpler arm; versus original, NLL improves -0.034812 but Brier worsens +0.009029. Thus this pilot shows a class-weighted tradeoff, not an established overall replacement benefit.

All off controls recover original logits exactly (maximum error 0, zero original-relative top-1 changes). Query-only uniform logits equal normal exactly. Uniform changes ref_only/interaction logits by at most 0.043771/0.042155, with zero top-1 changes; their NLL/Brier improve slightly. The saved predictions therefore do not show a top-1 benefit from learned reference weights in the added writer for this private order. Uniform retains transported reference content and original M3; this does not show that all reference information or query dependence is unnecessary.

| Arm | Total centered-increment energy | Common energy | Residual energy | Common fraction |
| --- | ---: | ---: | ---: | ---: |
| query_only | 479.990552 | 451.660995 | 28.329556 | 94.0979% |
| ref_only | 486.792510 | 459.507776 | 27.284734 | 94.3950% |
| interaction | 399.172863 | 369.749236 | 29.423627 | 92.6289% |

Interaction has a somewhat larger absolute residual despite lower total increment energy, but its update remains predominantly query-common. The decomposition is descriptive squared logit energy, not causal information or physical energy. Full definitions and decomposed endpoint metrics are in shared_shift.json.

Main-review cases: interaction versus CE, seed17; one distinct query per state selected by absolute fixed-rival margin change (no random sampling). IDs: rescued33658, harmed35728, CC8343, WW31141. Native VV/VH and actual legal same/rival references are in interaction_vs_ce/index.html, with original paths in native_sources.json. Query33658 VH is unavailable and marked accordingly. Four queries/eight panels were rendered; visual review remains with main. No image-based physical interpretation is claimed here.

Startup receipts for all three arms record initial_error=0 and baseline_batch_error=5.7220458984375e-06. The full saved original matches the prior CE archive exactly. Source coverage and reused schedule provenance are documented in ../SOURCE_CONTEXT.md; source audit was not repeated.

Five bounded synthetic CPU checks passed. The real private run completed with no missing inputs; no torch import, source prediction reads, final-test access, GPU work or Git operation occurred in this analysis task.
