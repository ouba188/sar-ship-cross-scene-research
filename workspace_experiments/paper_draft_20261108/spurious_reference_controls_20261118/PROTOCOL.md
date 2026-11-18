# Reference Dependence Controls

## Question and Scope

Test whether the frozen, trained routed PCEA/CCREL model's M3 contribution requires the original reference vessels or query-port reference images. This is a diagnostic of the current development candidate, not a new independent benchmark, formal-model replacement, or proof of absence of spurious correlations.

- Existing validation160 and expansion198 observations; 356 unique vessels in total.
- Existing routed seeds17/29/43, no optimization, no tuning or selecting banks by target predictions.
- All three continuations share the original model starting point. Repeated seeds/banks are not independent vessels.
- No 3075 test images, no new learned axis/relation operator.

## Fixed Conditions

Same query pixels, original native160 crop/pad preprocessing, labels, geometry, masks, all weights, 8 classes, 2 source references/class, and available-polarization reference budget. PCEA source statistics remain anchored to the original16 source images. Replacement reference features are obtained through `extract`, not `reference`, to avoid changing those statistics.

Compare M3 disabled (same weights, PCEA remains enabled), original M3, and six replacement banks:

1. Three same-port banks: match each original reference slot's category and port, but use different vessels and different passages.
2. Three outside-query-port banks: same category, different vessels/passages, and no reference from any of the five query ports or that slot's original reference port.

Alternative candidates come only from the15316 legal source-fit observations. Exclude all query and original-reference MMSIs and all their passages (satellite plus absolute orbit). All alternatives have both original available polarizations. Each bank has16 distinct vessels; rare source cells can repeat vessels between draws, which will be reported.

Within each category/port eligibility set, rank by mean squared log-size difference across VV/VH OBB lengths and widths relative to the original reference slot; randomly choose among the nearest20 valid candidates with fixed seeds2026111801/02/03. No target classifier scores enter selection. Record size mismatch and continuation-training exposure. This controls approximate size, not all imaging conditions. Different-port replacement also changes vessel/content/observation, so it does not isolate a pure causal port effect.

## Verification and Outcomes

- Original bank must replay cached predictions within2e-5.
- M3-off predictions must be invariant to bank selection within1e-7; otherwise fix hidden-state mixing before interpretation.
- Verify actual per-query legal references and effective polarization budget, not only nominal16.
- Report all banks, OA/BA, per-class and per-port paired net rescue/harm, stability of original rescues, and vessel-level averages. Keep adverse outcomes.
- Inspect native VV/VH chips and actual replacement references for persistent-correct, persistent-wrong, rescue, and harm cases.

Positive evidence requires net M3 benefits surviving at least some independently chosen reference changes. If benefits disappear, report reference sensitivity without declaring all features spurious. Different-port references cannot establish that the backbone ignores port information; it still sees each query's chip and source training covered these ports.

## Background Check Boundary

OBB exterior is not a certified sea-only mask. Extended responses exist beyond the nominal1.5L by2W neighborhood in reviewed chips. Full-background swapping is not run automatically: a credible local test would need manually verified donor/query sea patches, protected target extensions, fixed seams, and same-scene controls. This remains a separate evidence gap unless completed explicitly.
