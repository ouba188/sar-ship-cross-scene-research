# Same-Vessel Different-Acquisition Readout

Metadata-selected adjacent same-class observations from the frozen OSS2 cache. GRDH is primary; SLC is separate and descriptive. No new inference, fitting, support draws, model selection, or fusion construction was performed.

## Selection And Denominators

Within each product type, remove all rows in nonunique identity+datatake+SAR-time keys. Sort the remaining observations by time within vessel, select adjacent observations, and then require identical class, different datatake and positive time interval. No predictions enter selection. Preserve each cached query mask; both endpoints must be outside that draw's B64 support. The strict sensitivity additionally removes every pair whose vessel occurs anywhere in that support, including its other product/acquisition records.

- GRDH: 2746 cached rows; 12 ambiguous rows excluded; 498 pairs, 366 identities, 8 classes, 26 datatakes, 864 unique endpoints. Intervals in days min/median/max: 3.978449/71.999421/372.000058.
- SLC: 999 cached rows; 0 ambiguous rows excluded; 53 pairs, 52 identities, 6 classes, 10 datatakes, 105 unique endpoints. Intervals in days min/median/max: 12.000093/183.978646/457.474190.

## Results

All rates are percentages, T-R is percentage points. OA is ordinary mean accuracy over both pair endpoints; BA is the class-macro mean of endpoint accuracy. Adjacent pairs can share an endpoint. Classes receive equal weight for macro rates. Repeats are averaged within each pair before these summaries.

| Product / scope | Pairs / vessels / classes | R / T flip (micro) | R / T both-correct (macro) | R / T OA | R / T BA | T-R BA |
|---|---:|---:|---:|---:|---:|---:|
| GRDH / endpoint_excluded | 498 / 366 / 8 | 56.842369 / 61.700134 | 15.400216 / 16.237424 | 18.647088 / 26.067604 | 27.469513 / 30.631021 | +3.161508 |
| GRDH / vessel_excluded | 498 / 366 / 8 | 56.740295 / 61.502677 | 15.412413 / 16.283502 | 18.675535 / 26.231593 | 27.479206 / 30.678290 | +3.199085 |
| SLC / endpoint_excluded | 53 / 52 / 6 | 74.889937 / 74.355346 | 7.437364 / 7.657952 | 28.757862 / 30.935535 | 25.186717 / 27.949517 | +2.762800 |
| SLC / vessel_excluded | 53 / 52 / 6 | 75.078616 / 74.779874 | 7.365196 / 7.514978 | 28.545597 / 30.676101 | 25.101443 / 27.901178 | +2.799734 |

Primary GRDH T-R changes: endpoint BA +3.161508 pp, both-correct macro +0.837207 pp, top-1 flip micro +4.857764 pp. Here T improves classification while flipping more frequently; the data do not support a blanket lower-flip robustness claim. Both improvements and this limit remain visible after whole-vessel support exclusion.

The flip-event decomposition in acquisition_summary.csv separates R-only, T-only, both and neither flipping. Endpoint rescue minus harm exactly equals T-R endpoint accuracy. acquisition_classes.csv preserves rare-class denominators.

## Metadata And Limits

- Incidence and heading availability, differences, intervals and class/identity coverage are in acquisition_coverage.json and the pair manifest. Heading is the manifest image-coordinate head/tail orientation, not a verified geographic AIS bearing or sensor-relative aspect. Maximum circular discrepancy from atan2(head_x-tail_x, tail_y-head_y): 0.00000496 degrees.
- Natural revisits change multiple factors together: location, background, orientation, sea state, preprocessing and potentially annotation quality. Same vessel/class reduces identity variation but does not establish a sole physical cause. Same-product selection removes GRDH/SLC product mixing.
- A flip documents decision instability across observed acquisitions; it does not by itself prove a specific physical cue caused an error. Low flip rate can also mean consistent error; ordinary accuracy and both-correct rate must be read alongside it. T here is the existing augmented expert, not the full system or a newly fitted fusion. Fusion exploitation belongs to the main analysis.
- Two source contexts and five reused support draws are repeated predictions of the same observation pairs, not ten independent training seeds. No pseudo-seed tests, p-values or confidence intervals are reported. Class-macro uncertainty is poorly resolved for classes represented by only a few vessels; SLC additionally has singleton classes. Results are descriptive.
- Strict support exclusion is conditional on existing fitted support sets, not leave-vessel-out refitting. It does not purge source/pretraining exposure or overlap with other public releases. Repeated pairs can share ships, endpoints and scenes; the coverage is not independent replication.

## Verification And Reproduction

Verified manifest labels/identities and raw SAR timestamps against metadata; fold cache IDs/support/query masks align; probability argmax matches cached expert predictions; support size is 64; flip partition and rescue-minus-harm identities hold. These checks catch mismatched joins, cache axes, leakage and incorrect event accounting, and fail the run before artifacts are emitted.

Run acquisition_pairs.py with the existing ClearSAR Python and -B. It writes the generated CSV/JSON/report artifacts beside the script. All artifact names begin acquisition_; shared study/memory files are untouched.
