# Source-fit comparison: CPU numeric report

Matched source-fit first stage only: legacy vs spatial M1/M2 with explicit M1/M2 frequency branches removed and M3 retained. Protocol: same current init.pt, 1205 source-fit observations, fixed 16 references, 8 epochs, order seeds 17/29/43; fixed 160 historical development queries. This is NOT the full intervention-adapted final system, a fresh test, or test-based selection. Repeated queries/references across seeds are not independent observations. OA/BA are percentages; deltas are percentage points. BA averages present classes only (7 in the complete development cohort). NLL uses natural logs; Brier is the sum over all 8 classes. SD is sample SD across three order seeds, not a sampling confidence interval. Controlled success does not identify a physical cause.

| Arm | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |
|---|---:|---:|---:|---:|
| legacy | 41.666667 +/- 2.953635 | 33.222892 +/- 1.943356 | 2.647204 +/- 0.282654 | 0.887192 +/- 0.063569 |
| spatial | 41.041667 +/- 2.818281 | 33.199186 +/- 1.759171 | 2.664567 +/- 0.283742 | 0.893252 +/- 0.066502 |

| Paired spatial minus legacy | Mean | SD |
|---|---:|---:|
| OA | -0.625000 | 1.082532 |
| BA | -0.023706 | 1.074824 |
| NLL | 0.017363 | 0.018482 |
| Brier | 0.006060 | 0.005816 |

| State | Seed events | Unique query IDs |
|---|---:|---:|
| rescued | 1 | 1 |
| harmed | 4 | 3 |
| CC | 196 | 85 |
| WW | 279 | 111 |

IDs rescued in one seed and harmed in another: []

Class, port and pass metrics and transitions are in groups.csv and groups_summary.csv. Empty classes have blank metrics, not zero recall; subgroup BA uses that group's present classes. Seed events can repeat a query, ship, pass and reference pair. No independence or significance claim is made.

Initial-arm metrics are in initial_metrics.csv; seed_events.csv records all logits and both initial-fixed-rival and strongest-current-rival margins. Gallery case rivals are separately specified and then held fixed across all seeds.

[Native case gallery](cases/index.html): 8 unique queries, separate VV/VH, actual same-class/rival references. Images generated; pixels NOT reviewed by this analysis. Visible facts and physical explanations remain for the main review. Paired outcomes alone do not identify physical causes.
