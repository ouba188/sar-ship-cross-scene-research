# Posthoc pruning: training-path diagnostic

Optional posthoc path-dependency diagnostic, NOT a third fairly trained arm. Pruned predictions use each already-trained legacy model with explicit M1/M2 frequency branches removed by conversion, retained weights and M3 unchanged, and no refit. Pruned minus legacy measures removal after legacy-path training. Spatial minus pruned compares spatial-path refitting with removal after legacy-path training; it does not isolate a pure frequency effect or a new matched three-arm training experiment. The main legacy-versus-spatial summaries remain separate and unchanged. Matched source-fit first stage only: legacy vs spatial M1/M2 with explicit M1/M2 frequency branches removed and M3 retained. Protocol: same current init.pt, 1205 source-fit observations, fixed 16 references, 8 epochs, order seeds 17/29/43; fixed 160 historical development queries. This is NOT the full intervention-adapted final system, a fresh test, or test-based selection. Repeated queries/references across seeds are not independent observations. OA/BA are percentages; deltas are percentage points. BA averages present classes only (7 in the complete development cohort). NLL uses natural logs; Brier is the sum over all 8 classes. SD is sample SD across three order seeds, not a sampling confidence interval. Controlled success does not identify a physical cause.

Positive OA/BA deltas favor the alternative; negative NLL/Brier deltas favor the alternative. Means and sample SD describe three paired order seeds, not independent datasets.

| Comparison | Metric | Reference mean +/- SD | Alternative mean +/- SD | Paired delta mean +/- SD |
|---|---|---:|---:|---:|
| pruned_vs_legacy | OA | 41.666667 +/- 2.953635 | 41.666667 +/- 2.953635 | 0.000000 +/- 0.000000 |
| pruned_vs_legacy | BA | 33.222892 +/- 1.943356 | 33.222892 +/- 1.943356 | 0.000000 +/- 0.000000 |
| pruned_vs_legacy | NLL | 2.647204 +/- 0.282654 | 2.646117 +/- 0.281789 | -0.001088 +/- 0.001026 |
| pruned_vs_legacy | Brier | 0.887192 +/- 0.063569 | 0.886815 +/- 0.063469 | -0.000377 +/- 0.000318 |
| spatial_vs_pruned | OA | 41.666667 +/- 2.953635 | 41.041667 +/- 2.818281 | -0.625000 +/- 1.082532 |
| spatial_vs_pruned | BA | 33.222892 +/- 1.943356 | 33.199186 +/- 1.759171 | -0.023706 +/- 1.074824 |
| spatial_vs_pruned | NLL | 2.646117 +/- 0.281789 | 2.664567 +/- 0.283742 | 0.018451 +/- 0.017976 |
| spatial_vs_pruned | Brier | 0.886815 +/- 0.063469 | 0.893252 +/- 0.066502 | 0.006437 +/- 0.005601 |

| Comparison | State | Seed events | Unique query IDs |
|---|---|---:|---:|
| pruned_vs_legacy | rescued | 0 | 0 |
| pruned_vs_legacy | harmed | 0 | 0 |
| pruned_vs_legacy | CC | 200 | 86 |
| pruned_vs_legacy | WW | 280 | 111 |
| spatial_vs_pruned | rescued | 1 | 1 |
| spatial_vs_pruned | harmed | 4 | 3 |
| spatial_vs_pruned | CC | 196 | 85 |
| spatial_vs_pruned | WW | 279 | 111 |

pruned_vs_legacy: IDs rescued in one seed and harmed in another: []

All top-1 changes (including WW changes): 0 seed events / 0 unique IDs.


spatial_vs_pruned: IDs rescued in one seed and harmed in another: []

All top-1 changes (including WW changes): 10 seed events / 8 unique IDs.

No new checkpoint, hyperparameter or test selection is performed. These score comparisons do not identify a physical cause. Per-query logits, margins and exact seed/ID transitions are in seed_events.csv.
