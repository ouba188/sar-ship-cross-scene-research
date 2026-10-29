# oss2 development

Completed paired orders: [17, 29, 43]; three-order complete: True.

| Arm | Phase | Seed | OA | BA | NLL | Brier |
|---|---|---|---|---|---|---|
| saved_baseline | saved | 17 | 57.894737 | 50.123534 | 1.397182 | 0.622810 |
| raw_joint | after | 17 | 57.017544 | 49.561736 | 1.391520 | 0.621817 |
| center_joint | after | 17 | 58.333333 | 50.404433 | 1.390507 | 0.620214 |
| raw_joint | sourcefit_before | 17 | 57.894737 | 49.918019 | 1.477916 | 0.644914 |
| center_joint | sourcefit_before | 17 | 57.894737 | 49.918019 | 1.477916 | 0.644914 |
| saved_baseline | saved | 29 | 57.894737 | 47.837598 | 1.368724 | 0.622238 |
| raw_joint | after | 29 | 57.894737 | 47.837598 | 1.368462 | 0.622193 |
| center_joint | after | 29 | 57.894737 | 47.837598 | 1.366965 | 0.622011 |
| raw_joint | sourcefit_before | 29 | 57.894737 | 48.068943 | 1.433427 | 0.634495 |
| center_joint | sourcefit_before | 29 | 57.894737 | 48.068943 | 1.433427 | 0.634495 |
| saved_baseline | saved | 43 | 59.649123 | 50.948964 | 1.378791 | 0.610698 |
| raw_joint | after | 43 | 59.649123 | 50.948964 | 1.380148 | 0.610731 |
| center_joint | after | 43 | 59.649123 | 50.948964 | 1.378991 | 0.610711 |
| raw_joint | sourcefit_before | 43 | 58.771930 | 48.444598 | 1.441703 | 0.635541 |
| center_joint | sourcefit_before | 43 | 58.771930 | 48.444598 | 1.441703 | 0.635541 |

| Candidate - reference | Seed | Delta OA | Delta BA | Rescue | Harm | All top1 changes | WW changes |
|---|---|---|---|---|---|---|---|
| raw_joint - saved_baseline | 17 | -0.877192982456144 | -0.5617977528089924 | 0 | 2 | 3 | 1 |
| center_joint - saved_baseline | 17 | 0.43859649122806843 | 0.2808988764044855 | 1 | 0 | 1 | 0 |
| center_joint - raw_joint | 17 | 1.3157894736842124 | 0.8426966292134779 | 3 | 0 | 4 | 1 |
| raw_joint - saved_baseline | 29 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| center_joint - saved_baseline | 29 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| center_joint - raw_joint | 29 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| raw_joint - saved_baseline | 43 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| center_joint - saved_baseline | 43 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| center_joint - raw_joint | 43 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |

raw_joint and center_joint both start at sourcefit and use original six-epoch adaptation with LIVE plus M3 value project/output/gain at 3e-5.
saved_baseline is the saved original complete model, not the sourcefit before endpoint. Before is receipt-level only; no sample-level before predictions exist here.
BA averages present classes. Orders share queries/initialization; repeated query events are not independent ships.
Source adapt weights use float32 N/(present_classes*class_count); per-batch CE/KL normalizes by sum(weights). Normalized mass is coefficient exposure, not gradient size or a causal explanation.
Teacher-correct counts determine KL eligibility. Useful flags are reported separately; their counts alone do not determine margin-loss gradients.
Selected reader gains without gradients are not counted as training contributions. No new final-test data is read.
