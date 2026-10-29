# private development

Completed paired orders: [17, 29, 43]; three-order complete: True.

| Arm | Phase | Seed | OA | BA | NLL | Brier |
|---|---|---|---|---|---|---|
| saved_baseline | saved | 17 | 45.625000 | 34.928108 | 2.532032 | 0.835259 |
| raw_joint | after | 17 | 45.000000 | 34.278757 | 2.527341 | 0.833786 |
| center_joint | after | 17 | 45.000000 | 34.278757 | 2.530143 | 0.834213 |
| raw_joint | sourcefit_before | 17 | 41.250000 | 33.030458 | 2.774652 | 0.893470 |
| center_joint | sourcefit_before | 17 | 41.250000 | 33.030458 | 2.774652 | 0.893470 |
| saved_baseline | saved | 29 | 45.625000 | 36.383581 | 2.243900 | 0.826444 |
| raw_joint | after | 29 | 45.625000 | 36.383581 | 2.245207 | 0.828290 |
| center_joint | after | 29 | 45.000000 | 35.734230 | 2.251801 | 0.828448 |
| raw_joint | sourcefit_before | 29 | 43.750000 | 35.036642 | 2.342279 | 0.826641 |
| center_joint | sourcefit_before | 29 | 43.750000 | 35.036642 | 2.342279 | 0.826641 |
| saved_baseline | saved | 43 | 43.750000 | 35.889301 | 2.517035 | 0.861185 |
| raw_joint | after | 43 | 43.750000 | 35.889301 | 2.512740 | 0.858739 |
| center_joint | after | 43 | 43.750000 | 35.889301 | 2.512480 | 0.857398 |
| raw_joint | sourcefit_before | 43 | 38.125000 | 31.530458 | 2.876771 | 0.959644 |
| center_joint | sourcefit_before | 43 | 38.125000 | 31.530458 | 2.876771 | 0.959644 |

| Candidate - reference | Seed | Delta OA | Delta BA | Rescue | Harm | All top1 changes | WW changes |
|---|---|---|---|---|---|---|---|
| raw_joint - saved_baseline | 17 | -0.625 | -0.6493506493506374 | 0 | 1 | 2 | 1 |
| center_joint - saved_baseline | 17 | -0.625 | -0.6493506493506374 | 0 | 1 | 4 | 3 |
| center_joint - raw_joint | 17 | 0.0 | 0.0 | 0 | 0 | 4 | 4 |
| raw_joint - saved_baseline | 29 | 0.0 | 0.0 | 0 | 0 | 1 | 1 |
| center_joint - saved_baseline | 29 | -0.625 | -0.6493506493506587 | 0 | 1 | 4 | 3 |
| center_joint - raw_joint | 29 | -0.625 | -0.6493506493506587 | 0 | 1 | 3 | 2 |
| raw_joint - saved_baseline | 43 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| center_joint - saved_baseline | 43 | 0.0 | 0.0 | 0 | 0 | 2 | 2 |
| center_joint - raw_joint | 43 | 0.0 | 0.0 | 0 | 0 | 2 | 2 |

raw_joint and center_joint both start at sourcefit and use original six-epoch adaptation with LIVE plus M3 value project/output/gain at 3e-5.
saved_baseline is the saved original complete model, not the sourcefit before endpoint. Before is receipt-level only; no sample-level before predictions exist here.
BA averages present classes. Orders share queries/initialization; repeated query events are not independent ships.
Source adapt weights use float32 N/(present_classes*class_count); per-batch CE/KL normalizes by sum(weights). Normalized mass is coefficient exposure, not gradient size or a causal explanation.
Teacher-correct counts determine KL eligibility. Useful flags are reported separately; their counts alone do not determine margin-loss gradients.
Selected reader gains without gradients are not counted as training contributions. No new final-test data is read.
