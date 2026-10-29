# Development weight transplants

Completed orders: [17, 29, 43].

M3-only includes its trained M3 tensors and centering state on the saved full model; LIVE-only replaces LIVE on that model. Effects need not add under nonlinear joint inference.

| Arm | Component | Reference | Orders | Delta OA | Delta BA | Rescue/harm events |
|---|---|---|---|---|---|---|
| raw_joint | m3_only | saved_baseline | 3 | -0.20833333333333334 | -0.21645021645021956 | 0/1 |
| raw_joint | m3_only | raw_joint | 3 | 0.0 | -7.105427357601002e-15 | 1/1 |
| raw_joint | live_only | saved_baseline | 3 | -0.625 | -0.8116883116883097 | 1/4 |
| raw_joint | live_only | raw_joint | 3 | -0.4166666666666667 | -0.5952380952380972 | 1/3 |
| center_joint | m3_only | saved_baseline | 3 | -0.20833333333333334 | -0.5140692640692611 | 2/3 |
| center_joint | m3_only | center_joint | 3 | 0.20833333333333334 | -0.08116883116882907 | 3/2 |
| center_joint | live_only | saved_baseline | 3 | -0.4166666666666667 | -0.7204184704184703 | 2/4 |
| center_joint | live_only | center_joint | 3 | 0.0 | -0.2875180375180382 | 3/3 |
| center_joint | m3_only | raw_joint_m3_only | 3 | 0.0 | -0.2976190476190415 | 3/3 |
| center_joint | live_only | raw_joint_live_only | 3 | 0.20833333333333334 | 0.09126984126983946 | 4/3 |
