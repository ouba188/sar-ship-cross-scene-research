# Development weight transplants

Completed orders: [17, 29, 43].

M3-only includes its trained M3 tensors and centering state on the saved full model; LIVE-only replaces LIVE on that model. Effects need not add under nonlinear joint inference.

| Arm | Component | Reference | Orders | Delta OA | Delta BA | Rescue/harm events |
|---|---|---|---|---|---|---|
| raw_joint | m3_only | saved_baseline | 3 | -0.29239766081871466 | -0.18726591760299746 | 0/2 |
| raw_joint | m3_only | raw_joint | 3 | 0.0 | 0.0 | 0/0 |
| raw_joint | live_only | saved_baseline | 3 | 0.0 | 0.0 | 0/0 |
| raw_joint | live_only | raw_joint | 3 | 0.29239766081871466 | 0.18726591760299746 | 2/0 |
| center_joint | m3_only | saved_baseline | 3 | -0.14619883040935852 | -0.09578544061302807 | 1/2 |
| center_joint | m3_only | center_joint | 3 | -0.29239766081871466 | -0.18941839941452324 | 0/2 |
| center_joint | live_only | saved_baseline | 3 | 0.14619883040935613 | 0.09578544061302334 | 2/1 |
| center_joint | live_only | center_joint | 3 | 0.0 | 0.002152481811528162 | 2/2 |
| center_joint | m3_only | raw_joint_m3_only | 3 | 0.14619883040935613 | 0.09148047698996938 | 2/1 |
| center_joint | live_only | raw_joint_live_only | 3 | 0.14619883040935613 | 0.09578544061302334 | 2/1 |
