# PCEA paired development training

All values are validation-selected; not held-out confirmation.

| Scope | Arm | Seed | LR setting | Epoch | OA | BA | Last OA | Last BA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| joint | baseline | 17 | 1e-05 | 0 | 68.421 | 62.762 | 66.228 | 61.662 |
| joint | baseline | 17 | 3e-05 | 0 | 68.421 | 62.762 | 64.474 | 58.120 |
| joint | gate | 17 | 1e-05 | 0 | 68.421 | 62.762 | 66.228 | 61.662 |
| branch | gate | 17 | 1e-05 | 0 | 68.421 | 62.762 | 67.544 | 61.374 |
| joint | gate | 17 | 3e-05 | 0 | 68.421 | 62.762 | 64.035 | 57.833 |
| joint | independent | 17 | 1e-05 | 0 | 68.421 | 62.762 | 66.228 | 61.662 |
| branch | independent | 17 | 1e-05 | 0 | 68.421 | 62.762 | 67.544 | 61.374 |
| joint | independent | 17 | 3e-05 | 0 | 68.421 | 62.762 | 64.912 | 58.394 |
| joint | pcea | 17 | 1e-05 | 0 | 68.421 | 62.762 | 66.228 | 61.662 |
| branch | pcea | 17 | 1e-05 | 0 | 68.421 | 62.762 | 67.544 | 61.374 |
| joint | pcea | 17 | 3e-05 | 0 | 68.421 | 62.762 | 64.912 | 58.401 |

## Paired Predictions

| Seed | Candidate | Control | Rescue | Harm | Net | OA delta |
|---:|---|---|---:|---:|---:|---:|
| 17 | pcea | baseline | 0 | 0 | 0 | +0.000 |
| 17 | pcea | independent | 0 | 0 | 0 | +0.000 |
| 17 | pcea | gate | 0 | 0 | 0 | +0.000 |
| 17 | gate | baseline | 0 | 0 | 0 | +0.000 |
