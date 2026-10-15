# Source-Only Single-Backbone Reconstruction

All25 tasks, five fixed64-label draws; final checkpoint and both mapper architectures were frozen before target classification evaluation.

No SSL18 execution is required for these student inputs at inference. Original-view R50 and its existing C4 mean feed a fixed linear mapper or a 2,360,320-parameter MLP. The teacher is required during source training. End-to-end student latency has not been measured.

## Classification

| Mode | 24 ports OA / BA | F_0921 OA / BA | OSS2 OA / BA |
| --- | ---: | ---: | ---: |
| r50_expert | 56.46232 / 50.25693 | 30.50500 / 24.89561 | 31.05134 / 28.61074 |
| r50_late | 60.10845 / 52.18780 | 31.42281 / 25.96470 | 32.52920 / 30.25166 |
| r50_unified | 60.21204 / 52.11232 | 31.62001 / 25.67167 | 32.70307 / 30.23655 |
| linear_expert | 62.80878 / 53.77718 | 27.80020 / 22.16001 | 35.80549 / 30.12425 |
| linear_late | 63.54895 / 54.26512 | 29.52271 / 23.85217 | 36.65308 / 32.12141 |
| linear_unified | 63.97368 / 54.22659 | 30.14414 / 23.84372 | 37.35398 / 31.50134 |
| mlp_expert | 63.43291 / 54.38296 | 28.84602 / 22.80959 | 34.92529 / 28.94500 |
| mlp_late | 64.13735 / 55.04905 | 30.37180 / 24.65328 | 35.90329 / 31.00523 |
| mlp_unified | 64.45296 / 54.83677 | 30.98238 / 24.39691 | 36.10975 / 30.31493 |
| dual_refit_expert | 64.95792 / 55.37496 | 34.71387 / 26.77040 | 42.75469 / 33.93518 |
| dual_refit_late | 65.57430 / 55.98728 | 35.05335 / 27.72335 | 42.59712 / 34.67442 |
| dual_refit_unified | 65.66696 / 55.71262 | 35.04120 / 27.19600 | 42.09726 / 33.84831 |

## Transfer of Teacher Features

R2 is variance-weighted across source-standardized teacher coordinates, using the target cohort mean only in the diagnostic denominator. It is not class accuracy, causal information or a measure of physical mechanisms. Source MSE is in-sample.

| Dataset | Mapper | Source MSE | Target MSE | Target R2 | Centered cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| self24ports | linear | 0.28065 | 0.39603 | 0.50686 | 0.79174 |
| self24ports | mlp | 0.20372 | 0.35357 | 0.56800 | 0.81639 |
| f0921 | linear | 0.27623 | 7.57293 | -1.53419 | 0.46909 |
| f0921 | mlp | 0.22267 | 5.56558 | -0.86245 | 0.49119 |
| oss2 | linear | 0.27838 | 3.43579 | 0.01848 | 0.54565 |
| oss2 | mlp | 0.20376 | 3.35372 | 0.04192 | 0.55694 |

## Boundaries

- Original base predictions replay exactly for every sample/draw. Original matched-refit teacher expert predictions replay on every support-union sample before replacement.
- Source geometry, retained/complement normalization and 24-port stable/drift/angle operators are unchanged. A fresh source classifier is fitted on student features to avoid training/deployment feature mismatch.
- Mapper source rows equal the permitted source cohort; no target query or support labels train the mapper. All variants share the original64 target labels for residual adaptation.
- The GPU fitter loads inherited sklearn objects but never calls their transform/predict methods. Geometry prediction uses the original ClearSAR sklearn1.8.0 environment. Mapper CPU/GPU inference agrees on source probes.
- A failed linear/MLP reconstruction cannot establish that no single network can obtain the information. A successful student would support source-learned inference compression, not teacher-free training.
- Inherited development reuse and historical F_0921 SSL exposure remain. Five support draws are not five neural seeds or independent confirmation.
