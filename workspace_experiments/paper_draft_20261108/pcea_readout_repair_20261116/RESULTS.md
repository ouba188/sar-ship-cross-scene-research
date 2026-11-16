# PCEA readout repair: validation receipt summary

OSS2 existing validation, seed17; not an independent test. Epoch0 remains eligible.
Best and last are listed separately. A last-epoch diagnostic is not a selected gain.
bias_control is the source-trained four-class bias control with matched dual-valid eligibility and training budget.

| Arm | Best epoch | Best OA / BA | Last OA / BA |
|---|---:|---:|---:|
| coarse_raw | 1 | 68.421 / 62.762 | 67.982 / 61.662 |
| fine_raw | 2 | 68.860 / 63.042 | 67.982 / 61.662 |
| fine_std | 4 | 68.860 / 63.042 | 68.860 / 63.042 |
| fine_center_gate | 4 | 68.421 / 62.762 | 68.421 / 62.762 |
| fine_center_product | 4 | 68.421 / 62.762 | 68.421 / 62.762 |
| fine_independent | 4 | 68.860 / 63.042 | 68.860 / 63.042 |
| bias_control | 1 | 68.421 / 62.762 | 68.860 / 62.236 |

Fixed-weight diagnostics are saved separately as analysis_{arm}.json and gallery_{arm}/.
All comparison arms use the same best/last policy selected for that diagnostic arm.
Branch withdrawal and no_product probe trained computation paths, not retrained ablations.
Case selection and the fitted constant-shift approximation are descriptive validation analyses.
