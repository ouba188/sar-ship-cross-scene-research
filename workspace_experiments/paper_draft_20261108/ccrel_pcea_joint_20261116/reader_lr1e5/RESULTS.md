# Joint reader / PCEA validation summary

Existing OSS2 validation, seed17; not an independent test or a new whole architecture.
The trained read projection is a separate copy. Fixed-weight withdrawals diagnose paths, not retrained ablations.
Best includes epoch0. Last is reported separately and is not a selected improvement.

| Arm | Best epoch | Initial OA / BA | Best OA / BA | Last OA / BA |
|---|---:|---:|---:|---:|---:|
| joint | 1 | 68.421 / 62.762 | 68.421 / 62.762 | 68.421 / 62.749 |
| joint_fine | 1 | 68.421 / 62.762 | 68.421 / 62.762 | 67.544 / 61.374 |
| independent_fine | 1 | 68.421 / 62.762 | 68.421 / 62.762 | 67.982 / 61.662 |

Detailed outputs: analysis_{arm}.json, analysis_{arm}_scores.npz, gallery_{arm}/.
Best epoch0 automatically selects trained last for diagnosis, not as a validation-selected gain.
All last diagnostics use a _last suffix and do not overwrite best diagnostics.
Controls use the same best/last endpoint policy, with each actual selected epoch recorded.
No source recalibration, fitting, or independent-test evaluation occurs in this analysis.
