# Frozen OSS2 query OBB robustness

Full fixed weights, seeds 17/29/43, 378 paired queries, 32 unchanged source references. No training.
M2-only changes query local_slots/slots. M3 descriptors/masses/writes and its delta remain clean.
This run is M2-only. Pipeline sensitivity is not evaluated.

| Mode | Condition | OA | BA | OA change | BA change | Rescue/harm events |
|---|---|---:|---:|---:|---:|---:|
| m2 | across_m05 | 61.111 +/- 2.066 | 55.730 +/- 3.106 | +0.088 | +0.100 | 3/2 |
| m2 | across_m10 | 61.464 +/- 2.219 | 56.018 +/- 3.190 | +0.441 | +0.388 | 8/3 |
| m2 | across_m20 | 61.728 +/- 1.457 | 55.681 +/- 2.020 | +0.705 | +0.051 | 16/8 |
| m2 | across_p05 | 60.758 +/- 2.155 | 55.151 +/- 2.810 | -0.265 | -0.479 | 4/7 |
| m2 | across_p10 | 60.406 +/- 1.761 | 54.910 +/- 2.532 | -0.617 | -0.720 | 5/12 |
| m2 | across_p20 | 60.406 +/- 1.761 | 54.870 +/- 2.450 | -0.617 | -0.760 | 11/18 |
| m2 | along_m05 | 61.111 +/- 1.908 | 55.401 +/- 1.056 | +0.088 | -0.229 | 16/15 |
| m2 | along_m10 | 61.464 +/- 2.155 | 56.236 +/- 1.391 | +0.441 | +0.606 | 28/23 |
| m2 | along_m20 | 62.346 +/- 0.850 | 57.338 +/- 0.776 | +1.323 | +1.708 | 51/36 |
| m2 | along_p05 | 61.199 +/- 2.219 | 56.795 +/- 2.043 | +0.176 | +1.165 | 14/12 |
| m2 | along_p10 | 60.494 +/- 1.222 | 56.306 +/- 1.298 | -0.529 | +0.676 | 25/31 |
| m2 | along_p20 | 60.317 +/- 1.997 | 55.678 +/- 2.680 | -0.705 | +0.048 | 42/50 |
| m2 | angle_m05 | 60.847 +/- 2.306 | 55.246 +/- 2.897 | -0.176 | -0.384 | 5/7 |
| m2 | angle_m10 | 61.111 +/- 1.852 | 56.173 +/- 1.607 | +0.088 | +0.543 | 11/10 |
| m2 | angle_m20 | 61.464 +/- 2.003 | 56.936 +/- 2.451 | +0.441 | +1.306 | 28/23 |
| m2 | angle_p05 | 61.023 +/- 2.138 | 55.115 +/- 2.783 | +0.000 | -0.515 | 9/9 |
| m2 | angle_p10 | 60.935 +/- 1.358 | 54.876 +/- 2.265 | -0.088 | -0.754 | 18/19 |
| m2 | angle_p20 | 61.993 +/- 1.761 | 55.768 +/- 1.306 | +0.970 | +0.138 | 36/25 |
| m2 | length_m10 | 60.317 +/- 1.609 | 56.327 +/- 1.120 | -0.705 | +0.697 | 15/23 |
| m2 | length_m20 | 59.524 +/- 1.212 | 56.324 +/- 1.035 | -1.499 | +0.694 | 28/45 |
| m2 | length_p10 | 61.552 +/- 1.595 | 55.331 +/- 0.744 | +0.529 | -0.299 | 14/8 |
| m2 | length_p20 | 62.434 +/- 1.997 | 56.061 +/- 1.236 | +1.411 | +0.432 | 34/18 |
| m2 | scale_m10 | 60.141 +/- 1.305 | 56.691 +/- 1.275 | -0.882 | +1.061 | 22/32 |
| m2 | scale_m20 | 58.995 +/- 1.473 | 56.057 +/- 0.583 | -2.028 | +0.427 | 37/60 |
| m2 | scale_p10 | 61.199 +/- 1.193 | 55.070 +/- 0.717 | +0.176 | -0.560 | 19/17 |
| m2 | scale_p20 | 62.610 +/- 0.850 | 56.221 +/- 0.379 | +1.587 | +0.591 | 53/35 |
| m2 | width_m10 | 60.758 +/- 2.122 | 55.977 +/- 1.880 | -0.265 | +0.347 | 15/18 |
| m2 | width_m20 | 60.141 +/- 2.569 | 55.369 +/- 2.452 | -0.882 | -0.261 | 18/28 |
| m2 | width_p10 | 60.935 +/- 1.069 | 55.367 +/- 1.189 | -0.088 | -0.263 | 9/10 |
| m2 | width_p20 | 60.317 +/- 0.529 | 55.245 +/- 1.060 | -0.705 | -0.385 | 20/28 |
| m2 | zero | 61.023 +/- 2.610 | 55.630 +/- 3.498 | +0.000 | +0.000 | 0/0 |

Events count record x seed, not independent vessels; no test-selected preferred perturbation.
Area/coverage are relative to the supplied clean OBB, not segmentation ground truth. Zero mask change is not evidence of robustness.
Cases selected by cross-seed recurrence then manifest order, both signed extreme errors inspected. Examples do not estimate prevalence.
Panels: original display PNG / unchanged native160 model input / clean green and perturbed red OBB. No rescaling.
Source examples are first legal same-class/confusion-class members of the actual bank, not largest-contribution references.
Raw-chip review COMPLETED: 15 representative panels were directly inspected, covering rescued, harmed, stable-correct and stable-wrong cases. Observations, clean source-reference evidence and interpretation limits are recorded in FINDINGS_ZH.md. Appearance alone does not prove physical causality.

Net change is not total prediction turnover. Along +20%: 42 rescues / 50 harms, net -8 correct events, but 127 total flips (35 wrong-to-wrong). Scale -20%: 37/60, net -23, 138 flips. Angle +5 degrees: zero net OA change still contains 24 flips (9 rescues / 9 harms).
Scale -20% exchanges class recall: mean container -7.056 pp versus chemical +8.434 pp; scale +20% reverses this (-8.032 pp chemical / +6.083 pp container). Small aggregate curves must not hide these class tradeoffs.
Auxiliary audit correction: outside_footprint_px was independently recomputed for all 67,518 region records after finding an erroneous broadcast sum. Negative values fell from 44,926 to zero. Independent native area/change/coverage/IoU checks agree within 5e-13; masks, predictions and all performance tables are unchanged. See auxiliary_audit_fix.json and repair_mask_audit.py. The original audit is retained locally.
