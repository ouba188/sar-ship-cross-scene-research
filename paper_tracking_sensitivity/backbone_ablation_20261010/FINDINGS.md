# Fixed Backbone Necessity Ablation

Completed all 25 tasks, five original 64-label support draws each. All 1,250 task/draw/mode predictions were frozen before query evaluation.

Metrics below are OA / BA (%). Ports use an equal mean over 23 tasks, then five draws. No parameter was chosen using these results.

| Mode | 24 ports (23 tasks) | F_0921 | OSS2 |
| --- | --- | --- | --- |
| base | 61.19555 / 52.49943 | 30.52639 / 26.03478 | 33.05623 / 31.27830 |
| dual_original_expert | 64.95792 / 55.37496 | 31.87458 / 27.14747 | 39.78267 / 32.60824 |
| dual_original_late | 65.57430 / 55.98728 | 33.16621 / 28.00308 | 40.76610 / 33.91470 |
| dual_original_unified | 65.66696 / 55.71262 | 33.34965 / 27.60632 | 41.07036 / 33.66770 |
| dual_refit_expert | 64.95792 / 55.37496 | 34.71387 / 26.77040 | 42.75469 / 33.93518 |
| dual_refit_late | 65.57430 / 55.98728 | 35.05335 / 27.72335 | 42.59712 / 34.67442 |
| dual_refit_unified | 65.66696 / 55.71262 | 35.04120 / 27.19600 | 42.09726 / 33.84831 |
| r50_expert | 56.46232 / 50.25693 | 30.50500 / 24.89561 | 31.05134 / 28.61074 |
| r50_late | 60.10845 / 52.18780 | 31.42281 / 25.96470 | 32.52920 / 30.25166 |
| r50_unified | 60.21204 / 52.11232 | 31.62001 / 25.67167 | 32.70307 / 30.23655 |

## Matched R18 Increment

Positive values favor dual backbones over R50-only, with both source heads refitted on the same permitted source cohort and current frozen representation. Rescue/harm counts below are summed across five overlapping draws, not unique images.

| Dataset | Head | Delta OA (pp) | Delta BA (pp) | Rescued | Harmed | Net | BA-positive draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 24 ports (23 tasks) | expert | +8.49560 | +5.11802 | 23741 | 12077 | +11664 | 5/5 |
| 24 ports (23 tasks) | late | +5.46586 | +3.79948 | 14712 | 7096 | +7616 | 5/5 |
| 24 ports (23 tasks) | unified | +5.45492 | +3.60030 | 15961 | 8698 | +7263 | 5/5 |
| F_0921 | expert | +4.20887 | +1.87479 | 84697 | 58723 | +25974 | 5/5 |
| F_0921 | late | +3.63054 | +1.75866 | 58766 | 36361 | +22405 | 5/5 |
| F_0921 | unified | +3.42119 | +1.52432 | 64607 | 43494 | +21113 | 5/5 |
| OSS2 | expert | +11.70334 | +5.32444 | 3829 | 1675 | +2154 | 5/5 |
| OSS2 | late | +10.06792 | +4.42276 | 3064 | 1211 | +1853 | 5/5 |
| OSS2 | unified | +9.39419 | +3.61176 | 3225 | 1496 | +1729 | 5/5 |

## Prior Refit Control

Dual refit minus original dual separates the source-prior update from SSL18 removal. Port priors replay exactly; external priors move from pre-tail source fitting to the cached joint-tail deployment representation.

| Dataset | Head | Delta OA (pp) | Delta BA (pp) |
| --- | --- | --- | --- |
| 24 ports (23 tasks) | expert | +0.00000 | +0.00000 |
| 24 ports (23 tasks) | late | +0.00000 | +0.00000 |
| 24 ports (23 tasks) | unified | +0.00000 | +0.00000 |
| F_0921 | expert | +2.83930 | -0.37707 |
| F_0921 | late | +1.88714 | -0.27973 |
| F_0921 | unified | +1.69155 | -0.41032 |
| OSS2 | expert | +2.97202 | +1.32694 |
| OSS2 | late | +1.83102 | +0.75972 |
| OSS2 | unified | +1.02689 | +0.18061 |

## Per-Class Explanation

The following are each class contribution to the equal-task BA increment of adding R18 in late fusion. An absent class contributes zero for that task. Full per-class recalls, support counts, rescues and harms are in class_metrics.csv, class_effects.csv, and class_contributions.csv.

| Dataset | Class ID | BA contribution (pp) | Rescued | Harmed | Net |
| --- | --- | --- | --- | --- | --- |
| 24 ports (23 tasks) | 5 | -0.01603 | 621 | 418 | +203 |
| 24 ports (23 tasks) | 7 | +0.12143 | 564 | 322 | +242 |
| 24 ports (23 tasks) | 4 | +0.24897 | 816 | 429 | +387 |
| 24 ports (23 tasks) | 6 | +0.31706 | 632 | 251 | +381 |
| 24 ports (23 tasks) | 3 | +0.50145 | 3639 | 1497 | +2142 |
| 24 ports (23 tasks) | 1 | +0.56522 | 1059 | 264 | +795 |
| 24 ports (23 tasks) | 2 | +0.87292 | 2703 | 1905 | +798 |
| 24 ports (23 tasks) | 0 | +1.18846 | 4678 | 2010 | +2668 |
| F_0921 | 1 | -0.67466 | 522 | 1547 | -1025 |
| F_0921 | 7 | -0.57760 | 225 | 704 | -479 |
| F_0921 | 0 | -0.57369 | 5915 | 12539 | -6624 |
| F_0921 | 2 | -0.09658 | 1842 | 2134 | -292 |
| F_0921 | 5 | +0.51694 | 6669 | 3787 | +2882 |
| F_0921 | 4 | +0.86244 | 19137 | 8574 | +10563 |
| F_0921 | 6 | +1.01244 | 9057 | 3651 | +5406 |
| F_0921 | 3 | +1.28935 | 15399 | 3425 | +11974 |
| OSS2 | 2 | -1.44487 | 77 | 257 | -180 |
| OSS2 | 5 | -0.59643 | 39 | 74 | -35 |
| OSS2 | 6 | -0.06301 | 13 | 15 | -2 |
| OSS2 | 7 | +0.28409 | 19 | 13 | +6 |
| OSS2 | 0 | +0.69783 | 626 | 388 | +238 |
| OSS2 | 1 | +0.89410 | 259 | 87 | +172 |
| OSS2 | 3 | +2.04447 | 812 | 155 | +657 |
| OSS2 | 4 | +2.60657 | 1219 | 222 | +997 |

## Timing and Resource Boundary

- Two CPU threads per process. Total measured task time: 81.38 s; source input preparation plus both head fits: 10.31 s; shared construction of both expert kernels: 52.53 s.
- Per-draw cached-kernel adaptation times include solving and scoring all target rows; the unified timing also includes weighted kernel/prior assembly. They exclude backbone/image forward, image loading, extraction, source fitting, and feature-to-support kernel construction. Concurrent other experiments can affect wall time. This is not an isolated hardware throughput benchmark.
- Mean R50 two-head solve/score/readout: 5.197 ms; mean R50 unified solve/score: 6.066 ms.
- R50 expert retains 128 PCA and 2048 complementary coordinates. The removed SSL branch has 64 PCA and 512 complementary coordinates. Original divisors sqrt(192), sqrt(2560), source stable scaling, and all surviving feature scales are held fixed. This is a drop-only metric, not a search over single-backbone normalization.
- Each source coefficient matrix has 8 output columns. Each support residual coefficient matrix is 64x8; two heads solve two such systems, one unified head solves one. No backbone parameter/FLOP reduction or end-to-end acceleration was measured.
- One R50 backbone does NOT mean one forward: the unchanged full base retains C4 four-rotation forwards and the frozen frequency representation. Removing the SSL18 branch does not establish half the total compute. The concurrent C4 input ablation is a separate intervention.
- Prediction and small coefficient archives are saved. No feature copies, new model downloads, or old-result modifications.

## Verification and Limits

- Exact base, original expert, original equal-probability fusion and original unified predictions replay on every task and draw. Original expert scores match the existing tolerance of 3e-5 absolute / 1e-6 relative.
- Source rows equal the base source cohort; target ports or previously purged external identity rows are excluded. Source heads are balanced ridge1 with unpenalized intercept; target residuals use the original class-balanced ridge8 and equal fusion.
- ExpertContext helpers load the full source manifest, including labels for excluded rows. The fit receives only the permitted source slice. Target fits receive the stored 64 support labels. No claim is made that all label-containing files remained unopened.
- No source-episode hyperparameter selection or new source OOF gate was needed: rho=.5 and lambda=8 were frozen. Geometry and backbone weights remain inherited.
- The R50-only expert uses the original R50 directional and complementary information. The full base still uses its complete frozen condition/frequency representation. This experiment does not retrain a new end-to-end single model or exhaust single-model architectures.
- F_0921 was historically exposed to the SSL pretraining corpus; source head identity exclusion does not undo that exposure. Overlapping support draws are not independent replications.
- The concurrently supplied fusion_module_ablation_20261010 result motivates reading redundancy/blocking together with these contrasts; it did not change parameters, modes, or task selection. This ablation alone cannot identify the mechanism behind any R18 gain or loss.
