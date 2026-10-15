# Source-metric local residual expert: not promoted

## Result

The module and its two component ablations are implemented and the fixed
23-port/five-support experiment is complete. None improves the unchanged
linear expert at the primary5% handoff budget. Keep the main model unchanged;
do not sweep kernel widths or regularization against these target results.

Output: `../oof_local_metric_expert_results_20261007`.
Protocol: `LOCAL_METRIC_EXPERT_PROTOCOL_20261007.md`.
23 source covariance fits,345 new64-label support fits,1265 policy-metric
rows,21.29 seconds. No new backbone, PCA or original source-head fits.

## Equal-port BA (%)

| Expert / policy | Full-query expert | 5% handoff |
| --- | ---: | ---: |
| Existing linear residual | 52.894867 | 45.846390 |
| Source metric + linear residual | 52.837358 | 45.671809 |
| Local RBF residual, original geometry | 52.623730 | 45.803571 |
| Source metric + local RBF (candidate) | 52.563936 | 45.801135 |
| Existing bias-only residual | 51.756774 | 45.802450 |
| Previous action-advantage policy | not a new expert | 45.885986 |

At5%, candidate minus unchanged linear is -0.045256 percentage points:
10 positive/13 negative ports, descriptive port-bootstrap95% interval
[-0.237017,0.159363]pp. Four of five support-draw means are negative.
Accuracy also decreases0.042497pp. Mean admission fraction is identical
4.932516% for all arms, with integer5% cap rounding.

The candidate loses0.002437pp to isotropic RBF and gains0.129326pp over
metric-linear. The latter comparator is itself worse than the original;
it does not demonstrate a useful overall module. The candidate also loses
0.001315pp to bias-only and0.084851pp to the prior action-advantage policy.
Primary screen and the two-component-positive flag are bothFALSE.

## What The Module Does

1. Estimate source within-class covariance in the192 cached visual features
   with LedoitWolf shrinkage. Whiten using its inverse square root, then
   normalize SOURCE average feature energy to match the original geometry.
2. Form a support-local residual kernel1+exp(-distance_squared/h). Its width
   is the median of512 disjoint SOURCE-row pair distances, fixedseed1707.
3. Fit the original +/-1 class-score residual around the unchanged source
   classifier using the same64 target labels, class-balanced support weights
   and strength8. The constant kernel retains a penalized globalbias term.
4. Reuse the original frozen confidence gate, recalculated for each actual
   receiver, with unchanged observed-class guard and5% admission cap.

These are existing statistical components in a distinct local combination,
not an invented kernel/metric principle or established journal-level novelty.
Covariance is row-weighted across source classes/ports. Matching source
feature energy does not match effective regularization or model capacity
between kernels. The negative result does not exclude all local experts.

## Why This Does Not Pass

Full-query candidate BA also falls0.330931pp versus linear, with only5/23
ports improving; accuracy falls1.476382pp. Thus the present failure cannot
be explained solely by a bad budget selector hiding a stronger full expert.

The candidate recovers0.874676 BA points from cases where BOTH base and old
linear expert were wrong, but harms2.207332 BA points of previously correct
linear predictions. All recoveries relative to linear, including cases base
already gets right, total1.876401pp. Recovery minus harm gives the observed
-0.330931pp full-query difference. It changes genuine class decisions beyond
the original action pair, but the tradeoff is unfavorable in this configuration.

Full-query Mombasa improves4.604757pp, while Port Hedland loses2.054730pp,
Fujairah2.042559pp and Rotterdam1.873404pp. At5%, Mombasa improves1.556777pp
but Newcastle loses1.349468pp. Do not select per-port winners after readout.

In saved budget metrics, `linear_harm_ba` compares the final policy with the
FULL-query linear receiver, not the budgeted linear policy. It includes
missed opportunities from base fallback and must not be called incremental
harm of the new budgeted module. The paired BA comparisons use the correct
matched budget controls throughout.

## Verification

Five focused tests passed: independent primal/dual linear equivalence,
source-only geometry and energy normalization, positive kernel/local decay,
zero residual under an exact prior, and representational ability on nonlinear
synthetic data. The synthetic test does not establish held-out generalization.
Its initial argmax-accuracy assertion was replaced before the experiment by
the correct score-tie assertion; tiny ties cannot support a fixed argmax rate.

Independent replay completed in10.82seconds via
`verify_local_metric_expert.py`; output `VERIFICATION.json` recordsPASS.
All345 weighted dual fits,575 gate/mask policies,345 unchanged budget control
policies and1265 metric rows reproduce. Maximum score error2.83e-15; maximum
gate error6.00e-15. Geometry was independently refit for Antwerp-Bruges and
Mombasa; source bandwidth pair selections replay for all23 ports. Remaining
geometry transforms and upstream PCA/source-head exports are trusted inputs.
The verifier used NumPy1.26.4; an explicit float64 conversion after the
inherited float32 standardization reproduced the producer's promotion
semantics, resolving a6.7e-8 initial bandwidth difference without loosening
tolerances or changing experiment results.

All targets are development-exposed. Target support/query is row-disjoint
only; the inherited source head purges are a different boundary. All query
expert scores precede admission, so this is not a compute-budget speedup.
No manuscript or primary model has been upgraded based on this result.

## Native-scale Readiness Check (No New Experiment)

The old source-only descriptor probe at
`E:/临时会话/visual_reliable_baseline/artifacts/source_native_scale_probe_20260914/READOUT.md`
reported+2.1357pp from native crop scale on a different16741-row source
cohort. That is motivation, not evidence for the current eight-class model.

The current38091-row manifest has12435 missing entries in EACH of the six
native geometry fields (VV/VH crop side, box width, box height):32.645507%.
Observed crop sides span64..128 pixels with9 unique values; valid observed
widths/heights are positive. Origins are21195 legacy and16896 returned rows.
No missing dimensions were imputed, no rows dropped, no physical-meter
interpretation assigned, and no scale-based candidate trained this turn.
Related native-geometry experts already exist in `bdh_pixel_loss_controls`;
their provenance and results must be checked before declaring another new
geometry module. No follow-up job is silently queued.
