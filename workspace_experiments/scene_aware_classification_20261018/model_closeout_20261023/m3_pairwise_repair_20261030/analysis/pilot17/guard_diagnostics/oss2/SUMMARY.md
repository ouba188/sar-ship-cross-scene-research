# Separate source-offset diagnostic

Supplied source-class/vessel-balanced offsets, fixed multiplier 1. No target mean is used in corrected predictions.
Each guard is compared to its raw parent and baseline; rank/hybrid guards also compare to CE guard.
This is not a new attention mechanism or an integrated trained candidate.
A query-common offset leaves every class-pair gap unchanged; guard pair logistic/rank accuracy equal the raw parent.
The class-centered energy ratio is not an information fraction.

| Candidate - reference | Seeds | Delta OA | Delta BA | Delta NLL | Delta Brier |
|---|---|---|---|---|---|
| ce_guard - saved_baseline | [17] | 0.43859649122806843 | 0.2873563218390842 | -0.002532626992480802 | -0.0021818285982639907 |
| ce_guard - ce | [17] | 1.3157894736842124 | 1.7995215875208501 | -0.009117478391146738 | 0.003922478578388455 |
| rank_guard - saved_baseline | [17] | 0.0 | 0.0 | -0.004869374132065785 | -0.0018076101524178156 |
| rank_guard - rank | [17] | 0.0 | 0.006457445434591591 | -0.009618235438055178 | -6.752832496759176e-05 |
| hybrid_guard - saved_baseline | [17] | 0.0 | 0.0 | -0.0030379382139256172 | -0.0019455273669790607 |
| hybrid_guard - hybrid | [17] | 0.43859649122807554 | 1.224808943842696 | -0.009576295159952508 | 0.0036852085010815605 |
| rank_guard - ce_guard | [17] | -0.43859649122806843 | -0.2873563218390842 | -0.0023367471395849826 | 0.00037421844584617503 |
| hybrid_guard - ce_guard | [17] | -0.43859649122806843 | -0.2873563218390842 | -0.0005053112214448152 | 0.00023630123128492997 |
