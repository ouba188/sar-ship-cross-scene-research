# Separate source-offset diagnostic

Supplied source-class/vessel-balanced offsets, fixed multiplier 1. No target mean is used in corrected predictions.
Each guard is compared to its raw parent and baseline; rank/hybrid guards also compare to CE guard.
This is not a new attention mechanism or an integrated trained candidate.
A query-common offset leaves every class-pair gap unchanged; guard pair logistic/rank accuracy equal the raw parent.
The class-centered energy ratio is not an information fraction.

| Candidate - reference | Seeds | Delta OA | Delta BA | Delta NLL | Delta Brier |
|---|---|---|---|---|---|
| ce_guard - saved_baseline | [17] | -0.625 | -0.2820037105751325 | -0.02225066376979523 | -0.009574483515768839 |
| ce_guard - ce | [17] | 0.625 | 0.17207792207792494 | 0.03794627945197915 | -0.009912360649809271 |
| rank_guard - saved_baseline | [17] | -0.625 | -0.5714285714285623 | -0.004172984605455454 | -0.0020810851574822875 |
| rank_guard - rank | [17] | 1.25 | 1.0969387755102034 | 0.013051196571103318 | 0.0014241667566536886 |
| hybrid_guard - saved_baseline | [17] | 0.625 | 0.5714285714285694 | -0.020178848105995684 | -0.008698018992501888 |
| hybrid_guard - hybrid | [17] | 1.25 | 0.8214285714285623 | 0.041098842324940765 | -0.005971030131786792 |
| rank_guard - ce_guard | [17] | 0.0 | -0.2894248608534298 | 0.018077679164339777 | 0.0074933983582865515 |
| hybrid_guard - ce_guard | [17] | 1.25 | 0.8534322820037019 | 0.0020718156637995477 | 0.000876464523266951 |
