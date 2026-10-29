# Separate source-offset diagnostic

Supplied source-class/vessel-balanced offsets, fixed multiplier 1. No target mean is used in corrected predictions.
Each guard is compared to its raw parent and baseline; rank/hybrid guards also compare to CE guard.
This is not a new attention mechanism or an integrated trained candidate.
A query-common offset leaves every class-pair gap unchanged; guard pair logistic/rank accuracy equal the raw parent.
The class-centered energy ratio is not an information fraction.

| Candidate - reference | Seeds | Delta OA | Delta BA | Delta NLL | Delta Brier |
|---|---|---|---|---|---|
| ce_guard - saved_baseline | [17, 29, 43] | 0.14619883040935613 | 0.09578544061302807 | 0.0010295295024621935 | 0.00021987783399553726 |
| ce_guard - ce | [17, 29, 43] | 0.14619883040935852 | -0.045529669622114945 | -0.019864730531005126 | -0.0014034455938167028 |
| rank_guard - saved_baseline | [17, 29, 43] | -2.3684757858670005e-15 | 0.0 | -0.0035948803003214014 | -0.00207825605091431 |
| rank_guard - rank | [17, 29, 43] | -0.29239766081870994 | -0.11015091705083317 | -0.0026871920081605882 | -0.0003263686947613624 |
| hybrid_guard - saved_baseline | [17, 29, 43] | 0.14619883040935613 | 0.28735632183908183 | -0.0003169995993566784 | -0.00035488112241962205 |
| hybrid_guard - hybrid | [17, 29, 43] | 2.3684757858670005e-15 | 0.050255770990915494 | -0.017050962323883406 | -0.0010929738382737069 |
| rank_guard - ce_guard | [17, 29, 43] | -0.14619883040935852 | -0.09578544061302807 | -0.004624409802783595 | -0.0022981338849098476 |
| hybrid_guard - ce_guard | [17, 29, 43] | 0.0 | 0.19157088122605379 | -0.0013465291018188719 | -0.0005747589564151593 |
