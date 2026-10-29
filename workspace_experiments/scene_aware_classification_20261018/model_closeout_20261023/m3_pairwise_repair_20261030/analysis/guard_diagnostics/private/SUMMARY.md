# Separate source-offset diagnostic

Supplied source-class/vessel-balanced offsets, fixed multiplier 1. No target mean is used in corrected predictions.
Each guard is compared to its raw parent and baseline; rank/hybrid guards also compare to CE guard.
This is not a new attention mechanism or an integrated trained candidate.
A query-common offset leaves every class-pair gap unchanged; guard pair logistic/rank accuracy equal the raw parent.
The class-centered energy ratio is not an information fraction.

| Candidate - reference | Seeds | Delta OA | Delta BA | Delta NLL | Delta Brier |
|---|---|---|---|---|---|
| ce_guard - saved_baseline | [17, 29, 43] | -0.625 | -0.3784786641929519 | -0.014717934016301873 | -0.006164678491216306 |
| ce_guard - ce | [17, 29, 43] | 0.4166666666666667 | 0.32106782106781867 | 0.023201652661494876 | -0.009343530930694547 |
| rank_guard - saved_baseline | [17, 29, 43] | -0.625 | -1.083333333333331 | -0.0004426088917530713 | -0.0007108024299447383 |
| rank_guard - rank | [17, 29, 43] | -0.20833333333333334 | -0.2976190476190486 | 0.008020618870349328 | 0.004310969226039944 |
| hybrid_guard - saved_baseline | [17, 29, 43] | 0.0 | -0.025974025974029757 | -0.013181505829605348 | -0.005693462809717091 |
| hybrid_guard - hybrid | [17, 29, 43] | 0.20833333333333334 | -0.04978354978355526 | 0.031715996500714105 | -0.003620810977865374 |
| rank_guard - ce_guard | [17, 29, 43] | 0.0 | -0.7048546691403791 | 0.014275325124548802 | 0.0054538760612715675 |
| hybrid_guard - ce_guard | [17, 29, 43] | 0.625 | 0.35250463821892214 | 0.0015364281866965257 | 0.0004712156814992152 |
