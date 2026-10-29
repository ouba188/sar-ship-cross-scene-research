# private query interaction development analysis

Queries: 160; available seeds by arm: {'ce': [17, 29, 43], 'query_only': [17, 29, 43], 'ref_only': [17, 29, 43], 'interaction': [17, 29, 43]}.


| Arm - reference | Orders | Mean delta OA | SD OA | Mean delta BA | SD BA | Mean delta NLL | Mean delta Brier |
|---|---|---|---|---|---|---|---|
| query_only - saved_baseline | [17, 29, 43] | -0.8333333333333334 | 2.1949278651776543 | -0.37595341166769697 | 0.8763365608881154 | -0.0348099434075948 | 0.005396685474477543 |
| query_only - ce | [17, 29, 43] | 0.20833333333333334 | 0.7216878364870323 | 0.32359307359307365 | 0.8004174186492576 | 0.0031096432702019485 | 0.0022178330349993023 |
| ref_only - saved_baseline | [17, 29, 43] | -0.625 | 1.875 | -0.18547722119150478 | 0.5464768164023474 | -0.03946959238205494 | 0.005329338990089287 |
| ref_only - ce | [17, 29, 43] | 0.4166666666666667 | 0.36084391824351614 | 0.5140692640692658 | 0.4879248317924932 | -0.0015500057042581912 | 0.002150486550611045 |
| interaction - saved_baseline | [17, 29, 43] | -0.4166666666666667 | 2.1949278651776543 | 0.12744794887652233 | 0.825333466729988 | -0.0267808754320793 | 0.006998866381137299 |
| interaction - ce | [17, 29, 43] | 0.625 | 0.625 | 0.8269944341372929 | 0.7771566607265225 | 0.01113871124571745 | 0.003820013941659058 |
| interaction - query_only | [17, 29, 43] | 0.4166666666666667 | 0.36084391824351614 | 0.5034013605442192 | 0.11782658554890925 | 0.008029067975515503 | 0.0016021809066597559 |
| interaction - ref_only | [17, 29, 43] | 0.20833333333333334 | 0.3608439182435161 | 0.3129251700680271 | 0.2895754646774543 | 0.012688716949975642 | 0.0016695273910480128 |

| Arm - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm | CC | WW |
|---|---|---|---|---|---|---|---|---|---|
| query_only - saved_baseline | 17 | -0.625 | 0.11734693877551905 | -0.042912275787212906 | 0.005531967219505218 | 2 | 3 | 70 | 85 |
| query_only - ce | 17 | 0.625 | 0.5714285714285765 | 0.017284667434561474 | 0.0051940900854647865 | 1 | 0 | 71 | 88 |
| ref_only - saved_baseline | 17 | -0.625 | 0.11734693877551905 | -0.04802718337320311 | 0.005315543386300803 | 2 | 3 | 70 | 85 |
| ref_only - ce | 17 | 0.625 | 0.5714285714285765 | 0.012169759848571271 | 0.004977666252260371 | 1 | 0 | 71 | 88 |
| interaction - saved_baseline | 17 | -0.625 | 0.4846938775510239 | -0.03481186176196571 | 0.009028819025796442 | 2 | 3 | 70 | 85 |
| interaction - ce | 17 | 0.625 | 0.9387755102040813 | 0.025385081459808667 | 0.00869094189175601 | 2 | 1 | 70 | 87 |
| interaction - query_only | 17 | 0.0 | 0.36734693877550484 | 0.008100414025247193 | 0.0034968518062912235 | 1 | 1 | 71 | 87 |
| interaction - ref_only | 17 | 0.0 | 0.36734693877550484 | 0.013215321611237396 | 0.003713275639495639 | 1 | 1 | 71 | 87 |
| query_only - saved_baseline | 29 | -3.125 | -1.3877551020408205 | 0.04481711476619843 | 0.037471656971497014 | 0 | 5 | 68 | 87 |
| query_only - ce | 29 | -0.625 | -0.5714285714285765 | 0.005933443082632106 | 0.003033920269808421 | 0 | 1 | 68 | 91 |
| ref_only - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.03926611465525198 | 0.036373876205516575 | 0 | 4 | 69 | 87 |
| ref_only - ce | 29 | 0.0 | 0.0 | 0.0003824429716856592 | 0.0019361395038279827 | 0 | 0 | 69 | 91 |
| interaction - saved_baseline | 29 | -2.5 | -0.816326530612244 | 0.05126799931007131 | 0.038632627227038085 | 0 | 4 | 69 | 87 |
| interaction - ce | 29 | 0.0 | 0.0 | 0.012384327626504987 | 0.004194890525349493 | 0 | 0 | 69 | 91 |
| interaction - query_only | 29 | 0.625 | 0.5714285714285765 | 0.006450884543872881 | 0.0011609702555410717 | 1 | 0 | 68 | 91 |
| interaction - ref_only | 29 | 0.0 | 0.0 | 0.012001884654819328 | 0.00225875102152151 | 0 | 0 | 69 | 91 |
| query_only - saved_baseline | 43 | 1.25 | 0.14254792826221063 | -0.10633466920176993 | -0.0268135677675696 | 7 | 5 | 65 | 83 |
| query_only - ce | 43 | 0.625 | 0.970779220779221 | -0.013889180706587734 | -0.0015745112502753011 | 2 | 1 | 70 | 87 |
| ref_only - saved_baseline | 43 | 1.25 | 0.14254792826221063 | -0.1096477084282137 | -0.02570140262154952 | 7 | 5 | 65 | 83 |
| ref_only - ce | 43 | 0.625 | 0.970779220779221 | -0.017202219933031504 | -0.00046234610425521794 | 2 | 1 | 70 | 87 |
| interaction - saved_baseline | 43 | 1.875 | 0.7139764996907871 | -0.0967987638443435 | -0.02666484710942263 | 7 | 4 | 66 | 83 |
| interaction - ce | 43 | 1.25 | 1.5422077922077975 | -0.004353275349161301 | -0.0014257905921283287 | 2 | 0 | 71 | 87 |
| interaction - query_only | 43 | 0.625 | 0.5714285714285765 | 0.009535905357426433 | 0.00014872065814697244 | 1 | 0 | 72 | 87 |
| interaction - ref_only | 43 | 0.625 | 0.5714285714285765 | 0.012848944583870203 | -0.0009634444878731108 | 1 | 0 | 72 | 87 |

CE is the unchanged previous pairwise round endpoint; new arms retain the frozen original M3.
The saved original logits must match exactly across CE and available new arms after ID alignment.
Receipt LR, steps, batch size, source count and reference IDs are compared with CE. The main implementation reads the previous pairwise exposures/schedule_s{seed}.json directly; no new source coverage audit is performed. See analysis/SOURCE_CONTEXT.md.
OA/BA and differences are percent/percentage points; NLL/Brier are unscaled, lower is better.
BA uses present query classes. Missing arms/seeds stay pending; one-seed SD is null.
Repeated query events across seeds are not independent observations or vessels.
Mean and sample SD summarize available matched training orders sharing the original initialization source; they are not variation across independently pretrained models.
Arm comparisons measure trained-system differences; supplied dependence interventions require separate paired analysis.
Cases are outcome-selected diagnostics and require main-worker visual review; no physical cause is inferred.

Pending:
[]
