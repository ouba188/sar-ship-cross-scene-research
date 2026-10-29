# Centered logit increment decomposition

For each query, subtract the across-class mean from new-minus-original logits; common is the across-query mean of that centered increment.
Residual is each query centered increment minus common. All energies are sums of squared logit increments, not averages or physical energy.
Absolute total/common/residual energies accompany fractions: a larger residual fraction alone does not establish a larger useful update.
Means and sample SD are over available orders; energy fractions are averaged per order, not pooled. Summed decision counts are repeated query-order events.
Common-only and residual-only scores are descriptive decompositions of development predictions, not independently trained or deployable interventions.
These fractions are not causal information fractions or proof of useful query/reference dependence.


| Arm | Orders | Mean total energy | Mean common energy | Mean residual energy | Mean common fraction | Mean full-common OA | Mean full-common BA |
|---|---|---|---|---|---|---|---|
| query_only | [17, 29, 43] | 98.30552060197311 | 90.44955666086321 | 7.855963941109892 | 0.9325023741169897 | -0.14619883040936088 | -0.7016622774754945 |
| ref_only | [17, 29, 43] | 96.22327681257694 | 91.71031662910046 | 4.512960183476476 | 0.9546140899192808 | 2.3684757858670005e-15 | -0.07281003692909611 |
| interaction | [17, 29, 43] | 83.42455202193422 | 77.98932727458931 | 5.435224747344908 | 0.9398900586374758 | -0.5847953216374293 | -0.9868661175030505 |

| Arm | Seed | Total energy | Common energy | Residual energy | Common fraction | Residual fraction |
|---|---|---|---|---|---|---|
| query_only | 17 | 95.79614195611848 | 91.66731749854068 | 4.128824457577779 | 0.9568998878945554 | 0.043100112105444476 |
| query_only | 29 | 165.22713468922288 | 147.5679591204186 | 17.659175568804272 | 0.8931218192337501 | 0.10687818076624984 |
| query_only | 43 | 33.893285160577975 | 32.11339336363036 | 1.7798917969476267 | 0.9474854152226634 | 0.0525145847773369 |
| ref_only | 17 | 103.8988064220939 | 100.21142539275208 | 3.6873810293418208 | 0.9645098807549178 | 0.03549011924508216 |
| ref_only | 29 | 152.50251383503553 | 144.12726069237794 | 8.375253142657602 | 0.94508121255157 | 0.0549187874484302 |
| ref_only | 43 | 32.26851018060137 | 30.79226380217137 | 1.476246378430006 | 0.9542511764513546 | 0.045748823548645595 |
| interaction | 17 | 94.74338305674573 | 90.02982771095576 | 4.713555345789978 | 0.9502492396438195 | 0.049750760356180604 |
| interaction | 29 | 127.21514694419152 | 117.06011798768047 | 10.15502895651106 | 0.9201743723098792 | 0.07982562769012096 |
| interaction | 43 | 28.315126064865417 | 26.87803612513173 | 1.4370899397336858 | 0.9492465639587284 | 0.05075343604127148 |
