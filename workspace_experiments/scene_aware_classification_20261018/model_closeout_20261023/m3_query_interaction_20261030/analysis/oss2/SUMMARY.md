# oss2 query interaction development analysis

Queries: 228; available seeds by arm: {'ce': [17, 29, 43], 'query_only': [17, 29, 43], 'ref_only': [17, 29, 43], 'interaction': [17, 29, 43]}.


| Arm - reference | Orders | Mean delta OA | SD OA | Mean delta BA | SD BA | Mean delta NLL | Mean delta Brier |
|---|---|---|---|---|---|---|---|
| query_only - saved_baseline | [17, 29, 43] | -0.29239766081871466 | 0.5064476045523054 | -0.048103289179380226 | 1.4403327475090748 | 0.024859720823567095 | 0.003862515270014725 |
| query_only - ce | [17, 29, 43] | -0.29239766081871227 | 0.25322380227615066 | -0.18941839941452324 | 0.16963345762706658 | 0.003965460790099777 | 0.002239191842202485 |
| ref_only - saved_baseline | [17, 29, 43] | 0.0 | 0.877192982456144 | 0.6765343919800415 | 2.596549001014806 | 0.026693297633277435 | 0.003515150574452212 |
| ref_only - ce | [17, 29, 43] | 2.3684757858670005e-15 | 0.438596491228072 | 0.5352192817448985 | 1.1794771255963061 | 0.005799037599810116 | 0.001891827146639972 |
| interaction - saved_baseline | [17, 29, 43] | -0.5847953216374245 | 0.5064476045523013 | -0.31248420733453725 | 1.4790144448305398 | 0.023656240225180536 | 0.0031888880814806195 |
| interaction - ce | [17, 29, 43] | -0.5847953216374222 | 0.6699672068648823 | -0.4537993175696802 | 1.1338480340145987 | 0.0027619801917132167 | 0.0015655646536683794 |
| interaction - query_only | [17, 29, 43] | -0.29239766081870994 | 0.5064476045522971 | -0.264380918155157 | 1.004191877792703 | -0.00120348059838656 | -0.0006736271885341057 |
| interaction - ref_only | [17, 29, 43] | -0.5847953216374245 | 0.5064476045523013 | -0.9890185993145787 | 1.4173776870929253 | -0.0030370574080968993 | -0.00032626249297159254 |

| Arm - reference | Seed | Delta OA | Delta BA | Delta NLL | Delta Brier | Rescue | Harm | CC | WW |
|---|---|---|---|---|---|---|---|---|---|
| query_only - saved_baseline | 17 | -0.877192982456144 | -1.5057078202471814 | 0.007246252270376008 | -0.004265115140998255 | 3 | 5 | 127 | 93 |
| query_only - ce | 17 | 0.0 | 0.006457445434584486 | 0.0006614008717100717 | 0.0018391920356541913 | 1 | 1 | 129 | 97 |
| ref_only - saved_baseline | 17 | -0.877192982456144 | -1.5057078202471814 | 0.010351443539969862 | -0.0035135765770739624 | 3 | 5 | 127 | 93 |
| ref_only - ce | 17 | 0.0 | 0.006457445434584486 | 0.003766592141303926 | 0.0025907305995784835 | 1 | 1 | 129 | 97 |
| interaction - saved_baseline | 17 | -0.877192982456144 | -0.9245377311344356 | 0.005795991723962501 | -0.0048135675159475255 | 2 | 4 | 128 | 94 |
| interaction - ce | 17 | 0.0 | 0.5876275345473303 | -0.0007888596747034349 | 0.0012907396607049204 | 2 | 2 | 128 | 96 |
| interaction - query_only | 17 | 0.0 | 0.5811700891127458 | -0.0014502605464135065 | -0.0005484523749492709 | 1 | 1 | 129 | 97 |
| interaction - ref_only | 17 | 0.0 | 0.5811700891127458 | -0.004555451816007361 | -0.001299990938873563 | 1 | 1 | 129 | 97 |
| query_only - saved_baseline | 29 | 0.0 | 1.3743128435782097 | 0.048711151441281464 | 0.013669750333644481 | 3 | 3 | 129 | 93 |
| query_only - ce | 29 | -0.43859649122806843 | -0.2873563218390771 | 0.004677951252491486 | 0.00422012092657198 | 0 | 1 | 132 | 95 |
| ref_only - saved_baseline | 29 | 0.877192982456144 | 3.548225887056475 | 0.05052559751024832 | 0.011689886704105201 | 5 | 3 | 129 | 91 |
| ref_only - ce | 29 | 0.43859649122807554 | 1.886556721639188 | 0.006492397321458343 | 0.0022402572970326995 | 2 | 1 | 132 | 93 |
| interaction - saved_baseline | 29 | 0.0 | 1.3743128435782097 | 0.047441952381730434 | 0.012497083105207696 | 3 | 3 | 129 | 93 |
| interaction - ce | 29 | -0.43859649122806843 | -0.2873563218390771 | 0.0034087521929404563 | 0.003047453698135194 | 0 | 1 | 132 | 95 |
| interaction - query_only | 29 | 0.0 | 0.0 | -0.0012691990595510294 | -0.0011726672284367856 | 0 | 0 | 132 | 96 |
| interaction - ref_only | 29 | -0.877192982456144 | -2.173913043478265 | -0.0030836451285178867 | 0.0008071964011024946 | 0 | 2 | 132 | 94 |
| query_only - saved_baseline | 43 | 0.0 | -0.012914890869168971 | 0.018621758759043816 | 0.0021829106173979484 | 2 | 2 | 134 | 90 |
| query_only - ce | 43 | -0.43859649122806843 | -0.2873563218390771 | 0.0065570302460977725 | 0.0006582625643812845 | 0 | 1 | 136 | 91 |
| ref_only - saved_baseline | 43 | 0.0 | -0.012914890869168971 | 0.019202851849614122 | 0.002369141596325397 | 2 | 2 | 134 | 90 |
| ref_only - ce | 43 | -0.43859649122806843 | -0.2873563218390771 | 0.0071381233366680785 | 0.000844493543308733 | 0 | 1 | 136 | 91 |
| interaction - saved_baseline | 43 | -0.8771929824561298 | -1.3872277344473858 | 0.017730776569848672 | 0.0018831486551816878 | 2 | 4 | 132 | 90 |
| interaction - ce | 43 | -1.3157894736841982 | -1.661669165417294 | 0.005666048056902628 | 0.0003585006021650239 | 0 | 3 | 134 | 91 |
| interaction - query_only | 43 | -0.8771929824561298 | -1.3743128435782168 | -0.0008909821891951442 | -0.0002997619622162606 | 0 | 2 | 134 | 92 |
| interaction - ref_only | 43 | -0.8771929824561298 | -1.3743128435782168 | -0.0014720752797654502 | -0.00048599294114370917 | 0 | 2 | 134 | 92 |

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
