# Actual cohort distribution

Actual manifest counts only. Split names and absence of recorded overlap do not establish independent holdout. Fit and development share 3 ports (Jebel Ali, Port Klang, Singapore), so this is not strict leave-one-port-out (LOPO). The source1365 cache/index pool covers 1205 fit plus 160 development observations; its name does not define the training split. Training uses only indices['fit']; development indices are used for evaluation. Development is historically reused. References are source-fit records; same-ID OR same-known-MMSI query/reference pairs are excluded, not the reference observations from source fitting. Same-ID and same-MMSI exclusions overlap and must not be added. Repeated references, queries, ships, passes and seeds are not independent observations. Polarization means supplied manifest views, not pixel-verified target availability. No images, cache arrays or predictions were read by this summary.

| Cohort | Observations | Ships | Passes | Ports | Both views | VV only | VH only | Neither |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fit | 1205 | 1004 | 88 | 19 | 1036 | 169 | 0 | 0 |
| development | 160 | 158 | 3 | 3 | 145 | 15 | 0 | 0 |
| references | 16 | 16 | 13 | 9 | 15 | 1 | 0 | 0 |

Each class cell is observations / ships / passes / ports.

| Class | Fit | Development | References |
|---|---:|---:|---:|
| bulk_carrier | 367 / 306 / 75 / 18 | 25 / 24 / 3 / 3 | 2 / 2 / 2 / 2 |
| fishing | 285 / 232 / 16 / 6 | 0 / 0 / 0 / 0 | 2 / 2 / 2 / 1 |
| general_cargo | 193 / 162 / 57 / 15 | 12 / 12 / 3 / 3 | 2 / 2 / 2 / 2 |
| chemical_tanker | 170 / 141 / 33 / 12 | 70 / 70 / 3 / 3 | 2 / 2 / 2 / 1 |
| container | 69 / 62 / 26 / 7 | 22 / 22 / 3 / 3 | 2 / 2 / 2 / 2 |
| crude_oil_tanker | 66 / 55 / 23 / 8 | 16 / 15 / 3 / 3 | 2 / 2 / 2 / 2 |
| tug | 34 / 29 / 22 / 8 | 7 / 7 / 3 / 3 | 2 / 2 / 2 / 2 |
| offshore | 21 / 17 / 11 / 5 | 8 / 8 / 3 / 3 | 2 / 2 / 2 / 2 |

| Cohort pair | Dimension | Shared values | Left observations in overlap | Right observations in overlap |
|---|---|---:|---:|---:|
| fit / development | id | 0 | 0 | 0 |
| fit / development | mmsi | 0 | 0 | 0 |
| fit / development | pass_group | 0 | 0 | 0 |
| fit / development | port | 3 | 349 | 160 |
| fit / references | id | 16 | 16 | 16 |
| fit / references | mmsi | 16 | 18 | 16 |
| fit / references | pass_group | 13 | 340 | 16 |
| fit / references | port | 9 | 903 | 16 |
| development / references | id | 0 | 0 | 0 |
| development / references | mmsi | 0 | 0 | 0 |
| development / references | pass_group | 0 | 0 | 0 |
| development / references | port | 3 | 160 | 9 |

Reference IDs included in source fit: 16 / 16. Reference counts in the class order above: [2, 2, 2, 2, 2, 2, 2, 2]. Eligible counts can fall after ID/MMSI exclusion.

| Query cohort | Candidate pairs | Eligible pairs | Excluded union | Same ID | Same MMSI | Affected queries | Eligible refs/query | True refs/query |
|---|---:|---:|---:|---:|---:|---:|---|---|
| fit | 19280 | 19262 | 18 | 16 | 18 | 18 | [15, 16] | [1, 2] |
| development | 2560 | 2560 | 0 | 0 | 0 | 0 | [16, 16] | [2, 2] |

| Port | Fit | Development | References |
|---|---:|---:|---:|
| Antwerp-Bruges | 75 | 0 | 0 |
| Busan | 23 | 0 | 0 |
| Callao | 13 | 0 | 0 |
| Fujairah | 56 | 0 | 0 |
| Houston | 19 | 0 | 1 |
| Jebel Ali | 135 | 69 | 7 |
| Los Angeles | 1 | 0 | 0 |
| New York | 7 | 0 | 1 |
| Newcastle | 12 | 0 | 0 |
| Ningbo-Zhoushan | 300 | 0 | 2 |
| Port Hedland | 7 | 0 | 0 |
| Port Klang | 73 | 28 | 1 |
| Port Said | 22 | 0 | 1 |
| Richards Bay | 11 | 0 | 0 |
| Santos | 81 | 0 | 0 |
| Shanghai | 142 | 0 | 1 |
| Singapore | 141 | 63 | 1 |
| Sydney Botany | 23 | 0 | 0 |
| Tanger Med | 64 | 0 | 1 |

cohort_groups.csv contains every class, ship/MMSI, pass, port and supplied-polarization group with observation, unique-ID, ship, pass, port and polarization counts. cohort_overlap.csv includes directional observation participation; cohort_reference_pairs.csv records exclusions per query. The counts describe cohort composition and repeated exposure, not statistical independence or model benefit.
