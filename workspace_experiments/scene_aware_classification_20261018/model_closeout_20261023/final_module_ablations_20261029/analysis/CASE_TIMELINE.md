# Selected case prediction timeline

All panels remain not_reviewed. This is prediction evidence, not visual or physical interpretation.
Reference means the comparator model in the named contrast, not a support chip.
B = each arm's own sourcefit; A = its adaptation endpoint. C/W is correctness, not confidence.

| Query | Contrast | Selected state | Seed | Truth | Reference B>A | Alternative B>A | Pair B>A | True-rival margin B>A (ref;alt) |
|---|---|---|---:|---|---|---|---|---|
| 34599 | no_m1 vs full | rescued | 17 | bulk_carrier | bulk_carrier>bulk_carrier | bulk_carrier>bulk_carrier | CC>CC | 2.967>2.309; 3.227>2.422 |
| 34599 | no_m1 vs full | rescued | 29 | bulk_carrier | chemical_tanker>chemical_tanker | bulk_carrier>bulk_carrier | rescued>rescued | -0.455>-0.067; 0.118>0.387 |
| 34599 | no_m1 vs full | rescued | 43 | bulk_carrier | bulk_carrier>chemical_tanker | bulk_carrier>bulk_carrier | CC>rescued | 0.736>-0.008; 1.015>0.379 |
| 32936 | no_m2 vs full | rescued | 17 | crude_oil_tanker | chemical_tanker>chemical_tanker | crude_oil_tanker>crude_oil_tanker | rescued>rescued | -8.183>-6.073; 0.260>2.006 |
| 32936 | no_m2 vs full | rescued | 29 | crude_oil_tanker | chemical_tanker>chemical_tanker | chemical_tanker>crude_oil_tanker | WW>rescued | -4.905>-4.311; -0.446>0.590 |
| 32936 | no_m2 vs full | rescued | 43 | crude_oil_tanker | chemical_tanker>chemical_tanker | crude_oil_tanker>crude_oil_tanker | rescued>rescued | -3.110>-3.358; 0.137>1.086 |
| 33424 | no_m3 vs full | rescued | 17 | bulk_carrier | bulk_carrier>crude_oil_tanker | bulk_carrier>bulk_carrier | CC>rescued | 1.250>-0.377; 2.227>0.665 |
| 33424 | no_m3 vs full | rescued | 29 | bulk_carrier | container>container | bulk_carrier>bulk_carrier | rescued>rescued | -0.219>-0.523; 0.300>0.666 |
| 33424 | no_m3 vs full | rescued | 43 | bulk_carrier | bulk_carrier>bulk_carrier | bulk_carrier>bulk_carrier | CC>CC | 1.702>0.620; 2.507>1.292 |
| 7821 | plain vs full | rescued | 17 | general_cargo | offshore>offshore | general_cargo>general_cargo | rescued>rescued | -2.836>-3.444; 1.529>1.839 |
| 7821 | plain vs full | rescued | 29 | general_cargo | offshore>offshore | general_cargo>general_cargo | rescued>rescued | -1.636>-3.286; 1.425>0.387 |
| 7821 | plain vs full | rescued | 43 | general_cargo | offshore>offshore | general_cargo>general_cargo | rescued>rescued | -3.964>-4.477; 4.196>3.731 |
| 25585 | plain vs full_control | rescued | 17 | chemical_tanker | fishing>fishing | chemical_tanker>chemical_tanker | rescued>rescued | -2.184>-2.758; 3.623>2.259 |
| 25585 | plain vs full_control | rescued | 29 | chemical_tanker | fishing>fishing | chemical_tanker>chemical_tanker | rescued>rescued | -1.994>-2.537; 2.388>1.809 |
| 25585 | plain vs full_control | rescued | 43 | chemical_tanker | fishing>fishing | chemical_tanker>chemical_tanker | rescued>rescued | -1.158>-2.151; 3.668>4.545 |
| 11642 | no_m1 vs full | harmed | 17 | chemical_tanker | bulk_carrier>chemical_tanker | bulk_carrier>bulk_carrier | WW>harmed | -0.365>0.157; -0.366>-0.047 |
| 11642 | no_m1 vs full | harmed | 29 | chemical_tanker | chemical_tanker>chemical_tanker | chemical_tanker>chemical_tanker | CC>CC | 0.855>0.465; 0.548>0.227 |
| 11642 | no_m1 vs full | harmed | 43 | chemical_tanker | bulk_carrier>chemical_tanker | bulk_carrier>bulk_carrier | WW>harmed | -0.356>0.209; -0.602>-0.035 |
| 1659 | no_m1 vs full | CC | 17 | container | container>container | container>container | CC>CC | 6.042>5.032; 6.102>4.768 |
| 1659 | no_m1 vs full | CC | 29 | container | container>container | container>container | CC>CC | 6.437>5.502; 6.246>5.538 |
| 1659 | no_m1 vs full | CC | 43 | container | container>container | container>container | CC>CC | 3.383>2.091; 2.111>1.002 |
| 23221 | no_m1 vs full | WW | 17 | chemical_tanker | bulk_carrier>bulk_carrier | bulk_carrier>bulk_carrier | WW>WW | -5.080>-3.791; -5.093>-3.545 |
| 23221 | no_m1 vs full | WW | 29 | chemical_tanker | bulk_carrier>bulk_carrier | bulk_carrier>bulk_carrier | WW>WW | -3.489>-3.884; -4.262>-4.925 |
| 23221 | no_m1 vs full | WW | 43 | chemical_tanker | bulk_carrier>bulk_carrier | bulk_carrier>bulk_carrier | WW>WW | -5.344>-3.238; -5.528>-3.140 |
| 36892 | no_m1 vs full | rescued | 17 | offshore | fishing>fishing | fishing>offshore | WW>rescued | -0.862>-0.271; -0.840>0.062 |
| 36892 | no_m1 vs full | rescued | 29 | offshore | offshore>offshore | offshore>offshore | CC>CC | 0.635>1.527; 0.522>1.433 |
| 36892 | no_m1 vs full | rescued | 43 | offshore | fishing>offshore | fishing>offshore | WW>CC | -0.146>1.187; -0.075>1.237 |
| 9620 | no_m1 vs full | harmed | 17 | container | container>container | container>container | CC>CC | 3.690>2.057; 3.836>2.019 |
| 9620 | no_m1 vs full | harmed | 29 | container | container>container | container>container | CC>CC | 5.323>3.895; 5.192>3.884 |
| 9620 | no_m1 vs full | harmed | 43 | container | container>container | container>chemical_tanker | CC>harmed | 2.809>0.348; 2.289>-0.106 |
| 22766 | no_m2 vs full | harmed | 17 | offshore | fishing>offshore | fishing>fishing | WW>harmed | -0.615>0.215; -7.804>-6.712 |
| 22766 | no_m2 vs full | harmed | 29 | offshore | general_cargo>offshore | fishing>fishing | WW>harmed | -1.182>0.410; -8.407>-6.681 |
| 22766 | no_m2 vs full | harmed | 43 | offshore | general_cargo>offshore | fishing>fishing | WW>harmed | -1.020>0.620; -8.440>-6.920 |
| 10038 | no_m3 vs full | harmed | 17 | crude_oil_tanker | bulk_carrier>crude_oil_tanker | bulk_carrier>bulk_carrier | WW>harmed | -0.855>0.449; -2.246>-0.432 |
| 10038 | no_m3 vs full | harmed | 29 | crude_oil_tanker | crude_oil_tanker>crude_oil_tanker | crude_oil_tanker>crude_oil_tanker | CC>CC | 0.458>0.758; 0.838>0.540 |
| 10038 | no_m3 vs full | harmed | 43 | crude_oil_tanker | bulk_carrier>crude_oil_tanker | bulk_carrier>bulk_carrier | WW>harmed | -1.028>0.063; -1.593>-0.556 |
