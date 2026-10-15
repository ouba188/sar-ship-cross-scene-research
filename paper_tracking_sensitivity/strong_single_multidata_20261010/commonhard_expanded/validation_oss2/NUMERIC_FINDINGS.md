# Expanded Known-Single Common-Hard

Scope: explicit_input_validation_not_full_main.
Analyzed tasks: {'oss2': 1}; neural seeds in original set: [20261010, 20261011, 20261012].
Original common-hard flags are preserved and verified against stored predictions.
Added controls: sarclip, r50_c4_common_readout, sum, sum_difference.
This explicit expanded set is descriptive, not all possible or all historical singles.
Methods were already selected for inclusion; no selection or tuning is performed here.
Original R18 included in 0/1 tasks.

| Dataset | Unit | N | Old hard | Expanded hard | Old dual rescues | Expanded dual rescues | Eliminated rescues (union) | New reverse evidence (union) | Expanded dual wrong, any single correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oss2 | draw_exposure | 18405 | 7184 | 5682 | 888 | 596 | 292 | 1210 | 5479 |
| oss2 | unique_query_any_draw | 3745 | 2518 | 2155 | 640 | 453 | 245 | 835 | 2520 |
| oss2 | unique_query_all_draw | 3745 | 515 | 319 | 1 | 1 | 0 | 9 | 120 |

## Added-Control Rescue And Harm

Rescue = dual correct / added single wrong; harm = dual wrong / added single correct.
Removed rescue = old common-hard AND dual correct AND added single correct.
New reverse evidence = old common-hard AND dual wrong AND added single correct.

| Dataset | Unit | Added single | Removed rescues | New reverse evidence | Dual rescue | Dual harm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| oss2 | draw_exposure | sarclip | 116 | 568 | 4660 | 1535 |
| oss2 | draw_exposure | r50_c4_common_readout | 27 | 145 | 3837 | 1393 |
| oss2 | draw_exposure | sum | 0 | 4 | 3294 | 1823 |
| oss2 | draw_exposure | sum_difference | 192 | 632 | 3584 | 2024 |
| oss2 | unique_query_any_draw | sarclip | 104 | 369 | 2048 | 800 |
| oss2 | unique_query_any_draw | r50_c4_common_readout | 25 | 121 | 1846 | 801 |
| oss2 | unique_query_any_draw | sum | 0 | 4 | 1778 | 1184 |
| oss2 | unique_query_any_draw | sum_difference | 158 | 481 | 1875 | 1264 |
| oss2 | unique_query_all_draw | sarclip | 0 | 7 | 140 | 38 |
| oss2 | unique_query_all_draw | r50_c4_common_readout | 0 | 0 | 89 | 16 |
| oss2 | unique_query_all_draw | sum | 0 | 0 | 53 | 6 |
| oss2 | unique_query_all_draw | sum_difference | 0 | 2 | 62 | 14 |

Per-control counts overlap and must not be summed. Union columns count each exposure once.
Bitmask order is sarclip=1, r50_c4_common_readout=2, sum=4, sum_difference=8.
Unique-query any/all flags are computed over eligible draw exposures; queries can have both rescue and harm across draws.
Unique-query event totals do not form a partition and cannot be subtracted to obtain an accuracy gain.
Only exposure-level old-hard = expanded-hard + resolved is a partition; any/all aggregation does not preserve that identity.
The added union is an oracle correctness set, not a deployable classifier. Counts are not task-macro accuracy.
No independence of draws or neural seeds, architecture necessity, or physical specialization is inferred.
No neural execution, GPU, source fitting, network access or tuning. Original results are unchanged.
