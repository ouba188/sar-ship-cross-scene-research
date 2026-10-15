# Expanded Known-Single Common-Hard

Scope: full_main_72_task_seeds_23_ports_plus_F.
Analyzed tasks: {'f0921': 1, 'self24ports': 23}; neural seeds in original set: [20261010, 20261011, 20261012].
Original common-hard flags are preserved and verified against stored predictions.
Added controls: sarclip, r50_c4_common_readout, sum, sum_difference.
This explicit expanded set is descriptive, not all possible or all historical singles.
Methods were already selected for inclusion; no selection or tuning is performed here.
Original R18 included in 24/24 tasks.

| Dataset | Unit | N | Old hard | Expanded hard | Old dual rescues | Expanded dual rescues | Eliminated rescues (union) | New reverse evidence (union) | Expanded dual wrong, any single correct |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| f0921 | draw_exposure | 617125 | 227451 | 181999 | 7925 | 4910 | 3015 | 42437 | 223713 |
| f0921 | unique_query_any_draw | 123489 | 81573 | 70919 | 7008 | 4412 | 2783 | 27129 | 92909 |
| f0921 | unique_query_all_draw | 123489 | 12315 | 7749 | 1 | 0 | 0 | 304 | 6502 |
| self24ports | draw_exposure | 182775 | 34585 | 28474 | 675 | 403 | 272 | 5839 | 36006 |
| self24ports | unique_query_any_draw | 38027 | 12427 | 10588 | 620 | 377 | 260 | 4122 | 16363 |
| self24ports | unique_query_all_draw | 38027 | 2741 | 2079 | 0 | 0 | 0 | 36 | 1182 |

## Added-Control Rescue And Harm

Rescue = dual correct / added single wrong; harm = dual wrong / added single correct.
Removed rescue = old common-hard AND dual correct AND added single correct.
New reverse evidence = old common-hard AND dual wrong AND added single correct.

| Dataset | Unit | Added single | Removed rescues | New reverse evidence | Dual rescue | Dual harm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| f0921 | draw_exposure | sarclip | 1461 | 21061 | 118979 | 58084 |
| f0921 | draw_exposure | r50_c4_common_readout | 405 | 5052 | 93348 | 50700 |
| f0921 | draw_exposure | sum | 17 | 200 | 101663 | 58585 |
| f0921 | draw_exposure | sum_difference | 1682 | 20607 | 101666 | 66067 |
| f0921 | unique_query_any_draw | sarclip | 1358 | 12853 | 56702 | 29119 |
| f0921 | unique_query_any_draw | r50_c4_common_readout | 391 | 4176 | 50415 | 27834 |
| f0921 | unique_query_any_draw | sum | 17 | 200 | 55027 | 34341 |
| f0921 | unique_query_any_draw | sum_difference | 1554 | 14371 | 54583 | 37723 |
| f0921 | unique_query_all_draw | sarclip | 0 | 212 | 3083 | 1390 |
| f0921 | unique_query_all_draw | r50_c4_common_readout | 0 | 2 | 1377 | 677 |
| f0921 | unique_query_all_draw | sum | 0 | 0 | 1378 | 555 |
| f0921 | unique_query_all_draw | sum_difference | 0 | 72 | 1466 | 739 |
| self24ports | draw_exposure | sarclip | 127 | 3151 | 26565 | 12717 |
| self24ports | draw_exposure | r50_c4_common_readout | 37 | 669 | 21065 | 10244 |
| self24ports | draw_exposure | sum | 3 | 19 | 19779 | 11278 |
| self24ports | draw_exposure | sum_difference | 179 | 2907 | 21107 | 13643 |
| self24ports | unique_query_any_draw | sarclip | 119 | 2199 | 12255 | 6999 |
| self24ports | unique_query_any_draw | r50_c4_common_readout | 37 | 576 | 10459 | 6079 |
| self24ports | unique_query_any_draw | sum | 3 | 19 | 11300 | 7301 |
| self24ports | unique_query_any_draw | sum_difference | 172 | 2247 | 11818 | 8292 |
| self24ports | unique_query_all_draw | sarclip | 0 | 25 | 1098 | 316 |
| self24ports | unique_query_all_draw | r50_c4_common_readout | 0 | 1 | 683 | 156 |
| self24ports | unique_query_all_draw | sum | 0 | 0 | 304 | 95 |
| self24ports | unique_query_all_draw | sum_difference | 0 | 8 | 380 | 140 |

Per-control counts overlap and must not be summed. Union columns count each exposure once.
Bitmask order is sarclip=1, r50_c4_common_readout=2, sum=4, sum_difference=8.
Unique-query any/all flags are computed over eligible draw exposures; queries can have both rescue and harm across draws.
Unique-query event totals do not form a partition and cannot be subtracted to obtain an accuracy gain.
Only exposure-level old-hard = expanded-hard + resolved is a partition; any/all aggregation does not preserve that identity.
The added union is an oracle correctness set, not a deployable classifier. Counts are not task-macro accuracy.
No independence of draws or neural seeds, architecture necessity, or physical specialization is inferred.
No neural execution, GPU, source fitting, network access or tuning. Original results are unchanged.
