# Frozen frequency standalone results

Families: old, independent, standalone, pair. Training definitions belong to the main experiment records. All 30 modes and the original specialist CSV statistics are retained.

All 25 matching target tasks must be COMPLETE with policies present, and all source SELECTION.json rules are loaded before target-label reporting. Frozen source_selected predictions, row IDs, supports and support labels are checked by the reused report. No policies, predictions, training outputs or source selections are changed.

OA/BA/macro_f1_8 are percentages; deltas are percentage points. BA uses classes present in each evaluation set; macro F1 uses eight classes. Target summaries average five draws within task, then tasks equally within dataset. Base has no expert handoff; full uses each mode's route; frozen_route uses learned admissions.

Original metrics.csv, task_means.csv, summary.csv, draw_metrics.csv, class_metrics.csv and events.csv cover all stages. Their deltas are relative to same-stage learned. Complementarity, band_events and rescue_overlaps CSVs retain base/frozen_route details and summaries. Oracles use target labels and are nondeployable; union_minus_best_single_oracle measures the union against the best single-band oracle. Rescue/overlap counts accumulate query-draw events, including repeated samples. source_selection.csv retains frozen rules and source validation deltas.

This report is descriptive and makes no causal or training-effectiveness conclusions.

## Target BA

| dataset | mode | base | full | frozen_route |
| --- | --- | --- | --- | --- |
| f0921 | independent_equal | 26.1399 | 26.4456 | 26.4355 |
| f0921 | learned | 26.0348 | 26.3593 | 26.3593 |
| f0921 | old_equal | 26.1005 | 26.4069 | 26.3941 |
| f0921 | pair_equal | 26.1083 | 26.4157 | 26.4044 |
| f0921 | source_selected | 26.0277 | 26.3534 | 26.3442 |
| f0921 | standalone_equal | 26.1398 | 26.4459 | 26.4349 |
| oss2 | independent_equal | 31.1496 | 32.1821 | 32.1881 |
| oss2 | learned | 31.2783 | 32.3724 | 32.3724 |
| oss2 | old_equal | 30.9430 | 32.0829 | 31.9921 |
| oss2 | pair_equal | 31.1507 | 32.2454 | 32.1949 |
| oss2 | source_selected | 31.2420 | 32.3104 | 32.2930 |
| oss2 | standalone_equal | 31.1171 | 32.2214 | 32.1527 |
| self24ports | independent_equal | 52.3612 | 54.1313 | 54.0076 |
| self24ports | learned | 52.4994 | 54.2383 | 54.2383 |
| self24ports | old_equal | 52.3278 | 54.1356 | 53.9788 |
| self24ports | pair_equal | 52.3362 | 54.1465 | 53.9899 |
| self24ports | source_selected | 52.5426 | 54.3234 | 54.1909 |
| self24ports | standalone_equal | 52.3557 | 54.1642 | 54.0044 |

## Matched differences against independent (full)

| dataset | suffix | comparison | delta_oa | delta_ba | positive_draws_oa | positive_draws_ba | draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | equal | old-independent | -0.0425 | -0.0387 | 1 | 2 | 5 |
| f0921 | equal | pair-independent | -0.0233 | -0.0299 | 1 | 1 | 5 |
| f0921 | equal | standalone-independent | 0.0031 | 0.0003 | 3 | 1 | 5 |
| f0921 | support_local | old-independent | 0.0063 | 0.0061 | 3 | 2 | 5 |
| f0921 | support_local | pair-independent | 0.0181 | 0.0103 | 4 | 3 | 5 |
| f0921 | support_local | standalone-independent | 0.0151 | 0.0115 | 1 | 2 | 5 |
| f0921 | support_static | old-independent | -0.0481 | -0.0341 | 1 | 2 | 5 |
| f0921 | support_static | pair-independent | -0.0240 | -0.0323 | 2 | 3 | 5 |
| f0921 | support_static | standalone-independent | -0.0306 | 0.0106 | 0 | 3 | 5 |
| oss2 | equal | old-independent | -0.1250 | -0.0991 | 1 | 1 | 5 |
| oss2 | equal | pair-independent | -0.0217 | 0.0634 | 2 | 3 | 5 |
| oss2 | equal | standalone-independent | -0.0217 | 0.0394 | 1 | 3 | 5 |
| oss2 | support_local | old-independent | -0.0217 | -0.1576 | 1 | 0 | 5 |
| oss2 | support_local | pair-independent | -0.0435 | -0.0922 | 0 | 0 | 5 |
| oss2 | support_local | standalone-independent | -0.0109 | 0.0064 | 0 | 2 | 5 |
| oss2 | support_static | old-independent | 0.1902 | 0.2131 | 1 | 2 | 5 |
| oss2 | support_static | pair-independent | 0.1902 | 0.2131 | 1 | 2 | 5 |
| oss2 | support_static | standalone-independent | -0.0000 | 0.0378 | 1 | 1 | 5 |
| self24ports | equal | old-independent | -0.0368 | 0.0043 | 1 | 4 | 5 |
| self24ports | equal | pair-independent | -0.0070 | 0.0152 | 2 | 2 | 5 |
| self24ports | equal | standalone-independent | 0.0027 | 0.0329 | 3 | 4 | 5 |
| self24ports | support_local | old-independent | -0.0299 | -0.0240 | 0 | 0 | 5 |
| self24ports | support_local | pair-independent | -0.0156 | -0.0087 | 0 | 2 | 5 |
| self24ports | support_local | standalone-independent | 0.0077 | 0.0056 | 3 | 2 | 5 |
| self24ports | support_static | old-independent | 0.0083 | 0.0732 | 3 | 3 | 5 |
| self24ports | support_static | pair-independent | 0.0303 | 0.1088 | 4 | 4 | 5 |
| self24ports | support_static | standalone-independent | 0.0773 | 0.1383 | 4 | 4 | 5 |

matched_metrics.csv subtracts independent for the same dataset/task/draw/stage and the same equal, support_static or support_local suffix before aggregation. matched_task_means.csv, matched_draw_metrics.csv and matched_summary.csv retain all three stages and OA/BA/macro F1. positive_draws counts strictly positive dataset means over the five support draws; positive_task_draws counts positive individual task-draw differences. Counts are descriptive, not independent replications or tests.

## Source standalone ridge probe

| dataset | family | band | oa | ba | tasks | episodes | query_events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f0921 | independent | mean | 53.6388 | 52.6096 | 1 | 15 | 3357 |
| f0921 | pair | mean | 52.9066 | 52.3034 | 1 | 15 | 3357 |
| f0921 | standalone | mean | 54.2855 | 53.3377 | 1 | 15 | 3357 |
| oss2 | independent | mean | 45.9785 | 44.3258 | 1 | 20 | 4240 |
| oss2 | pair | mean | 46.4506 | 44.4498 | 1 | 20 | 4240 |
| oss2 | standalone | mean | 46.3085 | 44.3798 | 1 | 20 | 4240 |
| self24ports | independent | mean | 45.9712 | 43.3853 | 23 | 434 | 92527 |
| self24ports | pair | mean | 46.0975 | 43.4296 | 23 | 434 | 92527 |
| self24ports | standalone | mean | 46.5892 | 43.9066 | 23 | 434 | 92527 |

Every available encoded source episode for all 25 tasks is used for independent, standalone and pair. Labels come only from the original source episodes: the first 64 encoded indices fit the ridge and remaining indices supply evaluation labels. The kernel is 0.25 times unit-band similarity times exp(-squared condition distance/0.5). The existing class-balanced ridge helper fixes strength 8; the prior is zero and there is no shared kernel. OA/BA use all frozen source queries and eight-class argmax; the training-only eligibility filter is not applied to this readout. This diagnostic never selects a model or rule.

source_standalone_probe.csv contains episode-level OA/BA for LL/LH/HL/HH and mean. The mean row is the arithmetic mean of four band accuracies, not a fused predictor. source_standalone_probe_task_means.csv averages episodes within task; source_standalone_probe_summary.csv averages tasks equally within dataset and retains every band plus mean. Episode/query-event counts include reuse across tasks; these are source diagnostics, not independent representation-held evaluations.
