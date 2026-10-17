# HANDOFF: source diagnostic complete

Scope: cached fourfold, source fit+selection only, one fixed uint8_spatial alpha=.75. No training, downloads,
extraction, target query/support CSVs, target manifest, or target predictions. All writes are in this directory.

- 137 exact old source tasks, 3415 events, 1013 unique query observations/883 ships/22 acquisitions.
- Cross selection: unique-query macro Brier gain +0.000044820; old equal-task macro gain
  -0.000297942; BA -0.183824 pp. One rescue/four harms, all bulk.
- Query positive-only oracle 0.001124678: NONREALIZABLE true-label upper bound.
- Weak/inconsistent observability and ambiguity associations; no trained/validated gate. 184 queries overlap
  pooled fit and selection across different folds. Context repeats are not independent evidence.
- Same-scene fit AP macro +.126406 is mostly one crude-oil AP@1 change; consult source_class_coverage.csv.
- 37 queries have invalid descriptors; 3 additional fit queries have zero-coverage native contrast missingness.

Read FINDINGS.md and source_summary.csv first. Per-query continuous outcomes and both probability vectors:
source_query_events.csv. Unique-query aggregation: unique_source_queries.csv. Original episode identities:
source_task_rows.json. Support identities/classes: source_supports.csv. SAR measurements: source_observability.csv.
Class/acquisition/fold counts and associations are separate CSVs. selection_decision_changes.csv contains six flips.
VERIFICATION.json and SUMMARY_VERIFICATION.json contain the focused checks. No root/brain/model edits.

Loader caveat: original public.load_fold always opens forbidden target CSVs, so the local script uses its source-only
projection, exact old source rows, original public PCA and original task/predict helpers. Labels come only from source
split CSVs. Old train+val unsupervised PCA and source-train descriptor scaling exposure are preserved.

Main agent: training is yours; no learned-reader artifacts were created here. Findings describe the fixed previous
descriptor and cannot decide whether a trainable reader can use additional information or attain oracle gains.
