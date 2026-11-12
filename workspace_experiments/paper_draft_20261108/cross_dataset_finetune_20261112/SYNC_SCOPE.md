# Archive scope

This archive contains the source-only training/analysis scripts, frozen protocol,
aggregate results, training curves, checkpoint replay checks, and native-chip
review notes for the November 12-13 bidirectional fine-tuning experiment.

Source/target cohorts and staged source checkpoints are reused from
`../cross_dataset_retrained_20261111/`. They are not regenerated or replaced.

Raw chips, native caches, model/optimizer checkpoints, per-observation predictions,
manifests, and image galleries remain local. The local `index.html` contains 16
query-reference panels; 10 were inspected, as documented in `CASE_NOTES_ZH.md`.

The local manuscript's Section 6.3/Table 8 has been updated and rebuilt. Other
manuscript sections were being updated concurrently, so this experiment archive
does not overwrite the repository's manuscript or PDF. The values and scope to
insert are recorded in `RESULTS_ZH.md`.
