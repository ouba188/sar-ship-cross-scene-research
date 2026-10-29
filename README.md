# SAR Ship Cross-Scene Research Archive

October 29 supplemental source fitting is complete: [six matched runs and native-case diagnosis](RESEARCH_UPDATE_20261029_ZH.md). Removing M1/M2 explicit frequency gives OA41.0417/BA33.1992 versus legacy41.6667/33.2229 over three training orders, with slightly worse probability metrics. Deleting the paths from fixed trained legacy weights changes no top-1 prediction in 480 repeated evaluations. Keep the simplification as a candidate, not an accuracy improvement or established equivalence. The full intervention-selection/adaptation stage remains pending; all current jobs ended and default weights are unchanged.

October 29 closeout decision: [remove explicit frequency branches from M1/M2](workspace_experiments/sar_m123_current_20261027/SPATIAL_M12_CLOSEOUT_ZH.md), retaining existing M3 spatial-frequency matching. New architecture and checkpoint transfer are implemented; 4 new tests, 6 existing tests and a two-chip end-to-end smoke check pass. This is an unvalidated weight transfer, not a new accuracy result. Historical checkpoints and placement experiments are retained.

Latest incremental progress: [October 28 within-stage M2 depth comparison](RESEARCH_UPDATE_20261028_ZH.md). With one equal-width pre-add M2 per polarization, layer2[2] achieves OA47.0833/BA36.5317 on the fixed 160-query historical development cohort, versus layer2[3]46.6667/36.0543 and layer2[1]43.3333/34.0461. Middle placement gains BA over the last block in all three training orders but loses NLL; stage output still has the highest BA36.9726. Six new fits and nine replayed controls, no tuning or 3075-query selection. Ten native query VV/VH pairs and actual references were reviewed. This is a scoped candidate, not a new default or independent confirmation. Prior frozen3075 results remain below. The [October 27 engineering release](RESEARCH_UPDATE_20261027_ZH.md) remains unchanged. Weights, raw datasets and private manifests are excluded. Older descriptions below are historical.

Public research archive of the SAR ship-classification experiments through 2026-10-17. This is a research snapshot, not a release of the datasets or a claim that every candidate became part of the final method.

Start with [the latest October 17 supplement](RESEARCH_APPEND_20261017_ZH.md) for frozen-reader results, native-chip tracing, and the completed chip-quality study. [The earlier October 17 update](RESEARCH_UPDATE_20261017_ZH.md) retains the preceding studies and their limitations. Earlier findings are historical evidence, not silently replaced by the newest candidate.

## Current Study

The current strict protocol tests eight-class closed-set recognition across held ports or acquisitions. SAR-R18 self-supervision is limited to legal source chips for each fold. A frozen R50 C4 base and an R50 + SAR-R18 expert each receive source-prior-anchored adaptation from the same small labeled target support set; their probabilities are fused equally. The recent strict experiments use B=16 support examples, not a universal fixed budget.

The dual representation has scoped gains over the specified single-branch readout on OSS1, OSS2, and six held ports. This does not establish superiority over every strong single backbone. Fixed class relations and native crop/box geometry are conditional candidates; scene-mean subtraction did not transfer to the strict held-port probe. The 5% output handoff was retired from the active method. See `workspace_experiments/scene_cross_dataset_redesign_20261014/CURRENT_COMPONENTS_ZH.md` and the dated findings before citing any result.

The October 15 body/context probe found a small, scoped gain from a ring-only supplement on the six evaluated held-acquisition tasks (cross-acquisition OA/BA 52.433/46.405% versus 52.204/46.037% for its paired baseline). Three folds improved and three declined; learned conditional weighting added no target OA/BA gain. This remains a candidate, not a replacement for the main model. See `workspace_experiments/prepool_local_probe_20261015/body_context/FINDINGS_ZH.md` for the exact protocol, per-fold results, and limitations.

## Contents

- `workspace_experiments/`: experiment code, notes, plots, configs, and result tables from the dated October 3-17 workspace studies, including negative results.
- `workspace_tools/`: the repaired OBB preview utility and its regression tests used by the geometry investigations.
- `paper_tracking_sensitivity/`: October 2026 SAR sensitivity code and reports with its small result tables.
- `legacy_paper_20261007/`: older manuscript notes from the preceding repository, retained as historical context only; they do not describe the current strict model.
- `snapshot.ps1`: exact local file-selection and compression procedure used to assemble this archive.
- `archive_manifest.json`: source scopes, file counts, and size accounting generated at snapshot time.

CSV tables larger than 1 MiB are stored as `.csv.gz` at the same relative path. Standard gzip tools or Python's `gzip.open` can read them. Excluded are original SAR imagery, chips, pretrained/model weights, full feature tensors, PDF literature copies, and large cache arrays. Absolute paths in scripts describe the original Windows workstation and must be adjusted for another machine.

Image panels in reports are diagnostic figures, not a complete image dataset. External environment caches and vendored dependencies are excluded. The completed October 17 chip-quality study is now included; its multi-object detections are screening candidates, not independently confirmed labels or a measured cause of classification loss. Private manual-review state and browser-test exports are excluded. The review-tool sources are archived, but local imagery, compressed CSV restoration, and the Lucide browser bundle from its declared npm dependency are needed to run that local application.

This archive is a development record. The datasets have been used repeatedly for model development; support draws are not independent backbone seeds; some held folds contain fewer than eight query classes. Refer to the individual reports for each protocol and denominator. Do not add improvements from different protocols or treat diagnostic oracle interventions as deployable methods.
