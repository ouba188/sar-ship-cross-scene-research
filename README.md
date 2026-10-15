# SAR Ship Cross-Scene Research Archive

Private working archive of the SAR ship-classification experiments through 2026-10-15. This is a research snapshot, not a release of the datasets or a claim that every candidate became part of the final method.

## Current Study

The current strict protocol tests eight-class closed-set recognition across held ports or acquisitions. SAR-R18 self-supervision is limited to legal source chips for each fold. A frozen R50 C4 base and an R50 + SAR-R18 expert each receive source-prior-anchored adaptation from the same small labeled target support set; their probabilities are fused equally. The recent strict experiments use B=16 support examples, not a universal fixed budget.

The dual representation has scoped gains over the specified single-branch readout on OSS1, OSS2, and six held ports. This does not establish superiority over every strong single backbone. Fixed class relations and native crop/box geometry are conditional candidates; scene-mean subtraction did not transfer to the strict held-port probe. The 5% output handoff was retired from the active method. See `workspace_experiments/scene_cross_dataset_redesign_20261014/CURRENT_COMPONENTS_ZH.md` and the dated findings before citing any result.

## Contents

- `workspace_experiments/`: experiment code, notes, plots, configs, and result tables from the dated October 3-15 workspace studies, including negative results.
- `paper_tracking_sensitivity/`: October 2026 SAR sensitivity code and reports with its small result tables.
- `legacy_paper_20261007/`: older manuscript notes from the preceding repository, retained as historical context only; they do not describe the current strict model.
- `snapshot.ps1`: exact local file-selection and compression procedure used to assemble this archive.
- `archive_manifest.json`: source scopes, file counts, and size accounting generated at snapshot time.

CSV tables larger than 1 MiB are stored as `.csv.gz` at the same relative path. Standard gzip tools or Python's `gzip.open` can read them. Excluded are original SAR imagery, chips, pretrained/model weights, full feature tensors, PDF literature copies, and large cache arrays. Absolute paths in scripts describe the original Windows workstation and must be adjusted for another machine.

This archive is a development record. The datasets have been used repeatedly for model development; support draws are not independent backbone seeds; some held folds contain fewer than eight query classes. Refer to the individual reports for each protocol and denominator. Do not add improvements from different protocols or treat diagnostic oracle interventions as deployable methods.
