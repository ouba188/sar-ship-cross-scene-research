# Fixed Reader Calibration

Completed 2026-10-22. Read `FINDINGS_ZH.md` and the frozen `PROTOCOL.md`.

Run `run.py`, `analyze.py`, then `loss_diagnostic.py` in the existing ClearSAR
environment. Reuses sibling research modules and local feature/checkpoint caches;
these large artifacts and raw datasets are intentionally not redistributed.

Three calibration splits x three seeds are repeated evaluations, not an inference
ensemble or nine independent datasets. The old full-source comparator has only
three trained seeds: performance SD uses three, while paired rescue/harm retains
nine comparisons to different reduced-source references. See `summary_counts.csv`.

Same-reader reference OA/BA 39.535/41.314%, calibration score gate 39.507/41.267%.
Old full-source reference remains 41.512/42.707%. Preserve that old candidate;
do not promote the scalar gate. All zero-residual groups and negative results stay.

No new upstream model training, no source-calibration feedback into the reader,
no target-label checkpoint selection, no new independent confirmation claim.
