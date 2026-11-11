# Historical Reference-Bank Size Diagnostic

This is the 2026-10-30 corrected-anchor candidate diagnostic, NOT a complete sensitivity study of the current RCEW model.
Only saved observations are exported. No training, inference, score-array reading, interpolation, imputation or mock data is performed.

## Files

- reference_long.csv: one row per observed condition and metric; exact JSON pointer and CSV line provenance.
- reference_plot_ready.csv: one row per condition, OA/BA mean and sample SD, n, model/cohort/strategy identity.
- reference_bank_membership.csv: actual bank member IDs, class labels, ships and supplied polarizations. Supplied views are not a guarantee of valid regional evidence.
- provenance.json: actual class counts, exact query IDs, model paths (not loaded), sampling details, source-code locations and exclusions.

## Observations

| Dataset | Strategy | References/class | Total | OA (%) | BA (%) | n | SD |
|---|---|---:|---:|---:|---:|---:|---|
| oss2 | coverage | 8 | 32 | 57.894736842 | 50.123533739 | 1 | not estimable |
| oss2 | coverage | 16 | 64 | 57.894736842 | 50.123533739 | 1 | not estimable |
| oss2 | original | 8 | 32 | 57.894736842 | 50.123533739 | 1 | not estimable |
| oss2 | random | 8 | 32 | 57.894736842 | 50.123533739 | 1 | not estimable |
| oss2 | random | 16 | 64 | 57.894736842 | 50.123533739 | 1 | not estimable |
| private | coverage | 2 | 16 | 45.000000000 | 34.724025974 | 1 | not estimable |
| private | coverage | 4 | 32 | 45.000000000 | 34.724025974 | 1 | not estimable |
| private | original | 2 | 16 | 45.625000000 | 34.928107607 | 1 | not estimable |
| private | random | 2 | 16 | 45.625000000 | 34.928107607 | 1 | not estimable |
| private | random | 4 | 32 | 45.000000000 | 34.278756957 | 1 | not estimable |

## Interpretation And Plotting

Use separate panels for private and OSS2. Keep original, random and coverage separate; series_id defines a connectable series. Only random and coverage have two size points per dataset. Original is a single marker at its actual size, not an extra repeat or an observed horizontal baseline.
Private: 160 fixed historical development queries; 8-class model and source bank, but only 7 classes present in the query cohort. BA averages those 7 present classes. Sizes 2 and 4 mean 16 and 32 stored references.
OSS2: 228 fixed source-validation queries, four present classes. Sizes 8 and 16 mean 32 and 64 stored references. These are not the 378 final evaluation queries.
All points use one already-trained model-order endpoint (17). Random-bank seed 1030 is a sampling seed, not another model-training seed. Coverage is deterministic given the source descriptors/order. The original bank is inherited from its manifest; its construction seed is not inferred from the model seed.
The historical probe freezes network parameters while replacing source-fit references. Later center/control training uses seeds 17/29/43 and different parameter states; those results are excluded. This preparation itself performs no training or inference.
Each condition has n=1. Mean equals the saved observation. Sample SD is undefined and is an empty CSV field, not 0. Do not draw SD error bars or claim independent replication. Query count is not replicate n; strategies are not replicates.
Random selection resets default_rng(1030) per call and samples eligible observations without within-class MMSI reuse. Different budgets are not guaranteed nested banks. Coverage minimizes inverse-MMSI-weighted nearest-descriptor distance using the corrected protected anchor path. It uses source-fit data only.
Sizes count stored observations per class before query ID/MMSI masking and polarization/region validity. Exact eligible-count ranges and per-bank members are recorded; no hidden change of the x-axis to effective polarized evidence counts is made.
The private random doubled bank has lower OA/BA than the original; both coverage sizes have OA 45.0. All OSS2 OA/BA points coincide. Preserve overlapping values as separate series in the data; do not jitter numeric values or select the best bank. Probability metrics can change even when OA/BA agree.

## Missing Requested Points

Requested grid: 1, 2, 4, 8, 16 per class. Availability is scoped to the audited corrected historical probe, not a claim about every experiment in the workspace.
- private: observed [2, 4]; missing [1, 8, 16].
- oss2: observed [8, 16]; missing [1, 2, 4].
- The current RCEW model has no matching size-sweep observations in this export; none of these historical points may be relabeled as current RCEW.
- Missing budgets have no rows in the performance CSVs. Do not interpolate, add placeholder points or combine private 2/4 with OSS2 8/16 into a curve.

## Sources And Exclusions

Primary study: E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/m3_reference_coverage_20261030
Use anchor_corrected/{private,oss2}/results_s17.json, corroborated by analysis/anchor_corrected/{private,oss2}/endpoints.csv and saved NPZ ids/y. FINDINGS_ZH.md and the final correction section of PROTOCOL_ZH.md take precedence over the earlier uncorrected narrative.
The preserved legacy probe used live layer3 for statistics instead of protected anchor_last. Its results are not mixed into the corrected experiment.
Also inspected: E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/local_evidence_transport/conditional_relation_operator/reference_resampling/FINDINGS_ZH.md
The 2026-10-24 reference-resampling study has eight references/class at all six banks, a different conditional-relation candidate, different query cohorts, and a separate episodic-training phase. It cannot fill any size point of the 2026-10-30 model. Original and random banks there also differ in representativeness and pass coverage.
No new chip-level causal claims are made by this numeric preparation. Historical case analysis remains in the source study's CASE_NOTES_ZH.md; plotting these saved points is not a new experimental round.

## Reproduce

Run with an environment that already provides NumPy:

```powershell
python -B "E:/临时会话/paper_draft_20261108/editable_figures_20261111/prepare_reference.py"
```

The script verifies actual reference class counts, query ID/label equality and JSON/CSV metric agreement before writing. A mismatch stops output instead of silently changing study identity. It imports no project model/training module and writes only this reference_data directory.
