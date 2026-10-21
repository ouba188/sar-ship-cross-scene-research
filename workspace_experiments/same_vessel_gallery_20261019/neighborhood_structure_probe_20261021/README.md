# Neighborhood structure probe

Read `FINDINGS_ZH.md` first. Two tracks are deliberately separate: unchanged
whole-chip baseline for incremental utility; pre-convolution body-only baseline
for mechanism isolation. Training, source selection and original-image review
are complete. This is not an independently confirmed full support-system result.

## Local reproduction

Use the existing ClearSAR environment and prior experiment caches in adjacent
directories. The Git archive does not include data, model weights or features.

```powershell
$py = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
& $py -X utf8 run.py extract
& $py -X utf8 run.py train
& $py -X utf8 body_only.py
& $py -X utf8 analyze.py
& $py -X utf8 analyze.py --body
& $py -X utf8 posthoc.py
```

`PROTOCOL.md` records the body-only amendment before training results were
inspected. `body_only/` contains that track; root tables contain the whole-chip
track. `summary_by_seed.csv` and `summary_mean_std.csv` report pooled metrics;
`paired_uncertainty.csv` uses class-stratified MMSI cluster resampling conditional
on the trained models. Three paired runs share test data. `case_selected_steps.csv`
distinguishes actual trained use from zero-step fallback. Names `rescue_vs_D` in
generic summaries refer to each track's own original-R reference, not a shared head.

`extreme_score_diagnostic.csv` and `outlier_sensitivity.csv` preserve the flat
control artifact at10461. Primary rows are unchanged. `reference_constant.csv`
checks a source-average residual without fitting a new model. Read the report
before interpreting constant-input or fixed-head interventions.

Archive: code, protocols, reports, bounded native case panels, small JSON/CSV.
Large CSV are gzip compressed. No NPZ/PT/model/data/credentials are published.
