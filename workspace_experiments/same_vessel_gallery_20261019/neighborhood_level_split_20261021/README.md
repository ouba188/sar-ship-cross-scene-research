# Neighborhood level split, 2026-10-21

Read `FINDINGS_ZH.md` for results and limits; `PROTOCOL.md` for the pre-training
comparison. Three identity/date folds, three seeds, 27 residual-head fits on a
frozen stronger native classifier. Not the full target-support system.

Execution in the existing ClearSAR environment:

```powershell
python -X utf8 run.py extract
python -X utf8 run.py train
python -X utf8 analyze.py
python -X utf8 case_evidence.py
```

The scripts reuse sibling experiment modules, source image manifests and local
frozen weights. GitHub contains code, tables, protocol, review and native case
panels, not private raw datasets, model weights or feature caches. Tables over
1 MB are gzip-compressed in the archive. Paths in case tables refer to local
input provenance and are not bundled image files.

`summary_by_seed.csv` is the primary pooled test table. `metrics.csv` also has
per-fold source train/validation, terminal-training and selected scores.
`paired_uncertainty.csv` is conditional on fixed trained models and resamples
MMSI within class after averaging seed correctness. `case_interventions.csv`
records artificial fixed-head input removals, not physical causal interventions.

No tests or metrics choose new target-dependent hyperparameters. Independent
review is in `review/`. The main model was not overwritten.
