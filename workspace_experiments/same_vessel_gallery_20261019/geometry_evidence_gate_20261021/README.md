# Geometry evidence gate

Start with FINDINGS_ZH.md. The current main model is unchanged. This completed
probe has 45 source-only gates over an already frozen neighborhood residual;
correct geometry does not show clear incremental benefit over content/sham.

```powershell
$py = 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe'
& $py -X utf8 run.py
& $py -X utf8 analyze.py
& $py -X utf8 diagnose.py
```

Requires adjacent prior experiments and local input/feature/weight caches.
Archive omits NPZ/PT, original datasets and credentials; large CSV are gzipped.
`content` in machine-readable tables means content plus geometry-validity flags.
`reference` means frozen A+r, not A. Zero-step gate preserves A+r. `baseline`
means A. Sham uses one fixed donor realization per fold shared across seeds.

`oracle_family_*` uses true labels only AFTER fitting to diagnose which errors
could be repaired within the scalar-weight family. It is not deployable accuracy,
not a source-selected model, and never enters training or candidate selection.
`joint_zero_geometry` is an artificial numeric-slot ablation, not a physical angle.
Bootstrap intervals are conditional on current fitted models and development data.
