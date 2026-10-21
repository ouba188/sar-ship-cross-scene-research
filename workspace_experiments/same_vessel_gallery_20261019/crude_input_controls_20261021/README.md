# Crude tanker input controls

Start with `FINDINGS_ZH.md`, `PAPER_NOTES_ZH.md` and `PROTOCOL.md`.

This archive contains code, tables and diagnostic case panels, not the raw dataset, model weights or feature caches. Local source code dependencies remain in the adjacent `conditional_evidence_pilot` and `local_condition_module` directories. Image paths refer to the research machine. CSV files larger than 1 MiB may be stored as `.csv.gz` in the GitHub archive; the local working files remain ordinary CSV.

## Reproduce locally

Use the existing ClearSAR Python environment. No packages were installed in this round.

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 input_controls.py extract
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 input_controls.py fit
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 input_controls.py perturb
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 analyze_results.py
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 verify_results.py
```

The source-coverage and radiometry subdirectories contain their own scripts and reports. Main table `summary.csv` concerns newly fitted diagnostic heads, while source-coverage error association uses the previously frozen paired-relation baseline. They must not be conflated.

Completed: matched input heads, functional PNG perturbations, source geometry and displayed-response coverage, original input case review, local paper full-text review. Incomplete: verified calibrated two-date input pairs, physical attribution of display change, integration of these controls into the stronger complete classification pipeline.
