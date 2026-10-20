# Native axial strip sequence experiment

See FINDINGS_ZH.md for results, PROTOCOL_ZH.md for pre-result design,
DATA_RATIONALE_ZH.md for chip evidence, and VERIFICATION_ZH.md for checks.

Local prerequisites: native PNG cohort and metadata, frozen R50 checkpoint,
native_axis_probe_20261020 artifacts, and sampling_preservation_20261020
MSE keep_refs checkpoints. These data and weights are not in the archive.

```powershell
python -m unittest test_sequence -v
python run.py extract
python run.py train
# Only to continue interrupted, already-saved groups:
python run.py train --resume
python analyze.py
python coverage_diagnosis.py
python diagnose_content.py
```

Large feature arrays and checkpoints remain local. CSV files over 1 MiB may
be gzip-compressed in the research archive. Case panels are representative
diagnostics, not raw dataset exports. No complete support-adaptation system
or independent unseen dataset confirmation is claimed.
