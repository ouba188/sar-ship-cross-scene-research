# Cross-polarization factorial diagnosis and candidate redesign

Completed fixed-model diagnosis, OSS2 validation, 2026-11-16. Reuses prior regional single-polarization intervention caches; adds joint VV/VH interventions for three regions and three seeds. No SAR model retraining or formal model replacement.

Read `FINDINGS_ZH.md`, `CASES_ZH.md`, `REDESIGN_ZH.md` and `PROTOCOL_ZH.md`. Numeric records: `class_effects.csv`, `seed_effects.csv`, `pass_effects.csv`. All mixed logit-margin differences are numerical roundoff: the current frozen-reference classifier is additive across polarizations.

`pcea.py` is a candidate regional interaction operator, not an integrated trained model. `check_pcea.py` checks identity initialization, availability/region fallback, trainability, actual cross-polar influence and feature-map writeback. Its synthetic checks do not establish SAR accuracy.

Local reproduction from this folder, using the existing project environment and data:

```powershell
python run.py
python analyze.py
python check_pcea.py
```

`run.py` requires `../region_dependency_20261116/run.py`, previous FP32 caches and existing Full checkpoints. `analyze.py` uses the same local native chip cache; SVGs retain editable text. Local-only `gallery/index.html` contains original chip examples and source paths. Git archive excludes raw chips, gallery images/JSON, weights, large caches and per-observation IDs/MMSIs.

The formal network, training data and manuscript are unchanged. Do not interpret intervention rescue as improvement from an untrained PCEA candidate.
