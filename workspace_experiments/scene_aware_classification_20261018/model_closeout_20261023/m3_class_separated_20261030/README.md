# Class-separated M3 experiment

This is a scoped research experiment, not a new standalone model release. It imports the existing paired-repair runner and the `sar_m123_standalone` package used by that runner. Existing original checkpoints, local chip manifests and source schedules are required; they are not uploaded here.

- `class_match.py`: old M3 plus a zero-start, per-class distance writeback. Two matched arms: aligned local values and region-pooled values.
- `experiment.py`: source-only scale calibration and training of the new writeback. Existing M1/M2/backbone/head/M3 weights remain unchanged.
- `evaluate.py`: frozen broad-cohort replay of both new arms, original full model, and the exact same-budget old-M3 continuation.
- `analysis.py`: CPU ID-paired comparisons, source-mean offset diagnostic, class/pass outcome accounting and bounded native-chip galleries.
- `test_class_match.py`: local correspondence, reference-order invariance and missing-reference tests.
- `PROTOCOL_ZH.md`, `HISTORY_INTERFACE_ZH.md`, `CASE_NOTES_ZH.md`, `FINDINGS_ZH.md`: protocol, interface audit, visual review and results.

Run in the original local environment, with this directory on the import path:

```powershell
python -m unittest discover -s . -p test_class_match.py
python experiment.py --dataset private --seeds 17 29 43
python experiment.py --dataset oss2 --seeds 17 29 43
python evaluate.py --dataset private
python evaluate.py --dataset oss2
python analysis.py --dataset all --seeds 17 29 43 --scope evaluation --final-gallery
```

Existing training/evaluation outputs are intentionally not overwritten. Do not rerun completed GPU jobs merely to inspect reports. Training orders reuse the established initial models and are not independent pretraining repetitions. Evaluation uses historical research cohorts, not a new blind benchmark. Target labels never select the new branch scale or hyperparameters.

Related method: CrossTransformers: spatially-aware few-shot transfer (Doersch, Gupta, Zisserman, NeurIPS 2020), arXiv:2007.11498. The borrowed idea is a query-to-class-aligned-prototype distance, not an original invention of class-conditioned attention. This experiment instead tests a limited residual addition to the existing SAR pipeline.
