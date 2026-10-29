# Source non-regression objectives for M3

The operator and inference path remain the class-separated aligned M3 from the preceding experiment. This scoped pilot changes only the training loss for its zero-initialized writer. It imports the existing experiment helpers, checkpoints, native input manifests and source schedules; it is not a standalone data/weight release.

```powershell
python test_losses.py
python train.py --dataset private --seeds 17
python train.py --dataset oss2 --seeds 17
python source_analysis.py --gallery
```

The CPU test script explicitly invokes its three tests. Existing training receipts are not overwritten. Both dataset jobs are complete; consult the reports instead of restarting them.

- `losses.py`: focal logit matching with the paper's alpha=1, beta=5, lambda=1; and one-sided true-versus-rival margin preservation, plus the previous correct-teacher KL.
- `train.py`: same data, references, steps, initialization and new M3 writer; original M1/M2/backbone/head/M3 frozen. Source scale calibration, native replay and parameter-freeze checks retained.
- `source_analysis.py`: CPU-only retrospective source endpoint investigation for the preceding class-separated study.
- `guard_analysis.py`: current pilot source/development paired analysis and bounded original-chip panels.

Reference: Yan et al., Positive-Congruent Training: Towards Regression-Free Model Updates, CVPR 2021, sections 4-5, equations 4-9, Tables 2-3. Official PDF examined through text extraction, pages 3-6. FD-LM is borrowed, not a new loss claim. One-sided margin preservation is an experimental variant, not a verified new contribution. This study does not claim that stronger protection necessarily improves held-out recognition.

Archive excludes checkpoints, NPZ caches, full per-source ID tables and full per-query prediction tables. Summaries, code and bounded derived galleries are included. The formal model is not replaced by this pilot.
