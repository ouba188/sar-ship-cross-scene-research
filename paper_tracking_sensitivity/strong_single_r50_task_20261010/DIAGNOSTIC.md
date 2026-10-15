# Three-Seed CPU Diagnosis

Ready; not yet run on real completed seeds. Four synthetic CPU tests passed in
1.448 seconds. They check independent float64 kernel-form agreement, exact
float32 original-path replay, preserved mismatch receipts, unbalanced-class
rescue/BA accounting, and complete three-seed report aggregation.

After all three `seed_*/COMPLETE.json` receipts exist:

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' -u 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r50_task_diagnose.py'
```

The diagnostic is CPU-only, uses default seeds20261010/20261011/20261012, and
never modifies the live training implementation or original predictions.
Estimated duration: tens of seconds to a few minutes, depending on CPU/memory
pressure. Output goes to `strong_single_r50_task_20261010/diagnosis/`.

## Four Fixed Cells

1. Frozen prior + frozen kernel: exact original frozen comparator.
2. Trained prior + frozen kernel: complete prior-function intervention.
3. Frozen prior + trained kernel: support-relation intervention.
4. Trained prior + trained kernel: exact original trained classifier.

Each prior is evaluated on its own matching embedding, not on the crossed
kernel embedding. Every cell re-solves the balanced residual system at lambda8.
The main computation uses the original float32 Torch primal correction path,
not a rearranged direct kernel multiplication that could move borderline argmax.

Computed predictions are persisted before exact diagonal checks and before
reading target query truth. Each seed writes `seed_*_DIAGONAL_REPLAY.json` with
exact status and mismatch counts/locations. A mismatch aborts the formal report;
original predictions remain untouched. Approximate agreement is not silently
promoted to exact agreement.

## Outputs And Interpretation

- Four full eight-class classifiers: OA/BA per seed and support draw.
- Five-support-draw averages within each neural seed, then neural-seed mean and
  sample SD. Separate support-draw averages across neural seeds are also saved.
- Candidate rescues AND harms versus frozen single and `dual_refit_late` from
  the existing matched OSS2 dual reference. Net OA/BA accounting is checked.
- Per-class recall, signed net counts, support coverage and harms are retained.
- The existing dual is one fixed reference reused across comparisons; its five
  draws are not three newly trained dual seeds.
- Functional prior/kernel effects and nonlinear interactions can localize gains
  or degradation, but do not identify physical mechanisms or prove causality.
  These crossed cells are diagnostics, not query-selected deployment policies.

The generated `FINDINGS.md` summarizes actual outcomes only after real completion.
Synthetic test output is not a research result.
