# Ready For GPU Handoff

2026-10-10: eleven CPU tests pass (7.047 seconds in the project CUDA environment,
without CUDA execution). Actual checkpointed R50 tail receives gradients; its
frozen prefix does not. The balanced dual solve agrees with an independent
weighted primal solve. Support ordering/query chunks preserve predictions;
target-label mutation leaves input design unchanged; support/query overlap fails.
The source prior matches balanced sklearn RidgeClassifier with unpenalized
intercept; changing held rows/labels leaves its fit unchanged. Absent-support
classes retain distinct prior evidence, and actual prior-head gradients are nonzero.

`prepare` completed without CUDA:

- Exact established OSS2 source rows/exclusions and cached pixel IDs match.
- Source validation ports: Antwerp-Bruges, Hamburg, Jebel Ali, Santos.
- Inner training before identity purge:28,329; after:25,830.
- Shared MMSI removal:2,499. Additional shared-product removal:0.
- Validation:8,667 rows. All8 classes occur in train and validation.
- Immutable study meaning: one-seed bounded tail-adaptation pilot, not full
  strong-single qualification or established convergence.

This worker did not start GPU training. Main worker subsequently reported launch
of the corrected three-seed experiment; the live training file is now untouched.

Pre-launch correction: restore a same-backbone trainable Linear2049x8 source
prior, initialized by balanced source ridge alpha1 with unpenalized intercept.
Inner initialization uses inner train only; final refit initialization uses all
allowed source rows. Train layer4 AND prior using residual-query CE plus fixed
0.25 prior-query CE. Frozen comparator retains the all-source fitted initial
prior. This prevents forced identical scores for classes absent from a64-label
draw. The old zero-prior design was NOT RUN and is superseded, not a negative
result. Known support labels, not query metrics, motivated the correction.

GPU initialization now writes `CACHE_ALIGNMENT.json` before source validation:
actual current C4 geometry versus the initialization cache on the first16 inner
training rows, with relativeL2/max difference and source-prior logit RMS/argmax
agreement. Its actual values are produced by the main-worker run; no small-mismatch or
rounding-only conclusion is assumed. The eleven unchanged CPU tests were not
rerun for this bounded logging addition.

CPU four-cell diagnosis is ready in `DIAGNOSTIC.md`; execute after all three
seeds complete. The new diagnostic script does not change training parameters.

```powershell
& 'E:/Hermes/paper_tracking/envs/msfa-openmmlab-py310-rtx50-cu132/Scripts/python.exe' -u 'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/strong_single_r50_task.py' all
```

Default budget:6x32 source episodes, eight fixed validation episodes per epoch
including epoch0, source-selected refit of at most6x32 episodes, and two OSS2
extractions. Rough estimate10-30 minutes; not yet GPU-measured. If a smaller
runtime smoke is needed, use a separate `--out` under this result directory and
explicit `--epochs 1 --episodes 2`; do not merge that smoke with the default pilot.

Files written only: `strong_single_r50_task.py` in `oof_reliability_check` and
this dedicated result directory (test, protocol, ready note and source split).
SARCLIP files, shared brain and existing results are untouched.
