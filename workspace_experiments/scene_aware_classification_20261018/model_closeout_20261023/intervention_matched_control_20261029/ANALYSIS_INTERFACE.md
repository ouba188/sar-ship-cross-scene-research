# Matched intervention CPU analysis

Preparation only. Run formal analysis only after the main agent explicitly says
all six control prediction archives are complete. No prediction polling, training,
inference, GPU use, Git operations or new cohorts. Outputs stay in this directory's
`analysis/`; helpers are imported read-only from the adjacent full/refit analyses.

## Commands

Use the existing ClearSAR environment, without bytecode writes:

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/intervention_matched_control_20261029/test_analysis.py'
# Only after the main agent's completion notification:
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/intervention_matched_control_20261029/analysis.py'
```

`--package PATH` optionally selects the native input preparation package. Default:
`E:/临时会话/sar_m123_current_20261027`. The output root is fixed to this script's
directory. No model, runner or torch is imported. Tests use synthetic data and
temporary files strictly under `analysis/`; they never open real predictions.

## Input contract

- New: `predictions/{development,expanded}_control_s{17,29,43}.npz` (6).
- Reused from adjacent `spatial_m12_full_20261029`:
  `predictions/{development,expanded}_{before,after}_spatial_s{17,29,43}.npz`
  (12). `after` is named `guided` in outputs.
- Each NPZ: finite `score[N,8]` logits, unique `ids[N]`, integer `y[N]`.
  Existing helper aligns by ID and validates labels; order may differ.
- Reused `spatial_m12_full_20261029/cohorts.json`: fit1205, adapt136,
  development160, expanded3075, references16. No new manifest is produced.
- New receipts: `training/control_s{seed}/training.json` (3).
  Guided receipts: adjacent full `training/spatial_s{seed}/training.json` (3).
  Receipts are preserved verbatim with paths; no new receipt schema is required.
- Existing native image paths and input preparation must remain accessible.

Formal invocation checks all files once before loading predictions or writing
formal outputs. Missing inputs stop immediately. Six ready NPZs alone do not
substitute for completed training receipts. Source helper imports perform no
analysis. This prepared script has not certified the future runner's losses,
frozen parameters or reference use; those remain evidenced by its code/receipts.

## Conditions and attribution

Three conditions: `before`, `control`, `guided`. Main comparison:
`guided_minus_control`. Auxiliary: `control_minus_before`, `guided_minus_before`.
All deltas are alternative minus reference.

All use native160, the same source fit/init/order and reference16. Control uses
CE + 0.1 KL, coefficient=0, useful all False and no intervention-selection loss.
`teacher_original` reuses existing original scores from the same source-fitted
teacher; this is still the KL anchor. Guided additionally uses the original
intervention-selected `margin_floor`, coefficient 0.3683066666126251. Matched
continuations are planned for 6 epochs/54 steps. The main contrast isolates the
incremental selected-guidance training procedure conditional on this design;
it does not separately isolate selection versus floor loss, or prove a physical
mechanism. Guided-before alone also includes ordinary continuation.

Seeds17/29/43 change source-fit/adaptation training order with common initialization;
they are NOT independent initializations. Both evaluation cohorts are historical,
NOT blind tests. Repeated observations, vessels, reference exposures and seed
events are not independent samples.

## Outputs

- `metrics.csv`: each condition/cohort/seed OA, BA, NLL and Brier.
- `paired_metrics.csv`: three within-seed comparisons per cohort.
- `groups.csv`, `groups_summary.csv`: paired statistics by class, port,
  pass and actual available polarization; means, sample SD, transitions,
  unique query counts and prediction changes including wrong-to-wrong changes.
- `seed_events.csv`: all comparison/query/seed events, plus all three conditions'
  predictions and true-vs-top-rival/fixed-rival margins. Fixed rival is the
  strongest nontruth class of mean spatial-before logits across the three seeds.
- `summary.json`, `NUMERIC_REPORT.md`: condition means/sample SD, paired deltas,
  transitions and 2000 paired whole-pass bootstrap draws (seed20261029) for both
  cohorts. Bootstrap keeps observations/seeds paired, averages seed differences,
  uses present-class BA and counts resamples omitting an originally present class.
  Intervals are conditional on fitted models, not independent-training confidence;
  development has few passes. OA/BA percent and pp deltas; natural-log NLL;
  Brier sums over eight classes. Absent-class statistics are blank, not zero.
- Reused coverage helper emits `cohort_groups.csv`, `cohort_overlap.csv`,
  `reference_pairs.csv` from the existing manifest, without creating new cohorts.
- `cases/index.html`, `cases/cases.json`, `cases/native_sources.json` and PNGs:
  up to eight unique expanded guided-control cases, covering rescue/harm/CC/WW
  where present; at least four available unique queries are selected even with
  no flips. Net seed outcomes and fixed-margin changes guide selection; partial
  polarization is included when suitable. Outcome selection is not prevalence.
  Each case shows actual eligible same-class and rival references, excluding
  same ID/MMSI. Rival union covers all seeds and all three conditions, including
  before-only or other-seed wrong predictions. Per-seed predictions, logits and
  true-vs-rival margins are retained for all three conditions.
- VV/VH panels are separate, one source pixel per output pixel, no resize or
  contrast stretch; original display chip, exact native160 crop/pad and duplicate
  OBB overlay use the existing helper. Actual availability follows
  `prepare_view` plus `valid_target`, not mere path presence. Source paths,
  placement and reference IDs remain traceable. Display intensity is not raw
  radiometry. Generated cases and sources remain `visual_status=not_reviewed`;
  the main agent must inspect them and distinguish visible facts from hypotheses.
- `complete.json` is written only after numeric tables/report and gallery finish;
  it marks CPU generation complete, never human visual review complete.

## Focused verification

Seven synthetic CPU tests target wrong file routing/incomplete inputs, ID/label
misalignment, comparison sign/three-condition margins, known metric values and
sample SD, cluster bootstrap versus explicit whole-pass resampling, all-state
case selection/reference exclusions and native pixel scale/actual polarization.
No real cohort prediction or formal analysis is needed for these tests.
