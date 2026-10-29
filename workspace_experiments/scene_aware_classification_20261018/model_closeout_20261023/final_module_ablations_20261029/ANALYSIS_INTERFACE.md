# Private retrained ablations: CPU analysis interface

2026-10-29 completion update: explicit READY received; formal CPU analysis and
native gallery completed successfully (session70992 exit0). See
`analysis/CPU_HANDOFF_ZH.md`, `analysis/complete.json` and
`analysis/CASE_TIMELINE.md`. All24 native panels remain `not_reviewed`.

The following describes the preparation/run contract. Formal analysis must wait for an explicit main-agent READY
notification covering all predictions, training receipts and evaluation checks.
The training-active notification is not READY. No polling or real-NPZ loading is
needed to prepare/test this code. No training, model/runner import, torch, GPU or
Git operations are performed. All writes remain in the four authorized targets:
`analysis.py`, `test_analysis.py`, this document and `analysis/`.

## Commands

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/test_analysis.py'
# ONLY after the main agent explicitly confirms everything READY:
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/final_module_ablations_20261029/analysis.py' --main-agent-ready
```

The formal CLI without `--main-agent-ready` exits before reading input files.
This switch records an already received notification; its availability is not
permission to run early. `--package PATH` optionally selects the native-input
package; default is `E:/临时会话/sar_m123_current_20261027`.

## Input contract

Paths below are relative to this analysis directory unless explicitly marked as
sibling directories. No manifest/cohort or runner source is modified.

- New NPZs (48): `predictions/{development,expanded}_{plain,no_m1,no_m2,no_m3}_{before,after}_s{17,29,43}.npz`.
- Reused full NPZs (12): sibling `spatial_m12_full_20261029/predictions/{development,expanded}_{before,after}_spatial_s{17,29,43}.npz`.
- Reused full_control after (6): sibling `intervention_matched_control_20261029/predictions/{development,expanded}_control_s{17,29,43}.npz`.
- full_control before aliases the full before archive. New-arm before always uses
  that arm's own sourcefit predictions. There are 72 logical endpoints and 66
  unique archives; shared full/control before is loaded once per cohort/seed.
- NPZ keys: finite logits `score[N,8]`, unique `ids[N]`, integer `y[N]` in the
  existing eight-class order. The completed source-fit helper joins by ID and
  validates labels, so stored row order may differ from the cohort manifest.
- Cohorts: sibling `spatial_m12_full_20261029/cohorts.json`, unchanged fit1205,
  adapt136, development160, expanded3075, references16.
- New sourcefit receipts (12): `source_fit/training/{arm}_s{seed}/training.json`.
- New adapt receipts (12): `training/{arm}_s{seed}/training.json`.
- Reused fit receipts (3): sibling `spatial_m12_refit_20261029/training/spatial_s{seed}/training.json`.
- Reused guided receipts (3): sibling `spatial_m12_full_20261029/training/spatial_s{seed}/training.json`.
- Reused control receipts (3): sibling `intervention_matched_control_20261029/training/control_s{seed}/training.json`.
- Completion gate files: `checks.json` and `evaluation_check.json`, plus the
  explicit READY notification. Their complete JSON content is retained verbatim;
  the main agent owns their pass/fail interpretation. Their existence alone is
  not interpreted as a passed check.
- Native query/reference image paths and the completed `input.py` must remain accessible.

The loader checks the complete file set before opening any predictions. Receipt
seed, steps408/54, epochs8/6 and non-smoke status are checked before NPZ loading;
raw receipts are retained with paths. A mismatch stops analysis for a concrete
completion/path repair. It does not infer gradients, disabled paths, teacher
provenance or save/replay equivalence from logits. Those are runner/check evidence.

## Comparisons and interpretation

Main table: `full_after`, `plain_after`, `no_m1_after`, `no_m2_after`, `no_m3_after`.
Primary paired contrasts: `full_minus_{plain,no_m1,no_m2,no_m3}_after`.
Separate control: `full_control_minus_plain_after`. Supplementary source-fit
contrast: `full_minus_plain_before`. Each of the six models also has its own
`{arm}_after_minus_before` comparison, for 12 total contrasts in each cohort.
Every contrast is alternative minus reference: positive OA/BA and negative
NLL/Brier favor the alternative. The fixed rival for all comparisons is the
strongest nontruth class of mean full-sourcefit logits over the three order seeds.

full retains spatial M1/M2 and M3. no_m1/no_m2/no_m3 are retrained deletions of all
dedicated disabled-module paths and losses, certified by main-agent pre-save
checks, not runtime toggles on trained full. Each uses its own source-fit teacher,
intervention selection and guidance. Plain uses its own CE+0.1KL continuation and
has no reference contribution. full versus plain includes architecture and
training-objective differences, so it is not pure architecture attribution.
full_control versus plain matches the subsequent CE+KL objective, but its source
fit losses, architecture and own teachers still differ. Per-module rows compare
the complete retrained procedures, not a fixed-teacher single-operation effect.

All arms match samples, initialization, permitted update range and step counts;
they do not match parameter counts or FLOPs. Receipt trainable counts and seconds
are preserved, but differing caches and machine-memory conditions preclude a
formal speed claim. Seeds17/29/43 are training-order repetitions with shared init;
CUDA is not strict deterministic. They are not independent initializations.
Both cohorts are historical nonblind data. Shared ports preclude a strict
held-port claim. Repeated queries, vessels, seeds and references are not new
independent evidence. No checkpoint, threshold or arm is selected using results.

## Reuse and outputs

Read-only imports: completed `intervention_matched_control_20261029/analysis.py`
provides source routing and reference-detail helpers; its full-stage dependency
provides grouped paired statistics, pass bootstrap, case ranking and coverage;
the source-fit dependency provides ID joins, metrics and native panels. There is
no new statistics framework and no import of the actively edited runner/model.

- `main_after_metrics.csv`: only the five main final endpoints, per cohort/seed.
- `metrics.csv`: all 12 conditions and three seeds in each cohort.
- `paired_metrics.csv`, `groups.csv`, `groups_summary.csv`: paired OA/BA/NLL/Brier,
  class/port/pass/actual-polarization groups, sample SD, transitions, unique IDs,
  and wrong-to-wrong prediction changes. Absent-class metrics remain blank.
- `seed_events.csv`: each paired query/seed event, both predictions, fixed and
  true-top-rival margins, NLL/Brier, query/MMSI/pass identity. Events are not
  independent samples. Gallery cases retain all 12 conditions' logits/margins.
- `summary.json`, `NUMERIC_REPORT.md`: separated main/control/sourcefit tables,
  12 paired contrasts, transitions, raw training/check receipts, source paths,
  and 2000 paired whole-pass bootstrap draws per contrast (seed20261029).
- Bootstrap keeps all observations/seeds paired, averages fixed-seed deltas,
  uses classes present in each resample and counts missing-class resamples.
  These are conditional descriptive intervals, not independent-training CIs or
  multiplicity-adjusted significance claims. Development has only three passes.
- `cohort_groups.csv`, `cohort_overlap.csv`, `reference_pairs.csv`: reused cohort
  coverage/exclusion outputs, without changing the supplied cohorts. References
  describe the support-based bank; they do not imply plain consumes reference values.
- `cases/index.html`, `cases.json`, `native_sources.json`, `coverage.json`, PNGs:
  target 8-12 unique expanded query IDs, also distinct nonempty MMSIs, across
  the three module contrasts and both plain contrasts. Retain rescued/harmed/CC/WW
  where available, with partial-polarization coverage; explicit coverage gaps
  replace invented cases. Distinct vessels can still share a pass, so this is
  not statistical independence. Cases are outcome-selected, not prevalence.
- Cases include all seeds/endpoints, before/after evidence, all contrasted model
  outcomes, actual eligible true/rival reference IDs and native chips. Rival
  union covers before-only and nonrepresentative-seed competitors. Exclude same
  ID/MMSI references. Plain has NO reference contribution; no_m3 has NO M3
  reference transport, and displayed references provide comparator context.
- VV/VH are separate one-source-pixel-per-output-pixel panels: original display
  chip, exact native160 crop/pad, and duplicate OBB overlay. No resize/stretch.
  Actual availability comes from `prepare_view` plus `valid_target`, not file
  presence. Source paths/placement are retained; display intensity is not
  calibrated radiometry. Every case and panel remains `not_reviewed` for the
  main agent's original-image review and interpretation.
- `complete.json`: emitted last after numeric outputs and native gallery; means
  CPU generation finished, never that images were reviewed or mechanisms proved.

## Synthetic validation

Verified 2026-10-29: 10 synthetic CPU tests PASS, exit0, unittest elapsed0.888s.
No formal predictions were read during those tests. Formal analysis subsequently
completed after explicit READY; the historical preparation record and completion
update are retained in `analysis/PREPARATION_STATUS.md`.

Post-analysis timeline export:
`python -X utf8 -B analysis/derive_timeline.py` reads only the existing analysis
CSVs/JSONs and writes `timeline_seed_events.csv`, `timeline_summary.csv`,
`case_timeline.csv`, `CASE_TIMELINE.md` and `timeline_complete.json` in `analysis/`.
It does not reload NPZs or recompute the statistical framework. Temporal joins
are cross-checked against saved final transition totals for all ten cohort/primary
contrast combinations; 48525 events and36 case/seed rows were exported.

Tests use tiny invented arrays, receipts and images in temporary subdirectories
of `analysis/`, then remove those fixtures. They never read real predictions.
They detect early formal execution, wrong endpoint/receipt routing, incomplete
training receipts, ID/label mismatch, nonfinite scores, incorrect contrast sign,
absent-class treatment, seed-event overcounting, incorrect bootstrap clusters,
repeated-vessel gallery cases, missed module/state coverage, missing rivals,
plain/reference misattribution and native-pixel scaling errors. Failure changes
the corresponding adapter or routing before any formal result is generated.
