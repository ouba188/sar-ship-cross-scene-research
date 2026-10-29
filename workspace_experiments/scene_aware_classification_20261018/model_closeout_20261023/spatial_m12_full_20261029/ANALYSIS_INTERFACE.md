# CPU analysis interface

Prepared for the main agent. Formal analysis must wait for its explicit completion
notification. There is no polling, training, inference or parameter selection in
this script. It writes only `analysis/` under the selected run directory.

## Run after completion

```powershell
& 'D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe' -X utf8 -B 'E:/临时会话/scene_aware_classification_20261018/model_closeout_20261023/spatial_m12_full_20261029/analysis.py'
```

Optional `--root PATH` changes the input/output run directory; `--package PATH`
changes the native chip preparation package. The default package is the same
`sar_m123_current_20261027` used by the previous source-fit analysis. Metric,
ID-alignment and native-panel helpers are imported read-only from sibling
`spatial_m12_refit_20261029/analysis.py`; preserve that file when archiving.
No torch, GPU model, runner or feature cache is imported.

## Required inputs

- `cohorts.json`: `fit`1205, `adapt`136, `development`160, `expanded`3075,
  `references`16. Rows use the existing native manifest schema: `id`, `label`,
  `mmsi`, `port`, `pass_group`, `vv`, `vh`.
- 24 NPZs: `predictions/{scope}_{stage}_{arm}_s{seed}.npz` with
  scope `development/expanded`, stage `before/after`, arm `legacy/spatial`,
  seed `17/29/43`; `score` is finite N-by-8 **logits**, `ids` contains observation
  IDs, `y` contains integer labels. IDs may be reordered; the analysis joins by ID.
- Six JSON lists: `selection/{arm}_s{seed}.json`, one record per adapt observation,
  using the native `select_records` schema: `id`, `useful`, `original`, `altered`,
  `choice`, `selected_intervention` (and optional `label`). Final `useful` and
  earlier `choice.useful` remain distinct. Exact intervention intersection uses
  `kind/pol/region/band` after restricting to final-useful IDs.
- Six receipts: `training/{arm}_s{seed}/training.json`. Each receipt is retained
  verbatim in the analysis summary, including nested diagnostics.

Missing inputs stop the script before formal outputs. Native availability uses
the existing `input.prepare_view` validity rule for each polarization, not merely
the presence of a path. Original display chips must remain accessible for this
step and for the gallery. No raw radiometric interpretation is inferred.

## Outputs and interpretation

- `metrics.csv`: four model conditions, two cohorts, three paired order seeds.
- `paired_metrics.csv`, `groups.csv`, `groups_summary.csv`, `seed_events.csv`:
  final spatial-minus-legacy and each architecture's after-minus-before;
  class/port/pass/actual available-polarization groups; means, sample SD,
  rescue/harm/CC/WW counts, unique IDs and prediction changes including WW.
- `summary.json`, `NUMERIC_REPORT.md`: CPU numerical results, all receipts,
  scope limitations and expanded paired pass-bootstrap intervals (2000 draws,
  seed 20261029). The same pass resample applies to all fixed models; seed deltas
  are averaged. These intervals do not estimate independent-training uncertainty.
  BA uses classes present in each resample; omitted-class replicates are counted.
- `cohort_groups.csv`, `cohort_overlap.csv`, `reference_pairs.csv`: observation,
  ship/pass/port composition, directional overlap and native ID/MMSI exclusions.
- `selection_records.csv`, `selection_groups.csv`, `selection_groups_summary.csv`,
  `selection_summary.json`: final versus earlier screen, selection counts by
  class/polarization/region and architecture ID/intervention intersections.
- `cases/index.html`, `cases/cases.json`, `cases/native_sources.json`: eight unique
  expanded queries and sixteen separate native-size VV/VH panels with actual
  eligible true/rival references. Each case preserves all four conditions and
  all three seeds' logits and margins. Prefer net rescue/harm and consistent
  CC/WW, with partial-missing coverage when a suitable candidate exists. Cases
  are outcome-selected, not population estimates. Automated generation leaves
  `visual_status=not_reviewed` for the main agent's image review.
- `complete.json`: written only after numerical tables and gallery complete.

The same seed controls source-fit and adaptation order, with a common
initialization. Source-fit data are matched, source-adapt is separate from fit,
and the two architectures select their own teachers. The final comparison does
not isolate selection effects. Development and expanded data are historical
cohorts; neither becomes a fresh blind test. Only layer3.5.conv1/2/3 are adapted;
frozen heads still participate in differentiable computation. The CPU script
does not certify gradient flow or internal reference usage from logits alone.

## Synthetic verification

Run `test_analysis.py` with the same Python and `-X utf8 -B`. It tests known
OA/BA/NLL/Brier values, ID reordering and label disagreement, paired transitions
and missing classes, pass bootstrap against explicit whole-pass resampling,
final-useful selection intersections, case prioritization, native pixel scale
and syntax. It does not read real predictions or start formal analysis.
