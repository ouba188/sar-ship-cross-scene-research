# OSS2 CPU analysis interface

2026-10-29 update: explicit READY received; formal CPU analysis completed with exit 0. All 12 prediction archives, 12 training receipts and 12 evaluation gates were checked. Eight unique queries and sixteen native panels were generated, all not_reviewed. See `analysis/HANDOFF_ZH.md`. No torch import, GPU use, inference or training was performed by this analysis branch. The preparation-only statements below describe the earlier synthetic-test phase.

## Ownership and entry point

This branch writes only `analysis.py`, `test_analysis.py`, `analysis/`, and this interface document. It does not edit the runner, protocol, converter, shared model, private branch or Git state.

Only after the main agent explicitly says READY:

```powershell
python -B analysis.py --ready
```

Run from this directory. `--ready` represents the human/main-agent notification; file existence alone is not approval. Without it, the command stops before reading formal NPZs. There is no polling, scheduling or automatic launch. Preparation tests are independent:

```powershell
python -B -m unittest test_analysis -v
```

## Inputs and completion gates

- Twelve prediction archives: `predictions/evaluation_{full,plain}_{fit,adapt}_s{17,29,43}.npz`.
- Twelve training receipts: `training/{arm}_s{seed}/{fit,adapt}/training.json`.
- GPU gate: `gpu_checks.json`, status `passed`, explicit original class indices `[0,2,3,4]`.
- Evaluation gates: `evaluation/{arm}_{fit,adapt}_s{seed}.json`, six fit files and six adapt files. All twelve must exist before opening any NPZ.
- Current manifests: source_fit478, source_adapt241, validation228, selection194, evaluation378 and references32.

Training gates verify matching arm/stage/seed, fit160/adapt96 steps, 8/6 epochs, batch24/16, source count, reference count32, class names/indices, fixed-endpoint and frozen-state receipt, and save/native replay errors. Adapt coefficient must be full `.3683066666126251` or plain `0`. Evaluation gates verify matching arm/stage/seed, evaluation378, no checkpoint selection and native replay error. These are supplied completed-run checks; CPU logits are not treated as independent proof of GPU implementation correctness.

NPZ fields must be `score[N,4]`, `ids[N]`, integer `y[N]`, and `classes[4]`. Archives join by exact unique sample IDs, reorder to the manifest, and reject label mismatch, nonfinite scores, eight-class shape or incorrect class order. Output class names are explicitly:

| New label | Original label | Name |
|---:|---:|---|
|0|0|bulk_carrier|
|1|2|general_cargo|
|2|3|product_chemical_tanker|
|3|4|container_ship|

Evaluation must contain 378 queries, 369 ships and five pass groups. The actual 32 references must be independent MMSIs, eight per class and source-fit identities. The two development roles are counted as available only, with no scores requested or checkpoint choice.

## Statistics

Primary comparison: `full_adapt_minus_plain_adapt`. Auxiliary comparisons: `full_fit_minus_plain_fit`, `full_adapt_minus_full_fit`, and `plain_adapt_minus_plain_fit`. Every delta is alternative minus reference; higher OA/BA and lower NLL/Brier are favorable.

Report each order17/29/43 and mean/sample SD of OA, BA, NLL and four-class sum-Brier. Group tables cover four classes, pass group and effective polarization. Absent-class groups are blank; group BA averages present classes. Transition tables distinguish seed events, unique queries and unique ships, including queries rescued in one order but harmed in another.

For every comparison, 2000 paired whole-pass bootstrap draws use bootstrap seed20261029. Draw five passes uniformly with replacement, retain all paired observations in each selected pass, keep all three fixed model orders together, and average paired order deltas. Percentile95% intervals condition on these fitted models and only five clusters. Missing-class replicates are counted; BA uses classes present in that replicate. Repeated seeds, ships and support exposures do not become independent evidence.

This is a new architecture evaluated on historical public fold0, not a new blind test. Full/plain comparisons include differing architectures and fitting/adaptation objectives, not an isolated causal architecture effect. The three seeds share initialization.

## Native cases and review

Choose eight unique queries from the primary comparison and render sixteen separate VV/VH panels. First include one available net-rescue, net-harm, consistent-CC and consistent-WW query; then cover vv_only and vh_only where available; then fill remaining slots using the completed-round deterministic ranking. Absent outcome categories are recorded rather than invented. Outcome-selected cases do not estimate prevalence.

The fixed rival is the strongest nontruth class in mean full_fit logits across three orders, held fixed for all four endpoints/comparisons. Each case records all endpoint/order logits, predictions, top-rival margins and fixed margins. All eligible actual fixed-bank references are shown, with true/rival/other labels based on all displayed endpoint competitors. Candidate membership and labels do not imply high attention or a causal contribution. References apply to full only: plain is explicitly marked `NONE` and every panel receipt has `plain_reference_effect=false`.

Reuse the completed spatial-refit native panel helper, explicitly rebound to the four-class names in an isolated in-memory module instance. Read `sar_m123_standalone/sar_m123/input.py` directly without importing the torch-dependent package. Show original display pixels, exact native160 crop/pad, and separate OBB overlay with one original pixel per image pixel, no resize or stretch. Missing/invalid polarization is unavailable, not an observed black chip. Check selected native availability against converted manifests. Only selected queries and actual references need original image reads; no full-dataset image audit or feature cache is loaded.

Generated panels and descriptions remain `not_reviewed`. The main agent must inspect actual images and add visible facts, model-supported explanations, useful conditions and remaining uncertainty. No physical SAR cause is inferred from appearance, logits or an OBB alone.

## Outputs after READY

- `analysis/metrics.csv`, `paired_metrics.csv`, `groups.csv`, `groups_summary.csv`, `seed_events.csv`.
- `analysis/summary.json`: scope, explicit classes, cohort/ship/pass/polarization counts, original check receipts, four comparisons and bootstrap details.
- `analysis/RESULTS.md`: numerical tables and transition totals, with historical-fold and uncertainty boundaries.
- `analysis/cases/index.html`, `cases.json`, `native_sources.json`, `selection_receipt.json`, sixteen native PNG panels.
- `analysis/completion.json`: numerical completion only; image review remains pending.

No formal summary, accuracy claim, case selection or image-review result is generated during preparation.

## Preparation verification

Seven synthetic CPU tests passed on 2026-10-29. They cover READY/completion gating before NPZ reads; exact ID joining and four-class rejection paths; hand-checked Brier/BA/order SD; paired directions and all four comparisons; five-pass bootstrap against explicit repeated-pass enumeration; absent-state and missing-polarization case selection; reference exclusion/plain non-use; and eight synthetic queries/sixteen native panels with pixel-coordinate checks. A test import hook rejects torch and package imports. Synthetic fixtures were created only below `analysis/` and removed after tests.
