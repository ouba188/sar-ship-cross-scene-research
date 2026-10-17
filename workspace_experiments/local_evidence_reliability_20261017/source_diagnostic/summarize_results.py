"""Summarize saved source diagnostics and check their actual-data accounting."""
from pathlib import Path
import json
import os

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def table(frame, columns=None):
    frame = frame if columns is None else frame[columns]
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "NA" if pd.isna(value) else f"{value:.8g}"
        return str(value)
    return "\n".join(["| " + " | ".join(frame.columns) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |"
                      for row in frame.itertuples(index=False, name=None)])


def main():
    events = pd.read_csv(HERE / "source_query_events.csv", dtype={"mmsi": str})
    supports = pd.read_csv(HERE / "source_supports.csv", dtype={"mmsi": str})
    tasks = pd.read_csv(HERE / "source_tasks.csv")
    unique = pd.read_csv(HERE / "unique_source_queries.csv", dtype={"mmsi": str})
    summary = pd.read_csv(HERE / "source_summary.csv")
    acq = pd.read_csv(HERE / "source_acquisitions.csv")
    corr = pd.read_csv(HERE / "descriptive_associations.csv")
    classes = pd.read_csv(HERE / "source_classes.csv")
    verification = json.loads((HERE / "VERIFICATION.json").read_text())

    # A wrong class weight, serialization, or identity join changes these results.
    # Fail before writing a report if its accounting differs from the saved events.
    y = events.class_id.to_numpy(int)
    risks = []
    for prefix in ("baseline", "fixed"):
        p = events[[f"{prefix}_p{k}" for k in range(8)]].to_numpy()
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
        brier = ((p - np.eye(8)[y]) ** 2).sum(1)
        np.testing.assert_allclose(brier, events[f"{prefix}_brier"], atol=1e-12)
        risks.append(brier)
    np.testing.assert_allclose(risks[0] - risks[1], events.brier_gain, atol=1e-12)
    contribution = events.groupby("task_id").macro_brier_gain_contribution.sum()
    np.testing.assert_allclose(contribution, tasks.set_index("task_id").loc[contribution.index].macro_brier_gain,
                               atol=1e-12)
    query_means = events.groupby(["kind", "origin", "global_row"]).brier_gain.mean()
    saved_means = unique[~unique.origin.eq("pooled")].set_index(["kind", "origin", "global_row"])
    np.testing.assert_allclose(query_means, saved_means.loc[query_means.index].brier_gain, atol=1e-12)
    support_counts = supports.groupby(["task_id", "class_id"]).size()
    count_index = pd.MultiIndex.from_frame(events[["task_id", "class_id"]])
    np.testing.assert_array_equal(support_counts.reindex(count_index, fill_value=0), events.support_sameclass_count)
    assert supports.groupby("task_id").size().eq(16).all()
    support_ships = supports.groupby("task_id").mmsi.agg(set)
    assert all(not set(f.mmsi) & support_ships.loc[key] for key, f in events.groupby("task_id"))
    assert verification["source_episode_rows_equal_old"]

    coverage = []
    for keys, f in unique.groupby(["kind", "origin", "class_id", "class_name"]):
        matching = events[events.kind.eq(keys[0]) & events.class_id.eq(keys[2])]
        if keys[1] != "pooled":
            matching = matching[matching.origin.eq(keys[1])]
        coverage.append(dict(zip(["kind", "origin", "class_id", "class_name"], keys),
            unique_queries=len(f), unique_ships=f.mmsi.nunique(),
            descriptor_valid_queries=int(f.descriptor_valid.sum()),
            retrieval_eligible_queries=int(f.ap_delta.notna().sum()),
            retrieval_eligible_events=int(matching.ap_delta.notna().sum()),
            support_sameclass_min=int(matching.support_sameclass_count.min()),
            support_sameclass_max=int(matching.support_sameclass_count.max())))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(HERE / "source_class_coverage.csv", index=False)
    fold_summary = tasks.groupby(["kind", "origin", "fold"]).agg(
        episodes=("task_id", "size"), macro_brier_gain=("macro_brier_gain", "mean"),
        ba_delta_pp=("macro_ba_delta_pp", "mean"), rescues=("rescues", "sum"), harms=("harms", "sum")).reset_index()
    fold_summary.to_csv(HERE / "source_fold_summary.csv", index=False)
    flips = events[events.kind.eq("selection") & (events.rescue.eq(1) | events.harm.eq(1))]
    flips.to_csv(HERE / "selection_decision_changes.csv", index=False)

    cross = summary[summary.origin.eq("cross")]
    select = cross[cross.kind.eq("selection")].iloc[0]
    unique_cross = unique[unique.kind.eq("selection") & unique.origin.eq("cross")]
    acq_cross = acq[acq.kind.eq("selection") & acq.origin.eq("cross")]
    correlation = corr[corr.origin.eq("cross")].pivot(
        index="feature", columns=["kind", "unit"], values="spearman_brier_gain")
    correlation.columns = [f"{a}_{b}" for a, b in correlation.columns]
    correlation = correlation.reset_index()
    select_classes = classes[classes.kind.eq("selection") & classes.origin.eq("cross")]
    select_classes = select_classes.merge(coverage[coverage.kind.eq("selection") & coverage.origin.eq("cross")],
        on=["kind", "origin", "class_id", "class_name"], validate="one_to_one")
    report = f"""# Source-side fixed local evidence: completed diagnostic

Date: 2026-10-17. Fixed `uint8_spatial`, alpha=0.75, cached four folds, B16, both original fit/selection episode sets.
No new model, selector, alpha search, extraction, download, target prediction or target query/support CSV read.

## Findings

The fixed increment has small continuous source benefits, but usable selection benefit is not established.
For cross-acquisition selection, unique-query class-macro Brier gain is +{select.query_macro_brier_gain:.9f},
whereas the original equal-task macro objective changes by {select.task_mean_macro_brier_gain:.9f} (negative is harm).
Unique-query class-macro BA changes by {select.query_macro_ba_delta_pp:.6f} percentage points,
with one rescue and four harms, all bulk-carrier queries. True-class probability decreases despite the tiny
global class-macro Brier improvement. These are different estimands, not contradictory arithmetic.

The per-query positive-only oracle is {select.query_oracle_macro_brier_gain:.9f} Brier units,
or {100*select.query_oracle_macro_brier_gain/select.baseline_query_macro_brier:.4f}% of old source Brier.
It is a **NONREALIZABLE upper bound using true source query labels**, not a deployable gate or attainable gain.
Observed harm cancels {100*(1-select.query_macro_brier_gain/select.query_oracle_macro_brier_gain):.2f}% of this
query-oracle positive mass. Observable associations are weak, inconsistent between fit and selection,
or change with aggregation. No simple observability/confidence rule is established here.

## Actual Data And Exact Scope

- 110 fit + 27 selection tasks = 137; 3,415 source query-context rows.
- 875 distinct fit queries / 773 ships / 21 acquisitions; 322 distinct selection queries / 311 ships / 13 acquisitions.
- Their union is 1,013 observations / 883 ships / 22 acquisitions; 184 observations occur in fit in one fold and selection in another.
- Each fold keeps the old acquisition/MMSI-disjoint source split. All query/support/prior global rows exactly match the old `source_task_rows.json`.
- 128 query-context rows (37 distinct queries) have no valid descriptor and retain the old output. Three other fit queries (rows 522, 528, 951) have valid axes but zero native-region coverage and missing contrast; their predictions remain included.
- Same-class support is absent in 168 query-context rows. AP@R is undefined there, never imputed as zero; per-class retrieval eligibility is saved separately.
- Source pool union is 1,335 of the cached observations; 4-fold reuse does not create independent samples. Selection is held within a fold, not independent of the pooled fit data across folds.

The original `public.load_fold` unconditionally reads target query/support CSVs. To honor the source-only boundary,
`diagnose_source.py:source_fold` reuses its source file choice and cache-row mapping, strips non-source rows,
and reads labels only from `source_train.csv` / `source_validation.csv`. The descriptor sample-ID axis replaces
reading the global target manifest. This is the sole loader adaptation; original `public.pca`,
`probe.standardize`, `make_episodes`, `make_task`, `predict`, and `map_at_r` are reused.
PCA retains the original unsupervised train+selection source exposure and ordering. Descriptor scaling uses
valid source TRAIN rows only. Fit query/support rows therefore contribute unlabeled source scaling/PCA just as before.
Only the original analytic priors and support coefficients are reconstructed, with no learned new readout.

## Paired Continuous And Decision Results

Positive Brier gain means old minus fixed; true-probability/AP/BA deltas mean fixed minus old.
All Brier values are sums of squared error across eight classes, not divided by eight.
Unique-query results first average repeated fold/support contexts, then average within class and across observed classes.
Task results use the original per-task observed-class macro weights, then equal task weights.

{table(summary, ['kind','origin','unique_queries','query_macro_brier_gain','task_mean_macro_brier_gain','query_macro_true_probability_delta','query_macro_ap_delta','query_macro_ba_delta_pp'])}

Acquisition-class-macro cross-selection gain is {acq_cross.brier_gain.mean():.9f}, positive in
{int(acq_cross.brier_gain.gt(0).sum())}/{len(acq_cross)} acquisitions. Cross-fit task Brier improves in all four folds;
cross-selection improves in two and worsens in two. This does not establish reliable source-held improvement.

## Class And Support Composition

{table(select_classes, ['class_name','queries','ships','descriptor_valid_queries','retrieval_eligible_queries','mean_brier_gain','mean_true_probability_delta','mean_ap_delta','ba_delta_pp'])}

Bulk accounts for 204/322 cross-selection queries and loses Brier and BA; general cargo (27 queries) gains Brier.
Container (51) also loses Brier. Global eight-class averaging and acquisition/task observed-class averaging weight
these populations differently, explaining the reported sign change without attributing it to a physical mechanism.
Tug (5) and offshore supply (2) selection queries have no valid descriptor and stay unchanged.

The seemingly large fit/same AP delta (+0.126406 class-macro) is dominated by ONE eligible crude-oil query,
global row 132, with one same-class support and AP@1 changing 0 to 1. Only one of six fit/same crude-oil queries
has a same-class support in the observed contexts. This is not broad retrieval improvement; preserve its denominator.
Cross-selection has 291 retrieval-eligible unique queries; AP delta +0.000341 does not imply a decision benefit.
All six selection decision-change events have exactly zero AP delta, so the observed flips do not require AP-rank change.

{table(flips, ['fold','origin','global_row','mmsi','support_sameclass_count','rescue','harm','brier_gain','true_probability_delta','ap_delta'])}

## Local Observability And Ambiguity

Native measurements are existing calibrated sigma0 measurements, VV and VH separate; the fixed descriptor
itself remains the old uint8 representation. Contrast is calibrated body-versus-equal-region-neighbor dB,
top5 mass is the highest 5% of body pixels' response share, and extent ratios are native 90% response spans
relative to the detected box. Regions are not physical segmentation masks. Descriptor coverage and native
region coverage have different geometries and are reported separately. No AIS/environment feature enters prediction.
Ambiguity is the old full probability entropy / top-two gap; same-class support count uses source query truth and
is diagnostic composition information, not an inference-time gate feature.

Spearman associations below use unique query means or one class-balanced record per acquisition, never repeated
support events as independent samples. They are descriptive, not p-value tests, causal effects, or fitted gate evidence.
Unique-query counts are 875 fit / 322 selection before missingness; acquisition counts are only 21 / 13.
Ships can recur across acquisitions, so unique observations are not a claim of independent vessels.

{table(correlation)}

For example, selection-query VV contrast rho=+0.111 versus fit -0.059; native coverage -0.028 versus -0.074;
entropy +0.054 versus -0.007. VV contrast has acquisition rho=+0.495 in selection but -0.297 in fit.
Same-class support count reverses from +0.101 to -0.278 at query level, with different class/scene composition.
These patterns do not support a stable threshold. Source-fit median strata are saved without fitting a gate;
their class sets differ (e.g. high support count has only two classes), and folds overlap, so they are not independent validation.
Selection coverage-high strata can have better Brier yet worse BA. Do not label this brightness/background/wake causality.

## Oracle And Context Dependence

{table(summary, ['kind','origin','query_oracle_macro_brier_gain','event_oracle_query_macro_brier_gain','beneficial_queries','harmful_queries','zero_gain_queries','context_mixed_queries'])}

Primary oracle: clip each query's context-mean Brier gain at zero, then class-average. It makes an impossible
true-label-informed choice between the two already computed predictors for each query.
Secondary event oracle: clip each query-context gain before context averaging. It has additional access to support
context outcomes and is a looser upper bound. Neither trains, evaluates, or certifies an executable policy.
In pooled same/cross contexts, 538/875 fit and 163/322 selection queries have both positive and negative Brier changes.
That establishes support/fold dependence of the measured effect, not that an image-only reader can learn the oracle.
Cross-only selection has 130 beneficial, 182 harmful, 10 unchanged queries after averaging contexts.

## Focused Verification And Handoff

All 137 old source episode identities match. Query/support and prior/query-support MMSI overlaps are zero;
prior acquisitions exclude query/support acquisitions. Zero-dose full probability replay maximum error is
{verification['baseline_endpoint_max_error']:.3g}; query-weighted Brier versus old `risk` differs by at most
{verification['macro_risk_max_error']:.3g}. Invalid-query fallback is exact. Threadpools and Torch were limited to one.
Saved probabilities reconstruct saved Brier; per-event macro contributions reconstruct tasks; source support labels
reconstruct same-class counts; unique-query gains reconstruct the saved nonduplicated summary.
These checks detect actual baseline, weighting, join and identity errors; no broad audit or new synthetic test suite was added.

The main trainable-readout experiment may use these files to inspect source supervision, continuous risks, available
support classes and invalid local coverage. This sidecar neither duplicates that training nor recommends a new dose,
selector, or target experiment. Fixed-descriptor headroom is small and heterogeneous; an attainable learned gain remains
for the separately authorized main experiment to establish.

Run: `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B diagnose_source.py`, then
the same interpreter with `-X utf8 -B summarize_results.py`, from this directory. Only this directory is written.
"""
    (HERE / "FINDINGS.md").write_text(report, encoding="utf-8")
    handoff = f"""# HANDOFF: source diagnostic complete

Scope: cached fourfold, source fit+selection only, one fixed uint8_spatial alpha=.75. No training, downloads,
extraction, target query/support CSVs, target manifest, or target predictions. All writes are in this directory.

- 137 exact old source tasks, 3415 events, 1013 unique query observations/883 ships/22 acquisitions.
- Cross selection: unique-query macro Brier gain +{select.query_macro_brier_gain:.9f}; old equal-task macro gain
  {select.task_mean_macro_brier_gain:.9f}; BA {select.query_macro_ba_delta_pp:.6f} pp. One rescue/four harms, all bulk.
- Query positive-only oracle {select.query_oracle_macro_brier_gain:.9f}: NONREALIZABLE true-label upper bound.
- Weak/inconsistent observability and ambiguity associations; no trained/validated gate. 184 queries overlap
  pooled fit and selection across different folds. Context repeats are not independent evidence.
- Same-scene fit AP macro +.126406 is mostly one crude-oil AP@1 change; consult source_class_coverage.csv.
- 37 queries have invalid descriptors; 3 additional fit queries have zero-coverage native contrast missingness.

Read FINDINGS.md and source_summary.csv first. Per-query continuous outcomes and both probability vectors:
source_query_events.csv. Unique-query aggregation: unique_source_queries.csv. Original episode identities:
source_task_rows.json. Support identities/classes: source_supports.csv. SAR measurements: source_observability.csv.
Class/acquisition/fold counts and associations are separate CSVs. selection_decision_changes.csv contains six flips.
VERIFICATION.json and SUMMARY_VERIFICATION.json contain the focused checks. No root/brain/model edits.

Loader caveat: original public.load_fold always opens forbidden target CSVs, so the local script uses its source-only
projection, exact old source rows, original public PCA and original task/predict helpers. Labels come only from source
split CSVs. Old train+val unsupervised PCA and source-train descriptor scaling exposure are preserved.

Main agent: training is yours; no learned-reader artifacts were created here. Findings describe the fixed previous
descriptor and cannot decide whether a trainable reader can use additional information or attain oracle gains.
"""
    (HERE / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    check = dict(status="passed", probability_rows=len(events), tasks=len(tasks),
        identity_support_rows=len(supports), unique_kind_origin_queries=len(saved_means),
        checks=["saved probability normalization and Brier reconstruction", "task macro contribution accounting",
                "unique query context averaging", "sameclass counts from actual support labels",
                "B16 support sizes", "query-support MMSI exclusion", "original episode receipt"],
        selection_flip_events=len(flips), selection_cross_unique_retrieval_eligible=int(unique_cross.ap_delta.notna().sum()))
    (HERE / "SUMMARY_VERIFICATION.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    files = sorted(path.name for path in HERE.iterdir() if path.is_file())
    listing = "# Files Created\n\nAll are under this source_diagnostic directory; no pre-existing files were edited.\n\n"
    listing += "\n".join(f"- `{name}`" for name in files if name != "CHANGED_FILES.md")
    listing += "\n- `CHANGED_FILES.md`\n"
    (HERE / "CHANGED_FILES.md").write_text(listing, encoding="utf-8")
    print(json.dumps(check), flush=True)
    print("WROTE FINDINGS.md HANDOFF.md CHANGED_FILES.md", flush=True)


if __name__ == "__main__":
    main()
