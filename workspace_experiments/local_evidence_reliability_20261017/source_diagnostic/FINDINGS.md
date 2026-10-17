# Source-side fixed local evidence: completed diagnostic

Date: 2026-10-17. Fixed `uint8_spatial`, alpha=0.75, cached four folds, B16, both original fit/selection episode sets.
No new model, selector, alpha search, extraction, download, target prediction or target query/support CSV read.

## Findings

The fixed increment has small continuous source benefits, but usable selection benefit is not established.
For cross-acquisition selection, unique-query class-macro Brier gain is +0.000044820,
whereas the original equal-task macro objective changes by -0.000297942 (negative is harm).
Unique-query class-macro BA changes by -0.183824 percentage points,
with one rescue and four harms, all bulk-carrier queries. True-class probability decreases despite the tiny
global class-macro Brier improvement. These are different estimands, not contradictory arithmetic.

The per-query positive-only oracle is 0.001124678 Brier units,
or 0.1368% of old source Brier.
It is a **NONREALIZABLE upper bound using true source query labels**, not a deployable gate or attainable gain.
Observed harm cancels 96.01% of this
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

| kind | origin | unique_queries | query_macro_brier_gain | task_mean_macro_brier_gain | query_macro_true_probability_delta | query_macro_ap_delta | query_macro_ba_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit | cross | 875 | 8.3606661e-05 | 0.00016328626 | 8.6378781e-05 | -0.00024963352 | 0.0054824561 |
| fit | pooled | 875 | 0.00011302381 | 9.2505425e-05 | 0.00010997218 | 0.0064845865 | -0.028705761 |
| fit | same | 807 | 0.00022092004 | 1.0801834e-06 | 0.00013175993 | 0.12640625 | -0.068111024 |
| selection | cross | 322 | 4.4820024e-05 | -0.00029794233 | -7.3534343e-05 | 0.00034141971 | -0.18382353 |
| selection | pooled | 322 | 7.4553564e-05 | -0.00034571861 | 1.1608258e-06 | 0.00031168634 | -0.091911765 |
| selection | same | 300 | 0.00064990002 | -0.00040543897 | 0.00035426182 | 0.00058364417 | 0.066489362 |

Acquisition-class-macro cross-selection gain is -0.000294398, positive in
5/13 acquisitions. Cross-fit task Brier improves in all four folds;
cross-selection improves in two and worsens in two. This does not establish reliable source-held improvement.

## Class And Support Composition

| class_name | queries | ships | descriptor_valid_queries | retrieval_eligible_queries | mean_brier_gain | mean_true_probability_delta | mean_ap_delta | ba_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bulk_carrier | 204 | 195 | 204 | 204 | -0.0010016201 | -0.000709609 | 0.00047937357 | -1.4705882 |
| fishing_vessel | 5 | 5 | 4 | 1 | 0.00049742397 | 7.2289114e-05 | 0 | 0 |
| general_cargo | 27 | 27 | 26 | 25 | 0.0023447904 | 0.0010369356 | 0.0025 | 0 |
| product_chemical_tanker | 26 | 25 | 25 | 20 | 3.6342003e-05 | 2.4073817e-05 | 0.003125 | 0 |
| container_ship | 51 | 50 | 51 | 35 | -0.0009440148 | -0.00068069223 | -0.0033730159 | 0 |
| crude_oil_tanker | 2 | 2 | 2 | 1 | -0.00057436129 | -0.00033127202 | 0 | 0 |
| tug_towing | 5 | 5 | 0 | 4 | 0 | 0 | 0 | 0 |
| offshore_supply | 2 | 2 | 0 | 1 | 0 | 0 | 0 | 0 |

Bulk accounts for 204/322 cross-selection queries and loses Brier and BA; general cargo (27 queries) gains Brier.
Container (51) also loses Brier. Global eight-class averaging and acquisition/task observed-class averaging weight
these populations differently, explaining the reported sign change without attributing it to a physical mechanism.
Tug (5) and offshore supply (2) selection queries have no valid descriptor and stay unchanged.

The seemingly large fit/same AP delta (+0.126406 class-macro) is dominated by ONE eligible crude-oil query,
global row 132, with one same-class support and AP@1 changing 0 to 1. Only one of six fit/same crude-oil queries
has a same-class support in the observed contexts. This is not broad retrieval improvement; preserve its denominator.
Cross-selection has 291 retrieval-eligible unique queries; AP delta +0.000341 does not imply a decision benefit.
All six selection decision-change events have exactly zero AP delta, so the observed flips do not require AP-rank change.

| fold | origin | global_row | mmsi | support_sameclass_count | rescue | harm | brier_gain | true_probability_delta | ap_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pilot | same | 842 | 412419520 | 10 | 1 | 0 | 0.012799014 | 0.0063925091 | 0 |
| S1A_011246_011040 | cross | 47 | 412422480 | 3 | 1 | 0 | 0.0049353494 | 0.0027513824 | 0 |
| S1A_011246_011040 | cross | 342 | 412029000 | 3 | 0 | 1 | -0.0015621865 | -0.00089596779 | 0 |
| S1A_011246_011040 | cross | 315 | 351772000 | 3 | 0 | 1 | -0.0036557461 | -0.0014677761 | 0 |
| S1A_011421_0115FD | cross | 94 | 413352380 | 5 | 0 | 1 | -0.00056988642 | -0.00032587814 | 0 |
| S1A_011421_0115FD | cross | 716 | 566743000 | 14 | 0 | 1 | -0.0034233147 | -0.0016419398 | 0 |

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

| feature | fit_unique_query | fit_acquisition_class_macro | selection_unique_query | selection_acquisition_class_macro |
| --- | --- | --- | --- | --- |
| baseline_entropy | -0.0071802272 | 0.25844156 | 0.054462978 | 0.098901099 |
| baseline_top2_gap | 0.0057636067 | -0.014285714 | 0.042227342 | 0.032967033 |
| descriptor_min_coverage | -0.076714217 | 0.098701299 | -0.0042808181 | -0.21428571 |
| native_min_coverage | -0.073521163 | 0.11688312 | -0.028279984 | -0.24175824 |
| support_sameclass_count | 0.10143364 | -0.024033778 | -0.27846352 | 0.15427056 |
| vh_native_contrast_db | -0.010661905 | -0.35844156 | 0.018151912 | -0.42857143 |
| vh_native_long_extent_ratio | 0.028907186 | 0.17272727 | 0.024529835 | 0.071428571 |
| vh_native_top5_mass | 0.008926206 | -0.30779221 | 0.11962932 | 0.027472527 |
| vv_native_contrast_db | -0.058961548 | -0.2974026 | 0.11146509 | 0.49450549 |
| vv_native_long_extent_ratio | 0.011183021 | 0.20779221 | -0.080277986 | -0.44505495 |
| vv_native_top5_mass | 0.050635746 | -0.077922078 | 0.088565098 | -0.18131868 |

For example, selection-query VV contrast rho=+0.111 versus fit -0.059; native coverage -0.028 versus -0.074;
entropy +0.054 versus -0.007. VV contrast has acquisition rho=+0.495 in selection but -0.297 in fit.
Same-class support count reverses from +0.101 to -0.278 at query level, with different class/scene composition.
These patterns do not support a stable threshold. Source-fit median strata are saved without fitting a gate;
their class sets differ (e.g. high support count has only two classes), and folds overlap, so they are not independent validation.
Selection coverage-high strata can have better Brier yet worse BA. Do not label this brightness/background/wake causality.

## Oracle And Context Dependence

| kind | origin | query_oracle_macro_brier_gain | event_oracle_query_macro_brier_gain | beneficial_queries | harmful_queries | zero_gain_queries | context_mixed_queries |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit | cross | 0.00094845723 | 0.0010991489 | 417 | 424 | 34 | 220 |
| fit | pooled | 0.00072153031 | 0.0010938156 | 457 | 384 | 34 | 538 |
| fit | same | 0.00099868843 | 0.0011302998 | 407 | 367 | 33 | 177 |
| selection | cross | 0.0011246777 | 0.0011457003 | 130 | 182 | 10 | 13 |
| selection | pooled | 0.00099039819 | 0.0013623742 | 145 | 167 | 10 | 163 |
| selection | same | 0.0020221473 | 0.0020292236 | 160 | 130 | 10 | 11 |

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
2.22e-16; query-weighted Brier versus old `risk` differs by at most
3.33e-16. Invalid-query fallback is exact. Threadpools and Torch were limited to one.
Saved probabilities reconstruct saved Brier; per-event macro contributions reconstruct tasks; source support labels
reconstruct same-class counts; unique-query gains reconstruct the saved nonduplicated summary.
These checks detect actual baseline, weighting, join and identity errors; no broad audit or new synthetic test suite was added.

The main trainable-readout experiment may use these files to inspect source supervision, continuous risks, available
support classes and invalid local coverage. This sidecar neither duplicates that training nor recommends a new dose,
selector, or target experiment. Fixed-descriptor headroom is small and heterogeneous; an attainable learned gain remains
for the separately authorized main experiment to establish.

Run: `D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B diagnose_source.py`, then
the same interpreter with `-X utf8 -B summarize_results.py`, from this directory. Only this directory is written.
