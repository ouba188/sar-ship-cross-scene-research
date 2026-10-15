"""Frozen, metadata-selected same-vessel/different-acquisition R/T readout.

Write only acquisition_-prefixed artifacts next to this script.
"""

from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ports_root_cause_20261013"))
from analyze_factorial import load, metadata
from settings import labels, OUT, np, pd

FOLDS = (2, 3)
SCOPES = ("endpoint_excluded", "vessel_excluded")
FIELDS = ("incidence_xml", "heading_deg")
EVENTS = (
    "R_flip", "T_flip", "R_both_correct", "T_both_correct",
    "R_endpoint_accuracy", "T_endpoint_accuracy", "T_minus_R_endpoint_accuracy",
    "R_only_flip", "T_only_flip", "both_flip", "neither_flip",
    "T_rescue_endpoint", "T_harm_endpoint",
)


def describe(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return dict(n=len(values), minimum=float(values.min()) if len(values) else None,
                median=float(values.median()) if len(values) else None,
                maximum=float(values.max()) if len(values) else None)


def select_pairs(saved):
    ids = saved["ids"]
    frame = metadata("oss2", ids)
    detail = pd.read_csv(OUT / "dataset_audit/public_metadata.csv",
                         keep_default_na=False, low_memory=False)
    detail = detail.loc[detail.dataset.eq("oss2")].set_index("sample_id")
    assert detail.index.is_unique
    detail = detail.loc[ids].copy()
    y = labels("oss2", ids)
    identity = detail.identity.astype(str).str.replace(r"\.0$", "", regex=True)
    np.testing.assert_array_equal(identity, frame.identity)
    np.testing.assert_array_equal(pd.to_numeric(detail.class_id).to_numpy(int), y)
    detail["identity"] = identity
    detail["class_id"] = y
    detail["row"] = np.arange(len(ids))
    detail["sample_id"] = ids
    detail = detail.reset_index(drop=True)
    detail["time"] = pd.to_datetime(detail.sar_datetime, utc=True, errors="coerce")
    rawtime = pd.to_datetime(detail.sar_time, format="%Y/%m/%d.%H:%M:%S", utc=True)
    np.testing.assert_array_equal(rawtime, detail.time)
    for field in FIELDS:
        detail[field] = pd.to_numeric(detail[field], errors="coerce")
    keys = ["identity", "datatake_key", "time"]
    records, coverage = [], []
    for product in ("GRDH", "SLC"):
        group = detail.loc[detail.product_type.eq(product)].copy()
        valid = (group.identity.str.fullmatch(r"\d{9}")
                 & group.datatake_key.ne("") & group.time.notna())
        duplicated = group.duplicated(keys, keep=False)
        observations = group.loc[valid & ~duplicated].sort_values(
            ["identity", "time", "datatake_key", "sample_id"], kind="stable")
        candidates = sum(max(0, len(v) - 1) for _, v in observations.groupby("identity"))
        rejected = dict(same_datatake=0, nonpositive_interval=0, different_class=0)
        for ship, obs in observations.groupby("identity", sort=True):
            rows = list(obs.itertuples(index=False))
            for a, b in zip(rows[:-1], rows[1:]):
                # Adjacency is selected before class/datatake outcome-independent filtering.
                if a.datatake_key == b.datatake_key:
                    rejected["same_datatake"] += 1
                    continue
                if b.time <= a.time:
                    rejected["nonpositive_interval"] += 1
                    continue
                if a.class_id != b.class_id:
                    rejected["different_class"] += 1
                    continue
                record = dict(pair_id=f"{product}_{len(records):04d}", product=product,
                              identity=ship, class_id=int(a.class_id), class_name=a.class_name,
                              row_a=a.row, row_b=b.row,
                              interval_days=(b.time - a.time).total_seconds() / 86400.)
                for side, row in (("a", a), ("b", b)):
                    for field in ("sample_id", "datatake_key", "sar_time", "sar_datetime",
                                  "incidence_xml", "heading_deg"):
                        record[f"{field}_{side}"] = getattr(row, field)
                record["incidence_abs_difference_deg"] = abs(a.incidence_xml-b.incidence_xml)
                record["heading_circular_difference_deg"] = abs(
                    (b.heading_deg-a.heading_deg+180.) % 360.-180.)
                records.append(record)
        selected = pd.DataFrame([r for r in records if r["product"] == product])
        endpoints = np.union1d(selected.row_a, selected.row_b)
        coverage.append(dict(product=product, cached_rows=len(group),
                             cached_identities=int(group.identity.nunique()),
                             invalid_key_rows=int((~valid).sum()),
                             ambiguous_key_rows=int(duplicated.sum()),
                             ambiguous_keys=int(group.loc[duplicated, keys].drop_duplicates().shape[0]),
                             unique_observations=len(observations),
                             adjacent_candidates=candidates, rejected=rejected,
                             pairs=len(selected), paired_identities=int(selected.identity.nunique()),
                             paired_datatakes=len(set(selected.datatake_key_a) | set(selected.datatake_key_b)),
                             paired_unique_endpoints=len(endpoints),
                             represented_classes=int(selected.class_id.nunique()),
                             interval_days=describe(selected.interval_days),
                             incidence_abs_difference_deg=describe(selected.incidence_abs_difference_deg),
                             heading_circular_difference_deg=describe(selected.heading_circular_difference_deg),
                             incidence_available_rows=int(group.incidence_xml.notna().sum()),
                             heading_available_rows=int(group.heading_deg.notna().sum()),
                             paired_incidence_available_endpoints=int(detail.iloc[endpoints].incidence_xml.notna().sum()),
                             paired_heading_available_endpoints=int(detail.iloc[endpoints].heading_deg.notna().sum()),
                             classes=selected.groupby(["class_id", "class_name"]).agg(
                                 pairs=("pair_id", "size"), identities=("identity", "nunique"))
                                 .reset_index().to_dict("records")))
    pairs = pd.DataFrame(records)
    assert not pairs.duplicated(["product", "sample_id_a", "sample_id_b"]).any()
    np.testing.assert_array_equal(y[pairs.row_a], y[pairs.row_b])
    # Identify the metadata's coordinate convention; do not call this an AIS bearing.
    xy = detail[["head_x", "head_y", "tail_x", "tail_y"]].apply(pd.to_numeric)
    bearing = np.degrees(np.arctan2(xy.head_x-xy.tail_x, xy.tail_y-xy.head_y))
    error = abs((detail.heading_deg-bearing+180.) % 360.-180.)
    heading_check = dict(compared_rows=int(error.notna().sum()),
                         max_head_tail_image_bearing_error_deg=float(error.max()))
    return pairs, frame, y, coverage, heading_check


def evaluate(pairs, frame, y, caches):
    events, draws = [], []
    reference = caches[2][0]
    a, b = pairs.row_a.to_numpy(int), pairs.row_b.to_numpy(int)
    pair_y = y[a]
    for fold, (saved, probability) in caches.items():
        for key in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(saved[key], reference[key])
        assert saved["support"].shape == (5, 64)
        arms, heads = list(saved["arms"]), list(saved["heads"])
        hi = heads.index("expert")
        predictions = {}
        for arm in ("R", "T"):
            ai = arms.index(arm)
            p = np.asarray(probability[:, ai, hi])
            assert np.isfinite(p).all()
            np.testing.assert_allclose(p.sum(-1), 1., atol=2e-7, rtol=0.)
            predictions[arm] = p.argmax(-1)
            np.testing.assert_array_equal(predictions[arm], saved["predictions"][:, ai, hi])
        for draw in range(5):
            support = saved["support"][draw]
            support_ids = frame.identity.iloc[support]
            endpoint_clear = ~np.isin(a, support) & ~np.isin(b, support)
            query = saved["eval_masks"][draw]
            eligible = endpoint_clear & query[a] & query[b]
            strict = eligible & ~pairs.identity.isin(support_ids).to_numpy()
            for scope, mask in zip(SCOPES, (eligible, strict)):
                for product in ("GRDH", "SLC"):
                    selection = pairs.loc[mask & pairs["product"].eq(product)].copy()
                    draws.append(dict(fold=fold, draw=draw, product=product, scope=scope,
                                      n_pairs=len(selection), n_identities=selection.identity.nunique(),
                                      n_classes=selection.class_id.nunique(),
                                      n_unique_endpoints=len(set(selection.row_a) | set(selection.row_b))))
                for i in np.flatnonzero(mask):
                    r = predictions["R"][draw, [a[i], b[i]]]
                    t = predictions["T"][draw, [a[i], b[i]]]
                    rc, tc = r == pair_y[i], t == pair_y[i]
                    rf, tf = r[0] != r[1], t[0] != t[1]
                    events.append(dict(pair_id=pairs.pair_id.iloc[i], scope=scope, fold=fold, draw=draw,
                                       R_flip=float(rf), T_flip=float(tf),
                                       R_both_correct=float(rc.all()), T_both_correct=float(tc.all()),
                                       R_endpoint_accuracy=float(rc.mean()), T_endpoint_accuracy=float(tc.mean()),
                                       T_minus_R_endpoint_accuracy=float(tc.mean()-rc.mean()),
                                       R_only_flip=float(rf and not tf), T_only_flip=float(tf and not rf),
                                       both_flip=float(rf and tf), neither_flip=float(not rf and not tf),
                                       T_rescue_endpoint=float((~rc & tc).mean()),
                                       T_harm_endpoint=float((rc & ~tc).mean())))
    events = pd.DataFrame(events)
    # Each observation pair gets one vote after averaging available fold/draw repeats.
    pair_means = events.groupby(["pair_id", "scope"])[list(EVENTS)].mean().reset_index()
    repeats = events.groupby(["pair_id", "scope"]).agg(
        n_repeats=("draw", "size"), n_folds=("fold", "nunique"), n_draws=("draw", "nunique")).reset_index()
    pair_means = pair_means.merge(repeats, on=["pair_id", "scope"], validate="one_to_one")
    pair_means = pair_means.merge(pairs[["pair_id", "product", "identity", "class_id", "class_name"]],
                                  on="pair_id", validate="many_to_one")
    summaries, class_rows = [], []
    for (product, scope), part in pair_means.groupby(["product", "scope"]):
        class_mean = part.groupby(["class_id", "class_name"])[list(EVENTS)].mean()
        row = dict(product=product, scope=scope, n_pairs=len(part),
                   n_identities=int(part.identity.nunique()), n_classes=int(part.class_id.nunique()),
                   n_pair_repeats=int(part.n_repeats.sum()),
                   repeats_min=int(part.n_repeats.min()), repeats_max=int(part.n_repeats.max()))
        for event in EVENTS:
            row[event+"_micro_pct"] = float(100.*part[event].mean())
            row[event+"_macro_pct"] = float(100.*class_mean[event].mean())
        row["R_mean_endpoint_BA_pct"] = row["R_endpoint_accuracy_macro_pct"]
        row["T_mean_endpoint_BA_pct"] = row["T_endpoint_accuracy_macro_pct"]
        row["T_minus_R_mean_endpoint_BA_pp"] = row["T_minus_R_endpoint_accuracy_macro_pct"]
        summaries.append(row)
        for (class_id, class_name), cls in part.groupby(["class_id", "class_name"]):
            class_rows.append(dict(product=product, scope=scope, class_id=int(class_id), class_name=class_name,
                                   n_pairs=len(cls), n_identities=int(cls.identity.nunique()),
                                   **{event+"_pct": float(100.*cls[event].mean()) for event in EVENTS}))
    np.testing.assert_allclose(pair_means.R_only_flip+pair_means.T_only_flip+
                               pair_means.both_flip+pair_means.neither_flip, 1., atol=1e-14)
    np.testing.assert_allclose(pair_means.T_rescue_endpoint-pair_means.T_harm_endpoint,
                               pair_means.T_minus_R_endpoint_accuracy, atol=1e-14)
    return pair_means, pd.DataFrame(summaries), pd.DataFrame(class_rows), pd.DataFrame(draws)


def make_report(summary, coverage, heading_check):
    lines = ["# Same-Vessel Different-Acquisition Readout", "",
             "Metadata-selected adjacent same-class observations from the frozen OSS2 cache. "
             "GRDH is primary; SLC is separate and descriptive. No new inference, fitting, "
             "support draws, model selection, or fusion construction was performed.", "",
             "## Selection And Denominators", "",
             "Within each product type, remove all rows in nonunique identity+datatake+SAR-time "
             "keys. Sort the remaining observations by time within vessel, select adjacent "
             "observations, and then require identical class, different datatake and positive "
             "time interval. No predictions enter selection. Preserve each cached query mask; "
             "both endpoints must be outside that draw's B64 support. The strict sensitivity "
             "additionally removes every pair whose vessel occurs anywhere in that support, "
             "including its other product/acquisition records.", ""]
    for cov in coverage:
        lines.append(f"- {cov['product']}: {cov['cached_rows']} cached rows; "
                     f"{cov['ambiguous_key_rows']} ambiguous rows excluded; {cov['pairs']} pairs, "
                     f"{cov['paired_identities']} identities, {cov['represented_classes']} classes, "
                     f"{cov['paired_datatakes']} datatakes, {cov['paired_unique_endpoints']} unique endpoints. "
                     f"Intervals in days min/median/max: {cov['interval_days']['minimum']:.6f}/"
                     f"{cov['interval_days']['median']:.6f}/{cov['interval_days']['maximum']:.6f}.")
    lines.extend(["", "## Results", "",
                  "All rates are percentages, T-R is percentage points. OA is ordinary mean "
                  "accuracy over both pair endpoints; BA is the class-macro mean of endpoint "
                  "accuracy. Adjacent pairs can share an endpoint. Classes receive equal weight "
                  "for macro rates. Repeats are averaged within each pair before these summaries.", "",
                  "| Product / scope | Pairs / vessels / classes | R / T flip (micro) | "
                  "R / T both-correct (macro) | R / T OA | R / T BA | T-R BA |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for r in summary.itertuples():
        lines.append(f"| {r.product} / {r.scope} | {r.n_pairs} / {r.n_identities} / {r.n_classes} | "
                     f"{r.R_flip_micro_pct:.6f} / {r.T_flip_micro_pct:.6f} | "
                     f"{r.R_both_correct_macro_pct:.6f} / {r.T_both_correct_macro_pct:.6f} | "
                     f"{r.R_endpoint_accuracy_micro_pct:.6f} / {r.T_endpoint_accuracy_micro_pct:.6f} | "
                     f"{r.R_mean_endpoint_BA_pct:.6f} / {r.T_mean_endpoint_BA_pct:.6f} | "
                     f"{r.T_minus_R_mean_endpoint_BA_pp:+.6f} |")
    primary = summary.loc[summary["product"].eq("GRDH") & summary.scope.eq("endpoint_excluded")].iloc[0]
    lines.extend(["", f"Primary GRDH T-R changes: endpoint BA {primary.T_minus_R_mean_endpoint_BA_pp:+.6f} pp, "
                  f"both-correct macro {primary.T_both_correct_macro_pct-primary.R_both_correct_macro_pct:+.6f} pp, "
                  f"top-1 flip micro {primary.T_flip_micro_pct-primary.R_flip_micro_pct:+.6f} pp. "
                  "Here T improves classification while flipping more frequently; the data do not "
                  "support a blanket lower-flip robustness claim. Both improvements and this "
                  "limit remain visible after whole-vessel support exclusion."])
    lines.extend(["", "The flip-event decomposition in acquisition_summary.csv separates R-only, "
                  "T-only, both and neither flipping. Endpoint rescue minus harm exactly equals "
                  "T-R endpoint accuracy. acquisition_classes.csv preserves rare-class denominators.", "",
                  "## Metadata And Limits", "",
                  "- Incidence and heading availability, differences, intervals and class/identity "
                  "coverage are in acquisition_coverage.json and the pair manifest. Heading is "
                  "the manifest image-coordinate head/tail orientation, not a verified geographic "
                  "AIS bearing or sensor-relative aspect. Maximum circular discrepancy from "
                  f"atan2(head_x-tail_x, tail_y-head_y): {heading_check['max_head_tail_image_bearing_error_deg']:.8f} degrees.",
                  "- Natural revisits change multiple factors together: location, background, "
                  "orientation, sea state, preprocessing and potentially annotation quality. "
                  "Same vessel/class reduces identity variation but does not establish a sole "
                  "physical cause. Same-product selection removes GRDH/SLC product mixing.",
                  "- A flip documents decision instability across observed acquisitions; it does "
                  "not by itself prove a specific physical cue caused an error. Low flip rate "
                  "can also mean consistent error; ordinary accuracy and both-correct rate must "
                  "be read alongside it. T here is the existing augmented expert, not the full "
                  "system or a newly fitted fusion. Fusion exploitation belongs to the main analysis.",
                  "- Two source contexts and five reused support draws are repeated predictions "
                  "of the same observation pairs, not ten independent training seeds. No "
                  "pseudo-seed tests, p-values or confidence intervals are reported. Class-macro "
                  "uncertainty is poorly resolved for classes represented by only a few vessels; "
                  "SLC additionally has singleton classes. Results are descriptive.",
                  "- Strict support exclusion is conditional on existing fitted support sets, "
                  "not leave-vessel-out refitting. It does not purge source/pretraining exposure "
                  "or overlap with other public releases. Repeated pairs can share ships, "
                  "endpoints and scenes; the coverage is not independent replication.", "",
                  "## Verification And Reproduction", "",
                  "Verified manifest labels/identities and raw SAR timestamps against metadata; "
                  "fold cache IDs/support/query masks align; probability argmax matches cached "
                  "expert predictions; support size is 64; flip partition and rescue-minus-harm "
                  "identities hold. These checks catch mismatched joins, cache axes, leakage and "
                  "incorrect event accounting, and fail the run before artifacts are emitted.", "",
                  "Run acquisition_pairs.py with the existing ClearSAR Python and -B. It writes "
                  "the generated CSV/JSON/report artifacts beside the script. "
                  "All artifact names begin acquisition_; shared study/memory files are untouched.", ""])
    return "\n".join(lines)


def main():
    caches = {fold: load(fold, "oss2", "original", "original") for fold in FOLDS}
    pairs, frame, y, coverage, heading_check = select_pairs(caches[2][0])
    means, summary, classes, draws = evaluate(pairs, frame, y, caches)
    artifacts = {
        "acquisition_pairs.csv": pairs.to_csv(index=False),
        "acquisition_pair_means.csv": means.to_csv(index=False),
        "acquisition_summary.csv": summary.to_csv(index=False),
        "acquisition_classes.csv": classes.to_csv(index=False),
        "acquisition_draw_coverage.csv": draws.to_csv(index=False),
        "acquisition_coverage.json": json.dumps(dict(products=coverage, heading_check=heading_check), indent=2),
        "acquisition_FINDINGS.md": make_report(summary, coverage, heading_check),
        "acquisition_COMPLETE.json": json.dumps(dict(
            status="complete", folds=list(FOLDS), draws_per_fold=5, support_budget=64,
            input_cache=str(OUT / "predictions/original/original"),
            observation_pairs=len(pairs), pair_scope_rows=len(means),
            repeat_averaging="Per pair across eligible draws and both folds before summaries",
            prediction_based_selection=False, new_fits=0, new_models=0,
            confidence_intervals=False, pseudo_seed_tests=False,
            artifacts_written_by="acquisition_pairs.py", verification="passed"), indent=2),
    }
    for name, contents in artifacts.items():
        (HERE / name).write_text(contents, encoding="utf-8", newline="")
    columns = ["product", "scope", "n_pairs", "n_identities", "n_classes",
               "R_flip_micro_pct", "T_flip_micro_pct", "R_both_correct_macro_pct", "T_both_correct_macro_pct",
               "R_mean_endpoint_BA_pct", "T_mean_endpoint_BA_pct", "T_minus_R_mean_endpoint_BA_pp"]
    print(summary[columns].to_string(index=False))
    print("WROTE", len(artifacts), "acquisition_ artifacts", flush=True)


if __name__ == "__main__":
    main()
