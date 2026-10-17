"""Fixed-dose source evidence diagnostic; no target reads or new model training."""
from pathlib import Path
import importlib.util
import inspect
import json
import os
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_info, threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROBE = ROOT / "all_factor_probe_20261017"
OLD = ROOT / "oss1_joint_geometry_20261017"
POLAR = ROOT / "oss1_polar_environment_20261017"
ALPHA = .75
SPEC = importlib.util.spec_from_file_location(
    "source_reliability_fixed_descriptor_helpers_20261017", PROBE / "probe_classifier.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
old, public = probe.old, probe.public
# The old helper decorates the same analytic ridge solver with a two-thread limit.
# Unwrap only that runtime decorator; preserve its numerical implementation.
public.fit_head = inspect.unwrap(public.fit_head)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

OBSERVABLES = [
    "vv_native_contrast_db", "vh_native_contrast_db",
    "vv_native_top5_mass", "vh_native_top5_mass",
    "vv_native_long_extent_ratio", "vh_native_long_extent_ratio",
    "native_min_coverage", "descriptor_min_coverage",
    "baseline_entropy", "baseline_top2_gap", "support_sameclass_count",
]
CONTINUOUS = [
    "baseline_brier", "fixed_brier", "brier_gain", "true_probability_delta",
    "baseline_ap", "fixed_ap", "ap_delta", "rescue", "harm", "correctness_delta",
    "baseline_correct", "fixed_correct", "event_oracle_positive_brier_gain",
    "active", "support_valid_count", "baseline_true_probability", "fixed_true_probability",
    "baseline_entropy", "baseline_top2_gap", "support_sameclass_count",
]


def save_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def source_fold(args, sample_ids, c4):
    """Source-only projection of public.load_fold, then discard non-source rows.

    The original loader necessarily reads target query/support CSVs. Retain its
    source files and cache index mapping, but obtain labels only from source CSVs.
    Descriptor sample_ids provide the already verified global manifest row axis.
    """
    fold, cache, episode, _ = args
    source_folder = cache if (cache / "source_train.csv").exists() else episode
    train_frame = pd.read_csv(source_folder / "source_train.csv", dtype={"mmsi": str})
    val_frame = pd.read_csv(source_folder / "source_validation.csv", dtype={"mmsi": str})
    frame = pd.concat([train_frame, val_frame], ignore_index=True)
    global_index = pd.Series(np.arange(len(sample_ids)), index=sample_ids)
    rows = global_index.loc[frame.sample_id.astype(str)].to_numpy(int)
    with np.load(cache / "pilot_features.npz") as saved:
        cached_rows = (global_index.loc[saved["sample_ids"].astype(str)].to_numpy(int)
                       if "sample_ids" in saved else saved["rows"].astype(int))
        local_index = pd.Series(np.arange(len(cached_rows)), index=cached_rows)
        take = local_index.loc[rows].to_numpy(int)
        raw = {name: saved[name][take] for name in ("G50", "D18")}
    assert len(set(rows)) == len(rows)
    assert set(train_frame.class_id) == set(val_frame.class_id) == set(range(8))
    assert not set(train_frame.acquisition) & set(val_frame.acquisition)
    # Source-held ships must not enter priors through the other split.
    assert not set(train_frame.mmsi) & set(val_frame.mmsi)
    np.testing.assert_array_equal(sample_ids[rows], frame.sample_id.to_numpy(str))
    ntrain = len(train_frame)
    return dict(fold=fold, rows=rows, raw=raw, c4=np.asarray(c4[rows]),
                train=np.arange(ntrain), val=np.arange(ntrain, len(rows)),
                source_train=train_frame, source_val=val_frame,
                labels=frame.class_id.to_numpy(int), frame=frame)


def observability(source_rows, features):
    shape_columns = ["global_row", "sample_id", "pol", "representation", "min_coverage",
                     "body_top5_mass", "range_extent90_px", "azimuth_extent90_px",
                     "ship_long_extent90_px", "ship_transverse_extent90_px",
                     "long_extent_ratio", "transverse_extent_ratio"]
    shape = pd.read_csv(PROBE / "native_shape_measurements.csv", usecols=shape_columns)
    shape = shape[shape.global_row.isin(source_rows) & shape.representation.eq("sigma0")]
    contrast = pd.read_csv(POLAR / "image_descriptors.csv", usecols=[
        "dataset", "global_row", "pol", "representation", "body_neighbor_db",
        "body_neighbor_contrast", "min_outside_coverage"])
    contrast = contrast[contrast.global_row.isin(source_rows) &
                        contrast.dataset.eq("OSS1") & contrast.representation.eq("sigma0")]
    result = pd.DataFrame({"global_row": sorted(source_rows)}).set_index("global_row")
    for pol in ("vv", "vh"):
        s = shape[shape.pol.eq(pol)].set_index("global_row")
        c = contrast[contrast.pol.eq(pol)].set_index("global_row")
        assert s.index.is_unique and c.index.is_unique
        np.testing.assert_array_equal(s.sample_id.to_numpy(str), features["sample_ids"][s.index])
        result[f"{pol}_native_contrast_db"] = c.body_neighbor_db
        result[f"{pol}_native_contrast"] = c.body_neighbor_contrast
        for name in ("body_top5_mass", "range_extent90_px", "azimuth_extent90_px",
                     "ship_long_extent90_px", "ship_transverse_extent90_px",
                     "long_extent_ratio", "transverse_extent_ratio"):
            result[f"{pol}_native_{name.replace('body_top5_mass', 'top5_mass')}"] = s[name]
        result[f"{pol}_native_min_coverage"] = s.min_coverage
    result["native_min_coverage"] = result[["vv_native_min_coverage", "vh_native_min_coverage"]].min(axis=1)
    rows = result.index.to_numpy(int)
    cells = features["uint8_spatial"][rows].reshape(len(rows), 18, 6)
    valid = features["valid"][rows].astype(bool)
    result["descriptor_valid"] = valid
    result["descriptor_min_coverage"] = np.where(valid, cells[:, :, [1, 4]].min(axis=(1, 2)), np.nan)
    result["descriptor_mean_coverage"] = np.where(valid, cells[:, :, [1, 4]].mean(axis=(1, 2)), np.nan)
    return result.reset_index()


def run_fold(args, features, c4):
    start = time.perf_counter()
    data = source_fold(args, features["sample_ids"], c4)
    fold = args[0]
    source = np.r_[data["train"], data["val"]]
    g, d, c = [public.pca(array, source) for array in
               (data["raw"]["G50"], data["raw"]["D18"], data["c4"])]
    base = public.design_block(c, source)
    z = public.design_block((g, d), source)
    valid = torch.tensor(features["valid"][data["rows"]], dtype=torch.bool)
    raw = probe.standardize(features["uint8_spatial"][data["rows"]], valid.numpy(), data["train"])
    episodes = [(kind, i, e) for kind in ("fit", "selection")
                for i, e in enumerate(old.make_episodes(data, kind))]
    saved = json.loads((OLD / fold / "source_task_rows.json").read_text(encoding="utf-8"))
    serial = [dict(kind=kind, q=data["rows"][e["q"]].tolist(),
                   s=data["rows"][e["s"]].tolist(), prior=data["rows"][e["prior_rows"]].tolist())
              for kind, _, e in episodes]
    assert serial == saved, "Source episodes differ from the previous experiment"
    prior_cache = {}
    events, tasks, supports = [], [], []
    endpoint_errors, risk_errors, fallback_errors = [], [], []
    for kind, number, e in episodes:
        task_id = f"{fold}/{kind}/{number:03d}"
        q, s, prior = [np.array(e[key], copy=True) for key in ("q", "s", "prior_rows")]
        frame = data["frame"]
        assert not set(frame.iloc[q].mmsi) & set(frame.iloc[s].mmsi)
        assert not set(frame.iloc[prior].mmsi) & set(frame.iloc[np.r_[q, s]].mmsi)
        assert not set(frame.iloc[prior].acquisition) & set(frame.iloc[np.r_[q, s]].acquisition)
        task = old.make_task(data, base, z, e, prior_cache)
        p0, _ = old.predict(raw[task["ids"]], valid[task["ids"]], task, 0.)
        p1, h = old.predict(raw[task["ids"]], valid[task["ids"]], task, ALPHA)
        baseline, fixed = task["old"].numpy(), p1.numpy()
        endpoint_errors.append(float(np.max(np.abs(p0.numpy() - baseline))))
        assert endpoint_errors[-1] < 1e-7
        assert np.isfinite(fixed).all()
        np.testing.assert_allclose(fixed.sum(1), 1., atol=1e-12)
        y, sy = data["labels"][q], data["labels"][s]
        active = valid[q].numpy() & (int(valid[s].sum()) >= 2)
        fallback_errors.append(float(np.max(np.abs(fixed[~active] - baseline[~active])))
                               if (~active).any() else 0.)
        assert fallback_errors[-1] == 0.
        k0 = task["qs"].numpy()
        k1 = k0 + ALPHA * (h[len(s):] @ h[:len(s)].T).numpy()
        k1[~active] = k0[~active]
        ap0 = probe.map_at_r(k0, y, sy)
        ap1 = probe.map_at_r(k1, y, sy)
        b0 = ((baseline - np.eye(8)[y]) ** 2).sum(1)
        b1 = ((fixed - np.eye(8)[y]) ** 2).sum(1)
        weight = 1. / (len(np.unique(y)) * np.bincount(y, minlength=8)[y])
        before, after = baseline.argmax(1), fixed.argmax(1)
        gain = b0 - b1
        yt = torch.tensor(y, dtype=torch.long)
        for per, probability in ((b0, task["old"]), (b1, p1)):
            risk_errors.append(abs(float(per @ weight) - float(old.risk(probability, yt))))
        assert max(risk_errors[-2:]) < 1e-12
        sorted_p = np.sort(baseline, axis=1)
        entropy = -(baseline * np.log(baseline.clip(1e-15))).sum(1) / np.log(8)
        task_record = dict(task_id=task_id, fold=fold, kind=kind, task_number=number,
                           origin=e["origin"], acquisition=e["acquisition"],
                           support_acquisition=e["support_acquisition"], queries=len(q),
                           supports=len(s), prior_rows=len(prior), active_queries=int(active.sum()),
                           baseline_macro_brier=float(b0 @ weight), fixed_macro_brier=float(b1 @ weight),
                           macro_brier_gain=float(gain @ weight),
                           macro_true_probability_delta=float((fixed[np.arange(len(q)), y] -
                                                               baseline[np.arange(len(q)), y]) @ weight),
                           macro_ba_delta_pp=float(((after == y).astype(float) - (before == y)) @ weight * 100),
                           event_oracle_macro_brier_gain=float(np.maximum(gain, 0) @ weight),
                           retrieval_eligible=int(np.isfinite(ap0).sum()),
                           rescues=int(((before != y) & (after == y)).sum()),
                           harms=int(((before == y) & (after != y)).sum()))
        tasks.append(task_record)
        for pos, index in enumerate(s):
            r = frame.iloc[index]
            supports.append(dict(task_id=task_id, fold=fold, kind=kind, origin=e["origin"],
                                 position=pos, global_row=int(data["rows"][index]),
                                 sample_id=r.sample_id, mmsi=r.mmsi, acquisition=r.acquisition,
                                 class_id=int(sy[pos]), descriptor_valid=bool(valid[index])))
        for j, index in enumerate(q):
            r = frame.iloc[index]
            events.append(dict(task_id=task_id, fold=fold, kind=kind, origin=e["origin"],
                task_number=number, global_row=int(data["rows"][index]), sample_id=r.sample_id,
                acquisition=r.acquisition, support_acquisition=e["support_acquisition"], mmsi=r.mmsi,
                class_id=int(y[j]), class_name=r.class_name, alpha=ALPHA,
                baseline_prediction=int(before[j]), fixed_prediction=int(after[j]),
                baseline_brier=float(b0[j]), fixed_brier=float(b1[j]), brier_gain=float(gain[j]),
                task_macro_weight=float(weight[j]), macro_brier_gain_contribution=float(weight[j] * gain[j]),
                baseline_true_probability=float(baseline[j, y[j]]), fixed_true_probability=float(fixed[j, y[j]]),
                true_probability_delta=float(fixed[j, y[j]] - baseline[j, y[j]]),
                baseline_ap=float(ap0[j]), fixed_ap=float(ap1[j]), ap_delta=float(ap1[j] - ap0[j]),
                rescue=int(before[j] != y[j] and after[j] == y[j]),
                harm=int(before[j] == y[j] and after[j] != y[j]),
                correctness_delta=int(after[j] == y[j]) - int(before[j] == y[j]),
                baseline_correct=int(before[j] == y[j]), fixed_correct=int(after[j] == y[j]),
                event_oracle_positive_brier_gain=float(max(gain[j], 0)),
                support_sameclass_count=int((sy == y[j]).sum()), support_valid_count=int(valid[s].sum()),
                active=int(active[j]), baseline_entropy=float(entropy[j]),
                baseline_top2_gap=float(sorted_p[j, -1] - sorted_p[j, -2]),
                **{f"baseline_p{k}": float(baseline[j, k]) for k in range(8)},
                **{f"fixed_p{k}": float(fixed[j, k]) for k in range(8)}))
    receipt = dict(fold=fold, source_train=len(data["train"]), source_selection=len(data["val"]),
                   source_unique_ships=int(frame.mmsi.nunique()), source_acquisitions=int(frame.acquisition.nunique()),
                   fit_episodes=sum(k == "fit" for k, _, _ in episodes),
                   selection_episodes=sum(k == "selection" for k, _, _ in episodes),
                   source_episode_rows_equal_old=True, endpoint_max_error=max(endpoint_errors),
                   macro_risk_max_error=max(risk_errors), inactive_fallback_max_error=max(fallback_errors),
                   seconds=time.perf_counter() - start)
    return events, tasks, supports, [dict(fold=fold, **r) for r in serial], receipt, set(data["rows"])


def unique_queries(events, obs, by_origin):
    keys = ["kind"] + (["origin"] if by_origin else []) + ["global_row"]
    group = events.groupby(keys, sort=True)
    unique = group[CONTINUOUS].mean()
    unique["event_rows"] = group.size()
    unique["folds"] = group.fold.nunique()
    for name in ("sample_id", "mmsi", "acquisition", "class_id", "class_name"):
        assert group[name].nunique().max() == 1
        unique[name] = group[name].first()
    unique["query_oracle_positive_brier_gain"] = unique.brier_gain.clip(lower=0)
    unique["support_context_gain_min"] = group.brier_gain.min()
    unique["support_context_gain_max"] = group.brier_gain.max()
    unique["gain_sign_mixed_across_contexts"] = (unique.support_context_gain_min < 0) & (unique.support_context_gain_max > 0)
    if not by_origin:
        unique["origin"] = "pooled"
    return unique.reset_index().merge(obs, on="global_row", how="left", validate="many_to_one")


def macro(frame, column):
    return float(frame.groupby("class_id")[column].mean().mean())


def summarize(unique, tasks):
    output = []
    for (kind, origin), frame in unique.groupby(["kind", "origin"]):
        t = tasks[tasks.kind.eq(kind)]
        if origin != "pooled":
            t = t[t.origin.eq(origin)]
        output.append(dict(kind=kind, origin=origin, unique_queries=len(frame),
            unique_ships=int(frame.mmsi.nunique()), acquisitions=int(frame.acquisition.nunique()),
            classes=int(frame.class_id.nunique()), event_rows=int(frame.event_rows.sum()), tasks=len(t),
            active_query_equivalents=float(frame.active.sum()),
            baseline_query_macro_brier=macro(frame, "baseline_brier"),
            fixed_query_macro_brier=macro(frame, "fixed_brier"),
            query_macro_brier_gain=macro(frame, "brier_gain"),
            query_macro_true_probability_delta=macro(frame, "true_probability_delta"),
            query_macro_ap_delta=macro(frame, "ap_delta"),
            query_macro_ba_delta_pp=100 * macro(frame, "correctness_delta"),
            query_oracle_macro_brier_gain=macro(frame, "query_oracle_positive_brier_gain"),
            event_oracle_query_macro_brier_gain=macro(frame, "event_oracle_positive_brier_gain"),
            beneficial_queries=int((frame.brier_gain > 0).sum()), harmful_queries=int((frame.brier_gain < 0).sum()),
            zero_gain_queries=int(frame.brier_gain.eq(0).sum()),
            context_mixed_queries=int(frame.gain_sign_mixed_across_contexts.sum()),
            retrieval_eligible_unique_queries=int(frame.ap_delta.notna().sum()),
            rescue_query_equivalents=float(frame.rescue.sum()), harm_query_equivalents=float(frame.harm.sum()),
            task_mean_macro_brier_gain=float(t.macro_brier_gain.mean()),
            task_mean_ba_delta_pp=float(t.macro_ba_delta_pp.mean()),
            raw_event_rescues=int(t.rescues.sum()), raw_event_harms=int(t.harms.sum())))
    return pd.DataFrame(output)


def acquisition_summary(unique):
    output = []
    for (kind, origin, acq), f in unique.groupby(["kind", "origin", "acquisition"]):
        row = dict(kind=kind, origin=origin, acquisition=acq, queries=len(f), ships=f.mmsi.nunique(),
                   classes=f.class_id.nunique(), event_rows=int(f.event_rows.sum()))
        for col in list(dict.fromkeys(CONTINUOUS + OBSERVABLES + ["query_oracle_positive_brier_gain"])):
            row[col] = macro(f, col)
        output.append(row)
    return pd.DataFrame(output)


def associations(unique, acq):
    rows, strata = [], []
    for kind in ("fit", "selection"):
        for origin in ("same", "cross"):
            for unit, table in (("unique_query", unique), ("acquisition_class_macro", acq)):
                frame = table[table.kind.eq(kind) & table.origin.eq(origin)]
                for feature in OBSERVABLES:
                    values = frame[[feature, "brier_gain", "ap_delta"]].dropna(subset=[feature, "brier_gain"])
                    rho = (values[feature].rank().corr(values.brier_gain.rank())
                           if len(values) >= 3 and values[feature].nunique() > 1 and values.brier_gain.nunique() > 1
                           else np.nan)
                    ap = values.dropna(subset=["ap_delta"])
                    rho_ap = (ap[feature].rank().corr(ap.ap_delta.rank())
                              if len(ap) >= 3 and ap[feature].nunique() > 1 and ap.ap_delta.nunique() > 1
                              else np.nan)
                    rows.append(dict(kind=kind, origin=origin, unit=unit, feature=feature, n=len(values),
                                     missing=len(frame)-len(values), spearman_brier_gain=rho,
                                     n_ap=len(ap), spearman_ap_delta=rho_ap))
            frame = unique[unique.kind.eq(kind) & unique.origin.eq(origin)]
            fit = unique[unique.kind.eq("fit") & unique.origin.eq(origin)]
            for feature in OBSERVABLES:
                threshold = float(fit[feature].median())
                for level, mask in (("low", frame[feature].le(threshold)),
                                    ("high", frame[feature].gt(threshold))):
                    group = frame[mask]
                    if len(group):
                        strata.append(dict(kind=kind, origin=origin, feature=feature, level=level,
                            source_fit_median=threshold, queries=len(group), ships=group.mmsi.nunique(),
                            acquisitions=group.acquisition.nunique(), classes=group.class_id.nunique(),
                            macro_brier_gain=macro(group, "brier_gain"),
                            macro_ap_delta=macro(group, "ap_delta"),
                            macro_true_probability_delta=macro(group, "true_probability_delta"),
                            macro_ba_delta_pp=100 * macro(group, "correctness_delta")))
    return pd.DataFrame(rows), pd.DataFrame(strata)


def main():
    start = time.perf_counter()
    protocol = dict(alpha=ALPHA, descriptor="uint8_spatial", training="none; reconstruct only original analytic priors",
        dose_selection="none; .75 predeclared; zero dose used only for endpoint verification",
        source_loader="source-only projection of public.load_fold; no target query/support files or labels read",
        global_row_axis="spatial_descriptors.npz sample_ids; no target_manifest read",
        episodes="original make_episodes(fit/selection), exact q/s/prior comparison to old source_task_rows.json",
        pca="original public.pca; same train+validation source rows and order; transform source rows only",
        standardization="original probe.standardize; source train valid descriptors only; unchanged within-source exposure",
        integration="original make_task/predict, SS+QS increment, support solve, old priors and equal full fusion",
        brier="sum across 8 classes; gain=old-fixed, positive favors fixed; task weights 1/(observed_classes*class_query_count)",
        ap="original stable-sort MAP@R per query; NaN when no same-class support; inactive queries use old retrieval",
        primary_aggregation="average contexts/folds per unique sample within kind/origin, then equal observed class mean",
        secondary_aggregation="old equal-task macro Brier; class-balanced unique queries per acquisition",
        oracle="NONREALIZABLE: max(mean_context_brier_gain,0) per query; eventwise positive-only bound separately retained",
        associations=dict(features=OBSERVABLES, method="descriptive Spearman, unique queries and acquisition means; no p values",
                          strata="one source-fit median per feature/origin, reused unchanged in selection; no gate fitted"),
        limits="overlapping folds and fit/selection source memberships; source-only diagnostic, not independent confirmation or causal proof",
        threads=1, writes="this source_diagnostic directory only")
    save_json("PROTOCOL.json", protocol)
    with np.load(PROBE / "spatial_descriptors.npz") as stored:
        features = {k: stored[k] for k in ("sample_ids", "valid", "uint8_spatial")}
    c4 = np.load(old.FULL / "oss1_c4.npy", mmap_mode="r")
    folds = old.folds()
    assert len(folds) == 4, "This diagnostic is bounded to the four existing cached folds"
    events, tasks, supports, serial, receipts, source_rows = [], [], [], [], [], set()
    with threadpool_limits(limits=1), torch.no_grad():
        threads = [{k: item[k] for k in ("internal_api", "num_threads", "prefix")} for item in threadpool_info()]
        assert all(item["num_threads"] == 1 for item in threads)
        for args in folds:
            print("SOURCE_FOLD", args[0], flush=True)
            ev, ta, su, se, receipt, rows = run_fold(args, features, c4)
            events.extend(ev)
            tasks.extend(ta)
            supports.extend(su)
            serial.extend(se)
            receipts.append(receipt)
            source_rows.update(rows)
            print(json.dumps(receipt), flush=True)
    obs = observability(source_rows, features)
    events, tasks, supports = pd.DataFrame(events), pd.DataFrame(tasks), pd.DataFrame(supports)
    events = events.merge(obs, on="global_row", how="left", validate="many_to_one")
    assert not events.duplicated(["task_id", "global_row"]).any()
    assert events.descriptor_valid.notna().all()
    complete_region = events.descriptor_valid & events.native_min_coverage.gt(0)
    assert events.loc[complete_region, ["vv_native_contrast_db", "vh_native_contrast_db"]].notna().all().all()
    unique_origin = unique_queries(events, obs, True)
    unique_pooled = unique_queries(events, obs, False)
    unique = pd.concat([unique_origin, unique_pooled], ignore_index=True)
    assert not unique.duplicated(["kind", "origin", "global_row"]).any()
    summary = summarize(unique, tasks)
    acq = acquisition_summary(unique)
    corr, strata = associations(unique, acq)
    class_rows = []
    for keys, f in unique.groupby(["kind", "origin", "class_id", "class_name"]):
        class_rows.append(dict(zip(["kind", "origin", "class_id", "class_name"], keys),
            queries=len(f), ships=f.mmsi.nunique(), acquisitions=f.acquisition.nunique(),
            mean_brier_gain=f.brier_gain.mean(), mean_ap_delta=f.ap_delta.mean(),
            mean_true_probability_delta=f.true_probability_delta.mean(),
            ba_delta_pp=100*f.correctness_delta.mean(),
            query_oracle_mean_gain=f.query_oracle_positive_brier_gain.mean()))
    for name, frame in [("source_query_events.csv", events), ("source_tasks.csv", tasks),
                        ("source_supports.csv", supports), ("source_observability.csv", obs),
                        ("unique_source_queries.csv", unique), ("source_summary.csv", summary),
                        ("source_acquisitions.csv", acq), ("descriptive_associations.csv", corr),
                        ("fit_median_strata.csv", strata), ("source_classes.csv", pd.DataFrame(class_rows)),
                        ("fold_receipts.csv", pd.DataFrame(receipts))]:
        frame.to_csv(HERE / name, index=False)
    save_json("source_task_rows.json", serial)
    fit_ids = set(unique_pooled[unique_pooled.kind.eq("fit")].global_row)
    selection_ids = set(unique_pooled[unique_pooled.kind.eq("selection")].global_row)
    receipt = dict(status="complete", seconds=time.perf_counter()-start, folds=len(folds),
        events=len(events), unique_queries_all_roles=int(events.global_row.nunique()),
        unique_ships_all_roles=int(events.mmsi.nunique()), acquisitions_all_roles=int(events.acquisition.nunique()),
        source_pool_unique_rows=len(source_rows), fit_selection_query_overlap=len(fit_ids & selection_ids),
        task_count=len(tasks), source_episode_rows_equal_old=all(r["source_episode_rows_equal_old"] for r in receipts),
        baseline_endpoint_max_error=max(r["endpoint_max_error"] for r in receipts),
        macro_risk_max_error=max(r["macro_risk_max_error"] for r in receipts),
        inactive_fallback_max_error=max(r["inactive_fallback_max_error"] for r in receipts),
        same_mmsi_query_support_overlaps=0, prior_query_support_mmsi_overlaps=0,
        invalid_descriptor_query_events=int((~events.descriptor_valid).sum()),
        missing_contrast_unique_queries=int(events.loc[events.vv_native_contrast_db.isna(), "global_row"].nunique()),
        valid_axis_missing_contrast_unique_queries=int(events.loc[events.descriptor_valid & events.vv_native_contrast_db.isna(), "global_row"].nunique()),
        no_sameclass_support_query_events=int(events.support_sameclass_count.eq(0).sum()),
        positive_gain_convention="baseline Brier minus fixed Brier", threadpools=threads,
        target_query_or_support_csv_read=False, target_manifest_read=False, target_predictions_read=False,
        training_performed=False, no_new_extraction=True)
    save_json("VERIFICATION.json", receipt)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
