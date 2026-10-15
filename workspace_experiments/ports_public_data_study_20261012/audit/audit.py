"""Read-only metadata audit; all generated files stay next to this script."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

from pathlib import Path
import json
import itertools
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = Path("E:/Hermes/paper_tracking/sensitivity")
WORK = OUT.parent.parent
BASE = WORK / "visual_reliable_baseline"
SOURCE = BASE / "artifacts/eight_class_adaptive_20260916"
PUBLIC = WORK / "public_oss_comparison_20261012"
OLD = WORK / "dual_cue_decomposition_20261012"
DESC = ROOT / "c4_condition_diagnosis_20261009/descriptors"
SIDECAR = Path("E:/Hermes/paper_tracking/pixel_correction_review/chip_incidence_v1/incidence_by_instance.csv")
POOL = Path("E:/F_0921_pool_v1")
PROXIES = ["visible_footprint_proxy_fraction", "elongation_proxy", "contrast_proxy", "center_excess_proxy"]
NUMERIC = PROXIES + ["native_width_px_vv", "native_height_px_vv", "native_width_px_vh", "native_height_px_vh",
    "incidence_deg", "length_m", "width_m", "visible_length_px", "chip_width_px", "chip_height_px",
    "target_long_side_vv", "target_long_side_vh", "target_occupancy_vv", "target_occupancy_vh",
    "crop_side_vv", "crop_side_vh", "heading_deg", "center_distance_px"]


def read(path, **kwargs):
    return pd.read_csv(path, keep_default_na=False, low_memory=False, **kwargs)


def save(name, rows):
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    table.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    return table


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in keys}


def identities(values):
    s = values.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["", "0", "nan", "none", "unknown", "-1"]), "")


def identity_set(frame):
    return set(frame.loc[frame.identity.ne(""), "identity"])


def add_descriptors(frame, name):
    path = DESC / (name + ".csv")
    if not path.exists():
        frame["descriptor_available"] = False
        return frame
    desc = read(path)
    np.testing.assert_array_equal(desc.row_id, np.arange(len(frame)))
    np.testing.assert_array_equal(desc.sample_id, frame.sample_id)
    cols = ["sample_id"] + PROXIES + ["native_width_px_vv", "native_height_px_vv", "native_width_px_vh", "native_height_px_vh"]
    frame = frame.merge(desc[cols], on="sample_id", validate="one_to_one", sort=False)
    frame["descriptor_available"] = True
    return frame


def normalize(raw, name):
    frame = raw.copy()
    frame["manifest_row"] = np.arange(len(frame))
    if name == "self24ports":
        frame["sample_id"] = frame.instance_id
        frame["identity"] = identities(frame.mmsi)
        frame["actual_port"] = frame.port
    else:
        frame["identity"] = identities(frame.mmsi_or_ship_id)
        frame["product"] = frame.scene_id
        frame["actual_port"] = ""
    frame["valid_mmsi_format"] = frame.identity.str.fullmatch(r"[1-9][0-9]{8}")
    frame["product_type_parsed"] = frame["product"].str.extract(r"_(GRDH|SLC)_", expand=False).fillna("unknown")
    frame["satellite"] = frame["product"].str.extract(r"^(S1[A-Z])_", expand=False).fillna("unknown")
    frame["acquisition_time"] = frame["product"].str.extract(r"_(\d{8}T\d{6})_", expand=False).fillna("")
    frame["acquisition_day"] = frame.acquisition_time.str[:8]
    frame["acquisition_key"] = frame.satellite + "_" + frame.acquisition_time
    frame = add_descriptors(frame, name)
    if name == "self24ports":
        side = read(SIDECAR)
        keep = ["instance_id", "product", "port", "status", "incidence_deg", "coordinate_kind"]
        frame = frame.merge(side[keep], on=["instance_id", "product", "port"], how="left", validate="one_to_one", sort=False)
        frame.loc[frame.status.ne("ok"), "incidence_deg"] = np.nan
    frame["dataset"] = name
    return frame


def describe(frame, name, task="all"):
    counts = frame.class_id.value_counts()
    known = frame.identity.ne("")
    return dict(dataset=name, task=task, rows=len(frame), classes=len(counts),
        actual_ports=frame.loc[frame.actual_port.ne(""), "actual_port"].nunique() if name.startswith("self") else np.nan,
        actual_port_available=name.startswith("self"), products=frame["product"].nunique(),
        acquisition_keys=frame.acquisition_key.nunique(), known_identity_rows=int(known.sum()),
        known_identities=frame.loc[known, "identity"].nunique(), valid_mmsi_rows=int(frame.valid_mmsi_format.sum()),
        repeated_identity_rows=int(known.sum()-frame.loc[known, "identity"].nunique()),
        max_class_fraction=counts.max()/len(frame), min_present_class=int(counts.min()),
        date_min=frame.acquisition_day.min(), date_max=frame.acquisition_day.max(),
        grdh_rows=int(frame.product_type_parsed.eq("GRDH").sum()), slc_rows=int(frame.product_type_parsed.eq("SLC").sum()))


def numeric_summary(frame, dataset, task="all", class_id=-1):
    rows = []
    for key in NUMERIC:
        x = pd.to_numeric(frame[key], errors="coerce") if key in frame else pd.Series(np.nan, index=frame.index)
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(dataset=dataset, task=task, class_id=class_id, field=key,
            total=len(frame), valid=len(x), missing=len(frame)-len(x), coverage=len(x)/len(frame),
            mean=x.mean(), median=x.median(), p10=x.quantile(.1), p90=x.quantile(.9),
            min=x.min(), max=x.max()))
    return rows


def support_audit(frame, saved, dataset, task):
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), saved["ids"])
    result = []
    y = frame.class_id.to_numpy(int)
    target_counts = np.bincount(y, minlength=8)
    for draw, support in enumerate(saved["support"]):
        mask = saved["eval_masks"][draw]
        assert len(support) == len(np.unique(support)) == 64
        expected = np.ones(len(frame), bool)
        expected[support] = False
        np.testing.assert_array_equal(mask, expected)
        sf, qf = frame.iloc[support], frame.loc[mask]
        sc = np.bincount(y[support], minlength=8)
        qc = np.bincount(y[mask], minlength=8)
        np.testing.assert_array_equal(sc+qc, target_counts)
        result.append(dict(dataset=dataset, task=task, draw=draw, target_n=len(frame), support_n=64,
            query_n=int(mask.sum()), support_fraction=64/len(frame),
            target_classes=int((target_counts>0).sum()), support_classes=int((sc>0).sum()), query_classes=int((qc>0).sum()),
            missing_target_classes=int(((target_counts>0)&(sc==0)).sum()),
            query_support_unseen_class_rows=int(qc[sc==0].sum()),
            query_support_unseen_class_share=float(qc[sc==0].sum()/len(qf)),
            distinct_support_identities=len(identity_set(sf)),
            query_same_known_identity=int(qf.identity.isin(identity_set(sf)).sum()),
            query_same_product=int(qf["product"].isin(set(sf["product"])).sum()),
            common_query_n=int(np.all(saved["eval_masks"], axis=0).sum()),
            **{f"support_class_{c}":int(sc[c]) for c in range(8)}))
    return result


def source_audit(source, target, rows, dataset, task, exclusion_scope):
    sf = source.iloc[rows]
    held = source.actual_port.isin(set(target.actual_port)-{""})
    sameid = source.identity.isin(identity_set(target))
    sameproduct = source["product"].isin(set(target["product"]))
    return dict(dataset=dataset, task=task, source_n=len(sf), excluded_n=len(source)-len(sf),
        source_ports=sf.actual_port.nunique(), source_classes=sf.class_id.nunique(),
        heldport_source_rows=int(sf.actual_port.isin(set(target.actual_port)-{""}).sum()),
        target_identity_source_rows=int(sf.identity.isin(identity_set(target)).sum()),
        target_product_source_rows=int(sf["product"].isin(set(target["product"])).sum()),
        full_source_same_target_identity=int(sameid.sum()), full_source_same_target_product=int(sameproduct.sum()),
        full_source_heldport=int(held.sum()), additional_excluded_beyond_target_port=len(source)-len(sf)-int(held.sum()),
        source_melbourne_n=int(sf.actual_port.eq("Melbourne").sum()), scope=exclusion_scope,
        **{f"source_class_{c}":int((sf.class_id==c).sum()) for c in range(8)})


def pretraining_audit(frames):
    import torch
    ck = torch.load(BASE / "e206_out/ssl_resnet18.pt", map_location="cpu", weights_only=True)
    evidence = dict(checkpoint=str(BASE / "e206_out/ssl_resnet18.pt"), checkpoint_epoch=int(ck["epoch"]),
        checkpoint_keys=list(ck), script=str(BASE / "e206_ssl_pretrain.py"),
        log=str(BASE / "e206_ssl.txt"), epoch_log=str(BASE / "e206_out/pretrain_log.txt"),
        registered_corpus=str(POOL), logged_corpus_rows=345590, logged_epochs=40,
        recipe="All PNG paths recursively, no port/MMSI/product holdout; label-free SimCLR; shuffle=True, drop_last=True",
        exact_historical_per_epoch_sample_list_available=False,
        warning="Current pool metadata can establish corpus/identity overlap, not exact historical pixel membership or exact sample consumption.")
    del ck
    pool = read(POOL / "labels.csv", dtype={"mmsi":str})
    pool["identity"] = identities(pool.mmsi)
    pool["obs_day"] = pool.obs_time.astype(str).str.extract(r"^(\d{8})", expand=False).fillna("")
    evidence.update(current_pool_label_rows=len(pool), current_pool_unique_pngs=pool.png.nunique(),
        current_pool_identities=pool.loc[pool.identity.ne(""), "identity"].nunique(),
        current_pool_date_min=pool.obs_day.min(), current_pool_date_max=pool.obs_day.max())
    # Read only this known training corpus; no image decoding or filesystem-wide search.
    pngs = list((POOL / "chips").glob("**/*.png"))
    evidence["current_pool_png_count"] = len(pngs)
    current_names = {str(p.relative_to(POOL)).replace("/", "\\") for p in pngs}
    evidence["current_label_png_paths_found"] = int(pool.png.isin(current_names).sum())
    observed = pool[pool.png.isin(current_names)]
    poolids = set(observed.loc[observed.identity.ne(""), "identity"])
    pooldays = set(observed.obs_day)-{""}
    poolpairs = set(zip(observed.identity, observed.obs_day))
    records = []
    for name, f in frames.items():
        groups = [("all", f)] + (list(f.groupby("actual_port")) if name == "self24ports" else [])
        for task, part in groups:
            ids = identity_set(part)
            daypair = [(i,d) in poolpairs for i,d in zip(part.identity, part.acquisition_day)]
            records.append(dict(dataset=name, task=task, rows=len(part),
                target_identities=len(ids), shared_pool_identities=len(ids&poolids),
                target_rows_identity_in_pool=int(part.identity.isin(poolids).sum()),
                target_rows_day_in_pool=int(part.acquisition_day.isin(pooldays).sum()),
                target_rows_identity_and_day_in_pool=int(sum(daypair)),
                interpretation="Identity overlap only; matching day does not establish same acquisition or chip"))
    save("pretraining_overlap", records)
    domains = read(POOL / "domain_merged_counts.csv")
    save("pretraining_pool_domains", domains)
    write("PRETRAINING_EVIDENCE.json", evidence)
    return evidence


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {"self24ports":read(SOURCE / "manifest.csv", dtype={"mmsi":str})}
    for name in ("oss1", "oss2"):
        raw[name] = read(ROOT / f"opensarship{name[-1]}_transfer_20261007/target_manifest.csv", dtype={"mmsi_or_ship_id":str})
    fields = []
    for name, f in raw.items():
        for key in f:
            fields.append(dict(dataset=name, field=key, rows=len(f),
                nonempty=int((~f[key].astype(str).str.lower().isin(["", "nan", "none"])).sum()),
                distinct=f[key].nunique()))
    save("field_availability", fields)
    frames = {name:normalize(f,name) for name,f in raw.items()}
    source = frames["self24ports"]
    grdh = frames["oss2"].loc[frames["oss2"].product_type_parsed.eq("GRDH")].copy()
    grdh["dataset"] = "oss2_grdh"
    frames["oss2_grdh"] = grdh
    summaries, numbers, classes, supports, exclusions = [], [], [], [], []
    for name, frame in frames.items():
        summaries.append(describe(frame, name))
        numbers.extend(numeric_summary(frame, name))
        for c in range(8):
            part = frame[frame.class_id.eq(c)]
            classes.append(dict(dataset=name, task="all", class_id=c,
                class_name=source.loc[source.class_id.eq(c), "class_name"].iloc[0], n=len(part), share=len(part)/len(frame)))
            numbers.extend(numeric_summary(part, name, class_id=c))
    metacols = ["dataset", "manifest_row", "sample_id", "class_id", "class_name", "actual_port", "identity", "valid_mmsi_format",
        "product", "product_type_parsed", "satellite", "acquisition_time", "descriptor_available"] + NUMERIC
    save("metadata", pd.concat([f.reindex(columns=metacols) for f in frames.values()], ignore_index=True))
    eligible = sorted(p.name for p in (OLD / "targets/self24ports").iterdir() if p.is_dir())
    missing = sorted(set(source.actual_port)-set(eligible))
    assert missing == ["Melbourne"]
    for port, frame in source.groupby("actual_port"):
        summaries.append(describe(frame, "self24ports", port))
        numbers.extend(numeric_summary(frame, "self24ports", port))
        for c in range(8):
            classes.append(dict(dataset="self24ports", task=port, class_id=c,
                class_name=source.loc[source.class_id.eq(c), "class_name"].iloc[0], n=int(frame.class_id.eq(c).sum()), share=float(frame.class_id.eq(c).mean())))
        if port not in eligible:
            assert len(frame) == 64
            continue
        saved = npz(OLD / "targets/self24ports" / port / "predictions.npz", ["rows", "ids", "support", "eval_masks"])
        target = source.iloc[saved["rows"]].reset_index(drop=True)
        assert set(target.actual_port) == {port}
        np.testing.assert_array_equal(np.sort(saved["rows"]), frame.index.to_numpy())
        supports.extend(support_audit(target, saved, "self24ports", port))
        rows = npz(ROOT / "learned_frequency_residual_20261009/tasks/self24ports" / port / "target.npz", ["source_rows"])["source_rows"]
        exclusions.append(source_audit(source, target, rows, "self24ports", port, "historical per-port exclusion"))
    for name in ("oss1", "oss2", "oss2_grdh"):
        saved = npz(PUBLIC / "targets/shared_frozen" / name / "predictions.npz", ["ids", "support", "eval_masks"])
        frame = frames[name].set_index("sample_id", drop=False).loc[saved["ids"]].reset_index(drop=True)
        supports.extend(support_audit(frame, saved, name, name))
        origin = name.split("_")[0]
        rows = npz(ROOT / f"opensarship{origin[-1]}_transfer_20261007/source_split.npz", ["source_rows"])["source_rows"]
        exclusions.append(source_audit(source, frame, rows, name, name, "historical exclusion against full native target, not only selected fine-class cohort"))
    common = np.intersect1d(*[npz(ROOT / f"opensarship{i}_transfer_20261007/source_split.npz", ["source_rows"])["source_rows"] for i in (1,2)])
    for name in ("oss1", "oss2"):
        exclusions.append(source_audit(source, frames[name], common, name, "public_shared_source", "intersection of both native-target source purges"))
    ex = save("source_exclusions", exclusions)
    assert ex[["heldport_source_rows", "target_identity_source_rows", "target_product_source_rows"]].to_numpy().sum() == 0
    save("summary", summaries)
    save("class_counts", classes)
    save("numeric_summary", numbers)
    support = save("support_draws", supports)
    aggregates = support.groupby("dataset").agg(draw_contexts=("draw","size"), tasks=("task","nunique"),
        support_classes_min=("support_classes","min"), support_classes_mean=("support_classes","mean"), support_classes_max=("support_classes","max"),
        query_classes_min=("query_classes","min"), query_classes_max=("query_classes","max"),
        complete_target_class_draws=("missing_target_classes",lambda s:int((s==0).sum())),
        unseen_query_share_mean=("query_support_unseen_class_share","mean"),
        same_identity_query_min=("query_same_known_identity","min"), same_identity_query_max=("query_same_known_identity","max")).reset_index()
    save("support_summary", aggregates)
    overlap = []
    for a,b in itertools.combinations(frames,2):
        fa,fb = frames[a], frames[b]
        shared = identity_set(fa)&identity_set(fb)
        overlap.append(dict(dataset_a=a,dataset_b=b,shared_identities=len(shared),
            a_rows_shared_identity=int(fa.identity.isin(shared).sum()),b_rows_shared_identity=int(fb.identity.isin(shared).sum()),
            exact_products=len(set(fa["product"])&set(fb["product"])),
            satellite_starttime_keys=len(set(fa.acquisition_key)&set(fb.acquisition_key))))
    save("dataset_overlap", overlap)
    evidence = pretraining_audit(frames)
    receipt = dict(status="complete", rows_by_dataset={k:len(v) for k,v in frames.items()},
        historical_eligible_ports=eligible, missing_port="Melbourne", missing_port_rows=64,
        missing_reason="Historical budget B=64 leaves zero query rows; retained in source where not otherwise excluded",
        common_public_source_rows=len(common), support_contexts=len(support),
        descriptor_oss1="No existing matching descriptor cache; not recomputed or imputed",
        alignment_checks="Descriptor row IDs/sample IDs, target row IDs/sample IDs, support uniqueness/disjointness and class partitions",
        exclusion_checks="No heldport/known target identity/exact product in loaded historical source cohorts",
        model_fits=0, inference_runs=0, image_decodes=0, pretraining=evidence)
    write("COMPLETE.json", receipt)
    print(pd.DataFrame(summaries).query("task == 'all'").to_string(index=False), flush=True)
    print(aggregates.to_string(index=False), flush=True)
    print(json.dumps(evidence, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
