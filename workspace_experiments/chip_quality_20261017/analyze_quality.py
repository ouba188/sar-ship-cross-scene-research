"""Screen native SAR chips; detector flags are not adjudicated label errors."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
POLS = ("vv", "vh")
THRESHOLDS = (.08, .25, .50)


def load_manifest():
    frame = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    keys = ["instance_id", "pol", "target_center_x", "target_center_y", "crop_left", "crop_top"]
    restored = pd.read_csv(HERE / "ais_metadata/view_metadata.csv", usecols=keys, dtype=str, keep_default_na=False)
    for pol in POLS:
        v = restored[restored.pol.eq(pol)].set_index("instance_id")
        for key in keys[2:]:
            col = key+"_"+pol
            missing = frame[col].eq("")
            frame.loc[missing, col] = frame.loc[missing, "instance_id"].map(v[key])
    return frame


def read_detections(path):
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    result = {(r["global_row"], r["pol"]): r for r in records}
    assert len(result) == len(records), "Repeated image records"
    return result


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def usable(d, threshold=.25):
    return d["confidence"] >= threshold and d["shape_usable"] and d["center_inside"]


def select_recheck(chips):
    all_candidates = chips[chips.multi_any_25 | chips.length_screen_any].copy()
    all_candidates["recheck_reason"] = "all_multi_or_length_screen"
    width_only = chips[chips.width_screen_any & ~chips.global_row.isin(all_candidates.global_row)]
    width_sample = pd.concat([g.sample(n=min(16,len(g)), random_state=20261017) for _,g in width_only.groupby("port")]).copy()
    width_sample["recheck_reason"] = "width_only_port_stratified_sample"
    controls = chips[~(chips.multi_any_25 | chips.length_screen_any | chips.width_screen_any)].sample(n=96, random_state=20261017).copy()
    controls["recheck_reason"] = "random_unscreened_control"
    result = pd.concat([all_candidates,width_sample,controls]).sort_values("global_row")
    result[["global_row","recheck_reason"]].to_csv(HERE / "recheck_rows.csv", index=False)
    return result


def analyze_image(record, row, meta):
    pol = record["pol"]
    ax, ay = (number(row.get(f"target_chip_center_{a}_{pol}")) for a in ("x", "y"))
    anchor_source = "historical_target_chip_center"
    if not np.isfinite([ax, ay]).all():
        ax = number(row.get("target_center_x_"+pol)) - number(row.get("crop_left_"+pol))
        ay = number(row.get("target_center_y_"+pol)) - number(row.get("crop_top_"+pol))
        anchor_source = "historical_source_center_minus_crop_offset"
    anchor_valid = np.isfinite([ax, ay]).all() and 0 <= ax < record["native_w"] and 0 <= ay < record["native_h"]
    out = dict(global_row=record["global_row"], instance_id=record["instance_id"], pol=pol,
               anchor_x=ax, anchor_y=ay, anchor_valid=bool(anchor_valid), anchor_source=anchor_source,
               port=row["port"], class_name=row["class_name"], mmsi=row["mmsi"], product=row["product"],
               native_w=record["native_w"], native_h=record["native_h"], image=record["image"])
    for t in THRESHOLDS:
        out[f"n_{int(t*100):02d}"] = sum(usable(d, t) for d in record["detections"])
    out["partial_candidates_25"] = sum(d["confidence"] >= .25 and d["shape_usable"] and not d["center_inside"]
                                       for d in record["detections"])
    candidates = [(i, d) for i, d in enumerate(record["detections"]) if usable(d)]
    containing = [(i, d) for i, d in candidates if anchor_valid and
                  cv2.pointPolygonTest(np.array(d["corners"], np.float32), (ax, ay), False) >= 0]
    selected = None
    status = "no_valid_anchor" if not anchor_valid else "no_target_match"
    if containing:
        selected = min(containing, key=lambda x: np.hypot(x[1]["xywhr"][0]-ax, x[1]["xywhr"][1]-ay))
        status = "inside_unique_box" if len(containing) == 1 else "inside_multiple_boxes_ambiguous"
    elif anchor_valid and candidates:
        nearest = min(candidates, key=lambda x: np.hypot(x[1]["xywhr"][0]-ax, x[1]["xywhr"][1]-ay))
        distance = np.hypot(nearest[1]["xywhr"][0]-ax, nearest[1]["xywhr"][1]-ay)
        if distance <= max(8., .15*min(record["native_w"], record["native_h"])):
            selected, status = nearest, "near_anchor_not_containing"
    out.update(target_match_status=status, target_index=-1, target_confidence=np.nan, target_edge=False,
               target_length_px=np.nan, target_width_px=np.nan, target_length_m=np.nan, target_width_m=np.nan,
               length_ratio=np.nan, width_ratio=np.nan, length_screen=False, width_screen=False,
               size_comparable=False, primary_size_eligible=False)
    for key in ("ais_length_m", "ais_width_m", "ais_size_valid", "ais_size_scope", "ais_record_match_quality", "vessel_static_conflict"):
        out[key] = meta.get(key, "")
    out["scale_status"] = meta.get("ground_scale_status_"+pol, "missing")
    if selected is not None:
        idx, d = selected
        out.update(target_index=idx, target_confidence=d["confidence"], target_edge=d["touches_edge"],
                   target_length_px=d["length_px"], target_width_px=d["beam_px"])
        sx, sy = (number(meta.get(f"pixel_{a}_m_{pol}")) for a in ("x", "y"))
        if np.isfinite([sx, sy]).all() and min(sx, sy) > 0:
            # Source transforms use GDAL order [x0,a,b,y0,d,e], not Rasterio order.
            g = json.loads(meta["source_transform_json_"+pol])
            linear = np.array([[g[1], g[2]], [g[4], g[5]]], float)
            edges = np.diff(np.array(d["corners"], float)[:3], axis=0)
            lengths = np.linalg.norm(edges @ linear.T, axis=1)
            length_m, width_m = max(lengths), min(lengths)
            out.update(target_length_m=length_m, target_width_m=width_m)
            al, aw = number(meta.get("ais_length_m")), number(meta.get("ais_width_m"))
            if str(meta.get("ais_size_valid", "True")).lower() == "true" and np.isfinite([al, aw]).all() and al >= aw > 0:
                lr, wr = length_m/al, width_m/aw
                out.update(size_comparable=True, length_ratio=lr, width_ratio=wr,
                           length_screen=bool((lr < .5 or lr > 2.) and abs(length_m-al) > 30.),
                           width_screen=bool((wr < .5 or wr > 2.) and abs(width_m-aw) > 20.),
                           primary_size_eligible=bool(status == "inside_unique_box" and not d["touches_edge"] and out["n_25"] == 1))
    return out


def gallery(chips, views, detections, rows, name, title):
    rows = list(rows)[:8]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, len(rows)*3.2), squeeze=False)
    indexed = views.set_index(["global_row", "pol"])
    for ri, index in enumerate(rows):
        for pi, pol in enumerate(POLS):
            ax = axes[ri, pi]
            rec, v = detections[(int(index), pol)], indexed.loc[(index, pol)]
            with Image.open(rec["image"]) as im:
                ax.imshow(np.array(im.convert("L")), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            for di, d in enumerate(rec["detections"]):
                if d["confidence"] < .25 or not d["shape_usable"]:
                    continue
                color = "#32dd80" if di == v.target_index else "#ffb23c"
                if not d["center_inside"]:
                    color = "#ff5460"
                ax.add_patch(Polygon(d["corners"], fill=False, edgecolor=color, linewidth=1.1))
                ax.text(d["xywhr"][0], d["xywhr"][1], f'{d["confidence"]:.2f}', color=color, fontsize=7,
                        bbox=dict(facecolor="black", alpha=.6, pad=.5, edgecolor="none"))
            if v.anchor_valid:
                ax.plot(v.anchor_x, v.anchor_y, "+", color="#62e6ff", markersize=9)
            al, aw = number(v.ais_length_m), number(v.ais_width_m)
            ais = f"AIS {al:g} x {aw:g} m" if np.isfinite([al, aw]).all() else "AIS size unavailable"
            detected = (f"OBB {v.target_length_m:.0f} x {v.target_width_m:.0f} m" if np.isfinite(v.target_length_m)
                        else f"OBB {v.target_length_px:.1f} x {v.target_width_px:.1f} px")
            ax.set_title(f"#{index}  {v.port} / {pol.upper()} / {v.class_name}\nMMSI {v.mmsi} | {ais}\n{detected} | count {v.n_25}", fontsize=9)
            ax.set_xlim(-.5, rec["native_w"]-.5)
            ax.set_ylim(rec["native_h"]-.5, -.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title+"\nGreen: label-target match; orange: other candidate; cyan +: historical label anchor", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(HERE / (name+".png"), dpi=140)
    plt.close(fig)


def main():
    manifest = load_manifest()
    metadata = pd.read_csv(HERE / "ais_metadata/ais_metadata_sidecar.csv", dtype=str, keep_default_na=False).set_index("instance_id")
    detections = read_detections(HERE / "full/canvas512/detections.jsonl")
    assert len(detections) == len(manifest)*2, "Full detection incomplete"
    records = []
    for i, row in enumerate(manifest.to_dict("records")):
        meta = metadata.loc[row["instance_id"]].to_dict()
        for pol in POLS:
            rec = detections[(i, pol)]
            assert rec["instance_id"] == row["instance_id"]
            records.append(analyze_image(rec, row, meta))
    views = pd.DataFrame(records)
    views.to_csv(HERE / "per_view_quality.csv", index=False, encoding="utf-8-sig")
    objects = [dict(global_row=r["global_row"], instance_id=r["instance_id"], pol=r["pol"], detection_index=i,
                    **{k: json.dumps(v) if isinstance(v, list) else v for k,v in d.items()})
               for r in detections.values() for i,d in enumerate(r["detections"])]
    pd.DataFrame(objects).to_csv(HERE / "detected_objects.csv", index=False, encoding="utf-8-sig")
    chips = manifest[["instance_id", "port", "class_name", "mmsi", "product", "image_relpath_vv", "image_relpath_vh"]].copy()
    chips.insert(0, "global_row", np.arange(len(chips)))
    for pol in POLS:
        v = views[views.pol.eq(pol)].set_index("global_row")
        for key in ("n_08", "n_25", "n_50", "partial_candidates_25", "target_match_status", "target_edge",
                    "target_length_px", "target_width_px", "target_length_m", "target_width_m",
                    "length_ratio", "width_ratio", "length_screen", "width_screen", "size_comparable", "primary_size_eligible"):
            chips[key+"_"+pol] = v[key]
    for t in (8, 25, 50):
        chips[f"multi_any_{t:02d}"] = (chips[f"n_{t:02d}_vv"] >= 2) | (chips[f"n_{t:02d}_vh"] >= 2)
        chips[f"multi_both_{t:02d}"] = (chips[f"n_{t:02d}_vv"] >= 2) & (chips[f"n_{t:02d}_vh"] >= 2)
    for key in ("length_screen", "width_screen", "size_comparable", "primary_size_eligible"):
        chips[key+"_any"] = chips[key+"_vv"] | chips[key+"_vh"]
    chips["multi_including_partial_25"] = ((chips.n_25_vv + chips.partial_candidates_25_vv >= 2) |
                                          (chips.n_25_vh + chips.partial_candidates_25_vh >= 2))
    chips.to_csv(HERE / "per_chip_quality.csv", index=False, encoding="utf-8-sig")
    chips[chips.multi_any_25].to_csv(HERE / "multi_ship_candidates.csv", index=False, encoding="utf-8-sig")
    chips[chips.length_screen_any | chips.width_screen_any].to_csv(HERE / "size_mismatch_candidates.csv", index=False, encoding="utf-8-sig")
    summary = dict(chips=len(chips), images=len(views), thresholds={}, view_statistics={}, primary_size={})
    for t in (8, 25, 50):
        summary["thresholds"][str(t/100)] = dict(multi_any=int(chips[f"multi_any_{t:02d}"].sum()),
            multi_both=int(chips[f"multi_both_{t:02d}"].sum()),
            multi_vv=int((chips[f"n_{t:02d}_vv"] >= 2).sum()),
            multi_vh=int((chips[f"n_{t:02d}_vh"] >= 2).sum()),
            zero_both=int(((chips[f"n_{t:02d}_vv"] == 0) & (chips[f"n_{t:02d}_vh"] == 0)).sum()))
    for pol in POLS:
        v = views[views.pol.eq(pol)]
        primary = v[v.primary_size_eligible]
        summary["view_statistics"][pol] = dict(target_match=v.target_match_status.value_counts().to_dict(),
            size_comparable=int(v.size_comparable.sum()), edge=int(v.target_edge.sum()),
            length_screen=int(v.length_screen.sum()), width_screen=int(v.width_screen.sum()))
        summary["primary_size"][pol] = dict(n=len(primary), length_screen=int(primary.length_screen.sum()),
            width_screen=int(primary.width_screen.sum()), length_ratio_median=primary.length_ratio.median(),
            length_ratio_q10_q90=primary.length_ratio.quantile([.1, .9]).tolist(),
            width_ratio_median=primary.width_ratio.median(),
            length_within_25percent=int(primary.length_ratio.between(.75,1.25).sum()),
            by_ais_scope=primary.ais_size_scope.value_counts().to_dict())
    scope = views[views.primary_size_eligible].groupby(["pol", "ais_size_scope"]).agg(
        n=("global_row", "size"), length_screen=("length_screen", "sum"), width_screen=("width_screen", "sum"),
        length_ratio_median=("length_ratio", "median"), width_ratio_median=("width_ratio", "median"))
    scope.to_csv(HERE / "size_by_ais_scope.csv", encoding="utf-8-sig")
    summary["size_comparable_chips"] = int(chips.size_comparable_any.sum())
    summary["length_screen_chips"] = int(chips.length_screen_any.sum())
    summary["width_screen_chips"] = int(chips.width_screen_any.sum())
    summary["multi_including_partial_25"] = int(chips.multi_including_partial_25.sum())
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for group in ("port", "class_name"):
        stats = chips.groupby(group).agg(n=("global_row", "size"), multi_25=("multi_any_25", "sum"),
            multi_both_25=("multi_both_25", "sum"), size_comparable=("size_comparable_any", "sum"),
            length_screen=("length_screen_any", "sum"), width_screen=("width_screen_any", "sum"))
        stats["multi_percent"] = 100*stats.multi_25/stats.n
        stats["length_screen_percent_of_comparable"] = 100*stats.length_screen/stats.size_comparable.replace(0,np.nan)
        stats.to_csv(HERE / ("by_"+group+".csv"), encoding="utf-8-sig")
    recheck = select_recheck(chips)
    both = chips[chips.multi_both_25].sort_values(["n_50_vv", "n_50_vh"], ascending=False)
    gallery(chips, views, detections, both.groupby("port", sort=False).head(1).global_row, "multi_ship_examples", "Multiple-vessel candidates in both polarizations (not ground truth)")
    one = chips[chips.multi_any_25 & ~chips.multi_both_25]
    gallery(chips, views, detections, one.groupby("port", sort=False).head(1).global_row, "multi_ship_one_view", "Multiple-vessel candidates in only one polarization")
    primary = views[views.primary_size_eligible & views.pol.eq("vv")].copy()
    primary["size_deviation"] = np.abs(np.log(primary.length_ratio))
    bad = primary[primary.length_screen].sort_values("size_deviation", ascending=False)
    gallery(chips, views, detections, bad.groupby("port", sort=False).head(2).global_row, "size_mismatch_examples", "Isolated, untruncated target: large length discrepancy")
    good = primary[primary.length_ratio.between(.9,1.1)].sort_values("global_row")
    gallery(chips, views, detections, good.groupby("port", sort=False).head(1).global_row, "size_agreement_examples", "Length agreement examples (not proof of identity)")
    fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
    for pi, key in enumerate(("length", "width")):
        for pol, color in (("vv", "#167b8e"), ("vh", "#b03974")):
            v = views[views.primary_size_eligible & views.pol.eq(pol)]
            axes[pi].scatter(pd.to_numeric(v["ais_"+key+"_m"]), v["target_"+key+"_m"], s=3, alpha=.13, color=color, label=pol.upper(), rasterized=True)
        lim = (1, 1500) if key == "length" else (1, 300)
        axes[pi].plot(lim, lim, color="black", lw=1)
        for factor in (.5, 2):
            axes[pi].plot(lim, np.array(lim)*factor, "--", color="gray", lw=.8)
        axes[pi].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel=f"AIS {key} (m)", ylabel=f"SAR OBB {key} (m)", title=f"Isolated target: {key}")
        axes[pi].legend(markerscale=3)
    port = pd.read_csv(HERE / "by_port.csv").sort_values("multi_percent")
    axes[2].barh(port.port, port.multi_percent, color="#167b8e")
    axes[2].tick_params(axis="y", labelsize=6)
    axes[2].set(xlabel="Multi-candidate chips (%)", title="Any polarization, confidence >= 0.25")
    fig.tight_layout()
    fig.savefig(HERE / "quality_overview.png", dpi=180)
    plt.close(fig)
    a = read_detections(HERE / "pilot/canvas512/detections.jsonl")
    b = read_detections(HERE / "pilot/canvas1024/detections.jsonl")
    errors, same = [], 0
    for key in a:
        da, db = ([d for d in z[key]["detections"] if usable(d)] for z in (a,b))
        same += len(da) == len(db)
        if len(da) == len(db) == 1:
            errors.append(abs(db[0]["length_px"]/da[0]["length_px"]-1))
    pilot = dict(images=len(a), same_counts=same, common_single=len(errors), relative_length_difference_quantiles=np.quantile(errors,[.5,.95,1]).tolist())
    (HERE / "pilot_comparison.json").write_text(json.dumps(pilot, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Recheck chips:", len(recheck), flush=True)


if __name__ == "__main__":
    main()
