"""CPU-only native-chip geometry review; stdout statistics and local case panels."""
from pathlib import Path
import argparse
from collections import Counter
import importlib.util
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
spec = importlib.util.spec_from_file_location(
    "native_input_review", WORK / "sar_m123_standalone/sar_m123/input.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def quantiles(values):
    return np.round(np.quantile(values, [0, .1, .25, .5, .75, .9, 1]), 4).tolist()


def emit(name, value):
    print(name + " " + json.dumps(value, ensure_ascii=False), flush=True)


def geometry(row, pol):
    item = row.get(pol)
    if item is None:
        return dict(valid=False, reason="missing_spec")
    with Image.open(item["image"]) as im:
        size, mode = im.size, im.mode
    view = native.prepare_view(item)
    result = dict(size=list(size), mode=mode, valid=bool(view and view["valid_target"]))
    if not result["valid"]:
        target = item.get("target_index")
        if not item.get("alignment_valid", True):
            reason = "alignment_invalid"
        elif target is None or target == -1:
            reason = "no_target_index"
        else:
            det = item["detections"][target]
            reason = "shape_unusable" if not det.get("shape_usable", True) else (
                "confidence_below_0.25" if det["confidence"] < .25 else "empty_target_mask")
        result["reason"] = reason
        return result
    g, p = view["geometry"], view["placement"]
    result.update(length=float(g[2]), width=float(g[3]), confidence=float(g[5]),
                  padding=1-p["crop_width"]*p["crop_height"]/160**2,
                  target_pixels=p["retained_target_pixels"],
                  target_retention=p["retained_target_pixels"]/p["native_target_pixels"],
                  crop=bool(size[0] > 160 or size[1] > 160))
    return result


def panel(row):
    out = Image.new("RGB", (1010, 442), "white")
    draw = ImageDraw.Draw(out)
    draw.text((8, 4), row["id"] + " | native pixels; grids are geometric guides, not feature maps", fill="black")
    for p, pol in enumerate(("vv", "vh")):
        view = native.prepare_view(row.get(pol))
        if view is None:
            continue
        y = 35 + 202*p
        raw = Image.fromarray(view["pixels"]).convert("RGB")
        out.paste(raw, (8, y+20))
        draw.text((8, y), pol.upper()+" native 160 canvas", fill="black")
        det = row[pol]["detections"][row[pol]["target_index"]]
        pos = view["placement"]
        points = [(x+pos["pad_left"]-pos["crop_x"], yy+pos["pad_top"]-pos["crop_y"])
                  for x, yy in det["corners"]]
        marked = raw.copy()
        ImageDraw.Draw(marked).line(points+[points[0]], fill=(40, 230, 160), width=1)
        out.paste(marked, (178, y+20))
        draw.text((178, y), "OBB native | valid="+str(view["valid_target"]), fill="black")
        # Crop a central 80-pixel window only for the labeled nearest-neighbor detail.
        cx, cy = view["geometry"][:2] if view["geometry"] is not None else (80, 80)
        x0, y0 = max(0, min(80, int(cx)-40)), max(0, min(80, int(cy)-40))
        for x, stride in ((348, 8), (678, 16)):
            # Use an isotropic 2x inset centered within the 320-pixel column.
            zoom = raw.crop((x0, y0, x0+80, y0+80)).resize((160, 160), Image.Resampling.NEAREST)
            overlay = ImageDraw.Draw(zoom)
            for t in np.arange(stride/2, 160-stride/2, stride):
                if x0 <= t < x0+80:
                    overlay.line(((2*(t-x0), 0), (2*(t-x0), 159)), fill=(90, 110, 130))
                if y0 <= t < y0+80:
                    overlay.line(((0, 2*(t-y0)), (159, 2*(t-y0))), fill=(90, 110, 130))
            out.paste(zoom, (x, y+20))
            draw.text((x, y), f"stride {stride} | 80px detail x2", fill="black")
            g = view["geometry"]
            if g is not None:
                draw.text((x+165, y+35), f"L/s={g[2]/stride:.2f}\nW/s={g[3]/stride:.2f}\nL/4s={g[2]/4/stride:.2f}\nW/2s={g[3]/2/stride:.2f}", fill="black")
    out.save(ROOT/(row["id"]+"_data_scale.png"))


def reference_panel(case, lookup):
    sources = list(dict.fromkeys(s["id"] for s in case["sources"]))
    out = Image.new("RGB", (1080, 404), "white")
    draw = ImageDraw.Draw(out)
    for i, rid in enumerate(sources):
        row = lookup[rid]
        for p, pol in enumerate(("vv", "vh")):
            x = 180*(i*2+p)+8
            view = native.prepare_view(row.get(pol))
            draw.text((x, 4), rid[-6:]+f" C{row['label']} "+pol.upper(), fill="black")
            if view is None:
                continue
            raw = Image.fromarray(view["pixels"]).convert("RGB")
            out.paste(raw, (x, 25))
            mask = view["masks"][0].astype(bool)
            edge = mask & ~(np.roll(mask, 1, 0)&np.roll(mask, -1, 0)&np.roll(mask, 1, 1)&np.roll(mask, -1, 1))
            rgb = np.array(raw)
            rgb[edge] = (40, 230, 160)
            out.paste(Image.fromarray(rgb), (x, 225))
            draw.text((x, 199), "query" if i == 0 else "actual source reference", fill="black")
    out.save(ROOT/(case["id"]+"_data_references.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", action="store_true")
    args = parser.parse_args()
    rows = json.loads((WORK/"m1_coverage_20261101/manifest.json").read_text(encoding="utf-8"))
    groups = json.loads((WORK/"m1_coverage_20261101/groups.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in rows}
    emit("groups", {k: len(v) for k, v in groups.items()})
    used = sorted(set(groups["small"]+groups["validation"]+groups["evaluation"]))
    stats = {(i, p): geometry(rows[i], p) for i in used for p in ("vv", "vh")}
    available = np.load(WORK/"m1_coverage_20261101/cache/a/available.npy", mmap_mode="r")
    disagreements = [(rows[i]["id"], p) for i in used for j, p in enumerate(("vv", "vh"))
                     if bool(available[i, j]) != stats[i, p]["valid"]]
    emit("cache_availability_disagreements", disagreements)
    for group in ("small", "validation", "evaluation", "references"):
        ids = groups[group]
        emit("cohort "+group, dict(n=len(ids), ships=len({rows[i]["mmsi"] for i in ids}),
             passes=len({rows[i]["pass_group"] for i in ids}),
             labels=dict(Counter(rows[i]["label"] for i in ids)),
             availability=dict(Counter("".join("1" if stats[i,p]["valid"] else "0" for p in ("vv","vh")) for i in ids))))
        for pol in ("vv", "vh"):
            all_s = [stats[i, pol] for i in ids]
            ss = [s for s in all_s if s["valid"]]
            out = dict(valid=len(ss), invalid=dict(Counter(s["reason"] for s in all_s if not s["valid"])),
                       native_sizes=dict(Counter(str(s["size"]) for s in all_s if "size" in s)),
                       native_size_mismatch=[rows[i]["id"] for i in ids if "size" in stats[i,pol] and list(rows[i]["native_size"]) != stats[i,pol]["size"]],
                       cropped=sum(s["crop"] for s in ss), min_target_retention=min(s["target_retention"] for s in ss))
            for key in ("length", "width", "padding", "target_pixels"):
                out[key+"_min_p10_p25_p50_p75_p90_max"] = quantiles([s[key] for s in ss])
            for key in ("length", "width"):
                out[key+"_lt_thresholds"] = {str(t): sum(s[key] < t for s in ss) for t in (4, 8, 16, 32, 64)}
            emit(group+" "+pol, out)
            emit(group+" "+pol+" by_class", {str(c): dict(n=len(part),
                 length_median=float(np.median([s["length"] for s in part])),
                 width_median=float(np.median([s["width"] for s in part])))
                 for c in range(4) if (part := [stats[i,pol] for i in ids if rows[i]["label"] == c and stats[i,pol]["valid"]])})
    ids = groups["evaluation"]
    predictions = []
    for seed in (17, 29, 43):
        data = np.load(ROOT.parent/f"staged_ablation_20261109/full_s{seed}_lr1/evaluation.npz")
        if not np.array_equal(data["ids"], ids):
            raise ValueError("Prediction indices do not match evaluation cohort")
        predictions.append(data["score"])
    scores = np.stack(predictions)
    correct = scores.argmax(-1) == np.array([rows[i]["label"] for i in ids])[None]
    for pol in ("vv", "vh"):
        for lo, hi in ((0, 16), (16, 32), (32, float("inf"))):
            js = [j for j, i in enumerate(ids) if stats[i,pol]["valid"] and lo <= stats[i,pol]["length"] < hi]
            emit("baseline_length_stratum", dict(pol=pol, interval=f"[{lo},{hi})", n=len(js),
                 class_counts=dict(Counter(rows[ids[j]]["label"] for j in js)),
                 oa_by_seed=(100*correct[:,js].mean(1)).tolist()))
    selected = []
    for state in (True, False):
        candidates = [j for j, i in enumerate(ids) if np.all(correct[:,j] == state)
                      and all(stats[i,p]["valid"] for p in ("vv","vh"))]
        candidates.sort(key=lambda j: (max(stats[ids[j],p]["length"] for p in ("vv","vh")), rows[ids[j]]["id"]))
        for tag, j in (("small", candidates[0]), ("large", candidates[-1])):
            selected.append(ids[j])
            emit("deterministic_case", dict(size=tag, stable_correct=state, id=rows[ids[j]]["id"]))
        if not state:
            j = max(candidates, key=lambda j: min(stats[ids[j],p]["length"] for p in ("vv","vh")))
            selected.append(ids[j])
            emit("deterministic_case", dict(size="large_by_both_polarizations", stable_correct=False, id=rows[ids[j]]["id"]))
    selected += [i for i in ids if rows[i]["id"][-6:] in ("000141", "000295", "002557", "000132")]
    for i in dict.fromkeys(selected):
        j = ids.index(i)
        row = rows[i]
        y = row["label"]
        emit("case", dict(id=row["id"], label=y, mmsi=row["mmsi"], pass_group=row["pass_group"],
             predictions=scores[:,j].argmax(-1).tolist(),
             margins=(scores[:,j,y]-np.max(np.delete(scores[:,j], y, axis=1),axis=1)).tolist(),
             vv=stats[i,"vv"], vh=stats[i,"vh"], paths={p:row[p]["image"] for p in ("vv","vh")}))
        for pol in ("vv", "vh"):
            path = Path(row["source_paths"][pol])
            if path.exists():
                with Image.open(path) as src, Image.open(row[pol]["image"]) as chip:
                    emit("provenance", dict(id=row["id"], pol=pol, source=str(path),
                         source_size=src.size, source_mode=src.mode,
                         exact_display_pixels=bool(np.array_equal(np.array(src.convert("L")),np.array(chip.convert("L"))))))
        if args.panels:
            panel(row)
        rival = int(np.argmax(np.where(np.arange(4) == y, -np.inf, scores[0,j])))
        references = [next(rows[k] for k in groups["references"] if rows[k]["label"] == label
                          and rows[k]["mmsi"] != row["mmsi"] and rows[k]["pass_group"] != row["pass_group"])
                      for label in (y, rival)]
        emit("selected_references", dict(id=row["id"], rival_seed17=rival,
             references=[dict(id=r["id"], label=r["label"], vv=geometry(r,"vv"), vh=geometry(r,"vh")) for r in references]))
        if args.panels:
            reference_panel(dict(id=row["id"], sources=[dict(id=r["id"]) for r in [row]+references]), lookup)
    cases = json.loads((ROOT.parent/"m2_region_identity_20261111/cases_local.json").read_text(encoding="utf-8"))
    for rid in ("000141", "000295", "002557", "000132"):
        case = next(c for c in cases if c["id"].endswith(rid))
        emit("prior_case", {k:case[k] for k in ("id","before_arm","after_arm","kind","seed","margin_before","margin_after")})
        emit("case_reference_ids", dict(id=case["id"], sources=list(dict.fromkeys(s["id"] for s in case["sources"]))))
        if args.panels:
            reference_panel(case, lookup)


if __name__ == "__main__":
    main()
