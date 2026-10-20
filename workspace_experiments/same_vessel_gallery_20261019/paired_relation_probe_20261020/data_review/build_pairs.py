"""Identity-balanced test pairs and native-image cases; no fitting or GPU use."""
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent
STRIP = ROOT / "axial_strip_sequence_20261020"
LOCAL = ROOT / "local_condition_module"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows, fields=None):
    fields = fields or list(rows[0])
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def angular_gap(a, b):
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2) * 180 / math.pi


def target_ok(view):
    p = view["priors"]
    j = p.get("target_index", -1)
    ds = p.get("detections", [])
    return (p.get("match_status") == "inside_unique_box"
            and isinstance(j, int) and 0 <= j < len(ds)
            and ds[j].get("shape_usable", False)
            and ds[j].get("confidence", 0) >= .25
            and p.get("ship_axis") is not None)


def point_inside(corners, point):
    # Convex detector quadrilaterals; boundary inclusion matches the old audit.
    a = np.asarray(corners, dtype=float)
    e = np.roll(a, -1, axis=0) - a
    v = np.asarray(point, dtype=float) - a
    cross = e[:, 0] * v[:, 1] - e[:, 1] * v[:, 0]
    return bool(np.all(cross >= -1e-6) or np.all(cross <= 1e-6))


def unique_mapping(source, destination):
    p, q = source["priors"], destination["priors"]
    a = np.asarray(p["scale"]["transform"], dtype=float)
    b = np.asarray(q["scale"]["transform"], dtype=float)
    center = np.asarray(p["detections"][p["target_index"]]["xywhr"][:2])
    world = a[[0, 3]] + a[[1, 2, 4, 5]].reshape(2, 2) @ center
    mapped = np.linalg.solve(b[[1, 2, 4, 5]].reshape(2, 2), world - b[[0, 3]])
    matches = [j for j, d in enumerate(q["detections"])
               if d.get("shape_usable") and d["confidence"] >= .25
               and point_inside(d["corners"], mapped)]
    return matches == [q["target_index"]]


def geometry_fields(view):
    p = view["priors"]
    a = p["ship_axis"]
    radar = p.get("radar_axis")
    valid = p.get("radar_kind") == "orbit_los" and radar is not None and math.isfinite(radar)
    rel = ((a["angle"] - radar + math.pi / 2) % math.pi - math.pi / 2) if valid else None
    return dict(width=view["w"], height=view["h"], length_px=a["length_px"],
                width_px=a["width_px"], length_grid=a["length_px"] / 4,
                aspect=a["length_px"] / a["width_px"], axis_deg=math.degrees(a["angle"]),
                relative_axis_deg=math.degrees(rel) if rel is not None else "",
                incidence=p.get("radar_incidence") if valid else "",
                radar_valid=int(valid), other_boxes=sum(
                    d.get("shape_usable", False) and d["confidence"] >= .25 and j != p["target_index"]
                    for j, d in enumerate(p["detections"])),
                target_confidence=p["detections"][p["target_index"]]["confidence"])


def panel(pair, cohort, views, overlay=False):
    ids = [pair["obs_id_a"], pair["obs_id_b"]]
    scale = 3
    width = max(views[o]["images"][p]["w"] for o in ids for p in ("vv", "vh"))
    height = max(views[o]["images"][p]["h"] for o in ids for p in ("vv", "vh"))
    cw, ch = max(420, width * scale + 24), height * scale + 105
    canvas = Image.new("RGB", (2 * cw, 2 * ch), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for row, oid in enumerate(ids):
        for col, pol in enumerate(("vv", "vh")):
            view = views[oid]["images"][pol]
            with Image.open(cohort[oid]["image_" + pol]) as source:
                raw = source.convert("L").convert("RGB")
            img = raw.resize((raw.width * scale, raw.height * scale), Image.Resampling.NEAREST)
            if overlay:
                d = ImageDraw.Draw(img)
                pri = view["priors"]
                for j, det in enumerate(pri["detections"]):
                    if det.get("shape_usable") and det["confidence"] >= .25:
                        pts = [(scale * x, scale * y) for x, y in det["corners"]]
                        d.line(pts + [pts[0]], fill="lime" if j == pri["target_index"] else "orange", width=1)
                axis = pri["ship_axis"]
                for angle, color in [(axis["angle"], "cyan")]:
                    c = np.asarray(axis["center"])
                    along = np.array([math.cos(angle), math.sin(angle)])
                    across = np.array([-math.sin(angle), math.cos(angle)])
                    half = math.ceil((axis["length_px"] / 2 + 8) / 4)
                    for offset in [0, -axis["width_px"] / 2 - 4, axis["width_px"] / 2 + 4]:
                        line = [tuple((c + offset * across + t * along) * scale)
                                for t in [-half * 4, half * 4]]
                        d.line(line, fill=color, width=1)
            px, py = col * cw + 12, row * ch + 90
            canvas.paste(img, (px, py))
            side = "a" if row == 0 else "b"
            text = (f"{side.upper()} obs={oid} {pol.upper()} native={raw.width}x{raw.height}; display x3\n"
                    f"{cohort[oid]['time']} {cohort[oid]['port']}\n"
                    f"truth={pair['class_name']}\n"
                    f"pred={pair['prediction_' + side]} "
                    f"margin={pair['margin_' + side]:+.4f}")
            draw.multiline_text((px, row * ch + 8), text, fill="black", spacing=3)
    name = f"pair_{ids[0]}_{ids[1]}_{'sampling' if overlay else 'raw'}.png"
    canvas.save(OUT / name)
    return name


def main():
    all_cohort = read_csv(ROOT / "conditional_evidence_pilot/cohort.csv")
    cohort = {int(r["obs_id"]): r for r in all_cohort}
    excluded = {int(r["obs_id"]) for r in read_csv(LOCAL / "EXCLUSIONS.csv")} | {31237}
    pred = [r for r in read_csv(STRIP / "predictions.csv") if r["arm"] == "frozen_fivepoint"]
    baseline = {(int(r["seed"]), int(r["obs_id"])): r for r in pred}
    test_ids = {int(r["obs_id"]) for r in pred if int(r["seed"]) == 0}
    assert len(baseline) == len(pred) == len(test_ids) * 3
    alignment = {int(r["obs_id"]): r for r in read_csv(LOCAL / "crossview_alignment.csv")}
    with (LOCAL / "view_snapshot.json").open(encoding="utf-8") as stream:
        snapshots = json.load(stream)
    views = {v["obs_id"]: v for v in snapshots if v["obs_id"] in test_ids}
    del snapshots
    eligibility, eligible, reasons = [], [], Counter()
    for oid in sorted(test_ids):
        al = alignment[oid]
        vv, vh = (views[oid]["images"][p] for p in ("vv", "vh"))
        reason = "eligible"
        if oid in excluded:
            reason = "explicit_exclusion"
        elif int(al["disable_new_vh"]) or al["status"] == "other":
            reason = "crossview_conflict"
        elif al["status"] != "target":
            reason = "crossview_" + al["status"]
        elif not (target_ok(vv) and target_ok(vh)):
            reason = "nonunique_or_unusable_target"
        elif not (unique_mapping(vv, vh) and unique_mapping(vh, vv)):
            reason = "nonunique_or_nonreciprocal_mapping"
        row = cohort[oid]
        eligibility.append(dict(obs_id=oid, mmsi=row["mmsi"], fold=int(row["fold"]),
                                eligible=int(reason == "eligible"), reason=reason,
                                alignment_status=al["status"], crs_basis=al["crs_basis"]))
        reasons[reason] += 1
        if reason == "eligible":
            eligible.append(oid)
    write_csv("observation_eligibility.csv", eligibility)
    by_ship = defaultdict(list)
    for oid in eligible:
        by_ship[cohort[oid]["mmsi"]].append(oid)
    pairs, main_pairs = [], []
    for ship, ids in sorted(by_ship.items()):
        choices = []
        for ia, ib in itertools.combinations(ids, 2):
            ia, ib = sorted((ia, ib), key=lambda o: (cohort[o]["time"], o))
            a, b = cohort[ia], cohort[ib]
            ta, tb = datetime.fromisoformat(a["time"]), datetime.fromisoformat(b["time"])
            if (a["product"] == b["product"] or ta.date() == tb.date()
                    or a["class_id"] != b["class_id"] or a["fold"] != b["fold"]):
                continue
            pair = dict(pair_id=f"{ia}_{ib}", mmsi=ship, fold=int(a["fold"]),
                        class_id=int(a["class_id"]), class_name=a["class_name"],
                        obs_id_a=ia, obs_id_b=ib, time_a=a["time"], time_b=b["time"],
                        product_a=a["product"], product_b=b["product"],
                        port_a=a["port"], port_b=b["port"], platform_a=a["platform"], platform_b=b["platform"],
                        days_apart=(tb - ta).total_seconds() / 86400,
                        same_port=int(a["port"] == b["port"]))
            choices.append(pair)
        pairs.extend(choices)
        if choices:
            main_pairs.append(min(choices, key=lambda p: (-p["days_apart"], p["obs_id_a"], p["obs_id_b"])))
    assert len(main_pairs) == len({p["mmsi"] for p in main_pairs})
    write_csv("all_evaluable_pairs.csv", pairs)
    # Publish the prediction-independent join table before preparing illustrations.
    write_csv("main_pairs.csv", main_pairs)
    print(json.dumps(dict(main_pairs_ready=str(OUT / "main_pairs.csv"),
                          observations=len(test_ids), eligible_observations=len(eligible),
                          all_pairs=len(pairs), main_ships=len(main_pairs), reasons=reasons)), flush=True)
    score_lookup = {}
    for fold in range(3):
        for seed in range(3):
            with np.load(STRIP / f"scores_f{fold}_frozen_fivepoint_s{seed}.npz") as archive:
                for i, score in zip(archive["rows"], archive["scores"]):
                    row = all_cohort[int(i)]
                    oid, truth = int(row["obs_id"]), int(row["class_id"])
                    predicted = int(np.argmax(score))
                    assert predicted == int(baseline[seed, oid]["prediction"])
                    assert fold == int(baseline[seed, oid]["fold"])
                    rival = int(np.argmax(np.where(np.arange(8) == truth, -np.inf, score)))
                    score_lookup[seed, oid] = (predicted, float(score[truth] - score[rival]), rival)
    pair_predictions = []
    for pair in main_pairs:
        for seed in range(3):
            row = dict(pair, seed=seed)
            flags = []
            for side in ("a", "b"):
                prediction, margin, rival = score_lookup[seed, pair["obs_id_" + side]]
                correct = int(prediction == pair["class_id"])
                row.update({"prediction_" + side: prediction, "correct_" + side: correct,
                            "margin_" + side: margin, "rival_" + side: rival})
                flags.append("C" if correct else "W")
            row["outcome"] = "".join(flags)
            row["margin_delta_b_minus_a"] = row["margin_b"] - row["margin_a"]
            row["same_prediction"] = int(row["prediction_a"] == row["prediction_b"])
            pair_predictions.append(row)
    write_csv("main_pair_predictions.csv", pair_predictions)
    seed0 = [p for p in pair_predictions if p["seed"] == 0]
    write_csv("main_pairs_seed0.csv", seed0)
    view_rows, condition_rows = [], []
    for pair in main_pairs:
        condition = dict(pair_id=pair["pair_id"], mmsi=pair["mmsi"],
                         obs_id_a=pair["obs_id_a"], obs_id_b=pair["obs_id_b"])
        for pol in ("vv", "vh"):
            items = []
            for side in ("a", "b"):
                oid = pair["obs_id_" + side]
                fields = geometry_fields(views[oid]["images"][pol])
                view_rows.append(dict(pair_id=pair["pair_id"], side=side, obs_id=oid, pol=pol,
                                      image_path=cohort[oid]["image_" + pol], **fields))
                items.append(fields)
            a, b = items
            for key in ("length_px", "width_px", "length_grid", "other_boxes"):
                condition[f"{pol}_{key}_delta_b_minus_a"] = b[key] - a[key]
            condition[pol + "_axis_gap_deg"] = angular_gap(math.radians(a["axis_deg"]), math.radians(b["axis_deg"]))
            both = a["radar_valid"] and b["radar_valid"]
            condition[pol + "_relative_axis_gap_deg"] = angular_gap(
                math.radians(a["relative_axis_deg"]), math.radians(b["relative_axis_deg"])) if both else ""
            condition[pol + "_incidence_delta_b_minus_a"] = b["incidence"] - a["incidence"] if both and a["incidence"] is not None and b["incidence"] is not None else ""
        condition_rows.append(condition)
    write_csv("main_pair_views.csv", view_rows)
    write_csv("main_pair_conditions.csv", condition_rows)
    selected = []
    for outcome in ("CW", "WC"):
        options = [p for p in seed0 if p["outcome"] == outcome]
        if options:
            selected.append(min(options, key=lambda p: (-p["days_apart"], p["obs_id_a"], p["obs_id_b"])))
    if len(selected) < 2:
        rest = sorted(seed0, key=lambda p: (-p["days_apart"], p["obs_id_a"], p["obs_id_b"]))
        selected += [p for p in rest if p not in selected][:2 - len(selected)]
    cases = []
    for pair in selected:
        cases.append(dict(pair, raw_panel=panel(pair, cohort, views), sampling_panel=panel(pair, cohort, views, True)))
    write_csv("selected_cases.csv", cases)
    outcome_counts = {}
    for seed in range(3):
        ps = [p for p in pair_predictions if p["seed"] == seed]
        count = Counter(p["outcome"] for p in ps)
        count["WW_same_prediction"] = sum(p["outcome"] == "WW" and p["same_prediction"] for p in ps)
        outcome_counts[str(seed)] = dict(count)
    summary = dict(test_observations=len(test_ids), eligible_observations=len(eligible),
                   eligibility_reasons=dict(reasons), all_evaluable_pairs=len(pairs),
                   main_pairs=len(main_pairs), independent_unit="one MMSI; repeated dates and seeds are not independent",
                   main_pair_selection="maximum elapsed time; tie ascending obs_id_a then obs_id_b; no predictions used",
                   case_selection="seed0 CW and WC illustrations; greatest time gap within each outcome; not prevalence sampling",
                   pair_rule="different calendar dates/products; same MMSI/fold/class; both polarizations uniquely matched usable boxes; reciprocal unique center mapping",
                   identity_caveat="MMSI and reciprocal affine center-in-box checks are not manual physical identity verification; inherited CRS assumption retained",
                   class_counts=dict(Counter(p["class_name"] for p in main_pairs)),
                   fold_counts=dict(Counter(p["fold"] for p in main_pairs)),
                   cross_port_pairs=sum(not p["same_port"] for p in main_pairs),
                   outcomes_by_seed=outcome_counts,
                   margins="true-class score minus maximum other-class score, raw frozen_fivepoint logits; rival may differ between dates",
                   examples=[{k: p[k] for k in ("pair_id", "mmsi", "outcome", "days_apart", "margin_a", "margin_b", "raw_panel")} for p in cases])
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
