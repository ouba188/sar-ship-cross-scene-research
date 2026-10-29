"""CPU-only matched source-fit analysis. Full analysis requires six final NPZs.

D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -X utf8 -B analysis.py
Use --cohorts-only for manifest distributions while training is running.
Outputs: analysis/ and analysis/cases/. No training/model/runner imports.
NPZ logits are small and read one archive at a time; no feature cache is loaded.
Native rendering reuses the same input.prepare_view called by prepare_record,
without computing unused frequency tensors or importing torch.
"""
import argparse
from collections import Counter
import csv
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[2] / "sar_m123_current_20261027"
ARMS = ("legacy", "spatial")
SEEDS = (17, 29, 43)
METRICS = ("OA", "BA", "NLL", "Brier")
STATES = ("rescued", "harmed", "CC", "WW")
CLASSES = ("bulk_carrier", "fishing", "general_cargo", "chemical_tanker",
           "container", "crude_oil_tanker", "tug", "offshore")
SCOPE = (
    "Matched source-fit first stage only: legacy vs spatial M1/M2 with explicit "
    "M1/M2 frequency branches removed and M3 retained. Protocol: same current "
    "init.pt, 1205 source-fit observations, fixed 16 references, 8 epochs, order "
    "seeds 17/29/43; fixed 160 historical development queries. This is NOT the "
    "full intervention-adapted final system, a fresh test, or test-based selection. "
    "Repeated queries/references across seeds are not independent observations. "
    "OA/BA are percentages; deltas are percentage points. BA averages present "
    "classes only (7 in the complete development cohort). NLL uses natural logs; "
    "Brier is the sum over all 8 classes. SD is sample SD across three order "
    "seeds, not a sampling confidence interval. Controlled success does not "
    "identify a physical cause."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moments(values):
    values = [v for v in values if v is not None]
    return dict(mean=float(np.mean(values)) if values else None,
                sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                n=len(values))


def per_sample(score, y):
    z = np.asarray(score, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    prob = np.exp(logp)
    return dict(prediction=score.argmax(axis=1), NLL=-logp[np.arange(len(y)), y],
                Brier=((prob - np.eye(8)[y]) ** 2).sum(axis=1))


def metrics(score, y):
    if not len(y):
        return dict(n=0, present_classes=0, **{m: None for m in METRICS})
    values = per_sample(score, y)
    correct = values["prediction"] == y
    return dict(n=len(y), present_classes=len(np.unique(y)),
                OA=float(correct.mean() * 100),
                BA=float(np.mean([correct[y == c].mean() for c in np.unique(y)]) * 100),
                NLL=float(values["NLL"].mean()), Brier=float(values["Brier"].mean()))


def state(old, new, truth):
    return "CC" if old == truth and new == truth else "rescued" if new == truth else "harmed" if old == truth else "WW"


def eligible(query, ref):
    return str(query["id"]) != str(ref["id"]) and not (
        query.get("mmsi") and ref.get("mmsi") and str(query["mmsi"]) == str(ref["mmsi"]))


def load_score(path, rows):
    # mmap_mode cannot memory-map zipped members; only the tiny Nx8 logits load.
    with np.load(path, allow_pickle=False, mmap_mode="r") as pack:
        ids = pack["ids"].astype(str)
        y = pack["y"] if "y" in pack else pack["labels"]
        score = np.asarray(pack["score"], dtype=np.float64)
    expected = [str(r["id"]) for r in rows]
    if ids.shape != (len(rows),) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f"{path.name}: IDs do not exactly cover development")
    if y.shape != ids.shape or y.dtype.kind not in "iu" or score.shape != (len(rows), 8):
        raise ValueError(f"{path.name}: expected integer labels[N] and logits[N,8]")
    if not np.isfinite(score).all():
        raise ValueError(f"{path.name}: nonfinite logits")
    index = {key: i for i, key in enumerate(ids)}
    order = [index[key] for key in expected]
    if not np.array_equal(y[order], [r["label"] for r in rows]):
        raise ValueError(f"{path.name}: labels disagree with cohorts.json")
    return score[order]


def load_cohorts(root):
    cohorts = read_json(root / "cohorts.json")
    if "fit" not in cohorts and "source_fit" in cohorts:
        cohorts["fit"] = cohorts["source_fit"]
    cohorts = {key: cohorts[key] for key in ("fit", "development", "references")}
    for split, n in (("fit", 1205), ("development", 160), ("references", 16)):
        rows = cohorts[split]
        if len(rows) != n or len({str(r["id"]) for r in rows}) != n:
            raise ValueError(f"Unexpected {split} count or duplicate observation IDs")
        for row in rows:
            row["id"] = str(row["id"])
    return cohorts


def load_inputs(root):
    files = {(arm, seed): root / "predictions" / f"{arm}_s{seed}.npz"
             for arm in ARMS for seed in SEEDS}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise SystemExit("Waiting for all six final prediction files; no analysis run:\n" + "\n".join(missing))
    cohorts = load_cohorts(root)
    rows = cohorts["development"]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    if len(np.unique(y)) != 7 or not np.isin(y, np.arange(8)).all():
        raise ValueError("Expected 7 present development classes in 8-class taxonomy")
    scores = {key: load_score(path, rows) for key, path in files.items()}
    initial = {arm: load_score(root / "predictions" / f"initial_{arm}.npz", rows) for arm in ARMS}
    receipts = {}
    for arm, seed in files:
        path = root / "training" / f"{arm}_s{seed}" / "training.json"
        receipt = read_json(path)
        receipts[f"{arm}_s{seed}"] = dict(path=str(path), scalar_fields={
            k: v for k, v in receipt.items() if isinstance(v, (str, int, float, bool)) or v is None})
    return cohorts, y, scores, initial, receipts


def coverage(cohorts):
    result = {}
    for split, rows in cohorts.items():
        result[split] = dict(n=len(rows), unique_ids=len({r["id"] for r in rows}),
            ships=len({str(r["mmsi"]) for r in rows if r.get("mmsi")}),
            missing_ship_ids=sum(not r.get("mmsi") for r in rows),
            classes={CLASSES[c]: sum(r["label"] == c for r in rows) for c in range(8)},
            ports=dict(Counter(str(r["port"]) for r in rows)),
            passes=dict(Counter(str(r["pass_group"]) for r in rows)),
            supplied_views=dict(Counter("both" if r.get("vv") and r.get("vh") else
                "vv_only" if r.get("vv") else "vh_only" if r.get("vh") else "neither" for r in rows)))
    result["overlap_counts"] = {}
    for left, right in (("fit", "development"), ("fit", "references"), ("development", "references")):
        result["overlap_counts"][f"{left}__{right}"] = {field: len(
            {str(r[field]) for r in cohorts[left] if r.get(field) is not None} &
            {str(r[field]) for r in cohorts[right] if r.get(field) is not None})
            for field in ("id", "mmsi", "port", "pass_group")}
    return result


def polarization(row):
    return "both" if row.get("vv") and row.get("vh") else "vv_only" if row.get("vv") else "vh_only" if row.get("vh") else "neither"


def cohort_report(root, cohorts=None):
    cohorts = load_cohorts(root) if cohorts is None else cohorts
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    summary = coverage(cohorts)
    group_rows, overlap_rows, pair_rows = [], [], []
    for split, rows in cohorts.items():
        for field in ("label", "mmsi", "pass_group", "port", "polarization"):
            groups = {}
            for row in rows:
                value = polarization(row) if field == "polarization" else str(row.get(field) or "unknown") if field == "mmsi" else str(row[field])
                groups.setdefault(value, []).append(row)
            if field == "label":
                for c in range(8):
                    groups.setdefault(str(c), [])
            for value, members in sorted(groups.items()):
                counts = Counter(polarization(r) for r in members)
                group_rows.append(dict(split=split, dimension=field, group=value,
                    name=CLASSES[int(value)] if field == "label" else value,
                    n=len(members), percent=100 * len(members) / len(rows),
                    unique_ids=len({r["id"] for r in members}),
                    ships=len({str(r["mmsi"]) for r in members if r.get("mmsi")}),
                    passes=len({r["pass_group"] for r in members}), ports=len({r["port"] for r in members}),
                    **{key: counts[key] for key in ("both", "vv_only", "vh_only", "neither")}))
    for left, right in (("fit", "development"), ("fit", "references"), ("development", "references")):
        for field in ("id", "mmsi", "pass_group", "port"):
            shared = ({str(r[field]) for r in cohorts[left] if r.get(field) not in (None, "")} &
                      {str(r[field]) for r in cohorts[right] if r.get(field) not in (None, "")})
            overlap_rows.append(dict(left=left, right=right, dimension=field, shared_values=len(shared),
                left_observations=sum(str(r.get(field)) in shared for r in cohorts[left]),
                right_observations=sum(str(r.get(field)) in shared for r in cohorts[right])))
    refs = cohorts["references"]
    reference_ids = {r["id"] for r in refs}
    fit_ids = {r["id"] for r in cohorts["fit"]}
    summary["reference_membership"] = dict(reference_ids_in_fit=sorted(reference_ids & fit_ids),
        all_reference_ids_in_fit=reference_ids.issubset(fit_ids))
    summary["pair_exclusion"] = {}
    for split in ("fit", "development"):
        split_pairs = []
        for query in cohorts[split]:
            allowed = [r for r in refs if eligible(query, r)]
            item = dict(split=split, id=query["id"], label=query["label"], mmsi=query.get("mmsi"),
                candidate_references=len(refs), eligible_references=len(allowed),
                excluded_union=len(refs) - len(allowed),
                excluded_same_id=sum(query["id"] == r["id"] for r in refs),
                excluded_same_mmsi=sum(bool(query.get("mmsi") and r.get("mmsi") and
                    str(query["mmsi"]) == str(r["mmsi"])) for r in refs),
                eligible_true_references=sum(query["label"] == r["label"] for r in allowed))
            split_pairs.append(item)
        pair_rows.extend(split_pairs)
        summary["pair_exclusion"][split] = dict(queries=len(split_pairs),
            candidate_pairs=sum(r["candidate_references"] for r in split_pairs),
            eligible_pairs=sum(r["eligible_references"] for r in split_pairs),
            excluded_union=sum(r["excluded_union"] for r in split_pairs),
            excluded_same_id=sum(r["excluded_same_id"] for r in split_pairs),
            excluded_same_mmsi=sum(r["excluded_same_mmsi"] for r in split_pairs),
            queries_with_exclusions=sum(r["excluded_union"] > 0 for r in split_pairs),
            eligible_references_range=[min(r["eligible_references"] for r in split_pairs), max(r["eligible_references"] for r in split_pairs)],
            eligible_true_references_range=[min(r["eligible_true_references"] for r in split_pairs), max(r["eligible_true_references"] for r in split_pairs)])
    summary["source"] = str(root / "cohorts.json")
    summary["interpretation"] = (
        "Actual manifest counts only. Split names and absence of recorded overlap do not establish independent holdout. "
        "Fit and development share 3 ports (Jebel Ali, Port Klang, Singapore), so this is not strict leave-one-port-out (LOPO). "
        "The source1365 cache/index pool covers 1205 fit plus 160 development observations; its name does not define "
        "the training split. Training uses only indices['fit']; development indices are used for evaluation. "
        "Development is historically reused. References are source-fit records; same-ID OR same-known-MMSI "
        "query/reference pairs are excluded, not the reference observations from source fitting. "
        "Same-ID and same-MMSI exclusions overlap and must not be added. Repeated references, queries, ships, "
        "passes and seeds are not independent observations. Polarization means supplied manifest views, "
        "not pixel-verified target availability. No images, cache arrays or predictions were read by this summary."
    )
    dump(out / "cohort_summary.json", summary)
    write_csv(out / "cohort_groups.csv", group_rows)
    write_csv(out / "cohort_overlap.csv", overlap_rows)
    write_csv(out / "cohort_reference_pairs.csv", pair_rows)
    lines = ["# Actual cohort distribution", "", summary["interpretation"], "",
        "| Cohort | Observations | Ships | Passes | Ports | Both views | VV only | VH only | Neither |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in cohorts:
        c = summary[split]
        lines.append(f"| {split} | {c['n']} | {c['ships']} | {len(c['passes'])} | {len(c['ports'])} | " +
            " | ".join(str(c["supplied_views"].get(k, 0)) for k in ("both", "vv_only", "vh_only", "neither")) + " |")
    lines += ["", "Each class cell is observations / ships / passes / ports.", "",
        "| Class | Fit | Development | References |", "|---|---:|---:|---:|"]
    for c, name in enumerate(CLASSES):
        cells = []
        for split in cohorts:
            item = next(r for r in group_rows if r["split"] == split and r["dimension"] == "label" and r["group"] == str(c))
            cells.append(" / ".join(str(item[k]) for k in ("n", "ships", "passes", "ports")))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| Cohort pair | Dimension | Shared values | Left observations in overlap | Right observations in overlap |",
        "|---|---|---:|---:|---:|"]
    for item in overlap_rows:
        lines.append(f"| {item['left']} / {item['right']} | {item['dimension']} | {item['shared_values']} | {item['left_observations']} | {item['right_observations']} |")
    reference_class_counts = [sum(r["label"] == c for r in refs) for c in range(8)]
    lines += ["", f"Reference IDs included in source fit: {len(reference_ids & fit_ids)} / {len(refs)}. "
        f"Reference counts in the class order above: {reference_class_counts}. "
        "Eligible counts can fall after ID/MMSI exclusion.", "",
        "| Query cohort | Candidate pairs | Eligible pairs | Excluded union | Same ID | Same MMSI | Affected queries | Eligible refs/query | True refs/query |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for split, item in summary["pair_exclusion"].items():
        lines.append(f"| {split} | " + " | ".join(str(item[k]) for k in (
            "candidate_pairs", "eligible_pairs", "excluded_union", "excluded_same_id", "excluded_same_mmsi", "queries_with_exclusions", "eligible_references_range", "eligible_true_references_range")) + " |")
    lines += ["", "| Port | Fit | Development | References |", "|---|---:|---:|---:|"]
    for port in sorted({r["port"] for rows in cohorts.values() for r in rows}):
        lines.append(f"| {port} | " + " | ".join(str(summary[s]["ports"].get(port, 0)) for s in cohorts) + " |")
    lines += ["", "cohort_groups.csv contains every class, ship/MMSI, pass, port and supplied-polarization group with "
        "observation, unique-ID, ship, pass, port and polarization counts. cohort_overlap.csv includes directional "
        "observation participation; cohort_reference_pairs.csv records exclusions per query. "
        "The counts describe cohort composition and repeated exposure, not statistical independence or model benefit.", ""]
    (out / "COHORT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def event_rows(rows, y, scores, initial):
    rival_score = initial["legacy"].copy()
    rival_score[np.arange(len(y)), y] = -np.inf
    rivals = rival_score.argmax(axis=1)
    events = []
    for seed in SEEDS:
        old, new = (scores[arm, seed] for arm in ARMS)
        ov, nv = per_sample(old, y), per_sample(new, y)
        for i, row in enumerate(rows):
            truth, rival = int(y[i]), int(rivals[i])
            event = dict(id=row["id"], seed=seed, label=truth, class_name=CLASSES[truth],
                port=row["port"], pass_group=row["pass_group"], mmsi=row.get("mmsi"),
                state=state(ov["prediction"][i], nv["prediction"][i], truth),
                legacy_prediction=int(ov["prediction"][i]), spatial_prediction=int(nv["prediction"][i]),
                fixed_rival=rival, fixed_rival_rule="strongest nontruth initial_legacy logit")
            for arm, score, values in (("legacy", old, ov), ("spatial", new, nv)):
                event[f"{arm}_fixed_margin"] = float(score[i, truth] - score[i, rival])
                event[f"{arm}_top_rival_margin"] = float(score[i, truth] - np.delete(score[i], truth).max())
                for metric in ("NLL", "Brier"):
                    event[f"{arm}_{metric}"] = float(values[metric][i])
                for c in range(8):
                    event[f"{arm}_logit_{c}"] = float(score[i, c])
            event["delta_fixed_margin"] = event["spatial_fixed_margin"] - event["legacy_fixed_margin"]
            events.append(event)
    return events


def transition_summary(events):
    result = {key: dict(seed_events=sum(e["state"] == key for e in events),
                       unique_ids=len({e["id"] for e in events if e["state"] == key}),
                       ids=sorted({e["id"] for e in events if e["state"] == key}),
                       unique_ships=len({str(e["mmsi"]) for e in events if e["state"] == key and e.get("mmsi")}))
              for key in STATES}
    result["rescued_and_harmed_ids"] = sorted(set(result["rescued"]["ids"]) & set(result["harmed"]["ids"]))
    result["unique_ids"] = len({e["id"] for e in events})
    result["seed_events"] = len(events)
    return result


def grouped_tables(rows, y, scores, events):
    per_seed, aggregated = [], []
    for field in ("label", "port", "pass_group"):
        values = np.array([str(r[field]) for r in rows])
        groups = list(map(str, range(8))) if field == "label" else sorted(set(values))
        for group in groups:
            mask = values == group
            selected_ids = {r["id"] for r, keep in zip(rows, mask) if keep}
            these_events = [e for e in events if e["id"] in selected_ids]
            identity = dict(dimension=field, group=group,
                name=CLASSES[int(group)] if field == "label" else group, n=int(mask.sum()),
                ships=len({str(r["mmsi"]) for r, keep in zip(rows, mask) if keep and r.get("mmsi")}))
            group_rows = []
            for seed in SEEDS:
                item = dict(**identity, seed=seed)
                arm_metrics = {arm: metrics(scores[arm, seed][mask], y[mask]) for arm in ARMS}
                item["present_classes"] = arm_metrics["legacy"]["present_classes"]
                for metric in METRICS:
                    for arm in ARMS:
                        item[f"{arm}_{metric}"] = arm_metrics[arm][metric]
                    item[f"delta_{metric}"] = (arm_metrics["spatial"][metric] - arm_metrics["legacy"][metric]) if mask.any() else None
                item.update({key: sum(e["seed"] == seed and e["state"] == key for e in these_events) for key in STATES})
                group_rows.append(item)
            per_seed.extend(group_rows)
            total = dict(**identity)
            for name in [f"{arm}_{metric}" for arm in (*ARMS, "delta") for metric in METRICS]:
                stats = moments([r[name] for r in group_rows])
                total[name + "_mean"], total[name + "_sd"] = stats["mean"], stats["sd"]
            transitions = transition_summary(these_events)
            for key in STATES:
                total[key + "_seed_events"] = transitions[key]["seed_events"]
                total[key + "_unique_ids"] = transitions[key]["unique_ids"]
            total["rescued_and_harmed_ids"] = ";".join(transitions["rescued_and_harmed_ids"])
            aggregated.append(total)
    return per_seed, aggregated


def select_cases(events, count=8):
    # Preserve requested transition examples, then cover every available state.
    candidates = {}
    for key in STATES:
        pool = [e for e in events if e["state"] == key]
        frequency = Counter(e["id"] for e in pool)
        candidates[key] = sorted(pool, key=lambda e: (
            -frequency[e["id"]], -abs(e["delta_fixed_margin"]), e["id"], e["seed"]))
    selected, used = [], set()
    for query_id, key in (("37321", "rescued"), ("6194", "harmed"), ("31779", "harmed")):
        event = next((e for e in candidates[key] if e["id"] == query_id), None)
        if event is not None and len(selected) < count:
            selected.append(dict(event, selection_priority="requested transition case confirmed in final predictions"))
            used.add(query_id)
    for quota in (1, 2):
        for key in STATES:
            if len(selected) >= count or sum(e["state"] == key for e in selected) >= quota:
                continue
            event = next((e for e in candidates[key] if e["id"] not in used), None)
            if event is not None:
                selected.append(event.copy())
                used.add(event["id"])
    for key in ("rescued", "harmed"):
        for event in candidates[key]:
            if len(selected) < count and event["id"] not in used:
                selected.append(dict(event, selection_priority="remaining transition query before margin-only fill"))
                used.add(event["id"])
    for event in sorted(events, key=lambda e: (-abs(e["delta_fixed_margin"]), e["id"], e["seed"])):
        if len(selected) >= count:
            break
        if event["id"] not in used:
            selected.append(event.copy())
            used.add(event["id"])
    return selected[:count]


def native_tile(row, pol, native, role):
    view = native.prepare_view(row.get(pol))
    raw_w, raw_h = (view["raw"].shape[1], view["raw"].shape[0]) if view is not None else (0, 0)
    width, height = max(raw_w, 160) + 352, max(raw_h, 160) + 83
    tile = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(tile), ImageFont.load_default(size=12)
    draw.text((8, 6), f"{role} ID {row['id']} / {pol.upper()} / {CLASSES[row['label']]}", fill="black", font=font)
    receipt = dict(id=row["id"], pol=pol, role=role, source_path=row[pol]["image"] if row.get(pol) else None,
        available=bool(view is not None and view["valid_target"]), raw_size=[raw_w, raw_h],
        pixel_scale=1, contrast_stretch=False, visual_status="not_reviewed")
    if view is None:
        draw.text((8, 60), "UNAVAILABLE / invalid alignment (not an observed black chip)", fill="#a00000", font=font)
        return tile, receipt
    xcrop, xobb, top = max(raw_w, 160) + 16, max(raw_w, 160) + 184, 63
    draw.text((8, 42), f"Original {raw_w}x{raw_h}", fill="black", font=font)
    draw.text((xcrop, 42), "Model crop/pad 160", fill="black", font=font)
    draw.text((xobb, 42), "Same crop + target OBB", fill="black", font=font)
    tile.paste(Image.fromarray(view["raw"]).convert("RGB"), (8, top))
    crop = Image.fromarray(view["pixels"]).convert("RGB")
    tile.paste(crop, (xcrop, top))
    overlay = crop.copy()
    placement = view["placement"]
    spec = row[pol]
    target = spec.get("target_index")
    if view["valid_target"] and target is not None and int(target) >= 0:
        corners = np.asarray(spec["detections"][int(target)]["corners"], dtype=float)
        corners += [placement["pad_left"] - placement["crop_x"], placement["pad_top"] - placement["crop_y"]]
        points = [tuple(point) for point in corners]
        ImageDraw.Draw(overlay).line(points + points[:1], fill="#00ff66", width=1)
        receipt["target_corners_model_xy"] = corners.tolist()
    tile.paste(overlay, (xobb, top))
    if not view["valid_target"]:
        draw.text((8, 25), "NO VALID TARGET: this polarization is unavailable to the model", fill="#a00000", font=font)
    receipt.update(placement=placement, valid_target=bool(view["valid_target"]))
    return tile, receipt


def panel(rows, roles, pol, path, native):
    tiles, receipts = [], []
    for row, role in zip(rows, roles):
        tile, receipt = native_tile(row, pol, native, role)
        tiles.append(tile)
        receipts.append(receipt)
    width, height = max(t.width for t in tiles), max(t.height for t in tiles)
    columns = min(2, len(tiles))
    sheet = Image.new("RGB", (width * columns, height * ((len(tiles) + columns - 1) // columns)), "#dddddd")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * height))
        tile.close()
    sheet.save(path)
    sheet.close()
    return receipts


def gallery(out, cohorts, events, initial, package):
    spec = importlib.util.spec_from_file_location("sourcefit_native_input", package / "sar_m123/input.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    rows, refs = cohorts["development"], cohorts["references"]
    query_index = {r["id"]: (i, r) for i, r in enumerate(rows)}
    cases, sources, blocks = select_cases(events), [], []
    selection = ("Requested IDs 37321 (rescue), 6194 and 31779 (harm) take priority when those "
        "transitions occur in the final predictions. Then cover each available state and fill "
        "toward two distinct queries per state, including CC and WW. "
        "Rank by number of seeds in that state, then absolute initial-legacy-fixed-rival "
        "margin change, lexical ID and seed; deduplicate queries globally. Include remaining "
        "rescue/harm queries before margin-only fill to 8 "
        "by absolute margin change if a state is absent. Outcome-selected, not prevalence.")
    for number, case in enumerate(cases, 1):
        i, query = query_index[case["id"]]
        truth = case["label"]
        rival = case["legacy_prediction"] if case["legacy_prediction"] != truth else case["spatial_prediction"]
        if rival == truth:
            rival = case["fixed_rival"]
        allowed = [r for r in refs if eligible(query, r)]
        own = [r for r in allowed if r["label"] == truth]
        competing = [r for r in allowed if r["label"] == rival]
        extra = [r for r in allowed if r["label"] == case["fixed_rival"] and r not in competing]
        current_rivals = set()
        for arm in ARMS:
            logits = np.array([case[f"{arm}_logit_{c}"] for c in range(8)])
            logits[truth] = -np.inf
            current_rivals.add(int(logits.argmax()))
        current = [r for r in allowed if r["label"] in current_rivals and r not in own + competing + extra]
        group = [query, *own, *competing, *extra, *current]
        roles = ["QUERY", *["TRUE REF"] * len(own), *["CASE RIVAL REF"] * len(competing),
                 *["INITIAL RIVAL REF"] * len(extra), *["CURRENT RIVAL REF"] * len(current)]
        case.update(case_rival=rival, case_rival_name=CLASSES[rival],
            case_rival_rule="wrong legacy prediction, else wrong spatial prediction, else initial legacy nontruth argmax; chosen at selected seed, then fixed across seeds",
            eligible_reference_ids=[r["id"] for r in allowed],
            true_reference_ids=[r["id"] for r in own], rival_reference_ids=[r["id"] for r in competing],
            initial_rival_reference_ids=[r["id"] for r in extra],
            current_rival_reference_ids=[r["id"] for r in current],
            initial_logits={arm: initial[arm][i].tolist() for arm in ARMS},
            selection=selection, visual_status="not_reviewed", panels={}, per_seed=[])
        for event in [e for e in events if e["id"] == case["id"]]:
            detail = dict(seed=event["seed"], state=event["state"])
            for arm in ARMS:
                logits = [event[f"{arm}_logit_{c}"] for c in range(8)]
                detail.update({arm + "_logits": logits, arm + "_prediction": event[arm + "_prediction"],
                    arm + "_case_margin": logits[truth] - logits[rival]})
            case["per_seed"].append(detail)
        title = f"{number}. {case['state']} / ID {case['id']} / selected seed {case['seed']}"
        blocks.append(f"<section><h2>{html.escape(title)}</h2><p>Truth: {CLASSES[truth]}; case rival: {CLASSES[rival]}. "
            f"Port: {html.escape(str(case['port']))}; pass: {html.escape(str(case['pass_group']))}. "
            "Pixels not reviewed. CC/WW are per-seed states; all seed outcomes follow.</p>")
        blocks.append("<table><tr><th>Seed</th><th>State</th><th>Legacy prediction</th><th>Spatial prediction</th><th>Legacy truth-rival</th><th>Spatial truth-rival</th></tr>")
        for detail in case["per_seed"]:
            blocks.append(f"<tr><td>{detail['seed']}</td><td>{detail['state']}</td><td>{CLASSES[detail['legacy_prediction']]}</td>"
                f"<td>{CLASSES[detail['spatial_prediction']]}</td><td>{detail['legacy_case_margin']:.6f}</td><td>{detail['spatial_case_margin']:.6f}</td></tr>")
        blocks.append("</table><details><summary>Exact logits and reference IDs</summary><pre>" +
            html.escape(json.dumps({k: case[k] for k in ("eligible_reference_ids", "true_reference_ids", "rival_reference_ids", "initial_rival_reference_ids", "current_rival_reference_ids", "initial_logits", "per_seed")}, indent=2)) + "</pre></details>")
        for pol in ("vv", "vh"):
            filename = f"case_{number:02d}_{pol}.png"
            receipts = panel(group, roles, pol, folder / filename, native)
            for receipt in receipts:
                receipt.update(case_id=case["id"], selected_seed=case["seed"], panel=filename,
                    usable_with_query_polarization=bool(receipts[0]["available"] and receipt["available"]))
            sources.extend(receipts)
            case["panels"][pol] = str(folder / filename)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(title)} {pol.upper()}"></div>')
        blocks.append("</section>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Matched source-fit cases</title>"
        "<style>body{font:14px Arial;margin:24px;color:#222}section{border-top:1px solid #aaa;margin-top:24px}"
        "h1{font-size:22px}h2{font-size:18px}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}"
        ".image{overflow:auto}img{max-width:none;image-rendering:pixelated}pre{white-space:pre-wrap}</style>"
        "<h1>Legacy vs spatial M1/M2: native cases</h1><p>" + html.escape(SCOPE) + "</p><p>" + html.escape(selection) +
        "</p><p>One image pixel per source pixel, no resizing or contrast stretch. Original 8-bit display chips are not "
        "raw radiometry. Clean original, exact model crop/pad, and duplicate crop with target OBB appear side by side. "
        "VV/VH are separate. References come from the fixed 16 after the native ID/MMSI pair exclusion; missing "
        "polarizations are marked. Automated rendering is not pixel review.</p>" + "".join(blocks) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    dump(folder / "cases.json", cases)
    dump(folder / "native_sources.json", sources)
    return cases


def posthoc_pruning(root, rows, y, scores, initial):
    paths = {seed: root / "predictions" / f"pruned_s{seed}.npz" for seed in SEEDS}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        return dict(status="not_run", reason="All three posthoc pruning predictions required", missing=missing)
    pruned = {seed: load_score(path, rows) for seed, path in paths.items()}
    out = root / "analysis/posthoc_pruning"
    out.mkdir(parents=True, exist_ok=True)
    scope = (
        "Optional posthoc path-dependency diagnostic, NOT a third fairly trained arm. "
        "Pruned predictions use each already-trained legacy model with explicit M1/M2 frequency "
        "branches removed by conversion, retained weights and M3 unchanged, and no refit. "
        "Pruned minus legacy measures removal after legacy-path training. Spatial minus pruned "
        "compares spatial-path refitting with removal after legacy-path training; it does not "
        "isolate a pure frequency effect or a new matched three-arm training experiment. "
        "The main legacy-versus-spatial summaries remain separate and unchanged. " + SCOPE
    )
    rival_scores = initial["legacy"].copy()
    rival_scores[np.arange(len(y)), y] = -np.inf
    rivals = rival_scores.argmax(axis=1)
    metric_rows, events, comparisons = [], [], {}
    for name, reference, alternative in (("pruned_vs_legacy", "legacy", "pruned"),
                                          ("spatial_vs_pruned", "pruned", "spatial")):
        pair_metrics, pair_events = [], []
        for seed in SEEDS:
            old = pruned[seed] if reference == "pruned" else scores[reference, seed]
            new = pruned[seed] if alternative == "pruned" else scores[alternative, seed]
            om, nm = metrics(old, y), metrics(new, y)
            ov, nv = per_sample(old, y), per_sample(new, y)
            identity = dict(comparison=name, reference=reference, alternative=alternative, seed=seed)
            metric = dict(**identity, n=len(y), present_classes=len(np.unique(y)),
                          prediction_changes=int((ov["prediction"] != nv["prediction"]).sum()))
            for key in METRICS:
                metric.update({"reference_" + key: om[key], "alternative_" + key: nm[key], "delta_" + key: nm[key] - om[key]})
            for i, row in enumerate(rows):
                truth, rival = int(y[i]), int(rivals[i])
                event = dict(**identity, id=row["id"], label=truth, class_name=CLASSES[truth],
                    port=row["port"], pass_group=row["pass_group"], mmsi=row.get("mmsi"),
                    state=state(ov["prediction"][i], nv["prediction"][i], truth),
                    reference_prediction=int(ov["prediction"][i]), alternative_prediction=int(nv["prediction"][i]),
                    fixed_rival=rival, fixed_rival_rule="strongest nontruth initial_legacy logit")
                for prefix, score, values in (("reference", old, ov), ("alternative", new, nv)):
                    event[prefix + "_fixed_margin"] = float(score[i, truth] - score[i, rival])
                    event[prefix + "_logits"] = json.dumps(score[i].tolist())
                    for key in ("NLL", "Brier"):
                        event[prefix + "_" + key] = float(values[key][i])
                event["delta_fixed_margin"] = event["alternative_fixed_margin"] - event["reference_fixed_margin"]
                pair_events.append(event)
            metric.update({key: sum(e["seed"] == seed and e["state"] == key for e in pair_events) for key in STATES})
            pair_metrics.append(metric)
        comparisons[name] = dict(reference=reference, alternative=alternative,
            prediction_change_seed_events=sum(r["prediction_changes"] for r in pair_metrics),
            prediction_change_unique_ids=len({e["id"] for e in pair_events if e["reference_prediction"] != e["alternative_prediction"]}),
            metrics={key: moments([r[key] for r in pair_metrics])
                     for key in [f"{prefix}_{m}" for prefix in ("reference", "alternative", "delta") for m in METRICS]},
            transitions=transition_summary(pair_events))
        metric_rows.extend(pair_metrics)
        events.extend(pair_events)
    summary = dict(scope=scope, comparisons=comparisons, source_files={str(s): str(p) for s, p in paths.items()},
        pairing="Same query ID, truth, order seed, fixed reference bank and initial-legacy rival; repeated seed events are not independent.")
    write_csv(out / "metrics.csv", metric_rows)
    write_csv(out / "seed_events.csv", events)
    dump(out / "summary.json", summary)
    lines = ["# Posthoc pruning: training-path diagnostic", "", scope, "",
        "Positive OA/BA deltas favor the alternative; negative NLL/Brier deltas favor the alternative. "
        "Means and sample SD describe three paired order seeds, not independent datasets.", "",
        "| Comparison | Metric | Reference mean +/- SD | Alternative mean +/- SD | Paired delta mean +/- SD |",
        "|---|---|---:|---:|---:|"]
    for name, comparison in comparisons.items():
        for key in METRICS:
            stats = [comparison["metrics"][prefix + "_" + key] for prefix in ("reference", "alternative", "delta")]
            lines.append(f"| {name} | {key} | " + " | ".join(f"{s['mean']:.6f} +/- {s['sd']:.6f}" for s in stats) + " |")
    lines += ["", "| Comparison | State | Seed events | Unique query IDs |", "|---|---|---:|---:|"]
    for name, comparison in comparisons.items():
        for key in STATES:
            t = comparison["transitions"][key]
            lines.append(f"| {name} | {key} | {t['seed_events']} | {t['unique_ids']} |")
    for name, comparison in comparisons.items():
        lines += ["", f"{name}: IDs rescued in one seed and harmed in another: " + str(comparison["transitions"]["rescued_and_harmed_ids"]), ""]
        lines += [f"All top-1 changes (including WW changes): {comparison['prediction_change_seed_events']} seed events / "
            f"{comparison['prediction_change_unique_ids']} unique IDs.", ""]
    lines += ["No new checkpoint, hyperparameter or test selection is performed. These score comparisons "
        "do not identify a physical cause. Per-query logits, margins and exact seed/ID transitions are in seed_events.csv.", ""]
    (out / "NUMERIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return dict(status="complete", output=str(out), comparisons=comparisons)


def analyze(root, package):
    cohorts, y, scores, initial, receipts = load_inputs(root)
    cohort_report(root, cohorts)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    rows = cohorts["development"]
    events = event_rows(rows, y, scores, initial)
    metric_rows, paired_rows, initial_rows = [], [], []
    for arm in ARMS:
        initial_rows.append(dict(arm=arm, **metrics(initial[arm], y)))
        for seed in SEEDS:
            metric_rows.append(dict(arm=arm, seed=seed, **metrics(scores[arm, seed], y)))
    for seed in SEEDS:
        old, new = (metrics(scores[arm, seed], y) for arm in ARMS)
        paired_rows.append(dict(seed=seed, **{f"delta_{m}": new[m] - old[m] for m in METRICS},
            **{key: sum(e["seed"] == seed and e["state"] == key for e in events) for key in STATES}))
    grouped, grouped_summary = grouped_tables(rows, y, scores, events)
    reference_rows = [dict(id=q["id"], label=q["label"],
        eligible_n=sum(eligible(q, r) for r in cohorts["references"]),
        eligible_ids=";".join(r["id"] for r in cohorts["references"] if eligible(q, r)),
        true_ids=";".join(r["id"] for r in cohorts["references"] if eligible(q, r) and r["label"] == q["label"])) for q in rows]
    summary = dict(scope=SCOPE, arms={arm: {m: moments([r[m] for r in metric_rows if r["arm"] == arm]) for m in METRICS} for arm in ARMS},
        paired={m: moments([r[f"delta_{m}"] for r in paired_rows]) for m in METRICS},
        initial={r["arm"]: {m: r[m] for m in METRICS} for r in initial_rows},
        transitions=transition_summary(events), coverage=coverage(cohorts), training_receipts=receipts,
        alignment="All eight NPZs joined by unique observation ID to the same development labels; no prediction-order assumption.",
        pair_scope="Fixed references and native ID/MMSI eligibility reused for every arm/seed by protocol; logits alone cannot verify internal reference use.",
        visual_status="not_reviewed", present_class_ids=np.unique(y).tolist())
    for name, table in (("metrics.csv", metric_rows), ("paired_metrics.csv", paired_rows),
                        ("initial_metrics.csv", initial_rows), ("seed_events.csv", events),
                        ("groups.csv", grouped), ("groups_summary.csv", grouped_summary),
                        ("reference_pairs.csv", reference_rows)):
        write_csv(out / name, table)
    dump(out / "summary.json", summary)
    cases = gallery(out, cohorts, events, initial, package)
    lines = ["# Source-fit comparison: CPU numeric report", "", SCOPE, "",
        "| Arm | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |",
        "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        lines.append(f"| {arm} | " + " | ".join(f"{summary['arms'][arm][m]['mean']:.6f} +/- {summary['arms'][arm][m]['sd']:.6f}" for m in METRICS) + " |")
    lines += ["", "| Paired spatial minus legacy | Mean | SD |", "|---|---:|---:|"]
    for m in METRICS:
        lines.append(f"| {m} | {summary['paired'][m]['mean']:.6f} | {summary['paired'][m]['sd']:.6f} |")
    lines += ["", "| State | Seed events | Unique query IDs |", "|---|---:|---:|"]
    for key in STATES:
        value = summary["transitions"][key]
        lines.append(f"| {key} | {value['seed_events']} | {value['unique_ids']} |")
    lines += ["", "IDs rescued in one seed and harmed in another: " + str(summary["transitions"]["rescued_and_harmed_ids"]),
        "", "Class, port and pass metrics and transitions are in groups.csv and groups_summary.csv. "
        "Empty classes have blank metrics, not zero recall; subgroup BA uses that group's present classes. "
        "Seed events can repeat a query, ship, pass and reference pair. No independence or significance claim is made.",
        "", "Initial-arm metrics are in initial_metrics.csv; seed_events.csv records all logits and both initial-fixed-rival "
        "and strongest-current-rival margins. Gallery case rivals are separately specified and then held fixed across all seeds.",
        "", f"[Native case gallery](cases/index.html): {len(cases)} unique queries, separate VV/VH, "
        "actual same-class/rival references. Images generated; pixels NOT reviewed by this analysis. "
        "Visible facts and physical explanations remain for the main review. Paired outcomes alone do not identify physical causes.", ""]
    (out / "NUMERIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    dump(out / "complete.json", dict(final_prediction_files=6, initial_prediction_files=2,
        training_receipts=6, seed_events=len(events), unique_queries=len(rows), gallery_unique_queries=len(cases),
        panels=2 * len(cases), visual_status="not_reviewed", device="CPU", report=str(out / "NUMERIC_REPORT.md")))
    print(json.dumps(dict(output=str(out), arms=summary["arms"], paired=summary["paired"],
        gallery=str(out / "cases/index.html")), indent=2))
    print(json.dumps(dict(posthoc_pruning=posthoc_pruning(root, rows, y, scores, initial)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--cohorts-only", action="store_true", help="Summarize actual manifests without reading predictions or chips")
    mode.add_argument("--pruning-only", action="store_true", help="Write only the separate optional posthoc pruning comparison")
    args = parser.parse_args()
    if args.cohorts_only:
        summary = cohort_report(args.root.resolve())
        print(json.dumps(dict(report=str(args.root.resolve() / "analysis/COHORT_REPORT.md"),
            cohorts={s: {k: summary[s][k] for k in ("n", "ships", "classes", "supplied_views")} for s in ("fit", "development", "references")},
            overlap_counts=summary["overlap_counts"], pair_exclusion=summary["pair_exclusion"]), indent=2))
    elif args.pruning_only:
        cohorts, y, scores, initial, _ = load_inputs(args.root.resolve())
        print(json.dumps(posthoc_pruning(args.root.resolve(), cohorts["development"], y, scores, initial), indent=2))
    else:
        analyze(args.root.resolve(), args.package.resolve())
