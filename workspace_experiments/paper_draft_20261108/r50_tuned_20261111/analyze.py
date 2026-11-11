"""CPU-only post-training analysis of frozen OSS2 predictions.

Run only after selected_s17/29/43 have complete receipts and both NPZ files.
No training imports, model inference, cache construction, or model selection.
"""
import argparse
import csv
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
METRICS = ("oa", "ba", "macro_f1")
LABELS = ("Bulk carrier", "General cargo", "Chemical/product tanker", "Container ship")
KINDS = ("rescued", "harmed", "stable_correct", "stable_wrong")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def input_api():
    # This existing module has no import-time preparation or training side effects.
    sys.dont_write_bytecode = True
    path = ROOT.parent / "recent_baselines_20261111/data.py"
    spec = importlib.util.spec_from_file_location("r50_sidecar_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_run(folder, seed, selection, splits=("validation", "evaluation")):
    config = read_json(folder / "config.json")
    receipt = read_json(folder / "receipt.json")
    complete = (receipt.get("stop_reason") in ("validation_patience", "max_epochs")
                and receipt.get("epochs", 0) > 0 and receipt.get("steps", 0) > 0)
    if not complete:
        raise ValueError(f"Unfinished run (missing completed training receipt): {folder}")
    if (config["seed"] != seed or config["scope"] != selection["scope"]
            or float(config["lr"]) != float(selection["lr"])):
        raise ValueError(f"Run config disagrees with frozen selection: {folder}")
    for split in splits:
        if not (folder / f"{split}.npz").is_file():
            raise ValueError(f"Missing completed {split} predictions: {folder}")
    return {"path": str(folder), "config": config, "receipt": receipt}


def load_prediction(path, expected_ids, rows, require_label=False):
    expected_ids = np.asarray(expected_ids, dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        ids = data["ids"]
        score = data["score"]
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids)
                or not np.array_equal(np.sort(ids), np.sort(expected_ids))):
            raise ValueError(f"Wrong, duplicate, or missing manifest ids: {path}")
        if score.shape != (len(ids), 4) or not np.isfinite(score).all():
            raise ValueError(f"Expected finite score (N,4): {path}")
        if require_label and "label" not in data:
            raise ValueError(f"New predictions require label: {path}")
        label_key = "label" if "label" in data else "y" if "y" in data else None
        if label_key:
            truth = np.array([rows[int(i)]["label"] for i in ids])
            if not np.array_equal(data[label_key], truth):
                raise ValueError(f"Prediction labels disagree with manifest: {path}")
        index = {int(i): j for j, i in enumerate(ids)}
        return np.asarray(score[[index[int(i)] for i in expected_ids]], dtype=np.float64)


def confusion(y, prediction, weights=None):
    return np.bincount(4 * y + prediction, weights=weights, minlength=16).reshape(4, 4)


def metrics_from_cm(cm):
    """Percent metrics; BA over truth-present classes, F1 over fixed four labels."""
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    correct = np.diagonal(cm, axis1=-2, axis2=-1)
    recall = np.divide(correct, support, out=np.zeros_like(correct), where=support > 0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros_like(correct),
                   where=(support + predicted) > 0)
    total = support.sum(axis=-1)
    oa = np.divide(correct.sum(axis=-1), total, out=np.zeros_like(total), where=total > 0)
    count = (support > 0).sum(axis=-1)
    ba = np.divide(recall.sum(axis=-1), count, out=np.zeros_like(total), where=count > 0)
    return np.stack((oa, ba, f1.mean(axis=-1)), axis=-1) * 100


def metrics(y, prediction):
    if not len(y):
        raise ValueError("Empty cohorts have no metrics")
    return dict(zip(METRICS, map(float, metrics_from_cm(confusion(y, prediction)))))


def strata(rows, ids, available):
    av = np.asarray(available[ids], dtype=bool)
    masks = {"all": np.ones(len(ids), bool), "both": av.all(axis=1),
             "single": av.sum(axis=1) == 1, "vv_only": av[:, 0] & ~av[:, 1],
             "vh_only": av[:, 1] & ~av[:, 0]}
    for key in ("pass_group", "scene_id"):
        values = np.array([str(rows[int(i)][key]) for i in ids])
        for value in sorted(set(values)):
            masks[f"{key}:{value}"] = values == value
            masks[f"{key}:{value}|both"] = (values == value) & masks["both"]
    return {name: mask for name, mask in masks.items() if mask.any()}


def cluster_intervals(y, ship, candidate, baseline, replicates, seed):
    """Paired MMSI bootstrap, retaining all records and all three fixed seeds."""
    unique, inverse = np.unique(ship, return_inverse=True)
    cm = np.zeros((len(unique), 2, len(SEEDS), 4, 4), dtype=float)
    for method, predictions in enumerate((candidate, baseline)):
        for k, pred in enumerate(predictions):
            np.add.at(cm[:, method, k], (inverse, y, pred), 1)
    rng = np.random.default_rng(seed)
    # Integer ship multiplicities keep every observation from a sampled ship.
    multiplicity = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)),
                                  size=replicates)
    sampled = (multiplicity @ cm.reshape(len(unique), -1)).reshape(
        replicates, 2, len(SEEDS), 4, 4)
    value = metrics_from_cm(sampled)
    delta = (value[:, 0] - value[:, 1]).mean(axis=1)
    required = np.bincount(y, minlength=4) > 0
    ba_valid = (sampled[:, 0, 0].sum(axis=-1)[:, required] > 0).all(axis=1)
    point = (metrics_from_cm(cm.sum(axis=0))[0]
             - metrics_from_cm(cm.sum(axis=0))[1]).mean(axis=0)
    result = []
    for j, metric in enumerate(METRICS):
        valid = ba_valid if metric == "ba" else np.ones(replicates, bool)
        interval = np.quantile(delta[valid, j], [.025, .975]) if valid.any() else (None, None)
        result.append(dict(metric=metric, delta_pp=float(point[j]),
                           ci_low_pp=None if interval[0] is None else float(interval[0]),
                           ci_high_pp=None if interval[1] is None else float(interval[1]),
                           replicates=replicates, valid_replicates=int(valid.sum()),
                           ships=len(unique), bootstrap_seed=seed))
    return result


def outcome_masks(candidate, baseline, y):
    after, before = candidate == y, baseline == y
    return dict(rescued=after & ~before, harmed=~after & before,
                stable_correct=after & before, stable_wrong=~after & ~before)


def choose_cases(mask, y, ids, count):
    repeats = mask.sum(axis=0)
    # Balance example coverage across classes before taking a second of a class.
    queues = [sorted(np.flatnonzero((y == c) & (repeats > 0)),
                     key=lambda j: (-int(repeats[j]), int(ids[j]))) for c in range(4)]
    result = []
    while len(result) < count and any(queues):
        for c in sorted(range(4), key=lambda c: (-int(repeats[queues[c][0]])
                                               if queues[c] else 1, c)):
            if queues[c] and len(result) < count:
                result.append(queues[c].pop(0))
    return result


def draw_panel(native, query, references, path, caption):
    from PIL import Image, ImageDraw
    selected = [("Query", query)] + references
    canvas = Image.new("RGB", (max(760, 220 * len(selected)), 455), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), caption, fill="black")
    provenance = []
    for j, (role, row) in enumerate(selected):
        x = 10 + j * 220
        draw.text((x, 34), f"{role}: {row['id'].split('_')[-1]}", fill="black")
        draw.text((x, 50), LABELS[row["label"]], fill="black")
        for p, pol in enumerate(("vv", "vh")):
            top = 75 + 185 * p
            spec = row.get(pol)
            view = native.prepare_view(spec)
            usable = view is not None and bool(view["valid_target"])
            item = dict(role=role, id=row["id"], pol=pol, available=usable,
                        input_png=spec["image"] if spec else None,
                        original_source=row.get("source_paths", {}).get(pol))
            draw.text((x, top), pol.upper(), fill="black")
            if usable:
                canvas.paste(Image.fromarray(view["pixels"]).convert("RGB"), (x, top + 16))
                item.update(placement=view["placement"], raw_shape=list(view["raw"].shape),
                            footprint_pixels=int(view["footprint"].sum()))
            else:
                draw.text((x, top + 60), "Unavailable to model", fill="gray")
            provenance.append(item)
    canvas.save(path)
    return provenance


def make_gallery(out, api, rows, groups, ids, scores, count):
    y = np.array([rows[int(i)]["label"] for i in ids])
    candidate = scores["R50-Tuned"]
    native = api.native_module()
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    cases, coverage = [], []
    for arm in ("Full", "Plain"):
        baseline = scores[arm]
        states = outcome_masks(candidate.argmax(axis=2), baseline.argmax(axis=2), y)
        for kind in KINDS:
            mask = states[kind]
            chosen = choose_cases(mask, y, ids, count)
            coverage.append(dict(baseline=arm, kind=kind, seed_observation_events=int(mask.sum()),
                                 observations=int(mask.any(axis=0).sum()), panels=len(chosen)))
            for j in chosen:
                k = int(np.flatnonzero(mask[:, j])[0])
                index = int(ids[j])
                row, label = rows[index], int(y[j])
                competitor = int(baseline[k, j].argmax()) if kind == "rescued" else int(candidate[k, j].argmax())
                if competitor == label:
                    rival = candidate[k, j].copy()
                    rival[label] = -np.inf
                    competitor = int(rival.argmax())
                references, eligible_ids = [], []
                if arm == "Full":
                    eligible_ids = [int(i) for i in groups["references"]
                                    if str(rows[int(i)]["mmsi"]) != str(row["mmsi"])
                                    and rows[int(i)]["pass_group"] != row["pass_group"]]
                    for role, c in (("Same-class ref", label), ("Confusion ref", competitor)):
                        match = next((i for i in eligible_ids if rows[i]["label"] == c), None)
                        if match is not None:
                            references.append((role, rows[match]))
                name = f"{arm}_{kind}_{index}_s{SEEDS[k]}.png"
                cp, bp = int(candidate[k, j].argmax()), int(baseline[k, j].argmax())
                caption = f"R50 vs {arm} | {kind} | s{SEEDS[k]} | truth={label} R50={cp} {arm}={bp}"
                provenance = draw_panel(native, row, references, folder / name, caption)
                evidence = []
                for s, seed in enumerate(SEEDS):
                    evidence.append(dict(seed=seed, state=next(t for t in KINDS if states[t][s, j]),
                        candidate_prediction=int(candidate[s, j].argmax()),
                        baseline_prediction=int(baseline[s, j].argmax()),
                        candidate_score=candidate[s, j].tolist(), baseline_score=baseline[s, j].tolist(),
                        candidate_true_margin=float(candidate[s, j, label] - np.delete(candidate[s, j], label).max()),
                        baseline_true_margin=float(baseline[s, j, label] - np.delete(baseline[s, j], label).max())))
                cases.append(dict(baseline=arm, kind=kind, manifest_index=index, id=row["id"],
                    label=label, seed=SEEDS[k], repeats=int(mask[:, j].sum()), mmsi=str(row["mmsi"]),
                    pass_group=row["pass_group"], scene_id=row["scene_id"], file=f"cases/{name}",
                    actual_eligible_reference_indices=eligible_ids, evidence=evidence, paths=provenance,
                    review_status="pending_main_agent_visual_review"))
    write_json(out / "cases_local.json", cases)
    write_csv(out / "case_coverage.csv", coverage)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<title>R50-Tuned native cases</title><style>body{font:16px/1.6 system-ui;margin:24px}',
            'section{border-top:1px solid #aaa;padding:12px 0}.panel{overflow-x:auto}',
            'img{max-width:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
            '<h1>R50-Tuned native cases: pending visual review</h1>',
            '<p>Each chip is 160x160 pixels at 1:1 with no added contrast stretch. ',
            'Full references are examples from its legal fixed bank, not highest-contribution claims. ',
            'R50 and Plain have no reference bank. Margins use each model own score scale.</p>',
            '<p>Cases are deterministic representatives, not estimates of population proportions. ',
            'Outcome direction is historical baseline to R50-Tuned.</p>']
    for entry in coverage:
        if not entry["panels"]:
            body.append(f'<p>{entry["baseline"]} / {entry["kind"]}: no cases.</p>')
    for case in cases:
        body.extend([f'<section><h2>{case["baseline"]} / {case["kind"]} / {case["id"]}</h2>',
                     f'<p>Selected seed {case["seed"]}; same outcome {case["repeats"]}/3 seeds.</p>',
                     f'<div class="panel"><img src="{html.escape(case["file"])}"></div>',
                     '<details><summary>Predictions, margins and actual source paths</summary><pre>',
                     html.escape(json.dumps(case, ensure_ascii=False, indent=2)),
                     '</pre></details></section>'])
    (out / "index.html").write_text("\n".join(body) + "</html>", encoding="utf-8")
    return len(cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261111)
    parser.add_argument("--cases-per-kind", type=int, default=2)
    args = parser.parse_args(argv)
    if args.bootstrap < 100 or not 1 <= args.cases_per_kind <= 4:
        parser.error("Use at least 100 bootstrap replicates and 1-4 cases per kind")
    selection = read_json(ROOT / "selected.json")
    status = read_json(ROOT / "status.json")
    if status.get("state") != "training_and_test_complete" or status.get("complete") is not True:
        raise ValueError("Root status does not yet confirm completed training and selected-run testing")
    if selection["seeds"] != list(SEEDS):
        raise ValueError("Expected frozen seeds [17, 29, 43]")
    runs = [completed_run(ROOT / f"selected_s{s}", s, selection) for s in SEEDS]
    scout = Path(selection["selected_run"])
    if not scout.is_absolute():
        scout = ROOT / scout
    scout_record = completed_run(scout, 17, selection, splits=("validation",))
    api = input_api()
    rows, groups = api.records()
    available = np.load(api.ROOT / "native_cache/available.npy", mmap_mode="r")
    oldroot = ROOT.parent / "staged_ablation_20261109"
    old_selection = read_json(oldroot / "selected.json")
    cp_selection = read_json(api.ROOT / "selected.json")
    directories = {"R50-Tuned": [ROOT / f"selected_s{s}" for s in SEEDS]}
    for arm in ("Full", "Plain", "SE"):
        scale = old_selection["scales"][arm.lower()]
        directories[arm] = [oldroot / f"{arm.lower()}_s{s}_lr{scale:g}" for s in SEEDS]
    directories["CPINet"] = [api.ROOT / f"cpinet_s{s}_lr{cp_selection['lr']:g}" for s in SEEDS]
    scores, sources = {}, []
    # Validate all actual arrays before writing any result; no CSV/receipt metric is a score source.
    for split, expected_n in (("validation", 228), ("evaluation", 378)):
        ids = groups[split]
        if len(ids) != expected_n:
            raise ValueError(f"Unexpected {split} cohort size")
        scores[split] = {}
        for arm, folders in directories.items():
            arrays = []
            for seed, folder in zip(SEEDS, folders):
                path = folder / f"{split}.npz"
                arrays.append(load_prediction(path, ids, rows, require_label=arm == "R50-Tuned"))
                sources.append(dict(arm=arm, seed=seed, split=split, path=str(path)))
            scores[split][arm] = np.stack(arrays)
    scout_score = load_prediction(scout / "validation.npz", groups["validation"], rows, True)
    if not np.array_equal(scout_score, scores["validation"]["R50-Tuned"][0]):
        raise ValueError("selected_s17 validation must reuse the selected scout predictions")
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    search_rows, unique_receipts = [], []
    for candidate in selection["candidates"]:
        folder = ROOT / candidate["path"]
        receipt = read_json(folder / "receipt.json")
        curve = read_json(folder / "curve.json")
        unique_receipts.append(receipt)
        search_rows.append(dict(scope=receipt["scope"], lr=receipt["lr"], seed=receipt["seed"],
            epochs=receipt["epochs"], best_epoch=receipt["best"]["epoch"],
            validation_oa=receipt["best"]["metrics"]["oa"],
            validation_ba=receipt["best"]["metrics"]["ba"],
            last_train_oa=curve[-1]["train_oa"], last_validation_ba=curve[-1]["validation"]["ba"],
            steps=receipt["steps"], seconds=receipt["seconds"], trainable=receipt["trainable"],
            selected=candidate["path"] == selection["selected_run"]))
    for seed in SEEDS[1:]:
        unique_receipts.append(read_json(ROOT / f"selected_s{seed}" / "receipt.json"))
    write_csv(out / "validation_search.csv", search_rows)
    write_json(out / "training_budget.json", dict(
        unique_training_runs=len(unique_receipts), scout_runs=len(search_rows),
        selected_seed17_reused=True, epochs=sum(r["epochs"] for r in unique_receipts),
        optimizer_steps=sum(r["steps"] for r in unique_receipts),
        sample_exposures=sum(r["exposures"] for r in unique_receipts),
        fit_seconds=sum(r["seconds"] for r in unique_receipts),
        max_peak_gpu_gib=max(r["peak_gib"] for r in unique_receipts)))
    metric_rows, class_rows, paired_rows, ci_rows = [], [], [], []
    for split, arms in scores.items():
        ids = groups[split]
        y = np.array([rows[int(i)]["label"] for i in ids], dtype=np.int64)
        ships = np.array([str(rows[int(i)]["mmsi"]) for i in ids])
        masks = strata(rows, ids, available)
        predictions = {arm: score.argmax(axis=2) for arm, score in arms.items()}
        for subset, mask in masks.items():
            truth = y[mask]
            meta = dict(split=split, subset=subset, n=int(mask.sum()),
                        ships=len(set(ships[mask])), classes_present=len(set(truth)))
            for arm, pred in predictions.items():
                for k, seed in enumerate(SEEDS):
                    cm = confusion(truth, pred[k, mask])
                    metric_rows.append(dict(arm=arm, seed=seed, **meta, **metrics(truth, pred[k, mask])))
                    for c in range(4):
                        support, predicted = int(cm[c].sum()), int(cm[:, c].sum())
                        tp = int(cm[c, c])
                        class_rows.append(dict(arm=arm, seed=seed, **meta, label=c, class_name=LABELS[c],
                            support=support, predicted=predicted, correct=tp,
                            recall=100 * tp / support if support else None,
                            precision=100 * tp / predicted if predicted else None,
                            f1=200 * tp / (support + predicted) if support + predicted else 0.0))
            for arm in ("Full", "Plain"):
                cp, bp = predictions["R50-Tuned"][:, mask], predictions[arm][:, mask]
                states = outcome_masks(cp, bp, truth)
                for k, seed in enumerate(SEEDS):
                    a, b = metrics(truth, cp[k]), metrics(truth, bp[k])
                    paired_rows.append(dict(baseline=arm, seed=seed, **meta,
                        **{f"{m}_delta_pp": a[m] - b[m] for m in METRICS},
                        **{kind: int(states[kind][k].sum()) for kind in KINDS},
                        wrong_to_wrong_flips=int(((cp[k] != bp[k]) & states["stable_wrong"][k]).sum())))
                if split == "evaluation":
                    for interval in cluster_intervals(truth, ships[mask], cp, bp,
                                                      args.bootstrap, args.bootstrap_seed):
                        ci_rows.append({"baseline": arm, **meta, **interval})
    summary = []
    grouped = {}
    for row in metric_rows:
        grouped.setdefault((row["arm"], row["split"], row["subset"]), []).append(row)
    for (arm, split, subset), items in grouped.items():
        summary.append(dict(arm=arm, split=split, subset=subset, n=items[0]["n"],
            ships=items[0]["ships"], classes_present=items[0]["classes_present"], seeds=len(items),
            **{f"{m}_{stat}": float(np.mean([x[m] for x in items]) if stat == "mean"
                                   else np.std([x[m] for x in items], ddof=1))
               for m in METRICS for stat in ("mean", "std")}))
    class_summary = []
    class_groups = {}
    for row in class_rows:
        class_groups.setdefault((row["arm"], row["split"], row["subset"], row["label"]), []).append(row)
    for (arm, split, subset, label), items in class_groups.items():
        recalls = [r["recall"] for r in items if r["recall"] is not None]
        class_summary.append(dict(arm=arm, split=split, subset=subset, label=label,
            class_name=LABELS[label], support=items[0]["support"], seeds=len(items),
            recall_mean=float(np.mean(recalls)) if recalls else None,
            recall_std=float(np.std(recalls, ddof=1)) if len(recalls) > 1 else None,
            f1_mean=float(np.mean([r["f1"] for r in items])),
            f1_std=float(np.std([r["f1"] for r in items], ddof=1))))
    for name, records in (("metrics.csv", metric_rows), ("summary.csv", summary),
                          ("per_class.csv", class_rows), ("per_class_summary.csv", class_summary),
                          ("paired_by_seed.csv", paired_rows),
                          ("paired_ship_ci.csv", ci_rows)):
        write_csv(out / name, records)
    write_csv(out / "per_scene.csv", [r for r in metric_rows if r["subset"].startswith(("scene_id:", "pass_group:"))])
    write_json(out / "sources_local.json", dict(selection=selection, completion_status=status,
        runs=runs, selected_scout=scout_record,
        historical_staged_selection=old_selection, historical_cpinet_selection=cp_selection,
        prediction_sources=sources, availability_source=str(api.ROOT / "native_cache/available.npy"),
        manuscript_changed=False, test_selection=False))
    panels = make_gallery(out, api, rows, groups, groups["evaluation"], scores["evaluation"], args.cases_per_kind)
    lines = ["# R50-Tuned frozen-prediction analysis", "",
             "Status: computed from completed runs; native panels await main-agent visual review.", "",
             f"Selection: scope={selection['scope']}, backbone LR={selection['lr']}; seeds=17/29/43.",
             "Only the selected three R50 runs supply test predictions. Historical predictions are reused.", "",
             "| Method | Test OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |",
             "|---|---:|---:|---:|"]
    for row in summary:
        if row["split"] == "evaluation" and row["subset"] == "all":
            values = " | ".join(f"{row[m + '_mean']:.3f} +/- {row[m + '_std']:.3f}" for m in METRICS)
            lines.append(f"| {row['arm']} | {values} |")
    lines.extend(["", "SD is the sample SD of three trained seeds, not three independent scenes.", "",
                  "| R50 minus baseline | Metric | Delta (pp) | Paired ship 95% CI | Positive seeds |",
                  "|---|---|---:|---|---:|"])
    for row in ci_rows:
        if row["subset"] != "all":
            continue
        related = [r for r in paired_rows if r["baseline"] == row["baseline"]
                   and r["split"] == "evaluation" and r["subset"] == "all"]
        positive = sum(r[row["metric"] + "_delta_pp"] > 0 for r in related)
        lines.append(f"| {row['baseline']} | {row['metric']} | {row['delta_pp']:+.3f} | "
                     f"[{row['ci_low_pp']:+.3f}, {row['ci_high_pp']:+.3f}] | {positive}/3 |")
    lines.extend(["", "| Method | Polarization subset | N | OA mean | BA mean | Macro-F1 mean |",
                  "|---|---|---:|---:|---:|---:|"])
    for row in summary:
        if row["split"] == "evaluation" and row["subset"] in ("both", "single"):
            lines.append(f"| {row['arm']} | {row['subset']} | {row['n']} | {row['oa_mean']:.3f} | "
                         f"{row['ba_mean']:.3f} | {row['macro_f1_mean']:.3f} |")
    lines.extend(["", "| Method | Class | N | Recall mean +/- SD |",
                  "|---|---|---:|---:|"])
    for row in class_summary:
        if row["split"] == "evaluation" and row["subset"] == "all":
            lines.append(f"| {row['arm']} | {row['class_name']} | {row['support']} | "
                         f"{row['recall_mean']:.3f} +/- {row['recall_std']:.3f} |")
    lines.extend(["", "Metrics are percentages; differences are percentage points. BA averages truth-present classes;",
        "macro-F1 always uses four classes with zero-division set to zero. Subset class counts are explicit.",
        "Ship bootstrap retains all observations per MMSI and pairs methods and fixed seeds in each draw.",
        "It conditions on the existing scenes and trained runs, and does not estimate new-port or selection uncertainty.",
        "BA replicates missing an originally present class are excluded; valid replicate counts are recorded.",
        "Scene/pass-group tables include complete-dual subsets; inspect denominators before interpreting effects.", "",
        "R50-Tuned versus historical Plain/Full is a system comparison with different training recipes.",
        "Only the R50 scope comparisons share the new recipe; these results cannot isolate unfreezing alone.",
        "This historically used evaluation cohort is not a fresh blind test.", "",
        f"Native review: {panels} panels in index.html; cases_local.json records real paths, margins and all seed states.",
        "Rescue/harm direction is baseline to R50. Stable-wrong includes wrong-to-wrong flips, counted separately.",
        "Representative cases do not estimate prevalence; no physical explanation is generated automatically.",
        "Main agent must inspect raw VV/VH, Full references, and remaining failures before final conclusions.",
        "Local paths, predictions and raw panels are local artifacts; this script performs no Git actions."])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "analysis_receipt.json", dict(status="analysis_complete_visual_review_pending",
        seeds=list(SEEDS), bootstrap_replicates=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        panels=panels, results="RESULTS.md", gallery="index.html", new_training=False, new_inference=False))
    print(f"Analysis complete: {out / 'RESULTS.md'}; {panels} native panels await visual review.")


if __name__ == "__main__":
    main()
