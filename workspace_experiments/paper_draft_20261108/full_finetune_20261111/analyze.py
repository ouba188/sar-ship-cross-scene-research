"""CPU-only analysis of completed Full-Tuned OSS2 runs; no training or inference.

Run after status.json confirms completion and selected_s17/29/43 contain both
prediction files. Reuses only the R50 analysis statistics and native input API.
"""
import argparse
import html
import importlib.util
import json
import os
from pathlib import Path
import sys

# These settings affect this analysis process only, never the running trainer.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER = ROOT.parent
SEEDS = (17, 29, 43)
CANDIDATE = "Full-Tuned"
BASELINES = ("Full", "R50-Tuned", "Plain")
KINDS = ("rescued", "harmed", "stable_correct", "stable_wrong")
METRICS = ("oa", "ba", "macro_f1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_helpers():
    path = PAPER / "r50_tuned_20261111/analyze.py"
    spec = importlib.util.spec_from_file_location("full_tuned_statistics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def orbit(row):
    # Exact key used by model_task.legal; the remaining pass-group suffix is ignored.
    return "_".join(row["pass_group"].split("_")[:2])


def completed_inputs(h):
    selection = read_json(ROOT / "selected.json")
    if selection["seeds"] != list(SEEDS):
        raise ValueError("Expected selected seeds [17, 29, 43]")
    runs = [h.completed_run(ROOT / f"selected_s{s}", s, selection) for s in SEEDS]
    for run in runs:
        receipt = run["receipt"]
        if not 0 <= receipt["best"]["epoch"] <= receipt["epochs"]:
            raise ValueError(f"Invalid best epoch, including eligible epoch 0: {run['path']}")
    api = h.input_api()
    rows, groups = api.records()
    for split, count in (("small", 719), ("validation", 228), ("evaluation", 378)):
        ids = groups[split]
        if len(ids) != count or len(set(map(int, ids))) != count:
            raise ValueError(f"Expected {count} unique {split} observations")
        if set(rows[int(i)]["label"] for i in ids) != set(range(4)):
            raise ValueError(f"Expected the same four OSS2 classes: {split}")
    refs = groups["references"]
    if len(refs) != 32 or len(set(map(int, refs))) != 32 or not set(refs) <= set(groups["small"]):
        raise ValueError("Expected the actual fixed 32-source reference bank within small")
    for left, right in (("small", "validation"), ("small", "evaluation"), ("validation", "evaluation")):
        for key in ("id", "mmsi", "platform_orbit"):
            values = lambda split: {orbit(rows[int(i)]) if key == "platform_orbit"
                                    else str(rows[int(i)][key]) for i in groups[split]}
            if values(left) & values(right):
                raise ValueError(f"Overlapping {key}: {left} / {right}")
    availability_path = api.ROOT / "native_cache/available.npy"
    available = np.load(availability_path, mmap_mode="r", allow_pickle=False)
    if available.shape != (len(rows), 2) or not np.isin(available, (0, 1)).all():
        raise ValueError("Expected existing binary VV/VH availability for the manifest")
    if not np.asarray(available).any(axis=1).all():
        raise ValueError("Manifest includes an observation unavailable in both polarizations")
    old = PAPER / "staged_ablation_20261109"
    directories = {
        CANDIDATE: [ROOT / f"selected_s{s}" for s in SEEDS],
        "Full": [old / f"full_s{s}_lr1" for s in SEEDS],
        "Plain": [old / f"plain_s{s}_lr1" for s in SEEDS],
        "R50-Tuned": [PAPER / "r50_tuned_20261111" / f"selected_s{s}" for s in SEEDS],
    }
    scores, sources = {}, []
    for split in ("validation", "evaluation"):
        scores[split] = {}
        for method, folders in directories.items():
            arrays = []
            for seed, folder in zip(SEEDS, folders):
                path = folder / f"{split}.npz"
                arrays.append(h.load_prediction(path, groups[split], rows,
                                                require_label=method in (CANDIDATE, "R50-Tuned")))
                sources.append(dict(method=method, seed=seed, split=split, path=str(path)))
            scores[split][method] = np.stack(arrays)
    scout = ROOT / selection["path"]
    scout_score = h.load_prediction(scout / "validation.npz", groups["validation"], rows, True)
    if not np.array_equal(scout_score, scores["validation"][CANDIDATE][0]):
        raise ValueError("Selected seed 17 must reuse the validation-selected scout")
    return selection, runs, api, rows, groups, available, scores, sources, directories


def budget_tables(h, selection, directories):
    new_rows, search = [], []
    selected_path = (ROOT / selection["path"]).resolve()
    specs = [(ROOT / c["path"], 17, c, True) for c in selection["candidates"]]
    specs += [(ROOT / f"selected_s{s}", s, selection, False) for s in SEEDS[1:]]
    for folder, seed, cfg, is_scout in specs:
        run = h.completed_run(folder, seed, cfg, splits=("validation",))
        r = run["receipt"]
        if not 0 <= r["best"]["epoch"] <= r["epochs"]:
            raise ValueError(f"Invalid best epoch: {folder}")
        row = dict(method=CANDIDATE, path=str(folder), seed=seed, scout=is_scout,
                   selected=not is_scout or folder.resolve() == selected_path,
                   scope=r["scope"], lr=r["lr"], epochs=r["epochs"], best_epoch=r["best"]["epoch"],
                   best_is_epoch0=r["best"]["epoch"] == 0, optimizer_steps=r["steps"],
                   query_exposures=719 * r["epochs"], fit_seconds=r["seconds"],
                   trainable=r["trainable"], peak_gpu_gib=r["peak_gib"],
                   validation_oa=r["best"]["metrics"]["oa"], validation_ba=r["best"]["metrics"]["ba"])
        new_rows.append(row)
        if is_scout:
            search.append(row.copy())
    if len({r["path"] for r in new_rows}) != len(new_rows) or sum(r["selected"] for r in new_rows) != 3:
        raise ValueError("Training budget must contain unique scouts and three selected runs")
    history = []
    for method in ("Full", "Plain"):
        for seed, folder in zip(SEEDS, directories[method]):
            for phase in ("carrier", "intervention", "reference", "feedback", "readout"):
                path = folder / phase / "receipt.json"
                r = read_json(path)
                history.append(dict(method=method, seed=seed, phase=phase, path=str(path),
                    optimizer_steps=r["steps"], query_exposures=r["examples"],
                    fit_seconds=r["seconds"], trainable=r["trainable"]))
    r50root = PAPER / "r50_tuned_20261111"
    r50selection = read_json(r50root / "selected.json")
    r50paths = [r50root / c["path"] for c in r50selection["candidates"]]
    r50paths += [r50root / f"selected_s{s}" for s in SEEDS[1:]]
    r50runs = []
    for path in r50paths:
        r = read_json(path / "receipt.json")
        r50runs.append(dict(path=str(path), seed=r["seed"], epochs=r["epochs"],
            best_epoch=r["best"]["epoch"], optimizer_steps=r["steps"], query_exposures=r["exposures"],
            fit_seconds=r["seconds"], peak_gpu_gib=r["peak_gib"]))

    def total(records):
        return {key: sum(r[key] for r in records)
                for key in ("optimizer_steps", "query_exposures", "fit_seconds")}

    inherited = [r for r in history if r["method"] == "Full"]
    budget = dict(
        new_full_tuned_search_and_confirm=dict(unique_runs=len(new_rows), scout_runs=len(search),
            selected_seed17_reused=True, epochs=sum(r["epochs"] for r in new_rows),
            max_peak_gpu_gib=max(r["peak_gpu_gib"] for r in new_rows), **total(new_rows)),
        new_full_tuned_selected_runs=total([r for r in new_rows if r["selected"]]),
        inherited_full_selected_three_seeds=dict(phases=len(inherited), **total(inherited)),
        full_tuned_search_plus_inherited_unique_cost=total(new_rows + inherited),
        historical_plain_selected_three_seeds=total([r for r in history if r["method"] == "Plain"]),
        historical_r50_search_and_confirm=dict(unique_runs=len(r50runs), **total(r50runs)),
        r50_runs=r50runs,
        notes=["Full-Tuned starts from each seed's already-trained five-stage Full checkpoint.",
               "Seed17 is copied from one scout and counted once, not as another fit.",
               "Best epoch 0 is valid; attempted training after epoch 0 still incurs its full cost.",
               "Query exposures count repeated query presentations, not independent observations.",
               "Reference forward/backward exposures are not included in query exposures.",
               "Historical costs cover the three reused checkpoints, not all prior research/search.",
               "Receipt seconds exclude some preparation, evaluation and initialization work;",
               "inherited and continuation costs are separately reported, not equal-start ablations."])
    return budget, new_rows, search, history


def summarize(h, rows, groups, available, scores, args):
    metric_rows, class_rows, paired_rows, ci_rows, class_pairs = [], [], [], [], []
    for split, arms in scores.items():
        ids = groups[split]
        y = np.array([rows[int(i)]["label"] for i in ids], dtype=np.int64)
        ships = np.array([str(rows[int(i)]["mmsi"]) for i in ids])
        masks = h.strata(rows, ids, available)
        passes = np.array([orbit(rows[int(i)]) for i in ids])
        for value in sorted(set(passes)):
            masks[f"platform_orbit:{value}"] = passes == value
            both = (passes == value) & np.asarray(available[ids], bool).all(axis=1)
            if both.any():
                masks[f"platform_orbit:{value}|both"] = both
        predictions = {method: score.argmax(axis=2) for method, score in arms.items()}
        for subset, mask in masks.items():
            truth = y[mask]
            meta = dict(split=split, subset=subset, n=int(mask.sum()),
                        ships=len(set(ships[mask])), classes_present=len(set(truth)))
            cms = {}
            for method, pred in predictions.items():
                cms[method] = []
                for k, seed in enumerate(SEEDS):
                    cm = h.confusion(truth, pred[k, mask])
                    cms[method].append(cm)
                    metric_rows.append(dict(method=method, seed=seed, **meta, **h.metrics(truth, pred[k, mask])))
                    for c in range(4):
                        support, predicted, tp = int(cm[c].sum()), int(cm[:, c].sum()), int(cm[c, c])
                        class_rows.append(dict(method=method, seed=seed, **meta,
                            label=c, class_name=h.LABELS[c], support=support, predicted=predicted, correct=tp,
                            recall=100 * tp / support if support else None,
                            precision=100 * tp / predicted if predicted else None,
                            f1=200 * tp / (support + predicted) if support + predicted else 0.0))
            for baseline in BASELINES:
                cp, bp = predictions[CANDIDATE][:, mask], predictions[baseline][:, mask]
                states = h.outcome_masks(cp, bp, truth)
                for k, seed in enumerate(SEEDS):
                    a, b = h.metrics(truth, cp[k]), h.metrics(truth, bp[k])
                    paired_rows.append(dict(candidate=CANDIDATE, baseline=baseline, seed=seed, **meta,
                        **{f"{m}_delta_pp": a[m] - b[m] for m in METRICS},
                        **{kind: int(states[kind][k].sum()) for kind in KINDS},
                        wrong_to_wrong_flips=int(((cp[k] != bp[k]) & states["stable_wrong"][k]).sum())))
                    ca, cb = cms[CANDIDATE][k], cms[baseline][k]
                    for c in range(4):
                        support = int(ca[c].sum())
                        class_pairs.append(dict(candidate=CANDIDATE, baseline=baseline, seed=seed,
                            **meta, label=c, class_name=h.LABELS[c], support=support,
                            recall_delta_pp=100 * float(ca[c, c] - cb[c, c]) / support if support else None))
                if split == "evaluation":
                    for interval in h.cluster_intervals(truth, ships[mask], cp, bp,
                                                        args.bootstrap, args.bootstrap_seed):
                        ci_rows.append({"candidate": CANDIDATE, "baseline": baseline, **meta, **interval})

    def aggregate(records, keys, fields):
        grouped = {}
        for row in records:
            grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
        result = []
        for identity, items in grouped.items():
            entry = dict(zip(keys, identity))
            entry.update({key: items[0][key] for key in ("n", "ships", "classes_present")})
            entry["seeds"] = len(items)
            if "support" in items[0]:
                entry["support"] = items[0]["support"]
            for field in fields:
                values = [r[field] for r in items if r[field] is not None]
                entry[field + "_mean"] = float(np.mean(values)) if values else None
                entry[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
                if field.endswith("_delta_pp"):
                    entry[field + "_positive_seeds"] = sum(v > 0 for v in values)
            result.append(entry)
        return result

    summary = aggregate(metric_rows, ("method", "split", "subset"), METRICS)
    class_summary = aggregate(class_rows, ("method", "split", "subset", "label", "class_name"),
                              ("recall", "precision", "f1"))
    paired_summary = aggregate(paired_rows, ("candidate", "baseline", "split", "subset"),
                               tuple(m + "_delta_pp" for m in METRICS))
    class_pair_summary = aggregate(class_pairs,
        ("candidate", "baseline", "split", "subset", "label", "class_name"), ("recall_delta_pp",))
    return dict(metrics=metric_rows, summary=summary, per_class=class_rows,
        per_class_summary=class_summary, paired_by_seed=paired_rows, paired_summary=paired_summary,
        paired_ship_ci=ci_rows, per_class_paired=class_pairs, per_class_paired_summary=class_pair_summary,
        per_scene=[r for r in metric_rows if r["subset"].startswith(("scene_id:", "pass_group:", "platform_orbit:"))],
        per_polarization=[r for r in metric_rows if r["subset"] in ("both", "single", "vv_only", "vh_only")])


def choose_gallery_cases(h, states, y, ids, per_kind, minimum):
    chosen = []
    for kind in KINDS:
        chosen.extend((kind, int(j)) for j in h.choose_cases(states[kind], y, ids, per_kind))
    # Epoch-0 selection may produce no rescue/harm at all. Fill from real states.
    used = {j for _, j in chosen}
    for kind in KINDS:
        for j in h.choose_cases(states[kind], y, ids, len(ids)):
            if len(chosen) >= minimum:
                return chosen
            if int(j) not in used:
                chosen.append((kind, int(j)))
                used.add(int(j))
    return chosen


def reference_evidence(rows, groups, available, index, label, rival):
    query = rows[index]
    bank, legal = [], []
    for idx in map(int, groups["references"]):
        row = rows[idx]
        different_ship = str(row["mmsi"]) != str(query["mmsi"])
        different_orbit = orbit(row) != orbit(query)
        shared = np.asarray(available[index], bool) & np.asarray(available[idx], bool)
        active = different_ship and different_orbit and bool(shared.any())
        bank.append(dict(manifest_index=idx, id=row["id"], label=row["label"],
            mmsi=str(row["mmsi"]), pass_group=row["pass_group"], platform_orbit=orbit(row),
            different_mmsi=different_ship, different_platform_orbit=different_orbit,
            shared_polarizations=[p for p, ok in zip(("vv", "vh"), shared) if ok],
            legal_identity=different_ship and different_orbit, active=active,
            input_paths={p: row[p]["image"] if row.get(p) else None for p in ("vv", "vh")},
            source_paths=row.get("source_paths", {})))
        if active:
            legal.append(idx)
    displayed = []
    for role, cls in (("Same-class ref", label), ("Confusion ref", rival)):
        match = next((i for i in legal if rows[i]["label"] == cls), None)
        if match is not None:
            displayed.append((role, match))
    return bank, displayed


def make_gallery(h, out, api, rows, groups, available, scores, per_kind):
    ids = groups["evaluation"]
    y = np.array([rows[int(i)]["label"] for i in ids])
    candidate = scores[CANDIDATE]
    native = api.native_module()
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    cases, coverage = [], []
    for baseline, count, minimum in (("Full", per_kind, 8), ("R50-Tuned", 1, 4)):
        before = scores[baseline]
        states = h.outcome_masks(candidate.argmax(axis=2), before.argmax(axis=2), y)
        chosen = choose_gallery_cases(h, states, y, ids, count, minimum)
        for kind in KINDS:
            mask = states[kind]
            coverage.append(dict(candidate=CANDIDATE, baseline=baseline, kind=kind,
                seed_observation_events=int(mask.sum()), observations=int(mask.any(axis=0).sum()),
                ships=len({str(rows[int(ids[j])]["mmsi"]) for j in np.flatnonzero(mask.any(axis=0))}),
                panels=sum(t == kind for t, _ in chosen)))
        for kind, j in chosen:
            k = int(np.flatnonzero(states[kind][:, j])[0])
            index, label = int(ids[j]), int(y[j])
            query = rows[index]
            rival = int(before[k, j].argmax()) if kind == "rescued" else int(candidate[k, j].argmax())
            if rival == label:
                rival_scores = candidate[k, j].copy()
                rival_scores[label] = -np.inf
                rival = int(rival_scores.argmax())
            bank, displayed = reference_evidence(rows, groups, available, index, label, rival)
            name = f"{baseline}_{kind}_{index}_s{SEEDS[k]}.png"
            cp, bp = int(candidate[k, j].argmax()), int(before[k, j].argmax())
            caption = f"Full-Tuned vs {baseline} | {kind} | s{SEEDS[k]} | true={label} FT={cp} base={bp}"
            paths = h.draw_panel(native, query, [(role, rows[i]) for role, i in displayed], folder / name, caption)
            evidence = []
            for s, seed in enumerate(SEEDS):
                evidence.append(dict(seed=seed, state=next(t for t in KINDS if states[t][s, j]),
                    candidate_prediction=int(candidate[s, j].argmax()),
                    baseline_prediction=int(before[s, j].argmax()),
                    candidate_score=candidate[s, j].tolist(), baseline_score=before[s, j].tolist(),
                    candidate_true_margin=float(candidate[s, j, label] - np.delete(candidate[s, j], label).max()),
                    baseline_true_margin=float(before[s, j, label] - np.delete(before[s, j], label).max()),
                    candidate_true_vs_rival=float(candidate[s, j, label] - candidate[s, j, rival]),
                    baseline_true_vs_rival=float(before[s, j, label] - before[s, j, rival])))
            cases.append(dict(candidate=CANDIDATE, baseline=baseline, kind=kind, manifest_index=index,
                id=query["id"], label=label, class_name=h.LABELS[label], rival_label=rival,
                seed=SEEDS[k], repeats=int(states[kind][:, j].sum()), mmsi=str(query["mmsi"]),
                pass_group=query["pass_group"], platform_orbit=orbit(query), scene_id=query["scene_id"],
                query_available=np.asarray(available[index], bool).tolist(), file=f"cases/{name}",
                reference_bank=bank, displayed_references=[dict(role=role, manifest_index=i, id=rows[i]["id"])
                                                         for role, i in displayed],
                missing_reference_roles=[role for role in ("Same-class ref", "Confusion ref")
                                         if role not in {r for r, _ in displayed}],
                reference_owners=[CANDIDATE, "Full"] if baseline == "Full" else [CANDIDATE],
                evidence=evidence, paths=paths, review_status="pending_main_agent_visual_review"))
    h.write_json(out / "cases_local.json", cases)
    h.write_csv(out / "case_coverage.csv", coverage)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<title>Full-Tuned native chip cases</title><style>body{font:16px/1.6 system-ui;margin:24px}',
        'section{border-top:1px solid #aaa;padding:12px 0}.panel{overflow-x:auto}',
        'img{max-width:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>Full-Tuned native chip cases: visual review pending</h1>',
        '<p>Direction: historical baseline to Full-Tuned. Chips use the actual native160 crop/pad ',
        'at one image pixel per display pixel, VV/VH separate, without resizing or contrast stretch. ',
        'These are trained 8-bit display inputs, not calibrated physical backscatter.</p>',
        '<p>References come from the actual fixed 32-source bank: different MMSI and platform-orbit, ',
        'with at least one shared available polarization. Full and Full-Tuned share bank identities; ',
        'their learned reference descriptors can differ. R50-Tuned has no references. ',
        'Shown references are the first eligible class examples in bank order, not highest contributors. ',
        'Inspect reference_bank for per-polarization eligibility and paths.</p>',
        '<p>Cases are deterministically class-balanced and prioritized by repeated seed outcome, ',
        'then manifest index; the displayed seed is the first matching seed. Missing outcome groups ',
        'are reported; extra existing outcomes fill the panel budget. Cases do not estimate prevalence. ',
        'Margins are on each model own score scale; no physical cause is inferred automatically.</p>']
    for entry in coverage:
        body.append(f'<p>{entry["baseline"]} / {entry["kind"]}: '
                    f'{entry["observations"]} observations, {entry["seed_observation_events"]} seed events, '
                    f'{entry["panels"]} panels.</p>')
    for case in cases:
        title = html.escape(f'{case["baseline"]} / {case["kind"]} / {case["id"]}')
        body.extend([f'<section><h2>{title}</h2>',
            f'<p>Seed {case["seed"]}; same outcome {case["repeats"]}/3 seeds.</p>',
            f'<div class="panel"><img src="{html.escape(case["file"])}" alt="Native VV and VH query and source references"></div>',
            '<details><summary>IDs, paths, all-seed margins and actual reference bank</summary><pre>',
            html.escape(json.dumps(case, ensure_ascii=False, indent=2)), '</pre></details></section>'])
    (out / "index.html").write_text("\n".join(body) + "\n</html>\n", encoding="utf-8")
    return cases


def write_report(h, out, tables, budget, runs, selection, cases):
    lines = ["# Full-Tuned frozen-prediction analysis", "",
        "Computed from completed runs. Native panels still require main-agent visual review.", "",
        f"Selection: scope={selection['scope']}, backbone LR={selection['lr']}; seeds 17/29/43.",
        "Full-Tuned retains M1/M2/M3 and continues each seed's historical Full checkpoint.",
        "R50-Tuned is a separate reference-free baseline, not this complete method.", "",
        "| Method | Test OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |",
        "|---|---:|---:|---:|"]
    for row in tables["summary"]:
        if row["split"] == "evaluation" and row["subset"] == "all":
            values = " | ".join(f"{row[m + '_mean']:.3f} +/- {row[m + '_std']:.3f}" for m in METRICS)
            lines.append(f"| {row['method']} | {values} |")
    lines.extend(["", "| Full-Tuned minus baseline | Metric | Mean +/- SD (pp) | Paired MMSI 95% CI | Positive seeds |",
                  "|---|---|---:|---|---:|"])
    for row in tables["paired_ship_ci"]:
        if row["subset"] != "all":
            continue
        summary = next(r for r in tables["paired_summary"] if r["baseline"] == row["baseline"]
                       and r["split"] == "evaluation" and r["subset"] == "all")
        field = row["metric"] + "_delta_pp"
        low, high = row["ci_low_pp"], row["ci_high_pp"]
        ci = "unavailable" if low is None else f"[{low:+.3f}, {high:+.3f}]"
        lines.append(f"| {row['baseline']} | {row['metric']} | {summary[field + '_mean']:+.3f} +/- "
                     f"{summary[field + '_std']:.3f} | {ci} | {summary[field + '_positive_seeds']}/3 |")
    lines.extend(["", "| Seed | Executed epochs | Best epoch | Steps | Fit seconds |",
                  "|---|---:|---:|---:|---:|"])
    for run in runs:
        r = run["receipt"]
        lines.append(f"| {r['seed']} | {r['epochs']} | {r['best']['epoch']} | {r['steps']} | {r['seconds']:.2f} |")
    new = budget["new_full_tuned_search_and_confirm"]
    old = budget["inherited_full_selected_three_seeds"]
    lines.extend(["", f"New search/confirmation: {new['unique_runs']} unique fits, {new['epochs']} epochs, "
        f"{new['optimizer_steps']} steps, {new['query_exposures']} repeated query exposures, "
        f"{new['fit_seconds']:.2f} fit seconds; receipt peak {new['max_peak_gpu_gib']:.3f} GiB.",
        f"Inherited Full: {old['optimizer_steps']} steps and {old['fit_seconds']:.2f} fit seconds across three seeds.",
        "Epoch 0 may be best, but all attempted optimization is counted. Seed17 scout reuse is counted once.",
        "See training_budget.json and training_budget_by_run.csv for search versus selected and inherited costs.", "",
        "| Method | Class | N | Test recall mean +/- SD |",
        "|---|---|---:|---:|"])
    for row in tables["per_class_summary"]:
        if row["split"] == "evaluation" and row["subset"] == "all":
            lines.append(f"| {row['method']} | {row['class_name']} | {row['support']} | "
                         f"{row['recall_mean']:.3f} +/- {row['recall_std']:.3f} |")
    lines.extend(["", "| Method | Polarization | N | OA | BA | Macro-F1 |", "|---|---|---:|---:|---:|---:|"])
    for row in tables["summary"]:
        if row["split"] == "evaluation" and row["subset"] in ("both", "single", "vv_only", "vh_only"):
            lines.append(f"| {row['method']} | {row['subset']} | {row['n']} | {row['oa_mean']:.3f} | "
                         f"{row['ba_mean']:.3f} | {row['macro_f1_mean']:.3f} |")
    lines.extend(["", "Metrics are percentages; gains are percentage points. SD uses ddof=1 across three fixed seeds.",
        "BA averages truth-present classes. Macro-F1 uses all four labels, undefined F1 set to zero.",
        "MMSI bootstrap resamples ships jointly for both methods and all fixed seeds, retaining all observations.",
        "BA draws missing an originally present class are excluded; valid counts are in paired_ship_ci.csv.",
        "These intervals condition on observed scenes and trained runs, not new ports or model-selection uncertainty.",
        "Per-scene, pass-group, platform-orbit and complete-dual strata include denominators; empty strata are omitted.",
        "Scene and polarization associations do not prove missing-view or physical causes.",
        "Full-Tuned and R50-Tuned have different initialization, auxiliary losses and training budgets.",
        "This is a system/continuation comparison, not a matched pure module or unfreezing ablation.",
        "The historically used evaluation cohort is not a fresh blind test. No test-based model selection is performed.", "",
        f"Native review: {len(cases)} panels in index.html; cases_local.json includes IDs, source paths, scores,",
        "true-vs-best-rival and true-vs-fixed-rival margins, seeds, actual 32-reference bank and polarity eligibility.",
        "Stable-wrong includes wrong-to-wrong flips, counted separately in paired_by_seed.csv.",
        "Absent rescue/harm categories are reported honestly, including when epoch 0 retains historical decisions.",
        "Main agent must inspect raw VV/VH and source references before claiming image-based explanations.",
        "No physical mechanism or highest-contribution reference is inferred from these panels.",
        "No training, inference, cache construction, manuscript edits or Git actions are performed."])
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261111)
    parser.add_argument("--cases-per-kind", type=int, default=2)
    args = parser.parse_args(argv)
    if args.bootstrap < 100 or not 2 <= args.cases_per_kind <= 4:
        parser.error("Use >=100 bootstrap replicates and 2-4 Full cases per outcome kind")
    status_path = ROOT / "status.json"
    if not status_path.is_file():
        raise SystemExit("Not ready: status.json does not yet exist; no analysis written.")
    status = read_json(status_path)
    if status.get("complete") is not True or status.get("state") != "training_and_test_complete":
        raise SystemExit("Not ready: training and all selected test predictions must be complete.")
    required = [ROOT / f"selected_s{s}" / name for s in SEEDS
                for name in ("evaluation.npz", "validation.npz", "config.json", "receipt.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Not ready: missing selected-run outputs:\n" + "\n".join(missing))
    h = load_helpers()
    selection, runs, api, rows, groups, available, scores, sources, directories = completed_inputs(h)
    budget, budget_rows, search, history = budget_tables(h, selection, directories)
    # All prediction arrays and budget inputs must be valid before creating output.
    tables = summarize(h, rows, groups, available, scores, args)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    for name, records in tables.items():
        h.write_csv(out / f"{name}.csv", records)
    h.write_csv(out / "training_budget_by_run.csv", budget_rows)
    h.write_csv(out / "validation_search.csv", search)
    h.write_csv(out / "historical_training_budget.csv", history)
    h.write_json(out / "training_budget.json", budget)
    h.write_json(out / "sources_local.json", dict(candidate=CANDIDATE, baselines=list(BASELINES),
        selection=selection, completion_status=status, runs=runs, prediction_sources=sources,
        manifest=str(api.WORK / "m1_coverage_20261101/manifest.json"),
        groups=str(api.WORK / "m1_coverage_20261101/groups.json"),
        availability=str(api.ROOT / "native_cache/available.npy"),
        helper_path=str(PAPER / "r50_tuned_20261111/analyze.py"),
        reference_indices=list(map(int, groups["references"])),
        cohort_counts={split: len(groups[split]) for split in ("small", "validation", "evaluation")},
        class_names=list(h.LABELS), test_selection=False, manuscript_changed=False))
    cases = make_gallery(h, out, api, rows, groups, available, scores["evaluation"], args.cases_per_kind)
    write_report(h, out, tables, budget, runs, selection, cases)
    h.write_json(out / "analysis_receipt.json", dict(status="analysis_complete_visual_review_pending",
        method=CANDIDATE, seeds=list(SEEDS), bootstrap_replicates=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed, panels=len(cases),
        panels_by_baseline={b: sum(c["baseline"] == b for c in cases) for b in ("Full", "R50-Tuned")},
        results="RESULTS.md", gallery="index.html", new_training=False, new_inference=False, cpu_only=True))
    print(f"Full-Tuned analysis complete: {out / 'RESULTS.md'}; {len(cases)} panels await visual review.")


if __name__ == "__main__":
    main()
