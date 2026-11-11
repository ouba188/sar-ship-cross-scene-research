"""CPU-only frozen-prediction analysis; run after all A/B/C/D NPZs exist.

No trainer imports, model inference, status-file dependency, or model selection.
The caller is responsible for invoking this after training/evaluation completes.
"""
import argparse
from collections import Counter
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "2"
sys.dont_write_bytecode = True

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent.parent
SEEDS = (17, 29, 43)
SPLITS = ("validation", "evaluation")
METRICS = ("oa", "ba", "macro_f1")
KINDS = ("rescued", "harmed", "stable_correct", "stable_wrong")
COMPARISONS = (("D", "C"), ("D", "A"), ("C", "B"), ("B", "A"))
ARMS = {
    "A": "Original Full fine-tuning; original queries and references",
    "B": "A weights; restored queries, original references",
    "C": "A weights; restored queries and references",
    "D": "Matched fine-tuning with restored training, queries and references",
}
GROUPS = ("all", "original_both", "original_missing", "original_vv_only", "original_vh_only")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    hstats = load_module("consistent_recovery_hstats", ROOT.parent / "r50_tuned_20261111/analyze.py")
    recovery = load_module("consistent_recovery_policy", ROOT.parent / "crosspol_box_recovery_20261112/pilot.py")
    sys.path.insert(0, str(WORK / "sar_m123_standalone"))
    from sar_m123 import input as native
    return hstats, recovery, native


def orbit(row):
    return "_".join(row["pass_group"].split("_")[:2])


def availability(rows):
    return np.array([[row.get(p) is not None for p in ("vv", "vh")] for row in rows], dtype=bool)


def read_inputs(h):
    rows = h.read_json(WORK / "m1_coverage_20261101/manifest.json")
    groups = h.read_json(WORK / "m1_coverage_20261101/groups.json")
    for split, expected in (("small", 719), ("validation", 228), ("evaluation", 378), ("references", 32)):
        ids = np.asarray(groups[split])
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or len(ids) != expected or len(np.unique(ids)) != expected
                or np.any(ids < 0) or np.any(ids >= len(rows))):
            raise ValueError(f"Expected the original {expected} unique {split} manifest indices")
        groups[split] = ids.astype(np.int64)
    if not set(groups["references"]) <= set(groups["small"]):
        raise ValueError("The 32 references must belong to the source719 cohort")
    av = availability(rows)
    for split, missing in (("small", 48), ("validation", 22), ("evaluation", 30), ("references", 5)):
        subset = av[groups[split]]
        if not subset.any(axis=1).all() or int((~subset.all(axis=1)).sum()) != missing:
            raise ValueError(f"Original missing-view cohort changed: {split}, expected {missing}")
    return rows, groups, av


def data_scope(h, rows, groups, av):
    result = []
    for split in ("small", "references", "validation", "evaluation"):
        ids = groups[split]
        y = np.array([rows[int(i)]["label"] for i in ids])
        for label in (-1, 0, 1, 2, 3):
            selected = ids if label == -1 else ids[y == label]
            a = av[selected]
            result.append(dict(split=split, label=label, class_name="all" if label == -1 else h.LABELS[label],
                n=len(selected), ships=len({str(rows[int(i)]["mmsi"]) for i in selected}),
                original_both=int(a.all(axis=1).sum()), original_missing=int((~a.all(axis=1)).sum()),
                missing_vv=int((~a[:, 0]).sum()), missing_vh=int((~a[:, 1]).sum()),
                restored_both_expected=len(selected), references_are_subset_of_small=split == "references"))
    return result


def prediction_path(arm, seed, split):
    if arm == "A":
        return ROOT.parent / "full_finetune_20261111" / f"selected_s{seed}" / f"{split}.npz"
    if arm == "B":
        return ROOT.parent / "crosspol_box_recovery_20261112" / f"{split}_s{seed}.npz"
    prefix = "frozen_restored" if arm == "C" else "selected"
    return ROOT / f"{prefix}_s{seed}" / f"{split}.npz"


def load_prediction(path, expected_ids, rows, key="score"):
    with np.load(path, allow_pickle=False) as data:
        ids, score, labels = data["ids"], data[key], data["label"]
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != len(ids)
                or not np.array_equal(np.sort(ids), np.sort(expected_ids))):
            raise ValueError(f"Wrong, duplicate or missing prediction IDs: {path}")
        if score.shape != (len(ids), 4) or not np.isfinite(score).all():
            raise ValueError(f"Expected finite scores with shape (N,4): {path}")
        if not np.array_equal(labels, [rows[int(i)]["label"] for i in ids]):
            raise ValueError(f"Prediction labels disagree with original manifest: {path}")
        order = {int(i): j for j, i in enumerate(ids)}
        return np.asarray(score[[order[int(i)] for i in expected_ids]], dtype=np.float64)


def load_scores(rows, groups):
    return {split: {arm: np.stack([
        load_prediction(prediction_path(arm, seed, split), groups[split], rows,
                        "recovered" if arm == "B" else "score") for seed in SEEDS])
        for arm in ARMS} for split in SPLITS}


def strata(rows, ids, av):
    a = av[ids]
    masks = dict(all=np.ones(len(ids), bool), original_both=a.all(axis=1),
                 original_missing=~a.all(axis=1), original_vv_only=a[:, 0] & ~a[:, 1],
                 original_vh_only=a[:, 1] & ~a[:, 0])
    for key in ("scene_id", "pass_group", "platform_orbit"):
        values = np.array([orbit(rows[int(i)]) if key == "platform_orbit" else str(rows[int(i)][key]) for i in ids])
        for value in sorted(set(values)):
            mask = values == value
            masks[f"{key}:{value}"] = mask
            for group in ("original_both", "original_missing"):
                masks[f"{key}:{value}|{group}"] = mask & masks[group]
    return {name: mask for name, mask in masks.items() if mask.any()}


def aggregate(records, keys, fields):
    grouped = {}
    for row in records:
        grouped.setdefault(tuple(row[k] for k in keys), []).append(row)
    result = []
    for identity, items in grouped.items():
        entry = dict(zip(keys, identity))
        entry.update({key: items[0][key] for key in ("n", "ships", "classes_present", "support") if key in items[0]})
        entry["seeds"] = len(items)
        for field in fields:
            values = [r[field] for r in items if r[field] is not None]
            entry[field + "_mean"] = float(np.mean(values)) if values else None
            entry[field + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            if field.endswith("_delta_pp"):
                entry[field + "_positive_seeds"] = sum(v > 0 for v in values)
        result.append(entry)
    return result


def statistics(h, rows, groups, av, scores, replicates, bootstrap_seed):
    metric_rows, class_rows, paired, class_pairs, intervals = [], [], [], [], []
    for split, arms in scores.items():
        ids = groups[split]
        y = np.array([rows[int(i)]["label"] for i in ids], dtype=np.int64)
        ships = np.array([str(rows[int(i)]["mmsi"]) for i in ids])
        predictions = {arm: z.argmax(-1) for arm, z in arms.items()}
        for subset, mask in strata(rows, ids, av).items():
            truth = y[mask]
            meta = dict(split=split, subset=subset, n=int(mask.sum()),
                        ships=len(set(ships[mask])), classes_present=len(set(truth)))
            cms = {}
            for arm, pred in predictions.items():
                cms[arm] = []
                for k, seed in enumerate(SEEDS):
                    cm = h.confusion(truth, pred[k, mask])
                    cms[arm].append(cm)
                    metric_rows.append(dict(arm=arm, seed=seed, **meta, **h.metrics(truth, pred[k, mask])))
                    for c in range(4):
                        support, predicted, correct = int(cm[c].sum()), int(cm[:, c].sum()), int(cm[c, c])
                        class_rows.append(dict(arm=arm, seed=seed, **meta, label=c, class_name=h.LABELS[c],
                            support=support, predicted=predicted, correct=correct,
                            recall=100 * correct / support if support else None,
                            precision=100 * correct / predicted if predicted else None,
                            f1=200 * correct / (support + predicted) if support + predicted else 0.0))
            for candidate, baseline in COMPARISONS:
                cp, bp = predictions[candidate][:, mask], predictions[baseline][:, mask]
                states = h.outcome_masks(cp, bp, truth)
                for k, seed in enumerate(SEEDS):
                    after, before = h.metrics(truth, cp[k]), h.metrics(truth, bp[k])
                    paired.append(dict(candidate=candidate, baseline=baseline, seed=seed, **meta,
                        **{m + "_delta_pp": after[m] - before[m] for m in METRICS},
                        **{kind: int(states[kind][k].sum()) for kind in KINDS},
                        wrong_to_wrong_flips=int(((cp[k] != bp[k]) & states["stable_wrong"][k]).sum())))
                    ca, cb = cms[candidate][k], cms[baseline][k]
                    for c in range(4):
                        support = int(ca[c].sum())
                        denom_a, denom_b = support + int(ca[:, c].sum()), support + int(cb[:, c].sum())
                        fa = 200 * float(ca[c, c]) / denom_a if denom_a else 0.0
                        fb = 200 * float(cb[c, c]) / denom_b if denom_b else 0.0
                        class_pairs.append(dict(candidate=candidate, baseline=baseline, seed=seed, **meta,
                            label=c, class_name=h.LABELS[c], support=support, f1_delta_pp=fa - fb,
                            recall_delta_pp=100 * float(ca[c, c] - cb[c, c]) / support if support else None))
                if candidate == "D" and subset in GROUPS:
                    intervals.extend({"candidate": candidate, "baseline": baseline, **meta, **v}
                        for v in h.cluster_intervals(truth, ships[mask], cp, bp, replicates, bootstrap_seed))
    return dict(metrics=metric_rows,
        summary=aggregate(metric_rows, ("arm", "split", "subset"), METRICS),
        per_class=class_rows,
        per_class_summary=aggregate(class_rows, ("arm", "split", "subset", "label", "class_name"), ("recall", "precision", "f1")),
        paired_by_seed=paired,
        paired_summary=aggregate(paired, ("candidate", "baseline", "split", "subset"), tuple(m + "_delta_pp" for m in METRICS)),
        per_class_paired=class_pairs,
        per_class_paired_summary=aggregate(class_pairs, ("candidate", "baseline", "split", "subset", "label", "class_name"), ("recall_delta_pp", "f1_delta_pp")),
        paired_ship_ci=intervals,
        per_scene=[r for r in metric_rows if r["subset"].startswith(("scene_id:", "pass_group:", "platform_orbit:"))],
        per_original_availability=[r for r in metric_rows if r["subset"] in GROUPS[1:]])


def budget_tables(h):
    records, summary = [], []
    for arm in ("A", "D"):
        for seed in SEEDS:
            path = prediction_path(arm, seed, "evaluation").parent / "receipt.json"
            r = h.read_json(path) if path.is_file() else None
            complete = bool(r and r.get("stop_reason") in ("validation_patience", "max_epochs"))
            records.append(dict(arm=arm, seed=seed, receipt_available=r is not None, completed=complete,
                epochs=r.get("epochs") if r else None, steps=r.get("steps") if r else None,
                query_exposures=719 * r["epochs"] if r else None,
                best_epoch=r["best"]["epoch"] if r else None, seconds=r.get("seconds") if r else None,
                trainable=r.get("trainable") if r else None, peak_gpu_gib=r.get("peak_gib") if r else None))
        items = [r for r in records if r["arm"] == arm]
        complete = all(r["completed"] for r in items)
        summary.append(dict(arm=arm, completed_receipts=sum(r["completed"] for r in items),
            expected_runs=3, complete=complete,
            **{key: sum(r[key] for r in items) if complete else None
               for key in ("epochs", "steps", "query_exposures", "seconds")},
            peak_gpu_gib=max(r["peak_gpu_gib"] for r in items) if complete else None))
    return records, summary


def choose_cases(states, y, ids, missing, count):
    """Round-robin outcomes, then balance classes and original availability."""
    selected, used = [], set()
    class_count, group_count, kind_class = Counter(), Counter(), Counter()
    while len(selected) < count:
        added = False
        for kind in KINDS:
            repeats = states[kind].sum(axis=0)
            pool = [int(j) for j in np.flatnonzero(repeats) if int(j) not in used]
            if not pool:
                continue
            j = min(pool, key=lambda j: (class_count[int(y[j])], kind_class[kind, int(y[j])],
                group_count[bool(missing[j])], -int(repeats[j]), int(ids[j])))
            selected.append((kind, j))
            used.add(j)
            class_count[int(y[j])] += 1
            group_count[bool(missing[j])] += 1
            kind_class[kind, int(y[j])] += 1
            added = True
            if len(selected) == count:
                break
        if not added:
            break
    return selected


def legal_bank(rows, groups, av, index, arm):
    query = rows[index]
    qav = av[index] if arm == "A" else np.ones(2, bool)
    result = []
    for idx in map(int, groups["references"]):
        ref = rows[idx]
        rav = av[idx] if arm in ("A", "B") else np.ones(2, bool)
        shared = qav & rav
        legal = str(ref["mmsi"]) != str(query["mmsi"]) and orbit(ref) != orbit(query)
        result.append(dict(manifest_index=idx, id=ref["id"], label=int(ref["label"]),
            legal_identity=legal, active=bool(legal and shared.any()),
            shared_polarizations=[p for p, ok in zip(("vv", "vh"), shared) if ok],
            original_missing_polarizations=[p for p, ok in zip(("vv", "vh"), av[idx]) if not ok]))
    return result


def make_gallery(h, recovery, native, out, rows, groups, av, scores, count, include_a):
    from PIL import Image
    ids = groups["evaluation"]
    y = np.array([rows[int(i)]["label"] for i in ids])
    missing = ~av[ids].all(axis=1)
    predictions = {arm: z.argmax(-1) for arm, z in scores.items()}
    restored = {}

    def restore(index):
        if index not in restored:
            restored[index] = recovery.recover(rows[index])
        return restored[index]

    # The same recovered fixed bank is supplied to C and D, including its five restored members.
    for index in map(int, groups["references"]):
        restore(index)
    cases, coverage = [], []
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    for baseline, target in [("C", count)] + ([("A", 4)] if include_a else []):
        states = h.outcome_masks(predictions["D"], predictions[baseline], y)
        chosen = choose_cases(states, y, ids, missing, target)
        for kind in KINDS:
            for c in (-1, 0, 1, 2, 3):
                mask = np.ones(len(ids), bool) if c == -1 else y == c
                state = states[kind][:, mask]
                coverage.append(dict(candidate="D", baseline=baseline, kind=kind, label=c,
                    class_name="all" if c == -1 else h.LABELS[c], seed_observation_events=int(state.sum()),
                    observations=int(state.any(axis=0).sum()),
                    panels=sum(t == kind and (c == -1 or y[j] == c) for t, j in chosen)))
        for number, (kind, j) in enumerate(chosen, 1):
            index, label = int(ids[j]), int(y[j])
            k = int(np.flatnonzero(states[kind][:, j])[0])
            query, audit = restore(index)
            rival = int(predictions[baseline][k, j] if kind == "rescued" else predictions["D"][k, j])
            if rival == label:
                z = scores["D"][k, j].copy()
                z[label] = -np.inf
                rival = int(z.argmax())
            banks = {arm: legal_bank(rows, groups, av, index, arm) for arm in ARMS}
            displayed = []
            for role, c in (("Same-class ref", label), ("Confusion ref", rival)):
                eligible = [v["manifest_index"] for v in banks["D"] if v["active"] and v["label"] == c]
                # Show changed support evidence when present, never infer contribution rank.
                if eligible:
                    ref = min(eligible, key=lambda i: (bool(av[i].all()), list(groups["references"]).index(i)))
                    displayed.append((role, ref))
            name = f"D_vs_{baseline}_{number:02d}_{kind}.png"
            evidence = []
            for kk, seed in enumerate(SEEDS):
                evidence.append(dict(seed=seed, state=next(t for t in KINDS if states[t][kk, j]),
                    predictions={a: int(predictions[a][kk, j]) for a in ARMS},
                    scores={a: scores[a][kk, j].tolist() for a in ARMS},
                    true_margin={a: float(scores[a][kk, j, label] - np.delete(scores[a][kk, j], label).max()) for a in ARMS},
                    true_minus_displayed_rival={a: float(scores[a][kk, j, label] - scores[a][kk, j, rival]) for a in ARMS}))
            caption = f"D vs {baseline} | {kind} | s{SEEDS[k]} | truth={label} D={predictions['D'][k,j]} {baseline}={predictions[baseline][k,j]}"
            references = [(role, restore(i)[0]) for role, i in displayed]
            buffer = io.BytesIO()
            buffer.name = "restored.png"
            paths = h.draw_panel(native, query, references, buffer, caption + " | restored inputs")
            buffer.seek(0)
            with Image.open(buffer) as image:
                panel = image.copy()
            original_paths = None
            if baseline == "A":
                active_a = {r["manifest_index"] for r in banks["A"] if r["active"]}
                oldrefs = [(role, rows[i]) for role, i in displayed if i in active_a]
                before = io.BytesIO()
                before.name = "original.png"
                original_paths = h.draw_panel(native, rows[index], oldrefs, before, "A original inputs | " + caption)
                before.seek(0)
                with Image.open(before) as image:
                    combined = Image.new("RGB", (max(image.width, panel.width), image.height + panel.height), "white")
                    combined.paste(image, (0, 0))
                    combined.paste(panel, (0, image.height))
                panel = combined
            panel.save(folder / name)
            cases.append(dict(candidate="D", baseline=baseline, kind=kind, manifest_index=index,
                id=query["id"], label=label, rival=rival, mmsi=str(query["mmsi"]),
                scene_id=query["scene_id"], pass_group=query["pass_group"],
                seed=SEEDS[k], repeats=int(states[kind][:, j].sum()),
                original_available={p: bool(ok) for p, ok in zip(("vv", "vh"), av[index])},
                recovery_audit=audit, file=f"cases/{name}", evidence=evidence,
                actual_reference_bank_by_arm=banks,
                displayed_references=[dict(role=role, manifest_index=i, id=rows[i]["id"],
                    recovery_audit=restore(i)[1]) for role, i in displayed],
                restored_input_paths=paths, original_A_input_paths=original_paths,
                input_comparison="C and D use identical restored query/reference inputs" if baseline == "C"
                    else "Top: original A inputs; bottom: restored D inputs. Weights/training also differ.",
                review_status="pending_main_visual_review"))
    private = dict(cases=cases, reference_bank=[dict(manifest_index=i, id=rows[i]["id"],
        restored_row=restore(i)[0], recovery_audit=restore(i)[1]) for i in map(int, groups["references"])],
        prediction_sources=[dict(arm=a, split=s, seed=seed, path=str(prediction_path(a, seed, s)))
                            for a in ARMS for s in SPLITS for seed in SEEDS],
        sampling="Deterministic outcome round-robin; balance classes/availability, prefer repeated outcomes, then manifest ID; first matching seed.",
        reference_sampling="First legal example in bank order, prioritizing originally missing support within each requested class; not contribution ranking.")
    h.write_json(out / "cases_local.json", private)
    h.write_csv(out / "case_coverage.csv", coverage)
    return cases, coverage


def notes():
    return """# Analysis interpretation and interfaces

Run `python -B analyze.py` after all three seeds have both validation/evaluation
NPZs for A/B/C/D. Defaults: 2000 paired ship bootstrap draws, seed 20261112,
12 D-versus-C panels. `--panels 8` requests eight; `--include-d-vs-a` adds four
paired before/after panels. `--out PATH` changes the default ROOT/analysis output.
No training, inference, waiting, status-file access, formal edits, or Git actions.

A: original full fine-tuning, original inference inputs. B: A weights, recovered
queries only. C: A weights, recovered queries AND fixed32 references. D: matched
full/1e-4 fine-tuning from the SAME staged Full seed initialization as A, same
loss/RNG/schedule, with recovered training/reference/query inputs. D does not
continue A's fine-tuned checkpoint. These are the main trainer's protocol;
prediction arrays alone cannot establish initialization or RNG matching.

D-C isolates training/validation input organization under the same restored
inference inputs, subject to the matched recipe. Validation-selected epochs and
early stopping may differ; this is not a fixed-step optimization ablation.
D-A compares the complete process, including changed inference inputs. C-B
describes fixed-weight reference recovery; B-A describes query-only recovery.
Recovery can change pixels, masks, availability, geometry and reference relations.
The original absent view is an omitted input branch, not a zero-intensity SAR view.

data_scope.csv retains original missingness by split/class/polarization:
source719 / validation228 / test378 / fixed32 references; missing48/22/30/5.
References are a source subset, not 32 extra independent training observations.
The test original-missing30 group is a protocol-defined group, not all scene
ships, a detector recall denominator, or a causal missing-view experiment.
Class coverage differences offer explanations to investigate, not causal proof.
Scene/pass/orbit tables concern only the included manifest observations; they
cannot establish full-scene, unseen-port or all-vessel performance.

Metrics are percentages, differences percentage points. BA averages truth-present
classes; macro-F1 uses all four labels and zero-division=0. Absent-class recall
and precision with no predicted positives are blank. SD is sample SD over three
seeds; seeds share data and are not independent scenes. Bootstrap resamples MMSI
clusters with all observations retained, pairs arms and the same fixed three
seeds in each draw, then averages seed deltas. BA draws losing an originally
present class are excluded and valid counts recorded. Intervals condition on
these trained runs and scenes, excluding model-selection/new-scene uncertainty.
No hypothesis tests or p-values. Validation is not independent confirmation;
the historical evaluation cohort is not a fresh blind test.

Gallery panels reuse hstats.draw_panel and sar_m123.input.prepare_view: VV/VH
are separate 160x160 native crop/pad views with scale1, no resize or enhancement.
C/D show identical recovered queries and actual recovered fixed32-bank members.
Legality uses different MMSI and platform/orbit; shared usable polarization is
recorded separately. Supports are examples, not highest-contribution claims.
All32 restored rows/audits, per-arm legal bank, selected IDs/paths, crop placement,
per-seed scores/margins and original availability are in cases_local.json.
If a rescue/harm type is absent, case_coverage.csv records zero; stable examples
fill the requested count. Class balance is best effort within available outcomes.
Stable-wrong includes wrong-to-wrong flips, separately counted in paired tables.
Cases are illustrative, not prevalence estimates; score scales can differ across
trained weights. No physical causes are inferred automatically. Main must inspect
the panels, verify actual evidence and write findings after training completes.

Training budget tables read selected A/D receipts when available; absent receipts
remain explicitly incomplete. They count three selected continuations per arm,
not inherited staged training, A's unselected scouts, B/C inference, or preparation.
Query exposures=719*epochs are repeated presentations, not independent examples;
reference passes are excluded. Epoch0 selection is valid. Matched schedules need
not produce equal realized epochs/cost; these are recorded rather than assumed.

Aggregate CSVs, RESULTS.md, summary.json and these notes contain no raw chip paths
or per-record mappings. Keep cases_local.json, index.html and cases/ PRIVATE and
local; do not publish raw galleries, NPZs, weights or source mappings. No exports
are made automatically by this script. analysis_receipt.json marks visual review
pending, never completed research or verified training provenance.
"""


def write_reports(h, out, tables, cases, coverage, args):
    lines = ["# Consistent cross-polarization recovery", "",
        "Computed from frozen predictions. Main visual review and findings pending.", ""]
    lines += [f"- {arm}: {description}." for arm, description in ARMS.items()]
    lines += ["", "Primary D-C: matched training/validation organization under identical restored inference inputs.",
        "D-A: whole-process comparison; both training and inference inputs change.",
        "Original test missing30 is a protocol group; no full-scene claims.", "",
        "| Split | Arm | OA mean +/- SD | BA mean +/- SD | Macro-F1 mean +/- SD |",
        "|---|---|---:|---:|---:|"]
    for row in tables["summary"]:
        if row["subset"] == "all":
            values = " | ".join(f"{row[m+'_mean']:.3f} +/- {row[m+'_std']:.3f}" for m in METRICS)
            lines.append(f"| {row['split']} | {row['arm']} | {values} |")
    lines += ["", "| Test comparison | Metric | Delta pp | Paired MMSI 95% CI | Positive seeds |",
              "|---|---|---:|---|---:|"]
    for row in tables["paired_ship_ci"]:
        if row["split"] != "evaluation" or row["subset"] != "all":
            continue
        related = next(r for r in tables["paired_summary"] if r["split"] == row["split"]
            and r["subset"] == row["subset"] and r["candidate"] == row["candidate"] and r["baseline"] == row["baseline"])
        ci = "unavailable" if row["ci_low_pp"] is None else f"[{row['ci_low_pp']:+.3f}, {row['ci_high_pp']:+.3f}]"
        positive = related[row["metric"] + "_delta_pp_positive_seeds"]
        lines.append(f"| D-{row['baseline']} | {row['metric']} | {row['delta_pp']:+.3f} | {ci} | {positive}/3 |")
    lines += ["", "| Test original-input group | Arm | N | OA | BA | Macro-F1 |", "|---|---|---:|---:|---:|---:|"]
    for row in tables["summary"]:
        if row["split"] == "evaluation" and row["subset"] in ("original_both", "original_missing"):
            lines.append(f"| {row['subset']} | {row['arm']} | {row['n']} | {row['oa_mean']:.3f} | {row['ba_mean']:.3f} | {row['macro_f1_mean']:.3f} |")
    lines += ["", "Class recalls/F1, original availability and scene/pass/orbit denominators are in the CSV tables.",
        "data_scope.csv gives original missing VV/VH counts; class coverage is observational, not causal evidence.",
        "Selected A/D costs are in training_budget*.csv; these exclude scouts and inherited staged training.",
        "Sample SD is across three seeds. Ship intervals condition on existing runs/scenes and do not include selection uncertainty.",
        "Validation is descriptive; the evaluation cohort has been used historically. No hypothesis tests.",
        "See ANALYSIS_NOTES.md for metric definitions, design limits and interpretation.", "",
        f"Native panels: {len(cases)}. Main visual review pending. cases_local.json, index.html and cases/ remain PRIVATE."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "ANALYSIS_NOTES.md").write_text(notes(), encoding="utf-8")
    h.write_json(out / "summary.json", dict(arms=ARMS, metrics=tables["summary"],
        paired=tables["paired_summary"], intervals=tables["paired_ship_ci"],
        data_scope=tables["data_scope"], budget=tables["training_budget_summary"], review="pending"))
    body = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Consistent recovery: private native gallery</title>',
        '<style>body{font:16px/1.5 system-ui;margin:24px}section{border-top:1px solid #aaa;padding:16px 0}',
        '.panel{overflow-x:auto}img{max-width:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>',
        '<h1>Private native gallery: visual review pending</h1>',
        '<p>D-C uses identical restored query and reference inputs; only the trained weights differ. ',
        'A uses original query/reference inputs. Optional D-A panels show A above D.</p>',
        '<p>Separate VV/VH at 160x160 native crop/pad, scale1, no resize or display enhancement. ',
        'Reference examples belong to the actual legal fixed32 source bank and are restored for C/D. ',
        'They are not claimed to have the largest model contribution.</p>',
        '<p>Original test missing30 is a protocol group. No full-scene or causal claims. ',
        'Class-balanced examples are illustrative, not population proportions. No physical causes inferred.</p>']
    for row in coverage:
        if row["label"] == -1 and not row["observations"]:
            body.append(f"<p>D-{row['baseline']} / {row['kind']}: no observed cases.</p>")
    for case in cases:
        body += [f"<section><h2>D-{case['baseline']} / {case['kind']} / {html.escape(case['id'])}</h2>",
            f"<p>Truth: {html.escape(h.LABELS[case['label']])}; seed {case['seed']}; same outcome {case['repeats']}/3 seeds.</p>",
            f"<p>{html.escape(case['input_comparison'])} Original availability: {html.escape(str(case['original_available']))}.</p>",
            f'<div class="panel"><img src="{html.escape(case["file"])}" alt="Native query and legal reference VV/VH chips"></div>',
            '<details><summary>All-seed predictions and class margins</summary><pre>',
            html.escape(json.dumps(case["evidence"], indent=2)), '</pre></details></section>']
    (out / "index.html").write_text("\n".join(body) + "</body></html>\n", encoding="utf-8")
    h.write_json(out / "analysis_receipt.json", dict(status="analysis_complete_visual_review_pending",
        seeds=list(SEEDS), bootstrap_replicates=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        panels=len(cases), new_training=False, new_inference=False, formal_edits=False,
        training_provenance="Main trainer protocol; not independently verified by prediction analysis"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261112)
    parser.add_argument("--panels", type=int, default=12, choices=range(8, 13))
    parser.add_argument("--include-d-vs-a", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "analysis")
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap replicates")
    h, recovery, native = helpers()
    rows, groups, av = read_inputs(h)
    # Read every prediction before creating any outputs; partial trainer outputs fail here.
    scores = load_scores(rows, groups)
    tables = statistics(h, rows, groups, av, scores, args.bootstrap, args.bootstrap_seed)
    tables["data_scope"] = data_scope(h, rows, groups, av)
    tables["training_budget"], tables["training_budget_summary"] = budget_tables(h)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in tables.items():
        h.write_csv(args.out / f"{name}.csv", records)
    cases, coverage = make_gallery(h, recovery, native, args.out, rows, groups, av,
                                  scores["evaluation"], args.panels, args.include_d_vs_a)
    write_reports(h, args.out, tables, cases, coverage, args)
    print(f"Analysis complete: {args.out / 'RESULTS.md'}; {len(cases)} native panels await main visual review.")


if __name__ == "__main__":
    main()
