"""CPU analysis after ALL predictions, selections and training receipts are ready.

Reuses the preceding source-fit analysis' metrics, ID join and native panels.
Run: python -X utf8 -B analysis.py
No model/runner imports, training, parameter selection or prediction polling.
"""
import argparse
from collections import Counter
import html
import importlib.util
import itertools
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "spatial_m12_refit_20261029/analysis.py"


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = module_at("spatial_sourcefit_cpu", PREVIOUS)
read_json, dump, write_csv = base.read_json, base.dump, base.write_csv
metrics, per_sample, moments = base.metrics, base.per_sample, base.moments
CLASSES, SEEDS, ARMS, METRICS, STATES = base.CLASSES, base.SEEDS, base.ARMS, base.METRICS, base.STATES
CONDITIONS = ("legacy_before", "legacy_after", "spatial_before", "spatial_after")
COMPARISONS = {
    "spatial_after_minus_legacy_after": ("legacy_after", "spatial_after"),
    "legacy_after_minus_before": ("legacy_before", "legacy_after"),
    "spatial_after_minus_before": ("spatial_before", "spatial_after"),
}
SCOPES = ("development", "expanded")
SCOPE_NOTE = (
    "Full source-selected adaptation: M1/M2 explicit frequency branches removed in spatial; M3 retained. "
    "Each before endpoint is its own architecture/order-seed source fit; after is that fit plus its own "
    "source136 intervention selection and adaptation (planned 6 epochs/54 steps). Same source-fit1205 data; "
    "source-adapt136 is separate from fit. Fixed 16 references. Only layer3.5.conv1/2/3 are trained in "
    "adaptation; all remaining parameters including M1/M2/M3 heads are frozen but participate in differentiable computation. "
    "Each teacher independently selects interventions, so the final comparison includes selection differences "
    "and does NOT isolate their contribution from architecture or adaptation. "
    "Seeds 17/29/43 vary BOTH source-fit and adaptation data order with the same initialization, NOT independent initializations. "
    "Development160 is reused development, NOT a new blind test; expanded3075 is also a historical cohort. "
    "No target-based parameter selection. OA/BA are percentages and deltas percentage points. "
    "BA averages present classes; absent classes are blank, not zero. NLL uses natural logs and Brier "
    "sums over all eight classes. SD is sample SD across the three order seeds. "
    "Repeated seed events, ships, passes and reference exposures are not independent samples. "
    "Paired outcome changes and selected image cases do not establish a physical mechanism."
)


def load_inputs(root):
    paths = {(scope, condition, seed): root / "predictions" / f"{scope}_{stage}_{arm}_s{seed}.npz"
             for scope in SCOPES for arm in ARMS for stage in ("before", "after")
             for condition in (f"{arm}_{stage}",) for seed in SEEDS}
    selections = {(arm, seed): root / "selection" / f"{arm}_s{seed}.json" for arm in ARMS for seed in SEEDS}
    receipts = {(arm, seed): root / "training" / f"{arm}_s{seed}/training.json" for arm in ARMS for seed in SEEDS}
    missing = [str(p) for p in [root / "cohorts.json", *paths.values(), *selections.values(), *receipts.values()]
               if not p.is_file()]
    if missing:
        raise SystemExit("Incomplete inputs; no formal analysis started:\n" + "\n".join(missing))
    cohorts = read_json(root / "cohorts.json")
    cohorts = {key: cohorts[key] for key in ("fit", "adapt", "development", "expanded", "references")}
    for key, expected in (("fit", 1205), ("adapt", 136), ("development", 160), ("expanded", 3075), ("references", 16)):
        rows = cohorts[key]
        if len(rows) != expected or len({str(r["id"]) for r in rows}) != expected:
            raise ValueError(f"{key}: expected {expected} unique observations")
        for row in rows:
            row["id"] = str(row["id"])
            if row["label"] not in range(8):
                raise ValueError(f"{key}/{row['id']}: label outside 8-class taxonomy")
    scores = {key: base.load_score(path, cohorts[key[0]]) for key, path in paths.items()}
    selection = {key: read_json(path) for key, path in selections.items()}
    training = {f"{arm}_s{seed}": dict(path=str(path), receipt=read_json(path))
                for (arm, seed), path in receipts.items()}
    return cohorts, scores, selection, training


def annotate_available(cohorts, native):
    # Use the actual native input rule, including alignment and rasterized target,
    # instead of treating a supplied image path as an available polarization.
    cache = {}
    for rows in cohorts.values():
        for row in rows:
            flags = []
            for pol in ("vv", "vh"):
                key = (row["id"], pol)
                if key not in cache:
                    view = native.prepare_view(row.get(pol))
                    cache[key] = bool(view is not None and view["valid_target"])
                flags.append(cache[key])
            row["analysis_available"] = ("both" if all(flags) else "vv_only" if flags[0]
                                         else "vh_only" if flags[1] else "neither")


def group_masks(rows):
    for field in ("label", "port", "pass_group", "available"):
        values = np.array([str(r["analysis_available"] if field == "available" else r[field]) for r in rows])
        groups = list(map(str, range(8))) if field == "label" else sorted(set(values))
        for group in groups:
            yield field, group, values == group


def cohort_summary(cohorts, out):
    groups, pairs, overlaps = [], [], []
    refs = cohorts["references"]
    for split, rows in cohorts.items():
        for field, group, mask in [("all", "all", np.ones(len(rows), bool)), *group_masks(rows)]:
            selected = [r for r, keep in zip(rows, mask) if keep]
            groups.append(dict(split=split, dimension=field, group=group, n=len(selected),
                ships=len({r["mmsi"] for r in selected if r.get("mmsi")}),
                passes=len({r["pass_group"] for r in selected}), ports=len({r["port"] for r in selected})))
        if split != "references":
            for q in rows:
                eligible = [r for r in refs if base.eligible(q, r)]
                pairs.append(dict(split=split, id=q["id"], label=q["label"], available=q["analysis_available"],
                    eligible_n=len(eligible), eligible_ids=";".join(r["id"] for r in eligible),
                    true_ids=";".join(r["id"] for r in eligible if r["label"] == q["label"])))
    for left, right in itertools.combinations(cohorts, 2):
        for field in ("id", "mmsi", "port", "pass_group"):
            a = {str(r[field]) for r in cohorts[left] if r.get(field) not in (None, "")}
            b = {str(r[field]) for r in cohorts[right] if r.get(field) not in (None, "")}
            shared = a & b
            overlaps.append(dict(left=left, right=right, dimension=field, shared_values=len(shared),
                left_observations=sum(str(r.get(field)) in shared for r in cohorts[left]),
                right_observations=sum(str(r.get(field)) in shared for r in cohorts[right])))
    write_csv(out / "cohort_groups.csv", groups)
    write_csv(out / "cohort_overlap.csv", overlaps)
    write_csv(out / "reference_pairs.csv", pairs)
    return dict(cohorts=[r for r in groups if r["dimension"] == "all"], overlaps=overlaps,
                availability_rule="input.prepare_view: view is not None and valid_target; cached per observation/polarization",
                reference_rule="fixed supplied bank after native same-ID/MMSI exclusion; internal forward use requires runner receipt")


def pair_events(rows, scores, old_name, new_name):
    y = np.array([r["label"] for r in rows], dtype=int)
    # One source-fit rival per query, fixed across endpoints and order seeds.
    fixed = np.mean([scores["legacy_before", seed] for seed in SEEDS], axis=0)
    fixed[np.arange(len(y)), y] = -np.inf
    rivals = fixed.argmax(1)
    events = []
    for seed in SEEDS:
        old, new = scores[old_name, seed], scores[new_name, seed]
        ov, nv = per_sample(old, y), per_sample(new, y)
        for i, row in enumerate(rows):
            truth, rival = int(y[i]), int(rivals[i])
            event = dict(id=row["id"], seed=seed, label=truth, class_name=CLASSES[truth],
                port=row["port"], pass_group=row["pass_group"], mmsi=row.get("mmsi"),
                available=row["analysis_available"], state=base.state(ov["prediction"][i], nv["prediction"][i], truth),
                reference_prediction=int(ov["prediction"][i]), alternative_prediction=int(nv["prediction"][i]),
                fixed_rival=rival)
            for prefix, score, value in (("reference", old, ov), ("alternative", new, nv)):
                event[prefix + "_fixed_margin"] = float(score[i, truth] - score[i, rival])
                event[prefix + "_top_rival_margin"] = float(score[i, truth] - np.delete(score[i], truth).max())
                for metric in ("NLL", "Brier"):
                    event[prefix + "_" + metric] = float(value[metric][i])
            event["delta_fixed_margin"] = event["alternative_fixed_margin"] - event["reference_fixed_margin"]
            events.append(event)
    return events


def summarize_comparison(rows, scores, old_name, new_name, events):
    y = np.array([r["label"] for r in rows], dtype=int)
    detailed, aggregate = [], []
    for field, group, mask in [("all", "all", np.ones(len(y), bool)), *group_masks(rows)]:
        ids = {r["id"] for r, keep in zip(rows, mask) if keep}
        ev = [e for e in events if e["id"] in ids]
        identity = dict(dimension=field, group=group, n=int(mask.sum()),
                        present_classes=len(np.unique(y[mask])))
        seed_rows = []
        for seed in SEEDS:
            old, new = (metrics(scores[name, seed][mask], y[mask]) for name in (old_name, new_name))
            item = dict(**identity, seed=seed)
            for metric in METRICS:
                item.update({"reference_" + metric: old[metric], "alternative_" + metric: new[metric],
                             "delta_" + metric: new[metric] - old[metric] if mask.any() else None})
            item.update({state: sum(e["seed"] == seed and e["state"] == state for e in ev) for state in STATES})
            item["prediction_changes"] = sum(e["seed"] == seed and e["reference_prediction"] != e["alternative_prediction"] for e in ev)
            seed_rows.append(item)
        total = dict(**identity)
        for key in [f"{p}_{m}" for p in ("reference", "alternative", "delta") for m in METRICS] + list(STATES) + ["prediction_changes"]:
            stats = moments([r[key] for r in seed_rows])
            total[key + "_mean"], total[key + "_sd"] = stats["mean"], stats["sd"]
        transitions = base.transition_summary(ev)
        for state in STATES:
            total[state + "_seed_events"] = transitions[state]["seed_events"]
            total[state + "_unique_ids"] = transitions[state]["unique_ids"]
        total["prediction_changes_unique_ids"] = len({e["id"] for e in ev if e["reference_prediction"] != e["alternative_prediction"]})
        total["rescued_and_harmed_ids"] = ";".join(transitions["rescued_and_harmed_ids"])
        detailed.extend(seed_rows)
        aggregate.append(total)
    return detailed, aggregate


def pass_bootstrap(rows, scores, old_name, new_name, repeats=2000, seed=20261029):
    y = np.array([r["label"] for r in rows], dtype=int)
    passes, inverse = np.unique([str(r["pass_group"]) for r in rows], return_inverse=True)
    differences = []
    for order_seed in SEEDS:
        old, new = (per_sample(scores[name, order_seed], y) for name in (old_name, new_name))
        differences.append(np.column_stack([
            (new["prediction"] == y).astype(float) - (old["prediction"] == y),
            new["NLL"] - old["NLL"], new["Brier"] - old["Brier"]]))
    delta = np.mean(differences, axis=0)
    totals = np.zeros((len(passes), 3))
    counts = np.zeros((len(passes), 8))
    correct_delta = np.zeros_like(counts)
    np.add.at(totals, inverse, delta)
    np.add.at(counts, (inverse, y), 1)
    np.add.at(correct_delta, (inverse, y), delta[:, 0])
    rng = np.random.default_rng(seed)
    values = []
    missing_class_replicates = 0
    present = np.unique(y)
    for _ in range(repeats):
        draw = rng.integers(0, len(passes), len(passes))
        n = counts[draw].sum(0)
        numerator = totals[draw].sum(0)
        observed = n > 0
        missing_class_replicates += int(not observed[present].all())
        values.append([100 * numerator[0] / n.sum(),
                       100 * np.mean(correct_delta[draw].sum(0)[observed] / n[observed]),
                       numerator[1] / n.sum(), numerator[2] / n.sum()])
    values = np.asarray(values)
    point = {m: float(np.mean([metrics(scores[new_name, s], y)[m] - metrics(scores[old_name, s], y)[m] for s in SEEDS])) for m in METRICS}
    return dict(method="Paired pass-cluster percentile bootstrap; resample whole passes equally with replacement, keep all observations and paired fixed model seeds; then average seed deltas",
        interpretation="Conditional cohort uncertainty for these six fixed fitted models; NOT independent-training confidence, NOT a fresh-test claim",
        repeats=repeats, bootstrap_seed=seed, pass_count=len(passes), missing_class_replicates=missing_class_replicates,
        ba_rule="Average classes present in each resample; absent-class replicates explicitly counted",
        metrics={m: dict(point=point[m], low=float(np.quantile(values[:, i], .025)), high=float(np.quantile(values[:, i], .975))) for i, m in enumerate(METRICS)})


def selection_analysis(cohorts, selections, out):
    rows = {r["id"]: r for r in cohorts["adapt"]}
    records = []
    for (arm, seed), items in selections.items():
        if not isinstance(items, list) or len(items) != len(rows) or {str(e["id"]) for e in items} != set(rows):
            raise ValueError(f"selection/{arm}_s{seed}: expected one entry per adapt ID")
        for item in items:
            row = rows[str(item["id"])]
            truth = row["label"]
            if item.get("label", truth) != truth:
                raise ValueError(f"selection label mismatch: {row['id']}")
            choice, intervention = item["choice"], item["selected_intervention"]
            original, altered = (np.asarray(item[k], float) for k in ("original", "altered"))
            if original.shape != (8,) or altered.shape != (8,) or not np.isfinite([original, altered]).all():
                raise ValueError(f"selection score shape/nonfinite: {row['id']}")
            om, am = metrics(original[None], np.array([truth])), metrics(altered[None], np.array([truth]))
            records.append(dict(arm=arm, seed=seed, id=row["id"], label=truth, class_name=CLASSES[truth],
                port=row["port"], pass_group=row["pass_group"], mmsi=row.get("mmsi"), available=row["analysis_available"],
                useful=bool(item["useful"]), choice_useful=bool(choice["useful"]),
                original_prediction=int(original.argmax()), altered_prediction=int(altered.argmax()),
                delta_NLL=am["NLL"] - om["NLL"], delta_Brier=am["Brier"] - om["Brier"],
                delta_top_rival_margin=float(altered[truth] - np.delete(altered, truth).max() - original[truth] + np.delete(original, truth).max()),
                choice_rival=choice["rival"], structural_margin_gain=choice["structural_margin_gain"],
                chosen_index=choice["chosen_index"], control_index=choice["control_index"],
                **{k: intervention[k] for k in ("kind", "pol", "region", "band")}))
    grouped = []
    for arm in ARMS:
        for seed in SEEDS:
            pool = [r for r in records if r["arm"] == arm and r["seed"] == seed]
            for dimension in ("all", "label", "port", "pass_group", "available", "pol", "region", "band", "pol_region"):
                values = {str(r[dimension]) if dimension not in ("all", "pol_region") else
                          "all" if dimension == "all" else r["pol"] + "/" + r["region"] for r in pool}
                for value in sorted(values):
                    selected = [r for r in pool if (str(r[dimension]) if dimension not in ("all", "pol_region") else
                                "all" if dimension == "all" else r["pol"] + "/" + r["region"]) == value]
                    useful = [r for r in selected if r["useful"]]
                    grouped.append(dict(arm=arm, seed=seed, dimension=dimension, group=value, n=len(selected),
                        useful=len(useful), useful_fraction=len(useful) / len(selected),
                        choice_useful=sum(r["choice_useful"] for r in selected),
                        choice_rejected_by_final=sum(r["choice_useful"] and not r["useful"] for r in selected),
                        useful_ships=len({r["mmsi"] for r in useful if r.get("mmsi")})))
    intersections = []
    for seed in SEEDS:
        a, b = ({r["id"]: r for r in records if r["arm"] == arm and r["seed"] == seed and r["useful"]} for arm in ARMS)
        common, union = set(a) & set(b), set(a) | set(b)
        same = {i for i in common if all(a[i][k] == b[i][k] for k in ("kind", "pol", "region", "band"))}
        intersections.append(dict(seed=seed, legacy=len(a), spatial=len(b), intersection=len(common), union=len(union),
            jaccard=len(common) / len(union) if union else None, exact_intervention_intersection=len(same),
            intersection_ids=sorted(common), exact_intervention_ids=sorted(same),
            legacy_only_ids=sorted(set(a) - set(b)), spatial_only_ids=sorted(set(b) - set(a))))
    aggregated = []
    keys = sorted({(r["arm"], r["dimension"], r["group"]) for r in grouped})
    for arm, dimension, group in keys:
        entries = [r for r in grouped if (r["arm"], r["dimension"], r["group"]) == (arm, dimension, group)]
        item = dict(arm=arm, dimension=dimension, group=group)
        for field in ("n", "useful", "choice_useful", "choice_rejected_by_final", "useful_ships"):
            # Intervention categories absent in a seed contribute zero counts.
            stats = moments([next((r[field] for r in entries if r["seed"] == seed), 0) for seed in SEEDS])
            item[field + "_mean"], item[field + "_sd"] = stats["mean"], stats["sd"]
        aggregated.append(item)
    write_csv(out / "selection_records.csv", records)
    write_csv(out / "selection_groups.csv", grouped)
    write_csv(out / "selection_groups_summary.csv", aggregated)
    summary = dict(intersections=intersections, count_summary=[r for r in aggregated if r["dimension"] == "all"],
        note="Final useful is used for selection intersections. choice_useful is the earlier screen; non-useful fallback interventions are not selected teachers. Same source observations repeated across seeds are repeated exposure, not independent evidence.",
        inputs={f"{arm}_s{seed}": len(items) for (arm, seed), items in selections.items()})
    dump(out / "selection_summary.json", summary)
    return summary


def select_cases(events, count=8):
    by_id = {}
    for event in events:
        by_id.setdefault(event["id"], []).append(event)
    pools = {key: [] for key in STATES}
    for query_id, ev in by_id.items():
        counts = Counter(e["state"] for e in ev)
        net = counts["rescued"] - counts["harmed"]
        category = "rescued" if net > 0 else "harmed" if net < 0 else "CC" if counts["CC"] == len(SEEDS) else "WW" if counts["WW"] == len(SEEDS) else None
        if category:
            event = max((e for e in ev if e["state"] == category), key=lambda e: (abs(e["delta_fixed_margin"]), -e["seed"]))
            pools[category].append(dict(event, selection_category=category, net_rescue_seed_events=net,
                                        state_seed_counts=dict(counts)))
    for pool in pools.values():
        pool.sort(key=lambda e: (-abs(e["net_rescue_seed_events"]), -abs(e["delta_fixed_margin"]), e["id"]))
    selected, used = [], set()
    for quota in (1, 2):
        for key in STATES:
            candidates = [e for e in pools[key] if e["id"] not in used]
            if quota == 2 and not any(e["available"] != "both" for e in selected):
                candidates.sort(key=lambda e: e["available"] == "both")
            if candidates and len(selected) < count:
                selected.append(candidates[0])
                used.add(candidates[0]["id"])
    for event in sorted(events, key=lambda e: (e["state"] not in ("rescued", "harmed"), -abs(e["delta_fixed_margin"]), e["id"], e["seed"])):
        if len(selected) == count:
            break
        if event["id"] not in used:
            selected.append(dict(event, selection_category="mixed_seed_fallback"))
            used.add(event["id"])
    return selected


def gallery(out, cohorts, scores, events, native):
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    rows, refs = cohorts["expanded"], cohorts["references"]
    indices = {r["id"]: (i, r) for i, r in enumerate(rows)}
    cases, sources, blocks = select_cases(events), [], []
    rule = "Expanded final comparison only. Prefer net rescued/harmed across order seeds and all-seed CC/WW, up to two distinct queries each; prioritize available partial-missing coverage in second slots. Rank by absolute net rescue count, then absolute fixed-rival margin change and ID; representative seed has largest absolute margin change within state. Fill unavailable categories by transition then margin. Outcome-selected, not representative prevalence."
    for number, case in enumerate(cases, 1):
        i, query = indices[case["id"]]
        truth, selected_seed = query["label"], case["seed"]
        rival_classes = {case["fixed_rival"]}
        for condition in ("legacy_after", "spatial_after"):
            values = scores[condition, selected_seed][i].copy()
            values[truth] = -np.inf
            rival_classes.add(int(values.argmax()))
        allowed = [r for r in refs if base.eligible(query, r)]
        own = [r for r in allowed if r["label"] == truth]
        rivals = [r for r in allowed if r["label"] in rival_classes]
        group, roles = [query, *own, *rivals], ["QUERY", *["TRUE REF"] * len(own), *["RIVAL REF"] * len(rivals)]
        case.update(scope="expanded", selection_rule=rule, visual_status="not_reviewed",
            rival_classes=sorted(rival_classes), eligible_reference_ids=[r["id"] for r in allowed],
            true_reference_ids=[r["id"] for r in own], rival_reference_ids=[r["id"] for r in rivals],
            panels={}, per_seed=[])
        for seed in SEEDS:
            detail = dict(seed=seed, conditions={})
            for condition in CONDITIONS:
                score = scores[condition, seed][i]
                detail["conditions"][condition] = dict(logits=score.tolist(), prediction=int(score.argmax()),
                    truth_rival_margins={str(rival): float(score[truth] - score[rival]) for rival in sorted(rival_classes)})
            case["per_seed"].append(detail)
        title = f"{number}. {case['selection_category']} / {case['id']} / seed {selected_seed} / {case['available']}"
        blocks.append("<section><h2>" + html.escape(title) + "</h2><p>Truth: " + CLASSES[truth] +
                      "; port: " + html.escape(str(query["port"])) + "; pass: " + html.escape(str(query["pass_group"])) + "</p>")
        blocks.append("<table><tr><th>Seed</th>" + "".join(f"<th>{c}</th>" for c in CONDITIONS) + "</tr>")
        for detail in case["per_seed"]:
            blocks.append(f"<tr><td>{detail['seed']}</td>" + "".join("<td>" + CLASSES[detail["conditions"][c]["prediction"]] + "</td>" for c in CONDITIONS) + "</tr>")
        blocks.append("</table><details><summary>Logits, fixed rivals and reference IDs</summary><pre>" +
                      html.escape(base.json.dumps({k: case[k] for k in ("eligible_reference_ids", "true_reference_ids", "rival_reference_ids", "per_seed")}, indent=2)) + "</pre></details>")
        for pol in ("vv", "vh"):
            filename = f"case_{number:02d}_{pol}.png"
            receipts = base.panel(group, roles, pol, folder / filename, native)
            for receipt in receipts:
                receipt.update(case_id=case["id"], selected_seed=selected_seed, panel=filename,
                               usable_with_query_polarization=bool(receipts[0]["available"] and receipt["available"]))
            sources.extend(receipts)
            case["panels"][pol] = str(folder / filename)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(title)}"></div>')
        blocks.append("</section>")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Full adaptation native cases</title>"
        "<style>body{font:14px Arial;margin:24px;color:#222}section{border-top:1px solid #aaa;margin-top:24px}"
        "h1{font-size:22px}h2{font-size:18px}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}"
        ".image{overflow:auto}img{max-width:none;image-rendering:pixelated}pre{white-space:pre-wrap}</style>"
        "<h1>Full adaptation: spatial versus legacy native cases</h1><p>" + html.escape(SCOPE_NOTE) + "</p><p>" + rule +
        "</p><p>One source pixel per image pixel; no resize/contrast stretch. Original 8-bit display chips are not raw radiometry. "
        "Each row shows original, exact model crop/pad and duplicate OBB overlay; VV/VH separate. Actual fixed-bank true/rival "
        "references after ID/MMSI exclusions. Generation is NOT visual review; physical causes remain unproven.</p>" + "".join(blocks) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    dump(folder / "cases.json", cases)
    dump(folder / "native_sources.json", sources)
    return cases


def analyze(root, package):
    cohorts, all_scores, selections, training = load_inputs(root)
    native = module_at("full_adaptation_native_input", package / "sar_m123/input.py")
    annotate_available(cohorts, native)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    coverage = cohort_summary(cohorts, out)
    selection = selection_analysis(cohorts, selections, out)
    summary = dict(scope=SCOPE_NOTE, coverage=coverage, selection=selection, training_receipts=training,
        rival_rule="Strongest nontruth class of mean legacy_before logits across order seeds, fixed for each query",
        alignment="24 prediction archives joined by unique ID and validated against cohort labels",
        scopes={}, visual_status="not_reviewed")
    all_metrics, all_groups, all_group_summary, all_events = [], [], [], []
    expanded_scores, expanded_main_events = None, None
    for scope in SCOPES:
        rows = cohorts[scope]
        y = np.array([r["label"] for r in rows], dtype=int)
        scores = {(c, s): all_scores[scope, c, s] for c in CONDITIONS for s in SEEDS}
        metric_rows = [dict(scope=scope, condition=c, seed=s, **metrics(scores[c, s], y)) for c in CONDITIONS for s in SEEDS]
        all_metrics.extend(metric_rows)
        result = dict(conditions={c: {m: moments([r[m] for r in metric_rows if r["condition"] == c]) for m in METRICS} for c in CONDITIONS}, comparisons={})
        for name, (old, new) in COMPARISONS.items():
            events = pair_events(rows, scores, old, new)
            detailed, aggregated = summarize_comparison(rows, scores, old, new, events)
            all_events.extend(dict(scope=scope, comparison=name, **e) for e in events)
            all_groups.extend(dict(scope=scope, comparison=name, **r) for r in detailed)
            all_group_summary.extend(dict(scope=scope, comparison=name, **r) for r in aggregated)
            result["comparisons"][name] = dict(reference=old, alternative=new, overall=aggregated[0], transitions=base.transition_summary(events))
            if scope == "expanded":
                result["comparisons"][name]["pass_bootstrap"] = pass_bootstrap(rows, scores, old, new)
                if name == "spatial_after_minus_legacy_after":
                    expanded_scores, expanded_main_events = scores, events
        summary["scopes"][scope] = result
    for name, records in (("metrics.csv", all_metrics), ("groups.csv", all_groups),
                          ("groups_summary.csv", all_group_summary), ("seed_events.csv", all_events)):
        write_csv(out / name, records)
    write_csv(out / "paired_metrics.csv", [r for r in all_groups if r["dimension"] == "all"])
    dump(out / "summary.json", summary)
    cases = gallery(out, cohorts, expanded_scores, expanded_main_events, native)
    lines = ["# Full adaptation CPU numeric results", "", SCOPE_NOTE, ""]
    for scope, result in summary["scopes"].items():
        lines += [f"## {scope}", "", "| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |", "|---|---:|---:|---:|---:|"]
        for c, stats in result["conditions"].items():
            lines.append(f"| {c} | " + " | ".join(f"{stats[m]['mean']:.6f} +/- {stats[m]['sd']:.6f}" for m in METRICS) + " |")
        lines += ["", "| Comparison | Metric | Paired delta mean | SD |", "|---|---|---:|---:|"]
        for name, comp in result["comparisons"].items():
            for m in METRICS:
                r = comp["overall"]
                lines.append(f"| {name} | {m} | {r['delta_' + m + '_mean']:.6f} | {r['delta_' + m + '_sd']:.6f} |")
        lines += ["", "| Comparison | State | Seed events | Unique queries |", "|---|---|---:|---:|"]
        for name, comp in result["comparisons"].items():
            for state in STATES:
                t = comp["transitions"][state]
                lines.append(f"| {name} | {state} | {t['seed_events']} | {t['unique_ids']} |")
        if scope == "expanded":
            lines += ["", "Pass-cluster 95% percentile intervals below condition on these fixed fitted models; they are not independent-training confidence intervals.", "", "| Comparison | Metric | Low | High |", "|---|---|---:|---:|"]
            for name, comp in result["comparisons"].items():
                for m, value in comp["pass_bootstrap"]["metrics"].items():
                    lines.append(f"| {name} | {m} | {value['low']:.6f} | {value['high']:.6f} |")
    lines += ["", "Class/port/pass/actual-available groups and paired flips are in groups.csv and groups_summary.csv. "
        "selection_summary.json contains final-useful counts and architecture intersections; selection_groups.csv distinguishes "
        "final useful from the earlier choice screen. Raw training receipts are embedded in summary.json; the CPU analysis "
        "does not certify frozen weights or internal reference usage.", "", "Fixed rivals come from mean legacy_before nontruth logits. "
        "Full per-case logits and all four condition outcomes are in cases/cases.json. "
        "[Native gallery](cases/index.html): eight unique expanded queries, sixteen separate VV/VH panels, "
        "actual eligible true/rival refs. Generated pixels have NOT been reviewed. Main-agent visual review and "
        "mechanism interpretation remain pending.", ""]
    (out / "NUMERIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    dump(out / "complete.json", dict(prediction_files=24, selection_files=6, training_receipts=6,
        gallery_unique_queries=len(cases), panels=2 * len(cases), device="CPU", visual_status="not_reviewed"))
    print(base.json.dumps(dict(output=str(out), gallery=str(out / "cases/index.html")), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--package", type=Path, default=base.PACKAGE)
    args = parser.parse_args()
    analyze(args.root.resolve(), args.package.resolve())
