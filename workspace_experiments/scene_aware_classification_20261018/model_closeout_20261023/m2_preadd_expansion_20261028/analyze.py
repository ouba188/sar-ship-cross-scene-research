"""CPU-only frozen C/preadd vs E/stage expansion; no model/GPU imports.

Reuse prior expansion metric, bootstrap, selection and native-image helpers.
Run after predictions.npz is ready. --gallery-only adds main diagnostic cases
without recomputing metrics. All outputs belong to analysis/ or NUMERIC_REPORT.
"""
import argparse
from collections import Counter
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "preadd_expansion_helpers", HERE.parent / "shallow_m2_expansion_20261028/analyze.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
h = p.helpers()
ARMS = ("baseline", "stage", "preadd")
SEEDS = p.SEEDS
METRICS = p.METRICS
STATES = p.STATES
PAIRS = (("preadd", "baseline"), ("preadd", "stage"), ("stage", "baseline"))
SCOPE = (
    "Frozen historical 3075 observations, 1950 ships, 85 passes, 18 ports, 8 classes. "
    "No fitting, inference, tuning or model selection in this CPU analysis. "
    "Baseline/E reuse six historical scores after main-owned matched replay; only C has three new inference runs. "
    "Metrics are computed per seed 17/29/43 then averaged; no probability ensemble. "
    "OA/BA are percentages and deltas percentage points; BA averages classes present in the group. "
    "Natural-log NLL and summed eight-class Brier are lower-is-better. SD uses ddof=1. "
    "Historical targets are not a new blind test; seeds are not independent datasets."
)


def comparison(arm, reference):
    return f"{arm}_vs_{reference}"


def write_table(out, name, rows):
    h.write_csv(out / f"{name}.csv", rows, list(rows[0]))


def load_inputs():
    records = p.read_records(HERE / "query_records.json")
    with np.load(HERE / "predictions.npz", allow_pickle=False) as pack:
        ids, labels = pack["ids"].astype(str), pack["labels"]
        scores = pack["scores"].astype(np.float64)
        arms, seeds = tuple(pack["arms"].astype(str)), tuple(int(x) for x in pack["seeds"])
    if arms != ARMS or seeds != SEEDS or scores.shape != (3, 3, 3075, 8):
        raise ValueError(f"Expected baseline/stage/preadd x 17/29/43 x 3075 x 8; got {arms}/{seeds}/{scores.shape}")
    if ids.shape != (3075,) or labels.shape != (3075,) or labels.dtype.kind not in "iu":
        raise ValueError("Expected 3075 aligned IDs and integer labels")
    if not np.isfinite(scores).all() or not np.isin(labels, np.arange(8)).all():
        raise ValueError("Incomplete scores or invalid eight-class labels")
    index = {r["id"]: r for r in records}
    if len(index) != 3075 or len(records) != 3075 or len(set(ids)) != 3075 or set(ids) != set(index):
        raise ValueError("Prediction IDs must match the frozen query manifest without duplicates")
    rows = [index[x] for x in ids]
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise ValueError("Prediction labels differ from the aligned query records")
    counts = tuple(len({r[key] for r in rows}) for key in ("mmsi", "pass_group", "port"))
    if counts != (1950, 85, 18):
        raise ValueError(f"Wrong frozen ship/pass/port coverage: {counts}")
    refs = p.read_records(p.PACKAGE / "manifests/references.json")
    if len(refs) != 16 or Counter(r["label"] for r in refs) != Counter({c: 2 for c in range(8)}):
        raise ValueError("Expected fixed sixteen references, two per class")
    return ids, labels, scores, rows, refs


def pair_values(scores, labels, arm, reference):
    return p.sample_values(scores[[ARMS.index(reference), ARMS.index(arm)]], labels, h)


def tables(ids, labels, scores, rows, out):
    absolute, paired, absolute_summary, paired_summary = [], [], [], []
    values = {comparison(a, b): pair_values(scores, labels, a, b) for a, b in PAIRS}
    groups = {
        "all": ["all"] * len(rows),
        "class": [f"{r['label']}:{h.CLASSES[r['label']]}" for r in rows],
        "port": [r["port"] for r in rows],
        "pass_group": [r["pass_group"] for r in rows],
        "polarization": [p.polarization(r) for r in rows],
        "port_scope": ["old_three_ports" if r["port"] in p.OLD_PORTS else "other_fifteen_ports" for r in rows],
    }
    for dimension, names in groups.items():
        names = np.asarray(names)
        for name in sorted(set(names)):
            ii = np.flatnonzero(names == name)
            info = dict(dimension=dimension, group=str(name), n=len(ii),
                        ships=len({rows[i]["mmsi"] for i in ii}),
                        passes=len({rows[i]["pass_group"] for i in ii}),
                        present_classes=len(np.unique(labels[ii])))
            for arm in ARMS:
                pair = values["preadd_vs_baseline"] if arm != "stage" else values["stage_vs_baseline"]
                ai = 0 if arm == "baseline" else 1
                seed_rows = [dict(**info, arm=arm, seed=seed, **p.metric_at(pair, labels, ii, ai, s))
                             for s, seed in enumerate(SEEDS)]
                absolute.extend(seed_rows)
                summary = dict(**info, arm=arm)
                for m in METRICS:
                    summary.update({f"{m}_{key}": value for key, value in p.mean_std([r[m] for r in seed_rows]).items()})
                absolute_summary.append(summary)
            for arm, reference in PAIRS:
                name_pair = comparison(arm, reference)
                val = values[name_pair]
                pair_info = dict(**info, comparison=name_pair, arm=arm, reference=reference)
                seed_rows = []
                for s, seed in enumerate(SEEDS):
                    old, new = [p.metric_at(val, labels, ii, a, s) for a in (0, 1)]
                    row = dict(**pair_info, seed=seed, **{f"delta_{m}": new[m] - old[m] for m in METRICS})
                    row.update({state: int((val["state"][s, ii] == state).sum()) for state in STATES})
                    row["prediction_changes"] = int((val["prediction"][0, s, ii] != val["prediction"][1, s, ii]).sum())
                    seed_rows.append(row)
                paired.extend(seed_rows)
                summary = dict(**pair_info)
                for m in METRICS:
                    delta = [r[f"delta_{m}"] for r in seed_rows]
                    summary.update({f"delta_{m}_{key}": value for key, value in p.mean_std(delta).items()})
                    summary[f"improved_{m}_seeds"] = sum(d > 0 if m in ("OA", "BA") else d < 0 for d in delta)
                    summary[f"tied_{m}_seeds"] = sum(d == 0 for d in delta)
                for state in STATES:
                    mask = val["state"][:, ii] == state
                    selected = ii[mask.any(0)]
                    summary[f"{state}_events"] = int(mask.sum())
                    summary[f"{state}_unique_ids"] = len(selected)
                    summary[f"{state}_unique_ships"] = len({rows[i]["mmsi"] for i in selected})
                paired_summary.append(summary)
    for name, data in (("metrics_by_seed", absolute), ("metrics_summary", absolute_summary),
                       ("paired_by_seed", paired), ("paired_summary", paired_summary)):
        write_table(out, name, data)
    return values, absolute, absolute_summary, paired, paired_summary


def events_and_bootstrap(ids, labels, values, rows, out):
    events, observations, boots, outcomes, intervals = [], [], [], {}, {}
    for arm, reference in PAIRS:
        name = comparison(arm, reference)
        val = values[name]
        info = dict(comparison=name, arm=arm, reference=reference)
        for i, row in enumerate(rows):
            meta = dict(**info, id=str(ids[i]), mmsi=row["mmsi"], label=int(labels[i]),
                        port=row["port"], pass_group=row["pass_group"], polarization=p.polarization(row))
            observations.append(dict(**meta, **{f"{state}_events": int((val["state"][:, i] == state).sum()) for state in STATES},
                                     mean_margin_delta=float((val["margin"][1, :, i] - val["margin"][0, :, i]).mean())))
            for s, seed in enumerate(SEEDS):
                event = dict(**meta, seed=seed, state=str(val["state"][s, i]),
                             reference_prediction=int(val["prediction"][0, s, i]), prediction=int(val["prediction"][1, s, i]))
                for field in ("margin", "NLL", "Brier"):
                    old, new = [float(val[field][a, s, i]) for a in (0, 1)]
                    event.update({f"reference_{field}": old, field: new, f"delta_{field}": new-old})
                events.append(event)
        outcome = {}
        for state in STATES:
            count = (val["state"] == state).sum(0)
            ii = np.flatnonzero(count)
            outcome[state] = dict(events=int(count.sum()), unique_queries=len(ii),
                                  unique_ships=len({rows[i]["mmsi"] for i in ii}),
                                  queries_by_seed_count={str(k): int((count == k).sum()) for k in (1, 2, 3)})
        rescue, harm = [(val["state"] == state).any(0) for state in ("rescue", "harm")]
        outcome["both_rescue_and_harm_queries"] = int((rescue & harm).sum())
        outcome["both_rescue_and_harm_ships"] = len({rows[i]["mmsi"] for i in np.flatnonzero(rescue)} &
                                                    {rows[i]["mmsi"] for i in np.flatnonzero(harm)})
        outcomes[name] = outcome
        receipt, draws = p.cluster_bootstrap(val, labels, rows, replicates=2000)
        receipt.update(info)
        receipt["scope"] += " Same bootstrap random seed and pass draws used for all three comparisons; nine fixed models overall."
        intervals[name] = receipt
        boots.extend(dict(**info, replicate=i+1, **dict(zip(METRICS, draw))) for i, draw in enumerate(draws))
    for name, data in (("seed_events", events), ("query_outcomes", observations), ("bootstrap_replicates", boots)):
        write_table(out, name, data)
    return outcomes, intervals


def build_cases(ids, labels, scores, values, rows, refs, diagnostics):
    cases = []
    index = {str(x): i for i, x in enumerate(ids)}

    def make_case(i, arm, reference, state, seed, selection):
        a, b, s = ARMS.index(arm), ARMS.index(reference), SEEDS.index(seed)
        val = values[comparison(arm, reference)]
        truth = int(labels[i])
        rival_scores = scores[b, s, i].copy()
        rival_scores[truth] = -np.inf
        rival = int(rival_scores.argmax())
        predictions = scores[:, :, i].argmax(-1)
        confusion = sorted((set(predictions.ravel().tolist()) | {rival}) - {truth})
        eligible = [r for r in refs if r["id"] != ids[i] and r["mmsi"] != rows[i]["mmsi"]]
        case = dict(comparison=comparison(arm, reference), arm=arm, reference=reference,
                    reference_arm=reference, id=str(ids[i]), label=truth, state=state,
                    class_name=h.CLASSES[truth], port=rows[i]["port"], mmsi=rows[i]["mmsi"],
                    pass_group=rows[i]["pass_group"], product=rows[i].get("product", ""),
                    representative_seed=seed, fixed_rival=rival,
                    outcome_counts={x: int((val["state"][:, i] == x).sum()) for x in STATES},
                    mean_margin_delta=float((val["margin"][1, :, i] - val["margin"][0, :, i]).mean()),
                    confusion_classes=confusion, eligible_reference_ids=[r["id"] for r in eligible],
                    true_reference_ids=[r["id"] for r in eligible if r["label"] == truth],
                    confusion_reference_ids=[r["id"] for r in eligible if r["label"] in confusion],
                    selection=selection, visual_status="Generated native panels; manual visual review pending", per_seed=[])
        for j, run_seed in enumerate(SEEDS):
            entry = dict(seed=run_seed, state=str(val["state"][j, i]))
            for k, run_arm in enumerate(ARMS):
                logits = scores[k, j, i]
                losses = h.per_sample(logits[None], np.asarray([truth]))
                entry.update({f"{run_arm}_logits": logits.tolist(), f"{run_arm}_prediction": int(predictions[k, j]),
                              f"{run_arm}_fixed_margin": float(logits[truth]-logits[rival]),
                              f"{run_arm}_margin": float(logits[truth]-np.max(np.delete(logits, truth))),
                              f"{run_arm}_NLL": float(losses["NLL"][0]), f"{run_arm}_Brier": float(losses["Brier"][0])})
            for field in ("margin", "fixed_margin", "NLL", "Brier"):
                entry[f"delta_{field}"] = entry[f"{arm}_{field}"] - entry[f"{reference}_{field}"]
            case["per_seed"].append(entry)
        return case

    for arm, reference in PAIRS:
        pair_scores = scores[[ARMS.index(reference), ARMS.index(arm)]]
        chosen = p.select_cases(ids, labels, pair_scores, values[comparison(arm, reference)], rows, refs, h)
        for selected in chosen:
            cases.append(make_case(index[selected["id"]], arm, reference, selected["state"],
                                   selected["representative_seed"], selected["selection"]))
    if diagnostics and diagnostics.is_file():
        data = h.read_json(diagnostics)
        for item in data["cases"]:
            arm, reference = item.get("arm", "preadd"), item.get("reference", item.get("reference_arm", "baseline"))
            if item.get("comparison"):
                arm, reference = item["comparison"].split("_vs_")
            case_id, seed = str(item["id"]), int(item.get("seed", item.get("representative_seed", 17)))
            i = index[case_id]
            matches = [c for c in cases if c["id"] == case_id and c["comparison"] == comparison(arm, reference)]
            if not matches:
                state = str(values[comparison(arm, reference)]["state"][SEEDS.index(seed), i])
                matches = [make_case(i, arm, reference, state, seed, "Main path diagnostic ID; matched outcome retained")]
                cases.extend(matches)
            for case in matches:
                case["path_diagnosis"] = item
    return cases


def gallery(cases, rows, refs, out):
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    native = h.native_module(p.PACKAGE)
    queries, ref_index = {r["id"]: r for r in rows}, {r["id"]: r for r in refs}
    receipts, sections = [], []
    for case in cases:
        group = [queries[case["id"]], *[ref_index[x] for x in case["true_reference_ids"]],
                 *[ref_index[x] for x in case["confusion_reference_ids"]]]
        roles = ["QUERY", *(["TRUE REF"] * len(case["true_reference_ids"])),
                 *(["CONFUSION REF"] * len(case["confusion_reference_ids"]))]
        stem = f"{case['comparison']}_{case['state']}_{case['id']}"
        heading = f"{stem} / {case['class_name']} / {case['port']} / {case['outcome_counts']}"
        sections.append(f"<section><h2>{html.escape(heading)}</h2><p>Mean margin delta {case['mean_margin_delta']:+.6f}; seed {case['representative_seed']}; fixed rival {case['fixed_rival']}</p>")
        reviewed = {"34733", "31166", "26109", "32844", "7872", "2490", "35487", "23004"}
        if case["id"] in reviewed and case["comparison"].startswith("preadd_"):
            case["visual_status"] = "VV/VH and shown actual references reviewed; see CASE_NOTES_ZH.md"
        sections.append(f"<p>{html.escape(case['visual_status'])}</p>")
        if "path_diagnosis" in case:
            sections.append("<details><summary>Frozen C/preadd pathway diagnostic (not an E intervention)</summary><table><tr><th>Disabled / retained path</th><th>Prediction</th><th>True-class margin</th></tr>")
            for name, value in case["path_diagnosis"]["paths"].items():
                sections.append(f"<tr><td>{html.escape(name)}</td><td>{value['prediction']}</td><td>{value['top1_margin']:+.6f}</td></tr>")
            sections.append("</table><p>Changes affect query and reference jointly; frozen weights, no refit. Not physical or pixel attribution.</p></details>")
        case["panels"] = {}
        for pol in ("vv", "vh"):
            filename = f"{stem}_{pol}.png"
            items = h.panel(group, pol, folder / filename, native, roles, columns=min(4, len(group)))
            for receipt, record in zip(items, group):
                receipt.update(case_id=case["id"], comparison=case["comparison"], panel=filename,
                               input_spec=record.get(pol), product=record.get("product"))
            receipts.extend(items)
            case["panels"][pol] = filename
            sections.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(heading)}"></div>')
        sections.append("</section>")
    for pol in ("vv", "vh"):
        filename = f"reference_bank_{pol}.png"
        items = h.panel(refs, pol, folder / filename, native)
        for receipt, record in zip(items, refs):
            receipt.update(panel=filename, input_spec=record.get(pol), product=record.get("product"))
        receipts.extend(items)
        sections.append(f'<h2>Actual 16 references / {pol.upper()}</h2><div class="image"><img src="{filename}" alt="Actual references"></div>')
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Frozen preadd expansion cases</title>"
            "<style>body{font:15px Arial,sans-serif;margin:24px;color:#171717}h2{font-size:18px}"
            "section{border-top:1px solid #bbb;margin-top:24px}.image{overflow:auto}img{max-width:none;image-rendering:pixelated}</style>"
            "<h1>Frozen preadd expansion cases</h1><p>Native 1:1 raw display pixels and actual 160x160 crop/pad. VV/VH separate; no resize/contrast stretch. Missing views marked. Eight C-comparison cases reviewed; others are generated illustrations.</p><p><a href='../../CASE_NOTES_ZH.md'>Manual case notes</a> | <a href='../../FINDINGS_ZH.md'>Findings</a></p>"
            f"<p>{html.escape(SCOPE)}</p>" + "".join(sections) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    h.dump(folder / "cases.json", cases)
    h.dump(folder / "native_sources.json", receipts)


def report(summary, absolute, paired, cases):
    lines = ["# Frozen C/preadd expansion numeric report", "", SCOPE, "",
             "## Three-seed means and sample SD", "", "| Arm | OA | BA | NLL | Brier |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        stats = summary["by_arm"][arm]
        lines.append(f"| {arm} | " + " | ".join(f"{stats[m]['mean']:.6f} +/- {stats[m]['std']:.6f}" for m in METRICS) + " |")
    lines += ["", "## Paired differences", "", "| Comparison | Metric | Mean +/- SD | Improved seeds | 95% pass CI |", "|---|---|---:|---:|---:|"]
    for row in paired:
        if row["dimension"] != "all":
            continue
        name = row["comparison"]
        for m in METRICS:
            lo, hi = summary["bootstrap"][name]["intervals"][m]
            lines.append(f"| {name} | {m} | {row['delta_'+m+'_mean']:+.6f} +/- {row['delta_'+m+'_std']:.6f} | {row['improved_'+m+'_seeds']}/3 | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "2000 paired percentile bootstrap draws of 85 whole passes with replacement, RNG seed 20261028. Identical draws apply to all arms and seeds. Conditional on nine fixed models and the fixed reference bank; not retraining uncertainty, independent ships, or multiplicity-adjusted confirmation. A ship can recur across passes. BA averages present classes in each draw; all draws retained.", "",
              "## Rescue/harm and persistent outcomes", "", "| Comparison | State | Seed events | Unique IDs | Unique ships |", "|---|---|---:|---:|---:|"]
    for name, outcome in summary["outcomes"].items():
        for state in STATES:
            r = outcome[state]
            lines.append(f"| {name} | {state} | {r['events']} | {r['unique_queries']} | {r['unique_ships']} |")
    for name, outcome in summary["outcomes"].items():
        lines += ["", f"{name}: both rescued and harmed across seeds: {outcome['both_rescue_and_harm_queries']} IDs / {outcome['both_rescue_and_harm_ships']} ships.", ""]
    lines += ["Repeated seed events are not independent observations. Counts by one/two/three seeds are in summary.json.", "", "## Group differences", "",
              "Class-group OA/BA both equal recall. Old three ports are Jebel Ali, Port Klang and Singapore. Group mean/SD and unique-ID/ship outcomes are in paired_summary.csv; complete per-seed results include all 85 passes.", "",
              "| Comparison | Dimension | Group | n | Delta OA | Delta BA | Delta NLL | Delta Brier |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in paired:
        if r["dimension"] not in ("all", "pass_group"):
            lines.append(f"| {r['comparison']} | {r['dimension']} | {r['group']} | {r['n']} | " + " | ".join(f"{r['delta_'+m+'_mean']:+.6f}" for m in METRICS) + " |")
    lines += ["", "## Native cases", "", "Two deterministic rounds of rescue/harm/CC/WW per comparison, using prior expansion selection. C-vs-E receives its own cases. Main diagnostic IDs are appended when available. Cases are posthoc illustrations, not prevalence estimates or causal proof.", "",
              "Each case has comparison/arm/reference, representative_seed, fixed_rival and per_seed baseline_logits/stage_logits/preadd_logits. Actual same-class and confusion-class references plus the entire 16-reference bank are rendered separately for VV/VH, with source paths and preprocessing in native_sources.json.", "",
              "Generated panels do not establish visual review. Eight C-comparison queries have now been reviewed in VV/VH with actual references; see CASE_NOTES_ZH.md. Other panels remain generated illustrations. All case_paths interventions are on C/preadd, regardless of the comparison used to select the case.", "", "[Native gallery](analysis/cases/index.html)", ""]
    for c in cases:
        lines.append(f"- {c['comparison']} / {c['id']} / {c['state']} / {c['class_name']} / seed {c['representative_seed']}")
    (HERE / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gallery-only", action="store_true")
    parser.add_argument("--diagnostic-cases", type=Path)
    args = parser.parse_args()
    if args.diagnostic_cases is None:
        args.diagnostic_cases = next((HERE / name for name in ("case_paths.json", "path_diagnosis.json")
                                      if (HERE / name).is_file()), None)
    missing = [x for x in ("predictions.npz", "query_records.json") if not (HERE / x).is_file()]
    if missing:
        print("Prepared; waiting for " + ", ".join(missing))
        return
    ids, labels, scores, rows, refs = load_inputs()
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    if args.gallery_only:
        values = {comparison(a, b): pair_values(scores, labels, a, b) for a, b in PAIRS}
    else:
        values, absolute, absolute_summary, paired, paired_summary = tables(ids, labels, scores, rows, out)
        outcomes, bootstrap = events_and_bootstrap(ids, labels, values, rows, out)
    cases = build_cases(ids, labels, scores, values, rows, refs, args.diagnostic_cases)
    (out / "cases").mkdir(exist_ok=True)
    h.dump(out / "cases/cases.json", cases)
    print("Case IDs ready: " + ", ".join(f"{c['comparison']}:{c['state']}:{c['id']}" for c in cases), flush=True)
    gallery(cases, rows, refs, out)
    if not args.gallery_only:
        summary = dict(cohort=p.cohort_summary(rows, labels),
                       by_arm={a: {m: p.mean_std([r[m] for r in absolute if r["arm"] == a and r["dimension"] == "all"]) for m in METRICS} for a in ARMS},
                       paired_deltas={r["comparison"]: {m: dict(mean=r[f"delta_{m}_mean"], std=r[f"delta_{m}_std"]) for m in METRICS} for r in paired_summary if r["dimension"] == "all"},
                       outcomes=outcomes, bootstrap=bootstrap, scope=SCOPE, score_ensemble=False,
                       cases=[dict(comparison=c["comparison"], id=c["id"], state=c["state"]) for c in cases],
                       new_training=False, visual_review="pending main-agent review",
                       references_manifest=str(p.PACKAGE / "manifests/references.json"))
        h.dump(out / "summary.json", summary)
        report(summary, absolute, paired_summary, cases)
        print(json.dumps(dict(by_arm=summary["by_arm"], paired_deltas=summary["paired_deltas"]), indent=2))
    print(f"Cases: {out / 'cases/cases.json'} ({len(cases)} comparison-cases)")


if __name__ == "__main__":
    main()
