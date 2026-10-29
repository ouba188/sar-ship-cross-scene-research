"""CPU analysis for private retrained ablations; explicit main-agent ready required.

Imports completed analysis/native-input helpers only. No runner, model or torch.
"""
import argparse
from collections import Counter
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent / "spatial_m12_full_20261029"
CONTROL = ROOT.parent / "intervention_matched_control_20261029"
REFIT = ROOT.parent / "spatial_m12_refit_20261029"
spec = importlib.util.spec_from_file_location("private_matched_cpu", CONTROL / "analysis.py")
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)
full, base = matched.full, matched.base
SEEDS, METRICS, STATES, CLASSES = base.SEEDS, base.METRICS, base.STATES, base.CLASSES
SCOPES = full.SCOPES
NEW_ARMS = ("plain", "no_m1", "no_m2", "no_m3")
ARMS = ("full", "full_control", *NEW_ARMS)
CONDITIONS = tuple(f"{arm}_{stage}" for arm in ARMS for stage in ("before", "after"))
PRIMARY = {f"full_minus_{arm}_after": (f"{arm}_after", "full_after") for arm in NEW_ARMS}
OBJECTIVE_CONTROL = "full_control_minus_plain_after"
FIT_CONTROL = "full_minus_plain_before"
COMPARISONS = {
    **PRIMARY,
    OBJECTIVE_CONTROL: ("plain_after", "full_control_after"),
    FIT_CONTROL: ("plain_before", "full_before"),
    **{f"{arm}_after_minus_before": (f"{arm}_before", f"{arm}_after") for arm in ARMS},
}
SCOPE_NOTE = (
    "Private retrained final-module ablations: source-fit408 then adapt54, native160, fixed source reference16. "
    "full is the reused guided spatial M1/M2 + M3 endpoint; full_control is its matched CE+0.1KL continuation. "
    "no_m1/no_m2/no_m3 each use their OWN source-fit teacher, intervention selection and guided adaptation; "
    "plain uses its OWN teacher and CE+0.1KL, with no reference contribution. "
    "full-minus-plain includes architecture AND training-objective differences, not pure architecture. "
    "full_control-minus-plain matches the subsequent CE+KL objective, while source-fit objectives, architecture "
    "and own teachers still differ. Module deletion contrasts compare entire retrained procedures, not fixed-teacher causal effects. "
    "Before/after uses each arm's own source fit; only full/full_control share their before archive. "
    "Matched samples, initialization, permitted update range and step counts do NOT match parameter counts or FLOPs. "
    "The pre-save checks establish deletion of the disabled module's dedicated paths and losses; logits cannot establish this. "
    "Seeds17/29/43 vary training order with shared initialization, NOT independent initializations; CUDA is not strict deterministic. "
    "Receipt trainable counts and seconds are descriptive only: cache and machine-memory conditions differ, so no formal speed claim. "
    "Development160 and expanded3075 are historical, NOT blind tests; shared ports are not strict held-port evaluation. "
    "OA/BA are percent, paired deltas percentage points; BA uses present classes, natural-log NLL, sum-over-eight-class Brier. "
    "Sample SD is across order seeds. Pass bootstrap is conditional on these fixed models, not independent-training uncertainty. "
    "Repeated query/vessel/seed/reference exposure is not independent evidence. Intervals are descriptive, not multiplicity-adjusted. "
    "Native case selection is outcome-based, not prevalence; image-visible facts and physical mechanisms remain unreviewed."
)
CASE_RULE = (
    "Up to12 unique expanded query IDs AND distinct nonempty MMSIs; do not count repeated seed events as new cases. "
    "Reuse the full-stage selector within each of the four full-minus-arm comparisons and the CEKL control comparison. "
    "First retain a distinct query for each module contrast and the two plain contrasts, then add missing "
    "rescued/harmed/CC/WW categories, then partial-polarization coverage, then round-robin fill. "
    "Within each comparison prefer net rescued/harmed or all-seed CC/WW, ranked by net events, fixed-margin change and ID. "
    "If a category or module cannot be covered with distinct vessels, record the gap rather than duplicate evidence."
)


def input_paths(root=ROOT, source=FULL, control=CONTROL, refit=REFIT):
    reused, _ = matched.input_paths(control, source)
    predictions = {}
    for scope in SCOPES:
        for seed in SEEDS:
            for arm in NEW_ARMS:
                for stage in ("before", "after"):
                    predictions[scope, f"{arm}_{stage}", seed] = root / "predictions" / f"{scope}_{arm}_{stage}_s{seed}.npz"
            for arm in ("full", "full_control"):
                predictions[scope, f"{arm}_before", seed] = reused[scope, "before", seed]
            predictions[scope, "full_after", seed] = reused[scope, "guided", seed]
            predictions[scope, "full_control_after", seed] = reused[scope, "control", seed]
    receipts = {}
    for arm in ARMS:
        for seed in SEEDS:
            fit_dir = root / "source_fit/training" / f"{arm}_s{seed}" if arm in NEW_ARMS else refit / "training" / f"spatial_s{seed}"
            adapt_dir = (root / "training" / f"{arm}_s{seed}" if arm in NEW_ARMS else
                         source / "training" / f"spatial_s{seed}" if arm == "full" else
                         control / "training" / f"control_s{seed}")
            receipts[arm, "before", seed] = fit_dir / "training.json"
            receipts[arm, "after", seed] = adapt_dir / "training.json"
    return predictions, receipts


def load_inputs(root=ROOT, source=FULL, control=CONTROL, refit=REFIT):
    paths, receipt_paths = input_paths(root, source, control, refit)
    gate_paths = {name: root / f"{name}.json" for name in ("checks", "evaluation_check")}
    manifest = source / "cohorts.json"
    required = {manifest, *paths.values(), *receipt_paths.values(), *gate_paths.values()}
    missing = sorted(str(p) for p in required if not p.is_file())
    if missing:
        raise SystemExit("Incomplete inputs; no formal analysis started:\n" + "\n".join(missing))
    data = base.read_json(manifest)
    cohorts = {key: data[key] for key in matched.EXPECTED}
    for split, rows in cohorts.items():
        if len(rows) != matched.EXPECTED[split] or len({str(r['id']) for r in rows}) != len(rows):
            raise ValueError(f"{split}: wrong cohort size or duplicate IDs")
        for row in rows:
            row["id"] = str(row["id"])
            if row["label"] not in range(8):
                raise ValueError(f"{split}/{row['id']}: label outside eight-class taxonomy")
    training = {}
    for (arm, stage, seed), path in receipt_paths.items():
        receipt = base.read_json(path)
        steps, epochs = (408, 8) if stage == "before" else (54, 6)
        if (receipt.get("steps") != steps or receipt.get("completed_epochs") != epochs
                or receipt.get("seed") != seed or receipt.get("smoke_only", False)):
            raise ValueError(f"{path}: expected completed {steps}-step/{epochs}-epoch seed{seed} receipt")
        training[f"{arm}_{stage}_s{seed}"] = dict(path=str(path), receipt=receipt)
    gates = {name: dict(path=str(path), receipt=base.read_json(path)) for name, path in gate_paths.items()}
    # Archive existence is not readiness approval. The CLI requires the main-agent
    # notification; preserve its check receipts verbatim without guessing their schema.
    cache = {}
    scores = {}
    for key, path in paths.items():
        if path not in cache:
            cache[path] = base.load_score(path, cohorts[key[0]])
        scores[key] = cache[path]
    return cohorts, scores, training, gates


def pair_events(rows, scores, old, new):
    helper = {**scores, **{("legacy_before", s): scores["full_before", s] for s in SEEDS}}
    return full.pair_events(rows, helper, old, new)


def summarize_scores(cohorts, all_scores, repeats=2000):
    tables = {name: [] for name in ("metrics", "groups", "groups_summary", "seed_events")}
    results, case_events = {}, {}
    for scope in SCOPES:
        rows = cohorts[scope]
        y = np.array([r["label"] for r in rows], int)
        scores = {(c, s): all_scores[scope, c, s] for c in CONDITIONS for s in SEEDS}
        metric_rows = [dict(scope=scope, condition=c, seed=s, **base.metrics(scores[c, s], y))
                       for c in CONDITIONS for s in SEEDS]
        tables["metrics"].extend(metric_rows)
        result = dict(conditions={c: {m: base.moments([r[m] for r in metric_rows if r["condition"] == c])
                                     for m in METRICS} for c in CONDITIONS}, comparisons={})
        for comparison, (old, new) in COMPARISONS.items():
            events = pair_events(rows, scores, old, new)
            detailed, aggregate = full.summarize_comparison(rows, scores, old, new, events)
            for name, records in (("groups", detailed), ("groups_summary", aggregate), ("seed_events", events)):
                tables[name].extend(dict(scope=scope, comparison=comparison, **r) for r in records)
            result["comparisons"][comparison] = dict(reference=old, alternative=new, overall=aggregate[0],
                transitions=base.transition_summary(events),
                pass_bootstrap=full.pass_bootstrap(rows, scores, old, new, repeats=repeats))
            if scope == "expanded" and comparison in (*PRIMARY, OBJECTIVE_CONTROL):
                case_events[comparison] = events
        results[scope] = result
    return results, tables, case_events


def select_cases(event_sets, count=12):
    # Keep the established within-comparison ranking; add only cross-arm quotas
    # and vessel exclusion needed for an independent-query gallery.
    pools = {name: [dict(e, comparison=name) for e in full.select_cases(events, count=len({e['id'] for e in events}))]
             for name, events in event_sets.items()}
    selected, ids, ships = [], set(), set()

    def add(candidates):
        for event in candidates:
            ship = str(event["mmsi"]) if event.get("mmsi") else None
            if len(selected) < count and event["id"] not in ids and (ship is None or ship not in ships):
                selected.append(event)
                ids.add(event["id"])
                if ship is not None:
                    ships.add(ship)
                return True
        return False

    order = [f"full_minus_{arm}_after" for arm in ("no_m1", "no_m2", "no_m3", "plain")] + [OBJECTIVE_CONTROL]
    for name in order:
        add(pools.get(name, []))
    for state in STATES:
        if not any(e["selection_category"] == state for e in selected):
            add(e for name in order for e in pools.get(name, []) if e["selection_category"] == state)
    if not any(e["available"] != "both" for e in selected):
        add(e for name in order for e in pools.get(name, []) if e["available"] != "both")
    while len(selected) < count:
        changed = False
        for name in order:
            changed = add(pools.get(name, [])) or changed
        if not changed:
            break
    return selected


def case_details(case, index, query, refs, scores, event_sets):
    old, new = COMPARISONS[case["comparison"]]
    adapter = {(c, s): scores[source, s] for c, source in
               (("before", "full_before"), ("control", old), ("guided", new)) for s in SEEDS}
    detail, _, _ = matched.case_details(case, index, query, refs, adapter)
    truth = query["label"]
    rivals = {case["fixed_rival"]}
    for value in scores.values():
        score = value[index].copy()
        score[truth] = -np.inf
        rivals.add(int(score.argmax()))
    allowed = [r for r in refs if base.eligible(query, r)]
    own = [r for r in allowed if r["label"] == truth]
    wrong = [r for r in allowed if r["label"] in rivals]
    detail.update(rival_classes=sorted(rivals), true_reference_ids=[r["id"] for r in own],
                  rival_reference_ids=[r["id"] for r in wrong], per_seed=[], selection_rule=CASE_RULE,
                  reference_usage={arm: ("none: plain has no reference contribution" if arm == "plain" else
                                         "no M3 reference transport; bank displayed only as comparator context" if arm == "no_m3" else
                                         "actual eligible fixed-bank references; influence is not established by display") for arm in ARMS},
                  comparison_events={name: [e for e in events if e["id"] == query["id"]] for name, events in event_sets.items()})
    for seed in SEEDS:
        values = {}
        for condition in CONDITIONS:
            score = scores[condition, seed][index]
            sample = base.per_sample(score[None, :], np.array([truth]))
            values[condition] = dict(prediction=int(score.argmax()), logits=score.tolist(),
                fixed_margin=float(score[truth] - score[case["fixed_rival"]]),
                top_rival_margin=float(score[truth] - np.delete(score, truth).max()),
                NLL=float(sample["NLL"][0]), Brier=float(sample["Brier"][0]),
                truth_rival_margins={str(c): float(score[truth] - score[c]) for c in sorted(rivals)})
        detail["per_seed"].append(dict(seed=seed, conditions=values))
    return detail, [query, *own, *wrong], ["QUERY", *["TRUE REF"] * len(own), *["RIVAL REF"] * len(wrong)]


def gallery(out, cohorts, scores, event_sets, native):
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    indices = {r["id"]: (i, r) for i, r in enumerate(cohorts["expanded"])}
    selected = select_cases(event_sets)
    cases, sources, blocks = [], [], []
    for number, choice in enumerate(selected, 1):
        index, query = indices[choice["id"]]
        case, rows, roles = case_details(choice, index, query, cohorts["references"], scores, event_sets)
        title = f"{number}. {case['comparison']} / {case['selection_category']} / ID {query['id']} / seed {case['seed']}"
        blocks.append("<section><h2>" + html.escape(title) + "</h2><p>Truth: " + CLASSES[query["label"]] +
                      "; port: " + html.escape(str(query["port"])) + "; pass: " + html.escape(str(query["pass_group"])) +
                      "; actual polarization: " + case["available"] + "; not_reviewed</p>")
        blocks.append("<p>References apply to support-based arms. Plain: NO reference contribution. "
                      "no_m3: NO M3 reference transport; references are comparator context.</p>")
        blocks.append("<table><tr><th>Seed</th><th>Condition</th><th>Prediction</th><th>True-top-rival</th><th>Fixed margin</th></tr>")
        for seed_detail in case["per_seed"]:
            for condition, value in seed_detail["conditions"].items():
                blocks.append(f"<tr><td>{seed_detail['seed']}</td><td>{condition}</td><td>{CLASSES[value['prediction']]}</td>"
                              f"<td>{value['top_rival_margin']:.6f}</td><td>{value['fixed_margin']:.6f}</td></tr>")
        blocks.append("</table>")
        for pol in ("vv", "vh"):
            filename = f"case_{number:02d}_{pol}.png"
            receipts = base.panel(rows, roles, pol, folder / filename, native)
            for receipt in receipts:
                receipt.update(case_id=query["id"], selected_seed=case["seed"], comparison=case["comparison"],
                               panel=filename, visual_status="not_reviewed", plain_reference_contribution=False,
                               usable_with_query_polarization=bool(receipts[0]["available"] and receipt["available"]))
            sources.extend(receipts)
            case["panels"][pol] = str(folder / filename)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(title)}"></div>')
        blocks.append("<details><summary>All logits, contrasts and actual reference IDs</summary><pre>" +
                      html.escape(base.json.dumps(case, indent=2)) + "</pre></details></section>")
        cases.append(case)
    coverage = dict(requested_range=[8, 12], selected_queries=len(cases),
        unique_known_mmsi=len({str(c['mmsi']) for c in cases if c.get('mmsi')}),
        missing_mmsi_ids=[c['id'] for c in cases if not c.get('mmsi')],
        selection_categories=dict(Counter(c['selection_category'] for c in cases)),
        selection_comparisons=dict(Counter(c['comparison'] for c in cases)),
        missing_categories=[s for s in STATES if not any(c['selection_category'] == s for c in cases)],
        missing_comparisons=[name for name in event_sets if not any(c['comparison'] == name for c in cases)],
        note="Distinct queries/vessels still share passes and are not statistically independent samples.",
        visual_status="not_reviewed")
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Private retrained native cases</title>"
            "<style>body{font:14px Arial;margin:24px;color:#222}section{border-top:1px solid #aaa;margin-top:24px}"
            "h1{font-size:22px}h2{font-size:18px}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}"
            ".image{overflow:auto}img{max-width:none;image-rendering:pixelated}pre{white-space:pre-wrap}</style>"
            "<h1>Private retrained ablations: native cases</h1><p>" + html.escape(SCOPE_NOTE) + "</p><p>" + CASE_RULE +
            "</p><p>One source pixel per output pixel; no resizing or contrast stretch. Original display chips are not "
            "calibrated radiometry. VV/VH separately show original, exact native crop/pad and duplicate OBB overlay. "
            "Eligible true/rival references exclude same ID/MMSI; plain has no reference contribution. All images: not_reviewed.</p>" +
            "<pre>" + html.escape(base.json.dumps(coverage, indent=2)) + "</pre>" + "".join(blocks) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    base.dump(folder / "cases.json", cases)
    base.dump(folder / "native_sources.json", sources)
    base.dump(folder / "coverage.json", coverage)
    return cases, coverage


def numeric_report(out, summary, case_count):
    lines = ["# Private retrained ablation numerical results", "", SCOPE_NOTE, ""]
    for scope, result in summary["scopes"].items():
        lines += [f"## {scope}", ""]
        for title, conditions in (
            ("Main final endpoints", [f"{arm}_after" for arm in ("full", *NEW_ARMS)]),
            ("Matched subsequent objective", ["full_control_after", "plain_after"]),
            ("Supplementary source-fit endpoints", [f"{arm}_before" for arm in ("full", *NEW_ARMS)]),
        ):
            lines += [f"### {title}", "", "| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |",
                      "|---|---:|---:|---:|---:|"]
            for condition in conditions:
                stats = result["conditions"][condition]
                lines.append(f"| {condition} | " + " | ".join(f"{stats[m]['mean']:.6f} +/- {stats[m]['sd']:.6f}" for m in METRICS) + " |")
            lines.append("")
        lines += ["", "| Contrast | Metric | Paired delta | SD | Pass CI low | High |", "|---|---|---:|---:|---:|---:|"]
        for name, comparison in result["comparisons"].items():
            for metric in METRICS:
                row, ci = comparison["overall"], comparison["pass_bootstrap"]["metrics"][metric]
                lines.append(f"| {name} | {metric} | {row['delta_' + metric + '_mean']:.6f} | {row['delta_' + metric + '_sd']:.6f} | {ci['low']:.6f} | {ci['high']:.6f} |")
        lines += ["", "| Contrast | State | Seed events | Unique queries | Unique ships |", "|---|---|---:|---:|---:|"]
        for name, comparison in result["comparisons"].items():
            for state in STATES:
                item = comparison["transitions"][state]
                lines.append(f"| {name} | {state} | {item['seed_events']} | {item['unique_ids']} | {item['unique_ships']} |")
        lines += ["", "| Contrast | Passes | Resamples missing a present class | Total resamples |", "|---|---:|---:|---:|"]
        for name, comparison in result["comparisons"].items():
            boot = comparison["pass_bootstrap"]
            lines.append(f"| {name} | {boot['pass_count']} | {boot['missing_class_replicates']} | {boot['repeats']} |")
    lines += ["", "All deltas are alternative minus reference. Positive OA/BA favors the alternative; negative NLL/Brier favors it. "
              "Whole-pass resampling retains paired queries and all fixed seeds; development has only three passes. "
              "Groups CSVs preserve class/port/pass/actual-polarization differences and prediction transitions, including WW changes. "
              "Training/check receipts are retained verbatim in summary.json; logits alone do not certify optimization or mechanism.", "",
              f"[Native gallery](cases/index.html): {case_count} distinct queries, {2 * case_count} VV/VH panels, not_reviewed. "
              "Main-agent image inspection, visible facts, interpretation and study report remain pending.", ""]
    (out / "NUMERIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(package=base.PACKAGE, *, main_agent_ready=False):
    if not main_agent_ready:
        raise SystemExit("Formal analysis waits for explicit main-agent ready notification; synthetic tests remain available.")
    cohorts, scores, training, gates = load_inputs()
    native = full.module_at("private_ablation_native", package / "sar_m123/input.py")
    full.annotate_available(cohorts, native)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    results, tables, case_events = summarize_scores(cohorts, scores)
    paths, receipts = input_paths()
    summary = dict(scope=SCOPE_NOTE, primary_comparisons=list(PRIMARY), objective_control=OBJECTIVE_CONTROL,
        supplementary_fit_comparison=FIT_CONTROL,
        training_receipts=training, completion_checks=gates, scopes=results,
        coverage=full.cohort_summary(cohorts, out), visual_status="not_reviewed",
        rival_rule="Strongest nontruth mean full source-fit logit across order seeds; fixed for every comparison",
        input_paths=dict(cohorts=str(FULL / "cohorts.json"), predictions={
            f"{scope}_{condition}_s{seed}": str(path) for (scope, condition, seed), path in paths.items()}),
        alignment="72 logical prediction endpoints, 66 unique archives:48 new +18 reused; full/control before is aliased")
    for name, records in tables.items():
        base.write_csv(out / f"{name}.csv", records)
    base.write_csv(out / "main_after_metrics.csv", [r for r in tables["metrics"]
        if r["condition"] in {f"{arm}_after" for arm in ("full", *NEW_ARMS)}])
    base.write_csv(out / "paired_metrics.csv", [r for r in tables["groups"] if r["dimension"] == "all"])
    expanded = {(c, s): scores["expanded", c, s] for c in CONDITIONS for s in SEEDS}
    cases, coverage = gallery(out, cohorts, expanded, case_events, native)
    summary["case_coverage"] = coverage
    base.dump(out / "summary.json", summary)
    numeric_report(out, summary, len(cases))
    base.dump(out / "complete.json", dict(new_prediction_files=48, reused_prediction_files=18,
        unique_prediction_files=len(set(paths.values())), unique_training_receipts=len(set(receipts.values())),
        comparisons=len(COMPARISONS), gallery_unique_queries=len(cases), panels=2 * len(cases),
        device="CPU", visual_status="not_reviewed", main_agent_ready=True))
    print(base.json.dumps(dict(output=str(out), gallery=str(out / "cases/index.html")), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=base.PACKAGE)
    parser.add_argument("--main-agent-ready", action="store_true", help="Use only after explicit completion notification")
    args = parser.parse_args()
    analyze(args.package.resolve(), main_agent_ready=args.main_agent_ready)
