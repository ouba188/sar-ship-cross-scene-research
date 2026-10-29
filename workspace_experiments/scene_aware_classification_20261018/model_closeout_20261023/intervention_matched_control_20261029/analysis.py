"""CPU analysis; run only after the main agent confirms all six control NPZs.

No polling, model inference, training or GPU imports. Reuses sibling helpers.
"""
import argparse
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
FULL = ROOT.parent / "spatial_m12_full_20261029"
spec = importlib.util.spec_from_file_location("matched_full_cpu", FULL / "analysis.py")
full = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full)
base = full.base
SEEDS, METRICS, STATES, CLASSES = base.SEEDS, base.METRICS, base.STATES, base.CLASSES
SCOPES = full.SCOPES
CONDITIONS = ("before", "control", "guided")
COMPARISONS = {
    "guided_minus_control": ("control", "guided"),
    "control_minus_before": ("before", "control"),
    "guided_minus_before": ("before", "guided"),
}
EXPECTED = dict(fit=1205, adapt=136, development=160, expanded=3075, references=16)
SCOPE_NOTE = (
    "Matched spatial M1/M2 without explicit frequency branches, with M3 retained; native160 and fixed reference16. "
    "Primary attribution: guided minus control. Auxiliary comparisons: each after minus its shared before. "
    "Control is CE + 0.1 KL, intervention coefficient 0 and useful all False: no intervention-selection loss. "
    "Its teacher_original reuses the existing original scores from the same source-fitted teacher. "
    "Guided adds the original intervention-selected margin_floor at coefficient 0.3683066666126251. "
    "This estimates the incremental selected-guidance training procedure, not selection alone or a physical mechanism. "
    "Both continuations use the matched source fit and order, planned 6 epochs/54 steps on adapt136, fit1205 separate. "
    "Seeds 17/29/43 vary training order with the SAME initialization, NOT independent initializations. "
    "Development160 and expanded3075 are historical cohorts, NOT blind tests. "
    "OA/BA are percentages; differences are percentage points. BA averages present classes only. "
    "NLL uses natural logs; Brier sums over eight classes; SD is sample SD across the three order seeds. "
    "Pass bootstrap conditions on fixed fitted models; repeated seeds, vessels and references are not independent evidence. "
    "Protocol statements describe the supplied design; raw training receipts are retained, not inferred from logits. "
    "Generated native cases remain not_reviewed until the main agent inspects them."
)


def input_paths(root, source):
    predictions = {}
    for scope in SCOPES:
        for seed in SEEDS:
            predictions[scope, "control", seed] = root / "predictions" / f"{scope}_control_s{seed}.npz"
            for condition, stage in (("before", "before"), ("guided", "after")):
                predictions[scope, condition, seed] = source / "predictions" / f"{scope}_{stage}_spatial_s{seed}.npz"
    receipts = {(condition, seed): folder / "training" / f"{arm}_s{seed}/training.json"
                for condition, folder, arm in (("control", root, "control"), ("guided", source, "spatial"))
                for seed in SEEDS}
    return predictions, receipts


def load_inputs(root, source=FULL):
    paths, receipt_paths = input_paths(root, source)
    manifest = source / "cohorts.json"
    missing = [str(p) for p in (manifest, *paths.values(), *receipt_paths.values()) if not p.is_file()]
    if missing:
        raise SystemExit("Incomplete inputs; no formal analysis started:\n" + "\n".join(missing))
    data = base.read_json(manifest)
    cohorts = {key: data[key] for key in EXPECTED}
    for split, rows in cohorts.items():
        if len(rows) != EXPECTED[split] or len({str(r['id']) for r in rows}) != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} unique observations")
        for row in rows:
            row["id"] = str(row["id"])
    scores = {key: base.load_score(path, cohorts[key[0]]) for key, path in paths.items()}
    training = {f"{condition}_s{seed}": dict(path=str(path), receipt=base.read_json(path))
                for (condition, seed), path in receipt_paths.items()}
    return cohorts, scores, training


def pair_events(rows, scores, old, new):
    # The shared helper names its fixed-rival source legacy_before; supply the
    # spatial before scores under that internal key, without changing outputs.
    helper_scores = {**scores, **{("legacy_before", s): scores["before", s] for s in SEEDS}}
    events = full.pair_events(rows, helper_scores, old, new)
    y = np.array([r["label"] for r in rows], int)
    for seed in SEEDS:
        seed_events = [e for e in events if e["seed"] == seed]
        for condition in CONDITIONS:
            score = scores[condition, seed]
            values = base.per_sample(score, y)
            for i, event in enumerate(seed_events):
                truth, rival = int(y[i]), event["fixed_rival"]
                event[condition + "_prediction"] = int(values["prediction"][i])
                event[condition + "_fixed_margin"] = float(score[i, truth] - score[i, rival])
                event[condition + "_top_rival_margin"] = float(score[i, truth] - np.delete(score[i], truth).max())
    return events


def case_details(case, index, query, refs, scores):
    truth = query["label"]
    rivals = {case["fixed_rival"]}
    # Include actual wrong competitors for ALL displayed seeds and conditions.
    for seed in SEEDS:
        for condition in CONDITIONS:
            value = scores[condition, seed][index].copy()
            value[truth] = -np.inf
            rivals.add(int(value.argmax()))
    allowed = [r for r in refs if base.eligible(query, r)]
    own = [r for r in allowed if r["label"] == truth]
    wrong = [r for r in allowed if r["label"] in rivals]
    detail = dict(case, scope="expanded", visual_status="not_reviewed", rival_classes=sorted(rivals),
                  eligible_reference_ids=[r["id"] for r in allowed],
                  true_reference_ids=[r["id"] for r in own], rival_reference_ids=[r["id"] for r in wrong],
                  panels={}, per_seed=[])
    for seed in SEEDS:
        result = dict(seed=seed, conditions={})
        for condition in CONDITIONS:
            value = scores[condition, seed][index]
            result["conditions"][condition] = dict(prediction=int(value.argmax()), logits=value.tolist(),
                fixed_margin=float(value[truth] - value[case["fixed_rival"]]),
                top_rival_margin=float(value[truth] - np.delete(value, truth).max()),
                truth_rival_margins={str(c): float(value[truth] - value[c]) for c in sorted(rivals)})
        detail["per_seed"].append(result)
    return detail, [query, *own, *wrong], ["QUERY", *["TRUE REF"] * len(own), *["RIVAL REF"] * len(wrong)]


CASE_RULE = (
    "Select up to eight unique expanded guided-control queries using the full-stage helper: "
    "net rescue/harm across seeds and consistent CC/WW, up to two each; rank by net event count, "
    "absolute fixed-rival margin change and ID, with partial-polarization coverage when available. "
    "Fill absent categories by transition/margin; categories absent in the results are not invented. "
    "Representative seed maximizes absolute margin change within the selected state. "
    "These outcome-selected cases do not estimate population prevalence."
)


def gallery(out, cohorts, scores, events, native):
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    indices = {r["id"]: (i, r) for i, r in enumerate(cohorts["expanded"])}
    cases, sources, blocks = [], [], []
    for number, selected in enumerate(full.select_cases(events, count=8), 1):
        index, query = indices[selected["id"]]
        case, rows, roles = case_details(selected, index, query, cohorts["references"], scores)
        case["selection_rule"] = CASE_RULE
        title = f"{number}. {case['selection_category']} / ID {query['id']} / seed {case['seed']}"
        blocks.append("<section><h2>" + html.escape(title) + "</h2><p>Truth: " + CLASSES[query["label"]] +
                      "; port: " + html.escape(str(query["port"])) + "; pass: " + html.escape(str(query["pass_group"])) +
                      "; actual polarization: " + case["available"] + "; not_reviewed</p>")
        blocks.append("<table><tr><th>Seed</th>" + "".join(f"<th>{c}: pred / true-top-rival margin / fixed margin</th>" for c in CONDITIONS) + "</tr>")
        for seed_detail in case["per_seed"]:
            cells = []
            for condition in CONDITIONS:
                value = seed_detail["conditions"][condition]
                cells.append(f"<td>{CLASSES[value['prediction']]} / {value['top_rival_margin']:.6f} / {value['fixed_margin']:.6f}</td>")
            blocks.append(f"<tr><td>{seed_detail['seed']}</td>" + "".join(cells) + "</tr>")
        blocks.append("</table><details><summary>All logits, margins and actual reference IDs</summary><pre>" +
                      html.escape(base.json.dumps(case, indent=2)) + "</pre></details>")
        for pol in ("vv", "vh"):
            filename = f"case_{number:02d}_{pol}.png"
            receipts = base.panel(rows, roles, pol, folder / filename, native)
            for receipt in receipts:
                receipt.update(case_id=query["id"], selected_seed=case["seed"], panel=filename,
                               usable_with_query_polarization=bool(receipts[0]["available"] and receipt["available"]))
            sources.extend(receipts)
            case["panels"][pol] = str(folder / filename)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(title)}"></div>')
        blocks.append("</section>")
        cases.append(case)
    page = ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Matched intervention native cases</title>"
            "<style>body{font:14px Arial;margin:24px;color:#222}section{border-top:1px solid #aaa;margin-top:24px}"
            "h1{font-size:22px}h2{font-size:18px}td,th{border:1px solid #bbb;padding:6px}table{border-collapse:collapse}"
            ".image{overflow:auto}img{max-width:none;image-rendering:pixelated}pre{white-space:pre-wrap}</style>"
            "<h1>Guided versus matched CE + KL control</h1><p>" + html.escape(SCOPE_NOTE) + "</p><p>" + CASE_RULE +
            "</p><p>One source pixel per image pixel; no resize or contrast stretch. Original display chips are not raw radiometry. "
            "VV/VH separate; original, exact native crop/pad and duplicate OBB overlay. Actual eligible fixed-bank same-class "
            "and competing-class references after ID/MMSI exclusion. All cases: not_reviewed.</p>" + "".join(blocks) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    base.dump(folder / "cases.json", cases)
    base.dump(folder / "native_sources.json", sources)
    return cases


def analyze(package=base.PACKAGE):
    cohorts, all_scores, training = load_inputs(ROOT)
    native = full.module_at("matched_native_input", package / "sar_m123/input.py")
    full.annotate_available(cohorts, native)
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    summary = dict(scope=SCOPE_NOTE, primary_comparison="guided_minus_control", training_receipts=training,
        coverage=full.cohort_summary(cohorts, out), scopes={}, visual_status="not_reviewed",
        alignment="18 archives joined by unique ID and labels: 6 new control, 12 reused spatial before/guided",
        rival_rule="Strongest nontruth mean spatial-before logit across three seeds; fixed per query",
        input_paths={"cohorts": str(FULL / "cohorts.json"),
                     "predictions": {f"{scope}_{condition}_s{seed}": str(path)
                                     for (scope, condition, seed), path in input_paths(ROOT, FULL)[0].items()}})
    tables = {name: [] for name in ("metrics", "groups", "groups_summary", "seed_events")}
    for scope in SCOPES:
        rows = cohorts[scope]
        y = np.array([r["label"] for r in rows], int)
        scores = {(c, s): all_scores[scope, c, s] for c in CONDITIONS for s in SEEDS}
        metric_rows = [dict(scope=scope, condition=c, seed=s, **base.metrics(scores[c, s], y)) for c in CONDITIONS for s in SEEDS]
        tables["metrics"].extend(metric_rows)
        result = dict(conditions={c: {m: base.moments([r[m] for r in metric_rows if r["condition"] == c])
                                     for m in METRICS} for c in CONDITIONS}, comparisons={})
        for comparison, (old, new) in COMPARISONS.items():
            events = pair_events(rows, scores, old, new)
            detail, aggregate = full.summarize_comparison(rows, scores, old, new, events)
            for name, records in (("groups", detail), ("groups_summary", aggregate), ("seed_events", events)):
                tables[name].extend(dict(scope=scope, comparison=comparison, **r) for r in records)
            result["comparisons"][comparison] = dict(reference=old, alternative=new, overall=aggregate[0],
                transitions=base.transition_summary(events), pass_bootstrap=full.pass_bootstrap(rows, scores, old, new, repeats=2000))
            if scope == "expanded" and comparison == "guided_minus_control":
                expanded_scores, expanded_events = scores, events
        summary["scopes"][scope] = result
    for name, records in tables.items():
        base.write_csv(out / (name + ".csv"), records)
    base.write_csv(out / "paired_metrics.csv", [r for r in tables["groups"] if r["dimension"] == "all"])
    base.dump(out / "summary.json", summary)
    cases = gallery(out, cohorts, expanded_scores, expanded_events, native)
    lines = ["# Matched intervention numerical results", "", SCOPE_NOTE, ""]
    for scope, result in summary["scopes"].items():
        lines += [f"## {scope}", "", "| Condition | OA mean +/- SD | BA mean +/- SD | NLL mean +/- SD | Brier mean +/- SD |",
                  "|---|---:|---:|---:|---:|"]
        for condition, stats in result["conditions"].items():
            lines.append(f"| {condition} | " + " | ".join(f"{stats[m]['mean']:.6f} +/- {stats[m]['sd']:.6f}" for m in METRICS) + " |")
        lines += ["", "| Comparison | Metric | Paired delta mean | SD | Pass bootstrap low | High |", "|---|---|---:|---:|---:|---:|"]
        for name, comparison in result["comparisons"].items():
            for metric in METRICS:
                row, ci = comparison["overall"], comparison["pass_bootstrap"]["metrics"][metric]
                lines.append(f"| {name} | {metric} | {row['delta_' + metric + '_mean']:.6f} | {row['delta_' + metric + '_sd']:.6f} | {ci['low']:.6f} | {ci['high']:.6f} |")
            bootstrap = comparison["pass_bootstrap"]
            lines += ["", f"{name}: {bootstrap['pass_count']} passes; {bootstrap['missing_class_replicates']}/2000 resamples omit an originally present class.", ""]
        lines += ["| Comparison | State | Seed events | Unique queries |", "|---|---|---:|---:|"]
        for name, comparison in result["comparisons"].items():
            for state in STATES:
                transition = comparison["transitions"][state]
                lines.append(f"| {name} | {state} | {transition['seed_events']} | {transition['unique_ids']} |")
    lines += ["", "Primary attribution is guided-control; guided-before alone includes ordinary continuation. "
              "All paired differences, class/port/pass/actual-polarization groups and three-condition predictions/margins are in CSVs. "
              "Bootstrap resamples whole passes with all paired observations, averages fixed-seed differences, and uses present-class BA. "
              "The development cohort has few passes, so its conditional interval has limited resolution. "
              "Raw training receipts and input provenance are retained in summary.json.", "",
              f"[Native gallery](cases/index.html): {len(cases)} unique expanded queries, {2 * len(cases)} separate VV/VH panels. "
              "Cases include actual eligible same-class and rival references for all displayed seeds and conditions. "
              "Visual status: not_reviewed. Main-agent image review, visible facts and mechanism interpretation remain pending.", ""]
    (out / "NUMERIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    base.dump(out / "complete.json", dict(new_prediction_files=6, reused_prediction_files=12, training_receipts=6,
        gallery_unique_queries=len(cases), panels=2 * len(cases), device="CPU", visual_status="not_reviewed"))
    print(base.json.dumps(dict(output=str(out), gallery=str(out / "cases/index.html")), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=base.PACKAGE)
    args = parser.parse_args()
    analyze(args.package.resolve())
