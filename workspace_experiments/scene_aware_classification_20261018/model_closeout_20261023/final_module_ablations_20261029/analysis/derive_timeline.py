"""Derive temporal handoff tables from completed CPU outputs, without NPZ reload."""
import csv
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("private_timeline_analysis", OUT.parent / "analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def derive():
    summary = a.base.read_json(OUT / "summary.json")
    cases = a.base.read_json(OUT / "cases/cases.json")
    endpoints = {}
    with (OUT / "seed_events.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["comparison"]
            if name.endswith("_after_minus_before"):
                arm = name.removesuffix("_after_minus_before")
                endpoints[row["scope"], arm, int(row["seed"]), row["id"]] = (
                    int(row["reference_prediction"]), int(row["alternative_prediction"]),
                    int(row["label"]), row["mmsi"], row["pass_group"], row["available"])
    bins, rows = {}, []
    for scope in a.SCOPES:
        queries = [(seed, query_id) for s, arm, seed, query_id in endpoints if s == scope and arm == "full"]
        for name in (*a.PRIMARY, a.OBJECTIVE_CONTROL):
            reference, alternative = [c.removesuffix("_after") for c in a.COMPARISONS[name]]
            for seed, query_id in queries:
                old = endpoints[scope, reference, seed, query_id]
                new = endpoints[scope, alternative, seed, query_id]
                before = a.base.state(old[0], new[0], old[2])
                after = a.base.state(old[1], new[1], old[2])
                key = scope, name, before, after
                bucket = bins.setdefault(key, dict(events=0, ids=set(), ships=set()))
                bucket["events"] += 1
                bucket["ids"].add(query_id)
                bucket["ships"].add(old[3])
                rows.append(dict(scope=scope, comparison=name, seed=seed, id=query_id, label=old[2],
                    mmsi=old[3], pass_group=old[4], available=old[5], before_pair_state=before,
                    after_pair_state=after, reference_before=old[0], reference_after=old[1],
                    alternative_before=new[0], alternative_after=new[1],
                    reference_trajectory=a.base.state(old[0], old[1], old[2]),
                    alternative_trajectory=a.base.state(new[0], new[1], new[2])))
            # This cross-check detects a wrong stage/arm join against independent
            # final-pair totals already saved by the primary analysis.
            expected = summary["scopes"][scope]["comparisons"][name]["transitions"]
            for state in a.STATES:
                actual = sum(v["events"] for (s, c, _, after), v in bins.items()
                             if s == scope and c == name and after == state)
                if actual != expected[state]["seed_events"]:
                    raise ValueError(f"Timeline join mismatch: {scope}/{name}/{state}")
    a.base.write_csv(OUT / "timeline_seed_events.csv", rows)
    a.base.write_csv(OUT / "timeline_summary.csv", [dict(scope=s, comparison=c,
        before_pair_state=b, after_pair_state=f, seed_events=v["events"],
        unique_queries=len(v["ids"]), unique_ships=len(v["ships"])) for (s, c, b, f), v in bins.items()])
    lines = ["# Selected case prediction timeline", "",
        "All panels remain not_reviewed. This is prediction evidence, not visual or physical interpretation.",
        "Reference means the comparator model in the named contrast, not a support chip.",
        "B = each arm's own sourcefit; A = its adaptation endpoint. C/W is correctness, not confidence.", "",
        "| Query | Contrast | Selected state | Seed | Truth | Reference B>A | Alternative B>A | Pair B>A | True-rival margin B>A (ref;alt) |",
        "|---|---|---|---:|---|---|---|---|---|"]
    case_rows = []
    for case in cases:
        ref, alt = [c.removesuffix("_after") for c in a.COMPARISONS[case["comparison"]]]
        truth = case["label"]
        for seed_data in case["per_seed"]:
            values = seed_data["conditions"]
            rb, ra, ab, aa = [values[f"{arm}_{stage}"] for arm, stage in
                             ((ref, "before"), (ref, "after"), (alt, "before"), (alt, "after"))]
            before = a.base.state(rb["prediction"], ab["prediction"], truth)
            after = a.base.state(ra["prediction"], aa["prediction"], truth)
            row = dict(id=case["id"], comparison=case["comparison"], category=case["selection_category"],
                seed=seed_data["seed"], truth=a.CLASSES[truth],
                reference_before=a.CLASSES[rb["prediction"]], reference_after=a.CLASSES[ra["prediction"]],
                alternative_before=a.CLASSES[ab["prediction"]], alternative_after=a.CLASSES[aa["prediction"]],
                before_pair_state=before, after_pair_state=after, visual_status="not_reviewed")
            case_rows.append(row)
            margins = f"{rb['top_rival_margin']:.3f}>{ra['top_rival_margin']:.3f}; {ab['top_rival_margin']:.3f}>{aa['top_rival_margin']:.3f}"
            lines.append(f"| {case['id']} | {ref} vs {alt} | {case['selection_category']} | {seed_data['seed']} | {a.CLASSES[truth]} | "
                         f"{row['reference_before']}>{row['reference_after']} | {row['alternative_before']}>{row['alternative_after']} | "
                         f"{before}>{after} | {margins} |")
    a.base.write_csv(OUT / "case_timeline.csv", case_rows)
    (OUT / "CASE_TIMELINE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    a.base.dump(OUT / "timeline_complete.json", dict(device="CPU", source="existing seed_events/cases/summary",
        formal_npz_reloaded=False, events=len(rows), case_seed_rows=len(case_rows),
        checks="Final transition counts agree with the saved primary analysis in all10 cohort/contrast combinations",
        visual_status="not_reviewed"))
    print(a.base.json.dumps(dict(events=len(rows), case_seed_rows=len(case_rows), output=str(OUT)), indent=2))


if __name__ == "__main__":
    derive()
