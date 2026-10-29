"""CPU-only frozen-result analysis, keeping scopes and anchor versions separate.

Default scope is development. --scope final explicitly enables frozen final data.
Run with --gallery only when corrected paired receipts for all three orders exist.
Reuses prior metric/native panel helpers, never their dataset-loading entry points.
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
ROOT = HERE.parent
WORK = HERE.parents[2]
OUT = HERE / "analysis"
SEEDS = (17, 29, 43)
METRICS = ("OA", "BA", "NLL", "Brier")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prior = module("coverage_prior_analysis", ROOT / "m3_reference_delta_20261030/results/analyze.py")
read, dump, metric, events = prior.read, prior.dump, prior.metric, prior.events


def csv_file(path, records):
    if records:
        import csv
        fields = list(dict.fromkeys(k for row in records for k in row))
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)


def load_npz(path, group, classes):
    expected = [r["id"] for r in group]
    y = np.array([r["label"] for r in group])
    with np.load(path, allow_pickle=False) as pack:
        ids = pack["ids"].astype(str).tolist()
        assert len(ids) == len(set(ids)) == len(expected) and set(ids) == set(expected), path
        lookup = {key: i for i, key in enumerate(ids)}
        ix = [lookup[key] for key in expected]
        assert np.array_equal(pack["y"][ix], y), path
        values = {key: pack[key][ix].astype(float) for key in ("score", "baseline") if key in pack}
    for value in values.values():
        assert value.shape == (len(group), len(classes)) and np.isfinite(value).all(), path
    return values


def relation_rows(reports, group):
    labels = {r["id"]: r["label"] for r in group}
    output = []
    for seed, report in reports.items():
        raw = {(str(r["id"]), r["pol"]): r for r in report["raw_relation"]}
        centered = {(str(r["id"]), r["pol"]): r for r in report["centered_relation"]}
        assert raw.keys() == centered.keys()
        for pol in (-1, 0, 1):
            for label in (-1, *sorted(set(labels.values()))):
                keys = [k for k in raw if (pol == -1 or k[1] == pol)
                        and (label == -1 or labels[k[0]] == label)]
                if not keys:
                    continue
                row = dict(seed=seed, pol=pol, label=label, pairs=len(keys),
                           unique_queries=len({k[0] for k in keys}))
                for field in ("same_mass", "entropy"):
                    a = float(np.mean([raw[k][field] for k in keys]))
                    b = float(np.mean([centered[k][field] for k in keys]))
                    row.update({"raw_" + field: a, "centered_" + field: b, "delta_" + field: b - a})
                output.append(row)
    return output


def load_dataset(version, dataset):
    folder = (HERE if version == "legacy" else HERE / "anchor_corrected") / dataset
    manifest = (WORK / "sar_m123_current_20261027/manifests" if dataset == "private"
                else ROOT / "final_oss2_replication_20261029/manifests")
    group = prior.rows(manifest / ("development.json" if dataset == "private" else "validation.json"))
    classes = prior.PRIVATE if dataset == "private" else prior.PUBLIC
    scores, reports, receipts, pending, drift = {}, {}, [], [], []
    for seed in SEEDS:
        path = folder / "center" / f"results_s{seed}.json"
        if not path.is_file():
            pending.append(str(path))
            continue
        report = read(path)
        reports[seed] = report
        p = load_npz(folder / "center" / f"centered_s{seed}.npz", group, classes)
        scores["raw_source", seed] = p["baseline"]
        scores["direct_center", seed] = p["score"]
        for arm in ("control", "center"):
            receipt_path = folder / "trained_center" / f"{arm}_s{seed}.json"
            if not receipt_path.is_file():
                pending.append(str(receipt_path))
                continue
            receipt = read(receipt_path)
            assert receipt["seed"] == seed and receipt["arm"] == arm
            assert receipt["steps"] == (36 if dataset == "private" else 66)
            assert receipt["frozen_parameters_unchanged"] and receipt["saved_state_replay_error"] < 1e-4
            p = load_npz(folder / "trained_center" / f"{arm}_s{seed}.npz", group, classes)
            error = float(np.abs(p["baseline"] - scores["raw_source", seed]).max())
            assert error < 1e-4, (version, dataset, seed, "baseline replay", error)
            drift.append(dict(seed=seed, arm=arm, baseline_max_abs=error))
            name = "trained_center" if arm == "center" else "control"
            scores[name, seed] = p["score"]
            actual = metric(p["score"], np.array([r["label"] for r in group]))
            assert all(abs(actual[k] - receipt["after"][k.lower()]) < 1e-6 for k in METRICS)
            receipts.append(dict(seed=seed, arm=arm, path=str(receipt_path), steps=receipt["steps"],
                                 frozen_parameters_unchanged=True, replay_error=receipt["saved_state_replay_error"]))
    coverage = []
    path = folder / "results_s17.json"
    if path.is_file():
        probe = read(path)
        for arm, item in probe.items():
            name = "bank_" + arm
            scores[name, 17] = load_npz(folder / f"{arm}_s17.npz", group, classes)["score"]
            coverage.append(dict(arm=name, seed=17, references=item["reference_count"],
                                 class_macro_distance=float(np.mean([r["coverage"] for r in item["coverage"]]))))
    return dict(scope="development", version=version, dataset=dataset, folder=folder, group=group, classes=classes,
                refs=prior.rows(manifest / "references.json"), scores=scores, reports=reports,
                receipts=receipts, pending=pending, drift=drift, coverage=coverage,
                y=np.array([r["label"] for r in group]))


def load_final(dataset):
    folder = HERE / "anchor_corrected" / dataset / "evaluation"
    expected = {(a, s) for a in ("control", "center") for s in SEEDS}
    receipt_path = folder / "results.json"
    required = [receipt_path, *[folder / f"{a}_s{s}.npz" for a, s in sorted(expected)]]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return None, missing
    receipt = read(receipt_path)
    records = {(r["arm"], r["seed"]): r for r in receipt["records"]}
    assert len(receipt["records"]) == 6 and records.keys() == expected
    assert not receipt["parameter_selection_on_test"]
    assert max(receipt["native_fresh_checkpoint_errors"].values()) < 1e-4
    public = ROOT / "final_oss2_replication_20261029/manifests"
    manifest = (ROOT / "m2_preadd_expansion_20261028/query_records.json" if dataset == "private"
                else public / "evaluation.json")
    group = prior.rows(manifest)
    assert len(group) == (3075 if dataset == "private" else 378)
    classes = prior.PRIVATE if dataset == "private" else prior.PUBLIC
    y = np.array([r["label"] for r in group])
    scores = {}
    for arm, seed in sorted(expected):
        p = load_npz(folder / f"{arm}_s{seed}.npz", group, classes)
        if ("raw_source", seed) in scores:
            assert np.array_equal(scores["raw_source", seed], p["baseline"])
        scores["raw_source", seed] = p["baseline"]
        scores["trained_center" if arm == "center" else arm, seed] = p["score"]
        for field, key in (("score", "after"), ("baseline", "baseline")):
            actual = metric(p[field], y)
            assert all(abs(actual[k] - records[arm,seed][key][k.lower()]) < 1e-6 for k in METRICS)
    refpath = (WORK / "sar_m123_current_20261027/manifests/references.json" if dataset == "private"
               else public / "references.json")
    return dict(scope="final", version="anchor_corrected", dataset=dataset, folder=folder,
                group=group, classes=classes, refs=prior.rows(refpath), scores=scores, reports={},
                receipts=[dict(path=str(receipt_path), **receipt)], pending=[], drift=[], coverage=[], y=y), []


def output_folder(d):
    folder = OUT / d["version"] / d["dataset"]
    return folder / "evaluation" if d["scope"] == "final" else folder


def summarize(d):
    scores, y = d["scores"], d["y"]
    endpoints = [dict(arm=arm, seed=seed, n=len(y), **metric(z, y)) for (arm, seed), z in scores.items()]
    endpoint_means = []
    for arm in dict.fromkeys(r["arm"] for r in endpoints):
        rows = [r for r in endpoints if r["arm"] == arm]
        endpoint_means.append(dict(arm=arm, orders=len(rows), seeds=[r["seed"] for r in rows],
                                   **{k: float(np.mean([r[k] for r in rows])) for k in METRICS}))
    comparisons = [(arm, "raw_source") for arm in dict.fromkeys(a for a, s in scores) if arm != "raw_source"]
    comparisons += [("trained_center", "control"), ("trained_center", "direct_center")]
    paired, means, classes, flips = [], [], [], []
    for arm, reference in comparisons:
        seeds = [s for s in SEEDS if (arm, s) in scores and (reference, s) in scores]
        if not seeds:
            continue
        values, all_events = [], []
        for seed in seeds:
            a, b = scores[reference, seed], scores[arm, seed]
            before, after = metric(a, y), metric(b, y)
            es = events(a, b, d["group"], seed)
            counts = Counter(e["state"] for e in es)
            row = dict(arm=arm, reference=reference, seed=seed,
                       **{"delta_" + k: after[k] - before[k] for k in METRICS},
                       **{k: counts[k] for k in ("rescued", "harmed", "CC", "WW")},
                       flips=sum(e["changed"] for e in es))
            if arm == "trained_center" and reference == "control":
                saved = scores["raw_source", seed].argmax(1)
                row.update(
                    rescue_original_correct=int(sum(e["state"] == "rescued" and saved[i] == y[i] for i,e in enumerate(es))),
                    rescue_original_wrong=int(sum(e["state"] == "rescued" and saved[i] != y[i] for i,e in enumerate(es))),
                    harm_original_correct=int(sum(e["state"] == "harmed" and saved[i] == y[i] for i,e in enumerate(es))),
                    harm_original_wrong=int(sum(e["state"] == "harmed" and saved[i] != y[i] for i,e in enumerate(es))))
            paired.append(row); values.append(row); all_events.extend(es)
            if arm == "trained_center" and reference in ("raw_source", "control"):
                flips.extend(dict(e, arm=arm, reference=reference) for e in es if e["changed"])
        mean = dict(arm=arm, reference=reference, orders=len(seeds), seeds=seeds)
        for k in METRICS:
            x = [r["delta_" + k] for r in values]
            mean["delta_" + k] = float(np.mean(x))
            mean["sd_" + k] = float(np.std(x, ddof=1)) if len(x) > 1 else None
        for key in ("rescued", "harmed", "CC", "WW"):
            es = [e for e in all_events if e["state"] == key]
            mean[key + "_events"] = len(es)
            mean[key + "_queries"] = len({e["id"] for e in es})
        if arm == "trained_center" and reference == "control":
            for key in ("rescue_original_correct", "rescue_original_wrong", "harm_original_correct", "harm_original_wrong"):
                mean[key + "_events"] = sum(r[key] for r in values)
        means.append(mean)
        if arm == "trained_center" and reference in ("raw_source", "control"):
            for c, name in enumerate(d["classes"]):
                ix = y == c
                if not ix.any():
                    continue
                es = [e for e in all_events if e["label"] == c]
                classes.append(dict(arm=arm, reference=reference, label=c, class_name=name,
                    queries=int(ix.sum()), orders=len(seeds),
                    delta_recall_pp=100 * sum((e["new"] == c) - (e["old"] == c) for e in es) / len(es),
                    rescued=sum(e["state"] == "rescued" for e in es),
                    harmed=sum(e["state"] == "harmed" for e in es),
                    delta_NLL=float(np.mean([metric(scores[arm,s][ix],y[ix])["NLL"] - metric(scores[reference,s][ix],y[ix])["NLL"] for s in seeds]))))
    complete = all((a, s) in scores for a in ("raw_source", "control", "trained_center") for s in SEEDS)
    result = dict(scope=d["scope"], version=d["version"], dataset=d["dataset"], query_count=len(y),
                  present_classes=sorted(np.unique(y).tolist()), classes=d["classes"],
                  paired_three_orders_complete=complete, pending=d["pending"],
                  endpoint_means=endpoint_means, paired_means=means, coverage=d["coverage"],
                  relation=relation_rows(d["reports"], d["group"]), receipts=d["receipts"], baseline_drift=d["drift"],
                  anchor_checks=[dict(seed=s, actual=r.get("actual_anchor_error"), previous_live=r.get("previous_live_anchor_error")) for s,r in d["reports"].items()],
                  notes=["raw_source is the saved baseline logits from center NPZ; no separate raw_source file exists.",
                         "BA averages present classes only. Seeds are minibatch orders, not independent initialization or data.",
                         "Relation rows are direct-center snapshots, not a post-training remeasurement. same_mass averages all 20 slots including zero-mass slots; entropy is in nats.",
                         "Coverage is class-macro of within-class ship-weighted nearest source distance; self/same-ship distances were not masked."])
    if d["scope"] == "final":
        result["notes"] = ["raw_source = historical saved original-model logits; control = freshly evaluated matched value-path training; trained_center = the frozen corrected candidate.",
            "No final relation traces were saved; development relation measurements are not presented as final measurements.",
            "Candidate and hyperparameters were frozen before this final evaluation. Cohorts have historical exposure and are not newly blind tests.",
            "BA averages present classes; three orders share initialization and queries. Rescue/harm events across orders are not independent samples."]
    out = output_folder(d)
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "summary.json", result)
    for name, rows in (("endpoints", endpoints), ("paired", paired), ("paired_means", means),
                       ("class_tradeoff", classes), ("changed_queries", flips),
                       ("source_coverage", d["coverage"]), ("relation", result["relation"])):
        csv_file(out / (name + ".csv"), rows)
    lines = [f"# {d['scope']} / {d['version']} / {d['dataset']}", "", f"Complete three-order pair: {complete}. Queries: {len(y)}.",
             "", "raw_source = original saved-model logits; direct_center = no additional training; control and trained_center = matched value-path training."]
    prior.table(lines, ["Arm", "Orders", *METRICS], [[r["arm"], r["orders"], *[f"{r[k]:.6f}" for k in METRICS]] for r in endpoint_means])
    prior.table(lines, ["Arm - Reference", "Orders", *["Delta " + k for k in METRICS], "Rescue/Harm events"],
                [[r["arm"] + " - " + r["reference"], r["orders"], *[f"{r['delta_'+k]:+.6f}" for k in METRICS],
                  f"{r['rescued_events']}/{r['harmed_events']}"] for r in means])
    lines += ["", *result["notes"], "", "Interpret only this named scope; do not mix development and final outcomes."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def gallery(d):
    assert d["version"] == "anchor_corrected"
    scores, group = d["scores"], d["group"]
    if not all((a, s) in scores for a in ("raw_source", "control", "trained_center") for s in SEEDS):
        return {"status": "waiting_for_three_corrected_pairs"}
    helper = module("coverage_native_panel", ROOT / "spatial_m12_refit_20261029/analysis.py")
    helper.CLASSES = tuple(d["classes"])
    native = module("coverage_native_input", WORK / "sar_m123_standalone/sar_m123/input.py")
    folder = output_folder(d) / "cases"
    folder.mkdir(parents=True, exist_ok=True)
    lookup = {r["id"]: i for i, r in enumerate(group)}
    candidates = {ref: [e for s in SEEDS for e in events(scores[ref,s], scores["trained_center",s], group, s)]
                  for ref in ("control", "raw_source")}
    cases, sources, absent, used, blocks = [], [], [], set(), []
    requests = [(state, ("control", "raw_source"), state) for state in ("rescued", "harmed", "CC", "WW")]
    if d["scope"] == "final":
        requests = [(state, ("raw_source",), state) for state in ("rescued", "harmed", "CC", "WW")]
        requests += [(state, ("control",), "control_" + state) for state in ("rescued", "harmed")]
    for category, references, case_role in requests:
        selected = None
        for reference in references:
            pool = [e for e in candidates[reference] if e["state"] == category
                    and (e["id"] not in used or case_role.startswith("control_"))]
            freq = Counter(e["id"] for e in pool)
            pool.sort(key=lambda e: (-freq[e["id"]], -abs(e["margin_change"]), e["id"], e["seed"]))
            if pool:
                selected = dict(pool[0], reference=reference, state_orders=freq[pool[0]["id"]])
                break
        if selected is None:
            absent.append(case_role)
            continue
        used.add(selected["id"])
        i = lookup[selected["id"]]; q = group[i]; truth = q["label"]; rival = selected["rival"]
        case = dict(selected, case_role=case_role, scope=d["scope"], dataset=d["dataset"], version="anchor_corrected", classes=list(d["classes"]),
                    label_name=d["classes"][truth], rival_name=d["classes"][rival],
                    visual_status="not_reviewed", orders=[], panels={}, relation=[])
        for seed in SEEDS:
            row = dict(seed=seed)
            for arm in ("raw_source", "direct_center", "control", "trained_center"):
                if (arm, seed) not in scores:
                    continue
                z = scores[arm, seed][i]
                row[arm] = dict(prediction=int(z.argmax()), logits=z.tolist(), truth_rival_margin=float(z[truth]-z[rival]))
            for ref in ("control", "raw_source"):
                row["state_vs_"+ref] = prior.state(row[ref]["prediction"], row["trained_center"]["prediction"], truth)
            case["orders"].append(row)
            if seed in d["reports"]:
                for field in ("raw_relation", "centered_relation"):
                    case["relation"].extend(dict(r, seed=seed, phase=field) for r in d["reports"][seed][field] if str(r["id"]) == q["id"])
        legal = [r for r in d["refs"] if helper.eligible(q,r)]
        case["all_legal_reference_ids"] = [r["id"] for r in legal]
        for pol in ("vv", "vh"):
            own = [r for r in legal if r["label"] == truth and r.get(pol)]
            other = [r for r in legal if r["label"] == rival and r.get(pol)]
            shown = own[:2] + other[:2]
            members = [q, *shown]
            roles = ["QUERY"] + ["TRUE REF"]*len(own[:2]) + ["RIVAL REF"]*len(other[:2])
            name = f"{case_role}_{q['id']}_{pol}.png"
            receipts = helper.panel(members, roles, pol, folder/name, native)
            for r, entry in zip(members, receipts):
                entry.update(case_id=q["id"], panel=name, label=r["label"], class_name=d["classes"][r["label"]],
                             pair_legal=entry["role"] != "QUERY", usable_with_query=bool(q.get(pol) and entry["available"]))
            sources.extend(receipts)
            case["panels"][pol] = dict(file=name, same_class_available=len(own), rival_available=len(other),
                                        shown_reference_ids=[r["id"] for r in shown])
        cases.append(case)
        blocks += ["<h2>" + html.escape(f"{case_role}: {q['id']} against {selected['reference']}") + "</h2>",
                   "<pre>" + html.escape(json.dumps(case, ensure_ascii=False, indent=2)) + "</pre>"]
        blocks += [f'<h3>{p.upper()}</h3><img src="{case["panels"][p]["file"]}">' for p in ("vv", "vh")]
    dump(folder / "cases.json", cases)
    dump(folder / "native_sources.json", sources)
    comparison_rule = ("Four raw_source states plus one rescue and one harm versus control when present; absent states are not filled" if d["scope"] == "final"
                       else "control first, raw_source fallback")
    dump(folder / "selection.json", dict(scope=d["scope"], absent=absent, rule=f"One distinct query/state: {comparison_rule}; state frequency, absolute fixed-rival margin change, ID, seed. No random sampling.",
        displayed_references="First two available same-class and first two rival-class entries in the actual fixed bank order; not top attention. Query labels used for posthoc display only.",
        visual_status="not_reviewed", trained_relation_remeasured=False))
    page = '<!doctype html><meta charset="utf-8"><title>Corrected center cases</title><style>body{font:14px Arial;margin:24px}pre{white-space:pre-wrap}img{image-rendering:pixelated;max-width:none}</style>'
    page += '<h1>Anchor-corrected trained-center cases</h1><p>Native uint8 display chips, exact crop/pad and separate OBB overlay. VV/VH separately. Representative cases are not prevalence. Visual review is recorded in CASE_NOTES_ZH.md.</p>'
    (folder / "index.html").write_text(page + "".join(blocks), encoding="utf-8")
    return dict(status="rendered_not_reviewed", cases=len(cases), panels=len(cases)*2, absent=absent)


def compare_versions(data):
    output, relations = [], []
    for dataset in ("private", "oss2"):
        if ("legacy", dataset) not in data or ("anchor_corrected", dataset) not in data:
            continue
        old, new = data["legacy",dataset], data["anchor_corrected",dataset]
        for arm, seed in sorted(old["scores"].keys() & new["scores"].keys()):
            a, b = old["scores"][arm,seed], new["scores"][arm,seed]
            ma, mb = metric(a,new["y"]), metric(b,new["y"])
            output.append(dict(dataset=dataset, arm=arm, seed=seed, top1_changes=int((a.argmax(1)!=b.argmax(1)).sum()),
                max_abs_logits=float(np.abs(a-b).max()), **{"delta_"+k:mb[k]-ma[k] for k in METRICS}))
        for seed in sorted(old["reports"].keys() & new["reports"].keys()):
            for field in ("raw_relation", "centered_relation"):
                a = {(str(r["id"]),r["pol"]):r for r in old["reports"][seed][field]}
                b = {(str(r["id"]),r["pol"]):r for r in new["reports"][seed][field]}
                assert a.keys() == b.keys()
                relations.append(dict(dataset=dataset, seed=seed, phase=field, pairs=len(a),
                    **{"delta_"+k:float(np.mean([b[i][k]-a[i][k] for i in a])) for k in ("same_mass", "entropy")}))
    csv_file(OUT / "implementation_comparison.csv", output)
    csv_file(OUT / "implementation_relation_comparison.csv", relations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("private", "oss2", "all"), default="all")
    parser.add_argument("--version", choices=("legacy", "anchor_corrected", "all"), default="all")
    parser.add_argument("--scope", choices=("development", "final", "all"), default="development")
    parser.add_argument("--gallery", action="store_true")
    args = parser.parse_args()
    datasets = ("private", "oss2") if args.dataset == "all" else (args.dataset,)
    versions = ("legacy", "anchor_corrected") if args.version == "all" else (args.version,)
    OUT.mkdir(parents=True, exist_ok=True)
    data, status = {}, []
    scopes = ("development", "final") if args.scope == "all" else (args.scope,)
    loaded_final = False
    for scope in scopes:
        for version, dataset in [(v, ds) for v in versions for ds in datasets if scope != "final" or v == "anchor_corrected"]:
            if scope == "final":
                d, pending = load_final(dataset)
                if d is None:
                    item = dict(scope=scope, version=version, dataset=dataset, complete=False, pending=pending)
                    status.append(item)
                    print(json.dumps(item), flush=True)
                    continue
                loaded_final = True
            else:
                d = load_dataset(version, dataset)
                data[version,dataset] = d
            result = summarize(d)
            item = dict(scope=scope, version=version, dataset=dataset, complete=result["paired_three_orders_complete"], pending=result["pending"])
            if args.gallery and version == "anchor_corrected":
                item["gallery"] = gallery(d)
            status.append(item)
            print(json.dumps(item), flush=True)
    if data:
        compare_versions(data)
    assert "torch" not in sys.modules, "Analysis must not import model/GPU runtime"
    dump(OUT / f"run_{args.scope}_receipt.json", dict(status=status, cpu_only=True, torch_imported=False,
        final_test_manifests_read=loaded_final, final_test_predictions_read=loaded_final,
        implementation="Only this analysis.py and prior pure CPU helpers; no probe/center/train imports."))


if __name__ == "__main__":
    main()
