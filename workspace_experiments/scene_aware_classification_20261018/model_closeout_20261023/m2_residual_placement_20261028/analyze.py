"""CPU analysis for matched C/D/E residual placement; no runner/model imports.

Reuses the previous block-placement metrics, case selection and native gallery.
Run after predictions.npz is complete; --data-only prepares manifest previews.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "m2_block_placement_20261028/analyze.py"
spec = importlib.util.spec_from_file_location("residual_placement_helpers", HELPER)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
h = p.h
METRICS, STATES, CLASSES = p.METRICS, p.STATES, p.CLASSES
PACKAGE = p.PACKAGE
ARMS = ("baseline", "stage", "preadd", "postadd")
SEEDS = (17, 29, 43)
CAPACITY = "C/preadd, D/postadd, E/stage: 512x20x20, hidden32, extra197328 parameters each; baseline extra0"


def comparison_pairs(arms):
    return ([(a, "baseline") for a in arms]
            + [("preadd", "stage"), ("postadd", "stage"), ("postadd", "preadd")])


# Configure only this private imported module; the previous round stays intact.
p.comparison_pairs = comparison_pairs
p.PINNED = {}


def prediction_rows(ids, labels, scores, initial, arms, seeds, rows):
    rival_scores = initial.copy()
    rival_scores[np.arange(len(labels)), labels] = -np.inf
    rivals = rival_scores.argmax(1)
    output = []
    for arm, reference in comparison_pairs(arms):
        if arm == reference:
            continue
        a, b = arms.index(arm), arms.index(reference)
        for s, seed in enumerate(seeds):
            new, old = scores[a, s], scores[b, s]
            nv, ov = h.per_sample(new, labels), h.per_sample(old, labels)
            transitions = h.states(old, new, labels)
            for i, row in enumerate(rows):
                label, rival = int(labels[i]), int(rivals[i])
                before, after = float(old[i, label] - old[i, rival]), float(new[i, label] - new[i, rival])
                output.append(dict(arm=arm, reference=reference, seed=seed, id=str(ids[i]),
                    mmsi=row["mmsi"], port=row["port"], pass_group=row["pass_group"],
                    label=label, class_name=CLASSES[label],
                    reference_prediction=int(ov["prediction"][i]), prediction=int(nv["prediction"][i]),
                    state=next(state for state in STATES if transitions[state][i]),
                    fixed_rival=rival, reference_fixed_margin=before, fixed_margin=after,
                    delta_fixed_margin=after-before, reference_NLL=float(ov["NLL"][i]),
                    NLL=float(nv["NLL"][i]), delta_NLL=float(nv["NLL"][i]-ov["NLL"][i]),
                    reference_Brier=float(ov["Brier"][i]), Brier=float(nv["Brier"][i]),
                    delta_Brier=float(nv["Brier"][i]-ov["Brier"][i])))
    return output


def include_path_cases(cases, source, ids, labels, scores, initial, arms, seeds, rows, refs):
    path = source / "path_diagnosis.json"
    if not path.exists():
        return None
    diagnosis = h.read_json(path)
    for diagnostic in diagnosis["cases"]:
        arm, case_id = diagnostic["arm"], str(diagnostic["id"])
        selected = [c for c in cases if c["arm"] == arm and c["reference_arm"] == "baseline" and c["id"] == case_id]
        if not selected:
            i = list(ids).index(case_id)
            s = seeds.index(int(diagnostic["seed"]))
            transitions = h.states(scores[arms.index("baseline"), s], scores[arms.index(arm), s], labels)
            state = next(state for state in STATES if transitions[state][i])
            p.PINNED = {(arm, state): case_id}
            try:
                candidates, _ = p.select_cases(ids, labels, scores, initial, arms, seeds, rows, refs)
            finally:
                p.PINNED = {}
            selected = [c for c in candidates if c["arm"] == arm and c["reference_arm"] == "baseline" and c["id"] == case_id]
            if not selected:
                raise ValueError(f"Path diagnostic case could not be selected: {arm}/{case_id}")
            cases.extend(selected)
        for case in selected:
            case["selection"] = "Main path_diagnosis selected ID; baseline outcome retained, diagnostic kind recorded separately"
            case["path_diagnosis"] = diagnostic
            case["path_diagnosis_scope"] = diagnosis["scope"]
            case["path_fixed_rival_rule"] = "Strongest nontruth baseline seed17 logit; may differ from gallery initial-fixed rival"
            eligible = set(case["eligible_reference_ids"])
            rival_refs = [r["id"] for r in refs if r["id"] in eligible and r["label"] == diagnostic["rival"]]
            case["path_fixed_rival_reference_ids"] = rival_refs
            case["prediction_rival_reference_ids"] = list(dict.fromkeys(case["prediction_rival_reference_ids"] + rival_refs))
    return dict(source=str(path), scope=diagnosis["scope"], cases=len(diagnosis["cases"]),
                ids=sorted({str(c["id"]) for c in diagnosis["cases"]}), visual_status="pending_main_review")


def include_direct_flip_cases(cases, ids, labels, scores, initial, arms, seeds, rows, refs):
    # C/D can agree at seed17 yet differ at another matched training order.
    # Keep those supplementary cases separate from the seed17 outcome counts.
    for s, seed in enumerate(seeds):
        if seed == 17:
            continue
        selected, _ = p.select_cases(ids, labels, scores[:, s:s+1], initial, arms, [seed], rows, refs)
        for case in selected:
            if case["arm"] != "postadd" or case["reference_arm"] != "preadd" or case["state"] not in ("rescued", "harmed"):
                continue
            if any(c["arm"] == "postadd" and c["reference_arm"] == "preadd" and c["state"] == case["state"] and c["id"] == case["id"] for c in cases):
                continue
            i, rival = list(ids).index(case["id"]), case["fixed_rival"]
            case["per_seed"] = []
            for ss, order in enumerate(seeds):
                old, new = scores[arms.index("preadd"), ss, i], scores[arms.index("postadd"), ss, i]
                case["per_seed"].append(dict(seed=order, reference_prediction=int(old.argmax()), prediction=int(new.argmax()),
                    reference_fixed_margin=float(old[labels[i]]-old[rival]), fixed_margin=float(new[labels[i]]-new[rival])))
            case["selection"] = "Supplementary D-vs-C flip at non17 seed; max absolute initial-fixed-rival margin change; not part of seed17 group counts"
            cases.append(case)


def report(out, summary, metrics, classes):
    lines = ["# M2 C/D/E位置实验：CPU数值报告", "", h.SCOPE, "",
        "C=preadd：conv3+BN后、identity相加前；D=postadd：相加后、ReLU前；E=stage：ReLU后。",
        "C/D/E均512通道、每臂新增197328参数；仅位置与其输入/传播路径不同。",
        "协议为6次新训练（C/D各3seed）、6个复用模型（baseline/E各3seed）；实际运行记录见summary.json的runs。", "",
        "## 每臂均值", "", "| arm | OA | BA | NLL | Brier |", "|---|---:|---:|---:|---:|"]
    for arm, value in summary["by_arm"].items():
        lines.append(f"| {arm} | " + " | ".join(f"{value['metrics'][m]['mean']:.6f}" for m in METRICS) + " |")
    lines += ["", "## 当前判断", ""]
    for arm in ("preadd", "postadd"):
        value = summary["comparisons"][f"{arm}__vs__stage"]
        delta = value["paired"]
        favorable = value["favorable_delta_seeds"]
        lines.append(f"{arm}相对E：OA {delta['delta_OA']['mean']:+.6f}、BA {delta['delta_BA']['mean']:+.6f}个百分点；"
            f"NLL {delta['delta_NLL']['mean']:+.6f}、Brier {delta['delta_Brier']['mean']:+.6f}。"
            f"三seed中有利方向个数依次为{favorable['OA']}/{favorable['BA']}/{favorable['NLL']}/{favorable['Brier']}。")
        lines.append("")
    c, d, e = (summary["by_arm"][a]["metrics"] for a in ("preadd", "postadd", "stage"))
    if all(a["BA"]["mean"] < e["BA"]["mean"] and a["NLL"]["mean"] < e["NLL"]["mean"] and a["Brier"]["mean"] < e["Brier"]["mean"] for a in (c, d)):
        lines += ["C/D呈现更好的概率质量与较低BA的取舍，不能据此全面替代E；等容量比较支持位置影响类别竞争，但不单独证明ReLU或物理成因。", ""]
    lines += ["", "## 逐seed结果", "", "| arm | seed | OA | BA | NLL | Brier | dOA | dBA | dNLL | dBrier |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metrics:
        if row["reference"] == "baseline":
            lines.append(f"| {row['arm']} | {row['seed']} | " + " | ".join(
                f"{row[m]:.6f}" for m in (*METRICS, *[f"delta_{m}" for m in METRICS])) + " |")
    lines += ["", "## 同seed配对", "", "逐seed计算差值后取均值和样本标准差；不平均概率、不把480次预测当成480条独立样本。", "",
              "| comparison | dOA | dBA | dNLL | dBrier | favorable seeds OA/BA/NLL/Brier |",
              "|---|---:|---:|---:|---:|---|"]
    for pair, value in summary["comparisons"].items():
        if pair == "baseline__vs__baseline":
            continue
        lines.append(f"| {pair} | " + " | ".join(
            f"{value['paired'][f'delta_{m}']['mean']:.6f} +/- {value['paired'][f'delta_{m}']['std']:.6f}"
            for m in METRICS) + " | " + "/".join(str(value["favorable_delta_seeds"][m]) for m in METRICS) + " |")
    lines += ["", "## 救回与误伤", "", "事件数跨3seed累加；独立ID/船只数另列，同一ID可以在不同seed属于两组。", "",
              "| comparison | rescue events/IDs/ships | harm events/IDs/ships | both IDs |",
              "|---|---|---|---|---|"]
    for pair, value in summary["transition_totals"].items():
        lines.append(f"| {pair} | " + " | ".join(
            f"{value[state]['events']}/{value[state]['ids']}/{value[state]['ships']}" for state in ("rescued", "harmed"))
            + f" | {len(value['rescued_and_harmed_ids'])} |")
    lines += ["", "## 逐类召回变化", "", "以下为三seed平均百分点差；缺失类保留NA。", "",
              "| comparison | class | n | dRecall | dNLL | dBrier |", "|---|---|---:|---:|---:|---:|"]
    for arm, reference in comparison_pairs(summary["arms"]):
        if arm == reference:
            continue
        for c, name in enumerate(CLASSES):
            selected = [r for r in classes if r["arm"] == arm and r["reference"] == reference and r["class_id"] == c]
            values = ["NA" if selected[0][m] is None else f"{np.mean([r[m] for r in selected]):.6f}"
                      for m in ("delta_recall", "delta_NLL", "delta_Brier")]
            lines.append(f"| {arm} vs {reference} | {name} | {selected[0]['n']} | " + " | ".join(values) + " |")
    lines += ["", "## 案例与边界", "",
        "seed17按六组比较分别选择持续判对、持续判错、救回、误伤；空组如实保留0，不从其他seed补齐。",
        "每组按初始模型固定竞争类分差绝对变化最大选1例，ID字典序打破平局；包含D对C、C/D对E的直接比较。另补主线path_diagnosis选例，以及其他seed实际存在的D/C翻转代表例；补充例不计入seed17分组。",
        "cases.json同时保存全部3seed预测和固定竞争类分差、实际同类/竞争类参考ID；native_sources.json记录路径及有效性。",
        "path_diagnosis的竞争类按baseline seed17定义；图库主分差按initial定义，两者分开记录，并补齐诊断竞争类参考。",
        "[原始VV/VH及实际参考图库](CASES/index.html)。生成图像不代表已人工看图，本CPU报告不作图像物理成因判断。",
        "paired_predictions.csv保留所有逐样本配对；classes.csv、grouped.csv含类别及港口/过境/极化分层。",
        "bypass.csv为同权重关闭新增回写的诊断，不能替代重新训练对照；主线负责路径干预及最终解释。",
        "仅3个开发过境且开发集反复使用，结果不等于独立跨场景确认；不以训练秒数推断端到端效率。", ""]
    (out / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(predictions, out, package):
    predictions, out, package = Path(predictions).resolve(), Path(out).resolve(), Path(package).resolve()
    manifests = h.load_manifests(package)
    ids, labels, scores, bypasses, initial, arms, seeds, rows = p.load_predictions(predictions, manifests)
    if set(arms) != set(ARMS) or set(seeds) != set(SEEDS):
        raise ValueError(f"Expected C/D/E arms {ARMS} and seeds {SEEDS}; got {arms}, {seeds}")
    if [len(manifests[s]) for s in ("source_adapt", "development", "references")] != [136, 160, 16]:
        raise ValueError("Expected unchanged 136 source/160 development/16 references")
    out.mkdir(parents=True, exist_ok=True)
    source = predictions.parent
    runs = h.read_json(source / "runs.json") if (source / "runs.json").exists() else []
    diagnostics = h.read_json(source / "diagnostics.json") if (source / "diagnostics.json").exists() else None
    coverage, native = h.prepare_data_artifacts(out, package, manifests)
    metrics, classes, grouped = p.comparison_rows(arms, seeds, scores, labels, rows, runs)
    paired = prediction_rows(ids, labels, scores, initial, arms, seeds, rows)
    bypass = []
    for a, arm in enumerate(arms):
        for s, seed in enumerate(seeds):
            full, off = h.metrics(scores[a, s], labels), h.metrics(bypasses[a, s], labels)
            bypass.append(dict(arm=arm, seed=seed, **{f"full_{m}": full[m] for m in METRICS},
                **{f"off_{m}": off[m] for m in METRICS}, **{f"delta_{m}": full[m]-off[m] for m in METRICS},
                **{k: int(v.sum()) for k, v in h.states(bypasses[a, s], scores[a, s], labels).items()}))
    cases, counts = p.select_cases(ids, labels, scores, initial, arms, seeds, rows, manifests["references"])
    path_case_coverage = include_path_cases(cases, source, ids, labels, scores, initial, arms, seeds, rows, manifests["references"])
    include_direct_flip_cases(cases, ids, labels, scores, initial, arms, seeds, rows, manifests["references"])
    p.gallery(out, cases, rows, manifests["references"], native)
    notes = out / "CASES/CASE_NOTES_DRAFT_ZH.md"
    text = notes.read_text(encoding="utf-8").replace(
        "分组相对reference_arm，seed17；指定诊断例优先，其余按固定竞争类分差变化选取。",
        "分组相对reference_arm，seed以各标题为准；主体seed17，D/C额外翻转例来自其他seed。"
        "指定诊断例优先，其余按固定竞争类分差变化选取。额外竞争类参考也包含path_diagnosis的baseline固定竞争类，"
        "与initial固定竞争类分开记录于cases.json和case_index.csv。")
    notes.write_text(text, encoding="utf-8")
    h.write_csv(out / "case_index.csv", [dict(arm=c["arm"], reference=c["reference_arm"], seed=c["seed"],
        state=c["state"], id=c["id"], label=c["label"], fixed_rival=c["fixed_rival"],
        path_kind=c.get("path_diagnosis", {}).get("kind"), path_rival=c.get("path_diagnosis", {}).get("rival"),
        vv=c["panels"]["vv"], vh=c["panels"]["vh"], visual_status=c["visual_status"]) for c in cases],
        ["arm", "reference", "seed", "state", "id", "label", "fixed_rival", "path_kind", "path_rival", "vv", "vh", "visual_status"])
    summary = dict(status="numeric_complete_visual_review_pending", arms=arms, seeds=seeds, scope=h.SCOPE,
        predictions=str(predictions), helper=str(HELPER), manifest_directory=str(package / "manifests"),
        coverage=coverage, capacity_note=CAPACITY, planned_new_fits=6, planned_reused_models=6,
        score_semantics="unnormalized logits; float64 stable log-softmax", initial_metrics=h.metrics(initial, labels),
        by_arm={}, comparisons={}, transition_totals={}, case_group_counts_seed17=counts, cases=cases,
        runs=runs, diagnostics=diagnostics, same_weights_bypass=bypass,
        path_case_coverage=path_case_coverage,
        execution=h.read_json(source / "complete.json") if (source / "complete.json").exists() else None)
    for arm, reference in comparison_pairs(arms):
        selected = [r for r in metrics if r["arm"] == arm and r["reference"] == reference]
        value = dict(metrics=h.aggregate(selected, METRICS),
            paired=h.aggregate(selected, [f"delta_{m}" for m in METRICS]),
            favorable_delta_seeds={m: sum((r[f"delta_{m}"] > 0 if m in ("OA", "BA") else r[f"delta_{m}"] < 0)
                                         for r in selected) for m in METRICS})
        key = f"{arm}__vs__{reference}"
        summary["comparisons"][key] = value
        if reference == "baseline":
            summary["by_arm"][arm] = value
        if arm != reference:
            subset = [r for r in paired if r["arm"] == arm and r["reference"] == reference]
            total, id_sets = {}, {}
            for state in STATES:
                events = [r for r in subset if r["state"] == state]
                id_sets[state] = {r["id"] for r in events}
                total[state] = dict(events=len(events), ids=len(id_sets[state]), ships=len({r["mmsi"] for r in events}))
            total["rescued_and_harmed_ids"] = sorted(id_sets["rescued"] & id_sets["harmed"])
            summary["transition_totals"][key] = total
    for name, table in (("metrics", metrics), ("classes", classes), ("grouped", grouped),
                        ("bypass", bypass), ("paired_predictions", paired)):
        h.write_csv(out / f"{name}.csv", table, list(table[0]))
    h.dump(out / "summary.json", summary)
    report(out, summary, metrics, classes)
    for arm, value in summary["by_arm"].items():
        print(arm, {m: round(value["metrics"][m]["mean"], 6) for m in METRICS}, flush=True)
    print(f"Wrote {len(metrics)} metric rows, {len(paired)} paired observations, {len(cases)} cases: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.npz")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.data_only:
        args.out.mkdir(parents=True, exist_ok=True)
        h.prepare_data_artifacts(args.out, args.package, h.load_manifests(args.package))
        print("CPU manifest coverage and native reference previews ready; no predictions analyzed.")
    else:
        analyze(args.predictions, args.out, args.package)
