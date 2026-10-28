"""CPU-only placement analysis; reuses the prior verified metric/native helpers.

python -B analyze.py
python -B analyze.py --predictions ../shallow_m2_writeback_20261028/predictions.npz --out validation_old

Reads only small logits, manifests and selected native images; no cache/model/GPU.
The sibling shallow_m2_writeback_20261028/analyze.py is a required local helper.
"""
import argparse
import html
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "shallow_m2_writeback_20261028/analyze.py"
spec = importlib.util.spec_from_file_location("placement_analysis_helpers", HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
METRICS, STATES, CLASSES = h.METRICS, h.STATES, h.CLASSES
PACKAGE = HERE.parents[2] / "sar_m123_current_20261027"
PINNED = {("pre3x3", "harmed"): "34278", ("pre3x3", "CC"): "28657",
          ("pre3x3", "WW"): "28198", ("post3x3", "rescued"): "35666",
          ("post3x3", "harmed"): "20791", ("post3x3", "CC"): "35906",
          ("post3x3", "WW"): "20328"}


def load_predictions(path, manifests):
    with np.load(path, allow_pickle=False) as pack:
        data = {k: pack[k] for k in ("ids", "labels", "scores", "bypasses", "initial", "arms", "seeds")}
    ids, labels = data["ids"].astype(str), data["labels"]
    arms = list(data["arms"].astype(str))
    seeds = [int(s) for s in data["seeds"]]
    n = len(ids)
    if len(set(arms)) != len(arms) or "baseline" not in arms or len(set(seeds)) != len(seeds):
        raise ValueError("Expected unique arm/seed axes and a baseline arm")
    expected = (len(arms), len(seeds), n, len(CLASSES))
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or not n:
        raise ValueError("Expected nonempty integer labels[n]")
    for key, shape in (("scores", expected), ("bypasses", expected), ("initial", (n, len(CLASSES)))):
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"Incomplete or misaligned {key}: expected finite {shape}")
    dev = {r["id"]: r for r in manifests["development"]}
    if len(set(ids)) != n or set(ids) != set(dev):
        raise ValueError("IDs must exactly cover the current release development manifest")
    rows = [dev[i] for i in ids]
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise ValueError("Labels disagree with the current release manifest")
    return ids, labels, data["scores"], data["bypasses"], data["initial"], arms, seeds, rows


def comparison_pairs(arms):
    pairs = [(a, "baseline") for a in arms]
    if "stage" in arms:
        pairs += [(a, "stage") for a in ("pre3x3", "post3x3") if a in arms]
    if "pre3x3" in arms and "post3x3" in arms:
        pairs.append(("post3x3", "pre3x3"))
    return pairs


def comparison_rows(arms, seeds, scores, labels, rows, runs):
    metrics, classes, grouped = [], [], []
    run_index = {(r["arm"], int(r["seed"])): r for r in runs}
    for arm, reference in comparison_pairs(arms):
        a, b = arms.index(arm), arms.index(reference)
        for s, seed in enumerate(seeds):
            new, base = scores[a, s], scores[b, s]
            nv, bv = h.metrics(new, labels), h.metrics(base, labels)
            transitions = h.states(base, new, labels)
            identity = dict(arm=arm, reference=reference, seed=seed)
            run = run_index.get((arm, seed), {})
            metrics.append(dict(**identity, n=len(labels), **nv,
                **{f"delta_{m}": nv[m] - bv[m] for m in METRICS},
                **{k: int(v.sum()) for k, v in transitions.items()},
                trainable=run.get("trainable"), seconds=run.get("seconds")))
            pv, pb = h.per_sample(new, labels), h.per_sample(base, labels)
            for c, name in enumerate(CLASSES):
                mask = labels == c
                row = dict(**identity, class_id=c, class_name=name, n=int(mask.sum()),
                    ships=len({r["mmsi"] for i, r in enumerate(rows) if mask[i]}),
                    **{k: int(v[mask].sum()) for k, v in transitions.items()})
                row.update({k: None for k in ("recall", "reference_recall", "delta_recall", "NLL", "Brier", "delta_NLL", "delta_Brier")})
                if mask.any():
                    row["recall"] = float((pv["prediction"][mask] == c).mean() * 100)
                    row["reference_recall"] = float((pb["prediction"][mask] == c).mean() * 100)
                    row["delta_recall"] = row["recall"] - row["reference_recall"]
                    for m in ("NLL", "Brier"):
                        row[m] = float(pv[m][mask].mean())
                        row[f"delta_{m}"] = float((pv[m][mask] - pb[m][mask]).mean())
                classes.append(row)
            for field in ("port", "pass_group", "polarization"):
                groups = np.array([("both" if r.get("vv") and r.get("vh") else "vv_only" if r.get("vv") else "vh_only")
                    if field == "polarization" else r[field] for r in rows])
                for value in sorted(set(groups)):
                    mask = groups == value
                    mv, mb = h.metrics(new[mask], labels[mask]), h.metrics(base[mask], labels[mask])
                    grouped.append(dict(**identity, dimension=field, group=str(value), n=int(mask.sum()),
                        ships=len({r["mmsi"] for i, r in enumerate(rows) if mask[i]}), **mv,
                        **{f"delta_{m}": mv[m] - mb[m] for m in METRICS},
                        **{k: int(v[mask].sum()) for k, v in transitions.items()}))
    return metrics, classes, grouped


def select_cases(ids, labels, scores, initial, arms, seeds, rows, refs):
    s = seeds.index(17) if 17 in seeds else 0
    rival_logits = initial.copy()
    rival_logits[np.arange(len(labels)), labels] = -np.inf
    rivals = rival_logits.argmax(1)
    cases, counts = [], {}
    for arm, reference in comparison_pairs(arms):
        if arm == reference or reference == "pre3x3":
            continue
        a, b = arms.index(arm), arms.index(reference)
        new, base = scores[a, s], scores[b, s]
        index = np.arange(len(labels))
        before = base[index, labels] - base[index, rivals]
        after = new[index, labels] - new[index, rivals]
        transitions = h.states(base, new, labels)
        counts[f"{arm}__vs__{reference}"] = {state: int(mask.sum()) for state, mask in transitions.items()}
        pv, pb = h.per_sample(new, labels), h.per_sample(base, labels)
        for state, mask in transitions.items():
            candidates = np.flatnonzero(mask)
            if not len(candidates):
                continue
            requested = PINNED.get((arm, state)) if reference == "baseline" and seeds[s] == 17 else None
            pinned = [i for i in candidates if ids[i] == requested]
            i = pinned[0] if pinned else sorted(candidates, key=lambda j: (-abs(float(after[j] - before[j])), str(ids[j])))[0]
            row, rival = rows[i], int(rivals[i])
            eligible = [r for r in refs if r["id"] != row["id"] and not
                (r.get("mmsi") and row.get("mmsi") and str(r["mmsi"]) == str(row["mmsi"]))]
            per_seed = []
            for ss, seed in enumerate(seeds):
                old, current = scores[b, ss, i], scores[a, ss, i]
                per_seed.append(dict(seed=seed, reference_prediction=int(old.argmax()), prediction=int(current.argmax()),
                    reference_fixed_margin=float(old[labels[i]] - old[rival]), fixed_margin=float(current[labels[i]] - current[rival])))
            cases.append(dict(arm=arm, reference_arm=reference, seed=seeds[s], state=state, id=str(ids[i]),
                label=int(labels[i]), class_name=CLASSES[labels[i]], port=row["port"], mmsi=row["mmsi"], pass_group=row["pass_group"],
                original_initial_prediction=int(initial[i].argmax()), baseline_prediction=int(base[i].argmax()),
                baseline_field_semantics=f"comparison reference {reference}, see reference_arm", prediction=int(new[i].argmax()),
                fixed_rival=rival, fixed_rival_name=CLASSES[rival], baseline_fixed_margin=float(before[i]),
                fixed_margin=float(after[i]), delta_fixed_margin=float(after[i] - before[i]),
                original_initial_fixed_margin=float(initial[i, labels[i]] - initial[i, rival]),
                baseline_NLL=float(pb["NLL"][i]), NLL=float(pv["NLL"][i]), baseline_Brier=float(pb["Brier"][i]),
                Brier=float(pv["Brier"][i]), baseline_logits=base[i].tolist(), logits=new[i].tolist(),
                eligible_reference_ids=[r["id"] for r in eligible],
                true_reference_ids=[r["id"] for r in eligible if r["label"] == labels[i]],
                fixed_rival_reference_ids=[r["id"] for r in eligible if r["label"] == rival],
                prediction_rival_reference_ids=[r["id"] for r in eligible if r["label"] in {int(base[i].argmax()), int(new[i].argmax())} and r["label"] != labels[i]],
                per_seed=per_seed,
                selection="User-specified path-diagnosis case" if pinned else "Largest absolute initial-fixed-rival margin change within comparison/state; lexical ID tie break",
                fixed_rival_rule="Strongest nontruth initial logit, held fixed across arms",
                visual_status="pending_manual_review"))
    return cases, counts


def gallery(out, cases, rows, refs, native):
    folder = out / "CASES"
    folder.mkdir(exist_ok=True)
    queries, ref_index = {r["id"]: r for r in rows}, {r["id"]: r for r in refs}
    receipts, blocks, notes = [], [], ["# 原图案例待复核", "", "自动选例不是已读图。分组相对reference_arm，seed17；指定诊断例优先，其余按固定竞争类分差变化选取。", ""]
    for case in cases:
        key = f"{case['arm']}_vs_{case['reference_arm']}_{case['state']}_{case['id']}"
        own = case["true_reference_ids"]
        fixed = case["fixed_rival_reference_ids"]
        extra = [i for i in case["prediction_rival_reference_ids"] if i not in own + fixed]
        ref_ids = own + fixed + extra
        group = [queries[case["id"]], *[ref_index[i] for i in ref_ids]]
        roles = ["QUERY", *(["TRUE REF"] * len(own)), *(["FIXED RIVAL REF"] * len(fixed)), *(["PREDICTED RIVAL REF"] * len(extra))]
        title = f"{case['arm']} vs {case['reference_arm']} / {case['state']} / ID {case['id']} / seed {case['seed']}"
        text = (f"truth={case['class_name']}; {CLASSES[case['baseline_prediction']]} -> {CLASSES[case['prediction']]}; "
                f"fixed rival={case['fixed_rival_name']}; margin {case['baseline_fixed_margin']:.6f} -> {case['fixed_margin']:.6f}; "
                f"NLL {case['baseline_NLL']:.6f} -> {case['NLL']:.6f}")
        blocks.append(f"<section><h2>{html.escape(title)}</h2><p>{html.escape(text)}</p>")
        notes += [f"## {title}", "", text, "", f"MMSI={case['mmsi']}; pass={case['pass_group']}; port={case['port']}",
                  f"同类参考{own}；固定竞争类{fixed}；额外预测竞争类{extra}。", "可见事实：待回看。模型证据：实际分差/预测及path_diagnosis.json；尚不能推断物理成因。", ""]
        case["panels"] = {}
        for pol in ("vv", "vh"):
            filename = f"{key}_{pol}.png"
            rec = h.panel(group, pol, folder / filename, native, roles, columns=len(group))
            qvalid = rec[0]["available"]
            for r in rec:
                r.update(case_key=key, panel=filename, allowed_for_query_polarization=bool(qvalid and r["available"]))
            receipts.extend(rec)
            case["panels"][pol] = str(folder / filename)
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(title)}"></div>')
            notes += [f"![{pol}]({filename})", ""]
        blocks.append("</section>")
    banks = "".join(f'<h2>Actual fixed references {p.upper()}</h2><div class="image"><img src="../data_preview/reference_bank_{p}.png"></div>' for p in ("vv", "vh"))
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>M2 placement cases</title>"
        "<style>body{font:15px Arial;margin:24px}h2{font-size:18px}section{border-top:1px solid #aaa}.image{overflow:auto}img{max-width:none;image-rendering:pixelated}</style>"
        "<h1>M2 placement / actual native chips</h1><p>左：原始8bit显示chip；右：模型160像素裁剪/补零。均1:1，不拉伸、不改变对比度。"
        "VV/VH分开；缺失/无效极化明确标记。参考为实际固定库经ID/MMSI排除后的样本；极化可用性见native_sources.json。"
        "代表例不估计总体比例，已读图状态以人工记录为准。</p>"
        "<p><a href='../CASE_NOTES_ZH.md'>人工原图与路径分析</a></p>" + "".join(blocks) + banks + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    (folder / "CASE_NOTES_DRAFT_ZH.md").write_text("\n".join(notes), encoding="utf-8")
    h.dump(folder / "native_sources.json", receipts)
    h.dump(folder / "cases.json", cases)


def report(out, summary, metric_rows):
    lines = ["# M2位置实验：自动数值报告", "", h.SCOPE, "",
        "baseline/stage按本轮协议复用旧模型并重放，pre/post共6次新训练；实际训练记录和诊断原样保存在summary.json。",
        "内部128通道，stage512通道；pre/post等容量，内部对stage包含容量与作用路径差异。", "",
        "## 各seed", "", "| arm | seed | OA | BA | NLL | Brier | dOA | dBA | dNLL | dBrier |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in metric_rows:
        if r["reference"] == "baseline":
            lines.append(f"| {r['arm']} | {r['seed']} | " + " | ".join(f"{r[m]:.6f}" for m in (*METRICS, *[f"delta_{x}" for x in METRICS])) + " |")
    lines += ["", "## 配对均值及seed标准差", "", "先计算每seed指标/差值，再取均值及样本标准差；不是集成模型结果。", "",
              "| arm vs reference | dOA | dBA | dNLL | dBrier |", "|---|---:|---:|---:|---:|"]
    for pair, values in summary["comparisons"].items():
        lines.append(f"| {pair} | " + " | ".join(f"{values['paired'][f'delta_{m}']['mean']:.6f} +/- {values['paired'][f'delta_{m}']['std']:.6f}" for m in METRICS) + " |")
    lines += ["", "## seed17案例组", "", "| comparison | rescued | harmed | CC | WW |", "|---|---:|---:|---:|---:|"]
    for pair, counts in summary["case_group_counts_seed17"].items():
        lines.append(f"| {pair} | " + " | ".join(str(counts[s]) for s in STATES) + " |")
    lines += ["", "classes.csv含逐类召回及风险；grouped.csv含港口/过境/极化。空类召回为空，不作为0进入BA。",
        "bypass.csv是同权重关闭新增回写的诊断，不是重新训练的无模块对照。",
        "每个存在的comparison/state选择至多1例；空组保留0，不以其他seed补齐。",
        "[原始VV/VH及实际参考](CASES/index.html)；[案例机器记录](CASES/cases.json)。",
        "图像事实由人工回看后补充；自动输出不证明物理机制、独立泛化或创新性。", ""]
    (out / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(predictions, out, package):
    predictions, out, package = Path(predictions).resolve(), Path(out).resolve(), Path(package).resolve()
    manifests = h.load_manifests(package)
    ids, labels, scores, bypasses, initial, arms, seeds, rows = load_predictions(predictions, manifests)
    out.mkdir(parents=True, exist_ok=True)
    source = predictions.parent
    runs = h.read_json(source / "runs.json") if (source / "runs.json").exists() else []
    diagnostics = h.read_json(source / "diagnostics.json") if (source / "diagnostics.json").exists() else None
    coverage, native = h.prepare_data_artifacts(out, package, manifests)
    metric_rows, class_rows, grouped = comparison_rows(arms, seeds, scores, labels, rows, runs)
    bypass_rows = []
    for a, arm in enumerate(arms):
        for s, seed in enumerate(seeds):
            full, off = h.metrics(scores[a, s], labels), h.metrics(bypasses[a, s], labels)
            bypass_rows.append(dict(arm=arm, seed=seed, **{f"full_{m}": full[m] for m in METRICS},
                **{f"off_{m}": off[m] for m in METRICS}, **{f"delta_{m}": full[m] - off[m] for m in METRICS},
                **{k: int(v.sum()) for k, v in h.states(bypasses[a, s], scores[a, s], labels).items()}))
    cases, counts = select_cases(ids, labels, scores, initial, arms, seeds, rows, manifests["references"])
    review = h.read_json(out / "VISUAL_REVIEW.json") if (out / "VISUAL_REVIEW.json").exists() else None
    if review:
        for case in cases:
            if case['id'] in review['query_ids']:
                case['visual_status'] = 'Native query/reference content reviewed; see ../CASE_NOTES_ZH.md for actual scope and current-model interpretation.'
    gallery(out, cases, rows, manifests["references"], native)
    summary = dict(status="numeric_complete_visual_review_pending", arms=arms, seeds=seeds, scope=h.SCOPE,
        predictions=str(predictions), manifest_directory=str(package / "manifests"), helper=str(HELPER), coverage=coverage,
        score_semantics="unnormalized logits; float64 stable log-softmax", initial_metrics=h.metrics(initial, labels),
        by_arm={}, comparisons={}, case_group_counts_seed17=counts, cases=cases,
        runs=runs, diagnostics=diagnostics, same_weights_bypass=bypass_rows,
        capacity_note="pre/post:128x20x20, stage:512x20x20; hidden=32; pre/post matched, internal-stage not capacity matched")
    if review:
        summary.update(status='numeric_complete_selected_visual_review_complete', visual_review=review)
    for arm, reference in comparison_pairs(arms):
        selected = [r for r in metric_rows if r["arm"] == arm and r["reference"] == reference]
        values = dict(metrics=h.aggregate(selected, METRICS), paired=h.aggregate(selected, [f"delta_{m}" for m in METRICS]),
            favorable_delta_seeds={m: sum((r[f"delta_{m}"] > 0 if m in ("OA", "BA") else r[f"delta_{m}"] < 0) for r in selected) for m in METRICS})
        summary["comparisons"][f"{arm}__vs__{reference}"] = values
        if reference == "baseline":
            summary["by_arm"][arm] = values
    for name, table in (("metrics", metric_rows), ("classes", class_rows), ("grouped", grouped), ("bypass", bypass_rows)):
        h.write_csv(out / f"{name}.csv", table, list(table[0]))
    h.dump(out / "summary.json", summary)
    report(out, summary, metric_rows)
    for arm, values in summary["by_arm"].items():
        print(arm, {m: round(values["metrics"][m]["mean"], 6) for m in METRICS}, flush=True)
    print(f"Wrote {len(metric_rows)} comparison rows, {len(cases)} cases ({len(set(c['id'] for c in cases))} unique queries): {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.npz")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    args = parser.parse_args()
    analyze(args.predictions, args.out, args.package)
