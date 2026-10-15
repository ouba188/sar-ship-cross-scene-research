"""Independent, evaluation-only audit of the frozen port prior-protection run.

Run --self-test before freeze. Run --freeze-notified only after the main agent
explicitly confirms freeze and paired.csv completion. No project solver imports,
fits, training, or writes outside this directory are performed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
OLD = ROOT.parent / "frozen_candidate_coverage_20261012/budget/tasks/self24ports"
KEYS = ["task", "budget", "draw", "layer", "arm"]
VALUES = ["oa", "ba", "macro_brier"]
PAIRED = VALUES + ["delta_" + name for name in VALUES]
GUARDED = ["guard", "count", "count_common", "confidence", "initial_guard"]
NEW_KEYS = ["query_ids", "support_ids", "raw", "prior_only"] + GUARDED
ATOL = 1e-8


def probability(logits):
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def measure(labels, probs, reference):
    """Class denominators come from this query, including unsupported classes."""
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weight = 1.0 / (len(classes) * counts[inverse])
    correct = probs.argmax(axis=1) == labels
    before = reference.argmax(axis=1) == labels
    target = np.eye(probs.shape[1])[labels]
    loss = ((probs - target) ** 2).sum(axis=1)
    reference_loss = ((reference - target) ** 2).sum(axis=1)
    rescue = correct & ~before
    harm = ~correct & before
    result = {
        "oa": 100.0 * correct.mean(),
        "ba": 100.0 * np.dot(weight, correct),
        "macro_brier": np.dot(weight, loss),
        "delta_oa": 100.0 * (correct.mean() - before.mean()),
        "delta_ba": 100.0 * np.dot(weight, correct.astype(float) - before),
        "delta_macro_brier": np.dot(weight, loss - reference_loss),
        "query_n": len(labels),
        "query_classes": len(classes),
        "rescue": int(rescue.sum()),
        "harm": int(harm.sum()),
        "rescue_ba_pp": 100.0 * weight[rescue].sum(),
        "harm_ba_pp": 100.0 * weight[harm].sum(),
    }
    result["oa_identity_error"] = abs(
        result["delta_oa"] - 100.0 * (rescue.sum() - harm.sum()) / len(labels)
    )
    result["ba_identity_error"] = abs(
        result["delta_ba"] - result["rescue_ba_pp"] + result["harm_ba_pp"]
    )
    per_class = []
    for cls, count in zip(classes, counts):
        mask = labels == cls
        r, h = int(rescue[mask].sum()), int(harm[mask].sum())
        per_class.append({
            "class_id": int(cls), "query_n": int(count),
            "query_classes": len(classes), "rescue": r, "harm": h,
            "recall": 100.0 * correct[mask].mean(),
            "delta_recall": 100.0 * (r - h) / count,
            "rescue_ba": 100.0 * r / (count * len(classes)),
            "harm_ba": 100.0 * h / (count * len(classes)),
            "delta_ba_contribution": 100.0 * (r - h) / (count * len(classes)),
        })
    return result, per_class


def aggregate(frame):
    groups = ["budget", "layer", "arm"]
    per_port = frame.groupby(["task"] + groups, as_index=False)[PAIRED].mean()
    sizes = frame.groupby(["task"] + groups, as_index=False).agg(draws=("draw", "nunique"))
    per_port = per_port.merge(sizes, on=["task"] + groups, validate="one_to_one")
    overall = per_port.groupby(groups, as_index=False)[PAIRED].mean()
    sizes = per_port.groupby(groups, as_index=False).agg(ports=("task", "nunique"))
    return per_port, overall.merge(sizes, on=groups, validate="one_to_one")


def compare(expected, actual, keys, values):
    merged = expected[keys + values].merge(
        actual[keys + values], on=keys, how="left", validate="one_to_one",
        suffixes=("_audit", "_main"), indicator=True,
    )
    merged["matched"] = merged["_merge"] == "both"
    for name in values:
        merged["error_" + name] = abs(merged[name + "_audit"] - merged[name + "_main"])
    errors = merged[["error_" + name for name in values]].to_numpy()
    merged["passed"] = merged.matched & np.isfinite(errors).all(axis=1) & (errors <= ATOL).all(axis=1)
    return merged.drop(columns="_merge")


def self_test():
    labels = np.array([1, 1, 1, 6])
    raw = np.eye(8)[[0, 1, 1, 6]]
    changed = np.eye(8)[[1, 1, 1, 0]]
    values, classes = measure(labels, changed, raw)
    np.testing.assert_allclose([values["oa"], values["ba"], values["macro_brier"]], [75, 50, 1])
    np.testing.assert_allclose(values["delta_ba"], -100.0 / 3)
    np.testing.assert_allclose(sum(row["delta_ba_contribution"] for row in classes), -100.0 / 3)
    assert values["rescue"] == values["harm"] == 1
    assert values["oa_identity_error"] < ATOL and values["ba_identity_error"] < ATOL
    records = []
    for port, draw, value in [("a", 0, 0.0), ("a", 1, 2.0), ("b", 0, 10.0)]:
        records.append(dict(task=port, draw=draw, budget=16, layer="fusion", arm="count",
                            **{name: value for name in PAIRED}))
    frame = pd.DataFrame(records)
    _, overall = aggregate(frame)
    np.testing.assert_allclose(overall.ba.iloc[0], 5.5)
    bad = frame.copy()
    bad.loc[0, "ba"] = 1.0
    assert compare(frame, bad, KEYS, PAIRED).passed.tolist() == [False, True, True]
    a, b = np.array([[0.0, 0.0]]), np.array([[0.0, 2.0]])
    assert a[0, 0] == b[0, 0] and probability(a)[0, 0] != probability(b)[0, 0]
    print("PASS: query-present BA/Brier, unequal draw counts, metric mismatch detection, rescue-harm, logit/probability distinction")


def audit():
    freeze = ROOT / "PREDICTIONS_FROZEN.json"
    if not freeze.is_file():
        raise FileNotFoundError("Main freeze receipt is required before loading query labels")
    frozen = json.loads(freeze.read_text(encoding="utf-8-sig"))
    paired = pd.read_csv(ROOT / "paired.csv")
    paths = sorted((ROOT / "predictions").glob("*/*.npz"))
    if not paths:
        raise ValueError("No completed predictions are available")
    rows, classes, protections, units = [], [], [], []
    tables = {}
    for path in paths:
        task = path.parent.name
        draw, budget = map(int, path.stem.split("_"))
        if budget not in (16, 64, 128) or draw not in range(5):
            continue
        directory = OLD / task
        receipt = json.loads((directory / "PREDICTIONS_FROZEN.json").read_text(encoding="utf-8-sig"))
        manifest = receipt["target_manifest"]
        if manifest not in tables:
            tables[manifest] = pd.read_csv(manifest, usecols=["class_id", "port"])
        table = tables[manifest]
        with np.load(path, allow_pickle=False) as new, np.load(directory / path.name, allow_pickle=False) as old:
            missing = set(NEW_KEYS) - set(new.files)
            if missing:
                raise ValueError(f"Incomplete prediction cache {path}: {sorted(missing)}")
            q, s = new["query_ids"], new["support_ids"]
            np.testing.assert_array_equal(q, old["query_ids"], err_msg=str(path))
            np.testing.assert_array_equal(s, old["support_ids"], err_msg=str(path))
            assert len(s) == budget and len(np.intersect1d(q, s)) == 0, str(path)
            # These IDs index the global manifest, never a reset port-only table.
            query, support = table.iloc[q], table.iloc[s]
            assert query.port.eq(task).all() and support.port.eq(task).all(), str(path)
            labels = query.class_id.to_numpy(dtype=int)
            support_labels = support.class_id.to_numpy(dtype=int)
            np.testing.assert_array_equal(support_labels, old["support_labels"], err_msg=str(path))
            counts = np.bincount(support_labels, minlength=old["base"].shape[1])
            key = dict(task=task, budget=budget, draw=draw)
            raw_error = float(np.max(abs(new["raw"] - old["expert_raw"])))
            base = probability(old["base"])
            raw = probability(old["expert_raw"])
            fusion = (base + raw) / 2
            fusion_error = float(np.max(abs(fusion - old["fusion_raw"])))
            units.append(dict(**key, query_n=len(q), query_classes=len(np.unique(labels)),
                              raw_replay_error=raw_error, fusion_replay_error=fusion_error))
            scores_by_arm = {arm: new[arm] for arm in ["raw", "prior_only"] + GUARDED}
            scores_by_arm.update({arm: old["expert_" + arm] for arm in ("initial", "simple")})
            for arm, scores in scores_by_arm.items():
                expert = probability(scores)
                for layer, p, ref in [("expert", expert, raw), ("fusion", (base + expert) / 2, fusion)]:
                    record_key = dict(**key, layer=layer, arm=arm)
                    result, cls_rows = measure(labels, p, ref)
                    rows.append(dict(**record_key, **result))
                    classes.extend(dict(**record_key, **row, support_n=int(counts[row["class_id"]])) for row in cls_rows)
            absent = counts == 0
            prior = new["prior_only"]
            prior_p = probability(prior)
            for arm in GUARDED:
                scores = new[arm]
                protections.append(dict(**key, arm=arm, absent_classes=int(absent.sum()),
                    absent_logit_error=float(abs(scores[:, absent] - prior[:, absent]).max()) if absent.any() else 0.0,
                    absent_probability_change=float(abs(probability(scores)[:, absent] - prior_p[:, absent]).max()) if absent.any() else 0.0))

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No predictions match the declared budgets/draws")
    result.to_csv(OUT / "recomputed.csv", index=False)
    class_frame = pd.DataFrame(classes)
    class_frame.to_csv(OUT / "class_contributions.csv", index=False)
    units = pd.DataFrame(units)
    units.to_csv(OUT / "units.csv", index=False)
    protections = pd.DataFrame(protections)
    protections.to_csv(OUT / "missing_column_checks.csv", index=False)
    checks = compare(result, paired, KEYS, PAIRED)
    checks.to_csv(OUT / "paired_checks.csv", index=False)
    per_port, overall = aggregate(result)
    per_port.to_csv(OUT / "perport_summary.csv", index=False)
    overall.to_csv(OUT / "equal_port_summary.csv", index=False)
    main_checks = {}
    summary_metrics = [name for name in PAIRED if name != "macro_brier"]
    for name, expected, keys, values in [
        ("port_summary", per_port, ["task", "budget", "layer", "arm"], summary_metrics),
        ("summary", overall, ["budget", "layer", "arm"], summary_metrics + ["ports"]),
        ("classes", class_frame, KEYS + ["class_id"],
         ["query_classes", "support_n", "recall", "delta_recall", "rescue_ba", "harm_ba", "delta_ba_contribution"]),
    ]:
        comparison = compare(expected, pd.read_csv(ROOT / (name + ".csv")), keys, values)
        comparison.to_csv(OUT / (name + "_checks.csv"), index=False)
        main_checks[name] = int((~comparison.passed).sum())
    eligible = units.loc[units.query_n >= 30, ["task", "budget", "draw"]]
    restricted = result.merge(eligible, on=["task", "budget", "draw"], validate="many_to_one")
    _, sensitivity = aggregate(restricted)
    sensitivity.to_csv(OUT / "query_size_sensitivity.csv", index=False)
    comparison = compare(sensitivity, pd.read_csv(ROOT / "query_size_sensitivity.csv"),
                         ["budget", "layer", "arm"], ["ports", "delta_oa", "delta_ba", "delta_macro_brier"])
    comparison.to_csv(OUT / "query_size_sensitivity_checks.csv", index=False)
    main_checks["query_size_sensitivity"] = int((~comparison.passed).sum())
    targets = []
    target_keys = ["task", "budget", "draw", "layer"]
    for candidate, reference in [("count", ref) for ref in ("guard", "count_common", "initial", "simple")] + [("initial_guard", "initial")]:
        joined = result[result.arm == candidate].merge(result[result.arm == reference],
            on=target_keys, suffixes=("", "_ref"), validate="one_to_one")
        for name in VALUES:
            joined["delta_" + name] = joined[name] - joined[name + "_ref"]
        joined["candidate"], joined["reference"] = candidate, reference
        targets.append(joined[target_keys + ["candidate", "reference"] + ["delta_" + name for name in VALUES]])
    target_frame = pd.concat(targets, ignore_index=True)
    target_groups = ["budget", "layer", "candidate", "reference"]
    target_values = ["delta_" + name for name in VALUES]
    target_ports = target_frame.groupby(["task"] + target_groups, as_index=False)[target_values].mean()
    target_summary = target_ports.groupby(target_groups, as_index=False)[target_values].mean()
    target_summary.to_csv(OUT / "targeted_summary.csv", index=False)
    comparison = compare(target_summary, pd.read_csv(ROOT / "targeted_summary.csv"), target_groups, target_values)
    comparison.to_csv(OUT / "targeted_summary_checks.csv", index=False)
    main_checks["targeted_summary"] = int((~comparison.passed).sum())
    expected_units = 23 * 3 * 5
    summary = {
        "completed_units": len(units), "expected_units": expected_units,
        "ports": int(units.task.nunique()), "metric_rows": len(result),
        "paired_failures": int((~checks.passed).sum()),
        "paired_max_error": float(checks.filter(like="error_").max().max()),
        "ba_identity_max_error": float(result.ba_identity_error.max()),
        "oa_identity_max_error": float(result.oa_identity_error.max()),
        "raw_replay_max_error": float(units.raw_replay_error.max()),
        "fusion_replay_max_error": float(units.fusion_replay_error.max()),
        "absent_logit_max_error": float(protections.absent_logit_error.max()),
        "absent_probability_max_change": float(protections.absent_probability_change.max()),
        "query_class_counts": sorted(int(x) for x in units.query_classes.unique()),
        "freeze_receipt": str(freeze), "freeze_prediction_units": frozen.get("prediction_units"),
        "coverage_complete": len(units) == expected_units and units.task.nunique() == 23,
        "main_table_failures": main_checks,
    }
    summary["passed"] = bool(
        summary["paired_failures"] == 0
        and not any(main_checks.values())
        and all(summary[name] <= ATOL for name in (
            "ba_identity_max_error", "oa_identity_max_error", "raw_replay_max_error",
            "fusion_replay_max_error", "absent_logit_max_error"))
    )
    (OUT / "AUDIT_RESULTS.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(summary, overall, sensitivity)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


def write_report(summary, overall, sensitivity):
    lines = [
        "# 港口扩展独立数值审计", "",
        f"已完成缓存数值检查：{'通过' if summary['passed'] else '发现差异，见 paired_checks.csv / AUDIT_RESULTS.json'}。",
        f"覆盖 {summary['ports']} 港口、{summary['completed_units']}/{summary['expected_units']} 个单元；"
        f"重算全部九个正式候选的 expert/fusion 共 {summary['metric_rows']} 行。",
        "未运行拟合或训练；未修改主程序、旧规则或旧缓存。", "",
        "- query/support 使用旧 receipt 指向的全局 manifest 行号，逐个核对旧缓存、港口归属及 support 标签。",
        f"- BA 和 macro Brier 按 query 实际出现类别等权，实际类别数为 {summary['query_class_counts']}；Brier 单样本误差仍覆盖全部输出类别。",
        f"- paired.csv 不一致行数 {summary['paired_failures']}，最大绝对误差 {summary['paired_max_error']:.3g}。",
        f"- rescue−harm 的 BA/OA 恒等式最大误差分别 {summary['ba_identity_max_error']:.3g}/{summary['oa_identity_max_error']:.3g} 个百分点。",
        "- perport_summary.csv 先平均同港口 draws，equal_port_summary.csv 再对港口等权；部分完成时仅使用已完成单元，并保留 draws/ports 分母。",
        f"- 主表逐港/总表/逐类/查询规模敏感性/指定候选对照的不一致行数：{summary['main_table_failures']}。",
        f"- 缺失 support 类相对 prior_only 的 logit 最大差 {summary['absent_logit_max_error']:.3g}；对应概率最大变化 {summary['absent_probability_max_change']:.6g}。固定 logit 不能保证 softmax 概率、argmax 或召回不变。",
        "- 独立源码检查：run.py:41/49 将全局 ID 映射到港口特征局部行；run.py:112 按全局 manifest 评价。规则与求解器仅收到 support 标签 sy（52/56/57 行），未发现 query 标签进入规则或参数选择。Features.target() 的旧港分支读取冻结 phi/prior，新港分支由特征与源模型计算；条件特征仅使用 incidence_deg。",
        "- summarize.py 先按 task/budget/layer/arm 平均 draws，再平均港口；逐类全局贡献除全部 23 港口，有 query 的类内召回另算，两者区分正确。", "",
        "| Budget | Ports | Count−raw OA (pp) | Count−raw BA (pp) | Macro Brier Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    selected = overall[(overall.layer == "fusion") & (overall.arm == "count")]
    for row in selected.itertuples(index=False):
        lines.append(f"| {row.budget} | {row.ports} | {row.delta_oa:.6f} | {row.delta_ba:.6f} | {row.delta_macro_brier:.8f} |")
    lines.extend(["", "预设 n_query>=30 敏感性：", ""])
    selected = sensitivity[(sensitivity.layer == "fusion") & (sensitivity.arm == "count")]
    for row in selected.itertuples(index=False):
        lines.append(f"- B{row.budget}：{row.ports} 港，OA {row.delta_oa:+.6f} pp，BA {row.delta_ba:+.6f} pp，macro Brier {row.delta_macro_brier:+.8f}。")
    lines.extend(["", "OA 与 BA 结果混合，不能概括为跨预算全面净收益；B64/B128 的 BA 退化须保留。三预算 macro Brier 均升高（变差）；排除小查询港口后 B16 的 BA 也转负。",
        "此处是固定规则的历史缓存扩展；五个 support draws 不等于五个训练种子，也不是全新独立数据确认。未审计新增事后缓存分数诊断。", ""])
    (OUT / "AUDIT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze-notified", action="store_true", help="Main agent explicitly confirmed freeze and metrics completion")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.freeze_notified:
        raise SystemExit(audit())
    else:
        parser.error("Use --self-test now, or --freeze-notified after the main completion notice")
