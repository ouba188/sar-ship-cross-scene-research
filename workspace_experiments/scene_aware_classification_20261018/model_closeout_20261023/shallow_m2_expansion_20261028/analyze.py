"""CPU-only frozen expansion analysis. Run: python -B analyze.py

Reads predictions.npz (logits) and query_records.json beside this script.
Never imports a runner, torch, checkpoints, or training/inference code.
Only writes NUMERIC_REPORT_ZH.md and analysis/ beside this script.
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
PACKAGE = HERE.parents[2] / "sar_m123_current_20261027"
ARMS = ("baseline", "layer2")
SEEDS = (17, 29, 43)
METRICS = ("OA", "BA", "NLL", "Brier")
STATES = ("rescue", "harm", "CC", "WW")
OLD_PORTS = ("Jebel Ali", "Port Klang", "Singapore")
BOOT_SEED = 20261028
SCOPE = (
    "仅分析冻结模型已有 logits，无推理、训练、模型选择或目标集调参。"
    "3075 条历史目标观测不是全新盲测；两个 arm 按 seed 17/29/43 配对。"
    "先逐 seed 计算指标再取均值，不对概率做集成。"
    "OA/BA 为百分数，差值为百分点；BA 对当前分组实际出现的类别等权平均。"
    "NLL 使用自然对数，Brier 为全部 8 类概率平方误差之和，二者越低越好。"
    "三 seed 是已有训练次序，不是三套独立数据或独立预训练。"
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helpers():
    return load_module(HERE.parent / "shallow_m2_writeback_20261028/analyze.py",
                       "shallow_expansion_helpers")


def read_records(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["id"] = str(row["id"])
        row["mmsi"] = str(row["mmsi"])
        for pol in ("vv", "vh"):
            if row.get(pol):
                image = Path(row[pol]["image"])
                row[pol]["image"] = str((image if image.is_absolute() else path.parent / image).resolve())
    return rows


def load_inputs():
    records = read_records(HERE / "query_records.json")
    with np.load(HERE / "predictions.npz", allow_pickle=False) as data:
        ids, labels = data["ids"].astype(str), data["labels"]
        scores = data["scores"].astype(np.float64)
        arms = tuple(data["arms"].astype(str))
        seeds = tuple(int(s) for s in data["seeds"])
    if arms != ARMS or seeds != SEEDS:
        raise ValueError(f"Expected axes {ARMS}/{SEEDS}; got {arms}/{seeds}")
    if ids.shape != (3075,) or labels.shape != (3075,) or scores.shape != (2, 3, 3075, 8):
        raise ValueError("Expected ids[3075], labels[3075], logits[2,3,3075,8]")
    if labels.dtype.kind not in "iu" or not np.isin(labels, np.arange(8)).all():
        raise ValueError("Labels must be integer class IDs 0..7")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite/incomplete logits; do not report results")
    index = {r["id"]: r for r in records}
    if len(index) != len(records) or len(set(ids)) != len(ids) or set(ids) != set(index):
        raise ValueError("Query manifest and prediction IDs must match exactly, without duplicates")
    rows = [index[x] for x in ids]
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise ValueError("Prediction labels differ from the aligned query manifest")
    if len({r["pass_group"] for r in rows}) != 85 or len({r["port"] for r in rows}) != 18:
        raise ValueError("Expected frozen 85-pass / 18-port cohort")
    refs = read_records(PACKAGE / "manifests/references.json")
    if len(refs) != 16 or Counter(r["label"] for r in refs) != Counter({c: 2 for c in range(8)}):
        raise ValueError("Expected actual fixed 16-reference bank, two per class")
    return ids, labels, scores, rows, refs


def sample_values(scores, labels, helper):
    n = len(labels)
    result = {m: np.empty(scores.shape[:3]) for m in ("correct", "NLL", "Brier", "margin")}
    result["prediction"] = scores.argmax(-1)
    for a in range(2):
        for s in range(3):
            values = helper.per_sample(scores[a, s], labels)
            result["correct"][a, s] = values["prediction"] == labels
            for m in ("NLL", "Brier"):
                result[m][a, s] = values[m]
            rivals = scores[a, s].copy()
            rivals[np.arange(n), labels] = -np.inf
            result["margin"][a, s] = scores[a, s, np.arange(n), labels] - rivals.max(-1)
    bc, nc = result["correct"].astype(bool)
    result["state"] = np.select([~bc & nc, bc & ~nc, bc & nc], STATES[:3], default="WW")
    return result


def metric_at(values, labels, indices, arm, seed):
    correct = values["correct"][arm, seed, indices]
    truth = labels[indices]
    return dict(OA=float(100 * correct.mean()),
                BA=float(100 * np.mean([correct[truth == c].mean() for c in np.unique(truth)])),
                NLL=float(values["NLL"][arm, seed, indices].mean()),
                Brier=float(values["Brier"][arm, seed, indices].mean()))


def mean_std(numbers):
    return dict(mean=float(np.mean(numbers)), std=float(np.std(numbers, ddof=1)))


def polarization(row):
    available = [p.upper() for p in ("vv", "vh")
                 if row.get(p) and row[p].get("alignment_valid", True)]
    return "+".join(available) if available else "unavailable"


def cohort_summary(rows, labels):
    return dict(n=len(rows), ships=len({r["mmsi"] for r in rows}),
                passes=len({r["pass_group"] for r in rows}), ports=len({r["port"] for r in rows}),
                class_counts={str(c): int((labels == c).sum()) for c in range(8)},
                polarization=dict(Counter(polarization(r) for r in rows)),
                port_counts=dict(sorted(Counter(r["port"] for r in rows).items())),
                old_three_ports=list(OLD_PORTS),
                polarization_rule="Manifest image present and alignment_valid; missing views are not observed black chips")


def numeric_tables(values, labels, rows, helper, out):
    full, grouped, summaries = [], [], []
    dimensions = {
        "all": ["all"] * len(rows),
        "class": [f"{r['label']}:{helper.CLASSES[r['label']]}" for r in rows],
        "port": [r["port"] for r in rows],
        "pass_group": [r["pass_group"] for r in rows],
        "polarization": [polarization(r) for r in rows],
        "port_scope": ["old_three_ports" if r["port"] in OLD_PORTS else "other_fifteen_ports" for r in rows],
    }
    for dimension, groups in dimensions.items():
        groups = np.asarray(groups)
        for group in sorted(set(groups)):
            indices = np.flatnonzero(groups == group)
            info = dict(dimension=dimension, group=str(group), n=len(indices),
                        ships=len({rows[i]["mmsi"] for i in indices}),
                        passes=len({rows[i]["pass_group"] for i in indices}),
                        present_classes=len(np.unique(labels[indices])))
            by_arm = {arm: [] for arm in ARMS}
            for s, seed in enumerate(SEEDS):
                base = metric_at(values, labels, indices, 0, s)
                events = {state: int((values["state"][s, indices] == state).sum()) for state in STATES}
                for a, arm in enumerate(ARMS):
                    metrics = metric_at(values, labels, indices, a, s)
                    row = dict(**info, arm=arm, seed=seed, **metrics,
                               **{f"delta_{m}": metrics[m] - base[m] for m in METRICS})
                    row.update(events if a else {state: 0 for state in STATES})
                    # CC/WW describe paired outcomes, so use actual baseline self-comparison here.
                    if not a:
                        row["CC"] = int(values["correct"][0, s, indices].sum())
                        row["WW"] = len(indices) - row["CC"]
                    by_arm[arm].append(row)
                    (full if dimension == "all" else grouped).append(row)
            for arm, seed_rows in by_arm.items():
                summary = dict(**info, arm=arm)
                for field in (*METRICS, *(f"delta_{m}" for m in METRICS)):
                    stats = mean_std([r[field] for r in seed_rows])
                    summary.update({f"{field}_{k}": v for k, v in stats.items()})
                for state in STATES:
                    summary[f"{state}_events"] = sum(r[state] for r in seed_rows)
                if arm == "layer2":
                    for state in ("rescue", "harm"):
                        summary[f"unique_{state}"] = int((values["state"][:, indices] == state).any(0).sum())
                else:
                    summary.update(unique_rescue=0, unique_harm=0)
                summaries.append(summary)
    for name, table in (("metrics_by_seed", full), ("breakdown_by_seed", grouped),
                        ("metrics_and_breakdown_summary", summaries)):
        helper.write_csv(out / f"{name}.csv", table, list(table[0]))
    return full, summaries


def event_tables(ids, labels, values, rows, helper, out):
    events, observations = [], []
    for i, row in enumerate(rows):
        info = dict(id=str(ids[i]), label=int(labels[i]), class_name=helper.CLASSES[labels[i]],
                    mmsi=row["mmsi"], port=row["port"], pass_group=row["pass_group"],
                    product=row.get("product", ""), polarization=polarization(row))
        observation = dict(**info)
        for state in STATES:
            observation[f"{state}_events"] = int((values["state"][:, i] == state).sum())
        observation["mean_margin_delta"] = float(np.mean(values["margin"][1, :, i] - values["margin"][0, :, i]))
        observations.append(observation)
        for s, seed in enumerate(SEEDS):
            event = dict(**info, seed=seed, state=str(values["state"][s, i]),
                         baseline_prediction=int(values["prediction"][0, s, i]),
                         layer2_prediction=int(values["prediction"][1, s, i]))
            for field in ("margin", "NLL", "Brier"):
                event.update({f"baseline_{field}": float(values[field][0, s, i]),
                              f"layer2_{field}": float(values[field][1, s, i]),
                              f"delta_{field}": float(values[field][1, s, i] - values[field][0, s, i])})
            events.append(event)
    helper.write_csv(out / "seed_events.csv", events, list(events[0]))
    helper.write_csv(out / "query_outcomes.csv", observations, list(observations[0]))
    summary = {}
    for state in STATES:
        count = (values["state"] == state).sum(0)
        indices = np.flatnonzero(count)
        summary[state] = dict(events=int(count.sum()), unique_queries=len(indices),
                              unique_ships=len({rows[i]["mmsi"] for i in indices}),
                              queries_by_seed_count={str(k): int((count == k).sum()) for k in (1, 2, 3)})
    rescue = (values["state"] == "rescue").any(0)
    harm = (values["state"] == "harm").any(0)
    summary["both_rescue_and_harm_queries"] = int((rescue & harm).sum())
    summary["both_rescue_and_harm_ships"] = len(
        {rows[i]["mmsi"] for i in np.flatnonzero(rescue)} &
        {rows[i]["mmsi"] for i in np.flatnonzero(harm)})
    return summary, observations, events


def cluster_bootstrap(values, labels, rows, replicates=2000, random_seed=BOOT_SEED):
    names, codes = np.unique([r["pass_group"] for r in rows], return_inverse=True)
    counts = np.zeros((len(names), 8))
    np.add.at(counts, (codes, labels), 1)
    # These additive per-observation deltas equal averaging seed metrics after resampling.
    delta = {m: (values[m][1] - values[m][0]).mean(0) for m in ("correct", "NLL", "Brier")}
    correct = np.zeros_like(counts)
    np.add.at(correct, (codes, labels), delta["correct"])
    loss = np.zeros((len(names), 2))
    np.add.at(loss, codes, np.column_stack([delta["NLL"], delta["Brier"]]))
    rng = np.random.default_rng(random_seed)
    boot = np.empty((replicates, 4))
    missing_class_draws = 0
    for b in range(replicates):
        draw = rng.integers(len(names), size=len(names))
        n, d = counts[draw].sum(0), correct[draw].sum(0)
        present = n > 0
        missing_class_draws += int(not present.all())
        boot[b] = [100 * d.sum() / n.sum(), 100 * (d[present] / n[present]).mean(),
                   *(loss[draw].sum(0) / n.sum())]
    intervals = np.quantile(boot, [.025, .975], axis=0)
    receipt = dict(group="pass_group", clusters=len(names), replicates=replicates,
                   random_seed=random_seed, confidence=0.95, method="paired percentile cluster bootstrap",
                   missing_class_draws=missing_class_draws,
                   missing_class_rule="BA averages classes present in that replicate; all draws retained",
                   intervals={m: intervals[:, j].tolist() for j, m in enumerate(METRICS)},
                   scope="Conditional on the six fixed models and fixed reference bank; averages seed metrics, not probabilities. Resample 85 whole passes with replacement with identical weights for both arms and all seeds. Same MMSI may occur across passes; not training-initialization uncertainty or independent-ship bootstrap.")
    return receipt, boot


def select_cases(ids, labels, scores, values, rows, refs, helper):
    counts = {state: (values["state"] == state).sum(0) for state in STATES}
    change = (values["margin"][1] - values["margin"][0]).mean(0)
    used, selected = set(), []

    def add(state, diversity=False):
        pool = [i for i in range(len(ids)) if i not in used and counts[state][i] > 0]
        if not pool:
            return
        previous = [i for i, category in selected if category == state]
        def rank(i):
            novel = (int(all(labels[i] != labels[j] for j in previous)) +
                     int(all(rows[i]["port"] != rows[j]["port"] for j in previous))) if diversity else 0
            return (-int(counts[state][i]), -novel, -abs(float(change[i])), str(ids[i]))
        i = min(pool, key=rank)
        used.add(i)
        selected.append((i, state))

    for state in STATES:
        add(state)
    for state in STATES:
        add(state, diversity=True)
    while len(selected) < min(8, len(ids)):
        before = len(selected)
        for state in STATES:
            if len(selected) < 8:
                add(state, diversity=True)
        if before == len(selected):
            break
    cases = []
    for i, state in selected:
        seeds = np.flatnonzero(values["state"][:, i] == state)
        s = min(seeds, key=lambda j: (-abs(values["margin"][1, j, i] - values["margin"][0, j, i]), SEEDS[j]))
        rival_scores = scores[0, s, i].copy()
        rival_scores[labels[i]] = -np.inf
        rival = int(rival_scores.argmax())
        eligible = [r for r in refs if r["id"] != ids[i] and r["mmsi"] != rows[i]["mmsi"]]
        predictions = values["prediction"][:, :, i]
        confusion = sorted((set(predictions.ravel().tolist()) | {rival}) - {int(labels[i])})
        case = dict(id=str(ids[i]), state=state, label=int(labels[i]), class_name=helper.CLASSES[labels[i]],
                    port=rows[i]["port"], mmsi=rows[i]["mmsi"], pass_group=rows[i]["pass_group"],
                    product=rows[i].get("product", ""), representative_seed=SEEDS[s],
                    outcome_counts={name: int(counts[name][i]) for name in STATES},
                    mean_margin_delta=float(change[i]), fixed_rival=rival, confusion_classes=confusion,
                    eligible_reference_ids=[r["id"] for r in eligible],
                    true_reference_ids=[r["id"] for r in eligible if r["label"] == labels[i]],
                    confusion_reference_ids=[r["id"] for r in eligible if r["label"] in confusion],
                    selection="Two rounds rescue/harm/CC/WW. Rank by 3-seed state count, then (round2) new class/port within state, then absolute mean top1-margin change, then lexical ID. Fill absent states from available groups to eight unique queries. Representative seed maximizes absolute margin change within the selected state.",
                    margin_rule="Top1 margin=true logit minus current strongest nontruth logit; fixed_rival is strongest baseline nontruth at representative seed and retained for paired fixed margins.",
                    visual_status="Generated native panels; manual visual review pending", per_seed=[])
        for j, seed in enumerate(SEEDS):
            item = dict(seed=seed, state=str(values["state"][j, i]),
                        baseline_prediction=int(predictions[0, j]), layer2_prediction=int(predictions[1, j]),
                        baseline_logits=scores[0, j, i].tolist(), layer2_logits=scores[1, j, i].tolist())
            for field in ("margin", "NLL", "Brier"):
                item.update({f"baseline_{field}": float(values[field][0, j, i]),
                             f"layer2_{field}": float(values[field][1, j, i]),
                             f"delta_{field}": float(values[field][1, j, i] - values[field][0, j, i])})
            for a, arm in enumerate(ARMS):
                item[f"{arm}_fixed_margin"] = float(scores[a, j, i, labels[i]] - scores[a, j, i, rival])
            case["per_seed"].append(item)
        cases.append(case)
    return cases


def make_panels(out, cases, rows, refs, helper):
    folder = out / "cases"
    folder.mkdir(exist_ok=True)
    native = helper.native_module(PACKAGE)
    queries = {r["id"]: r for r in rows}
    ref_index = {r["id"]: r for r in refs}
    receipts, sections = [], []
    notes = ["# 原始 chip 案例", "", "确定性事后代表例，不代表总体比例。三 seed 结果与分差见 cases.json。",
             "VV/VH 分开，原始 8-bit 显示值、1:1 像素及模型 crop/pad 同时展示；未作拉伸或对比度增强。",
             "原图面板已生成，尚待主代理逐图回看；不据外观断言科学原因。", ""]
    for case in cases:
        own = [ref_index[x] for x in case["true_reference_ids"]]
        rivals = [ref_index[x] for x in case["confusion_reference_ids"]]
        group = [queries[case["id"]], *own, *rivals]
        roles = ["QUERY", *(["TRUE REF"] * len(own)), *(["CONFUSION REF"] * len(rivals))]
        name = f"{case['state']}_{case['id']}"
        heading = f"{name} / {case['class_name']} / {case['port']} / {case['outcome_counts']}"
        sections.append(f"<section><h2>{html.escape(heading)}</h2><p>Mean margin delta: {case['mean_margin_delta']:+.6f}</p>")
        notes.extend([f"## {heading}", "", f"三 seed 平均 top1 分差变化 {case['mean_margin_delta']:+.6f}；展示 seed {case['representative_seed']}。",
                      f"实际同类参考 {case['true_reference_ids']}；实际混淆类参考 {case['confusion_reference_ids']}。", ""])
        for pol in ("vv", "vh"):
            filename = f"{name}_{pol}.png"
            items = helper.panel(group, pol, folder / filename, native, roles, columns=min(4, len(group)))
            for item, record in zip(items, group):
                item.update(case_id=case["id"], panel=filename,
                            input_spec=record.get(pol), product=record.get("product"))
            receipts.extend(items)
            sections.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(heading)}"></div>')
            notes.append(f"![{pol.upper()}]({filename})")
        sections.append("</section>")
        notes.append("")
    for pol in ("vv", "vh"):
        filename = f"reference_bank_{pol}.png"
        items = helper.panel(refs, pol, folder / filename, native)
        for item, record in zip(items, refs):
            item.update(panel=filename, input_spec=record.get(pol), product=record.get("product"))
        receipts.extend(items)
        sections.append(f'<h2>Actual 16-reference bank / {pol.upper()}</h2><div class="image"><img src="{filename}" alt="Actual reference bank"></div>')
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Frozen shallow M2 cases</title>"
            "<style>body{font:15px Arial,sans-serif;margin:24px;color:#171717}h2{font-size:18px}"
            "section{border-top:1px solid #bbb;margin-top:24px}.image{overflow:auto}"
            "img{max-width:none;image-rendering:pixelated}</style><h1>Frozen shallow M2 cases</h1>"
            f"<p>{html.escape(SCOPE)}</p><p>Native 1:1 pixels; missing views explicitly marked. Manual visual review pending.</p>"
            + "".join(sections) + "</html>")
    (folder / "index.html").write_text(page, encoding="utf-8")
    (folder / "CASE_NOTES_ZH.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    helper.dump(folder / "native_sources.json", receipts)
    helper.dump(folder / "cases.json", cases)


def write_report(summary, full, tables, cases):
    lines = ["# 冻结 shallow-M2 扩围数值报告", "", SCOPE, "",
             f"观测 {summary['cohort']['n']}；MMSI {summary['cohort']['ships']}；pass {summary['cohort']['passes']}；港口 {summary['cohort']['ports']}。",
             "", "## 逐 seed 指标", "",
             "| arm | seed | OA | BA | NLL | Brier | ΔOA | ΔBA | ΔNLL | ΔBrier |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = [*METRICS, *(f"delta_{m}" for m in METRICS)]
    for row in full:
        lines.append(f"| {row['arm']} | {row['seed']} | " + " | ".join(f"{row[m]:.6f}" for m in fields) + " |")
    lines.extend(["", "## 三 seed 均值和样本标准差", "", "差值为 layer2 减同 seed baseline，标准差 ddof=1。", "",
                  "| arm | OA | BA | NLL | Brier | ΔOA | ΔBA | ΔNLL | ΔBrier |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in tables:
        if row["dimension"] == "all":
            lines.append(f"| {row['arm']} | " + " | ".join(f"{row[m + '_mean']:.6f} ± {row[m + '_std']:.6f}" for m in fields) + " |")
    boot = summary["bootstrap"]
    lines.extend(["", "## 85-pass 配对聚类 bootstrap", "",
                  "固定六个已有模型和参考库，2000 次、每次有放回抽取 85 个完整 pass；同一抽样用于两个 arm 和三个 seed。",
                  "统计量是逐 seed 指标差值的均值，未集成概率。以下为条件性 95% percentile 区间，不包含重新训练/初始化不确定性。",
                  "同一 MMSI 可跨 pass，不能将其解释为独立船只 bootstrap。",
                  f"随机种子 {boot['random_seed']}；缺类重复 {boot['missing_class_draws']}，BA 按该次存在类别计算，保留全部 2000 次。", "",
                  "| 指标 | 配对均值差 | 95% 区间 |", "|---|---:|---:|"])
    for m in METRICS:
        lo, hi = boot["intervals"][m]
        lines.append(f"| {m} | {summary['paired_deltas'][m]['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines.extend(["", "## 翻转与重复事件", "", "| 状态 | seed 事件 | 去重查询 | 去重 MMSI | 1/2/3 seed 查询数 |",
                  "|---|---:|---:|---:|---|"])
    for state in STATES:
        row = summary["outcomes"][state]
        lines.append(f"| {state} | {row['events']} | {row['unique_queries']} | {row['unique_ships']} | {row['queries_by_seed_count']} |")
    lines.extend(["", f"既被救回又被误伤的查询：{summary['outcomes']['both_rescue_and_harm_queries']}。事件为重复 seed 结果，不能作为独立样本。",
                  "", "## 分组差值", "", "类别组 OA/BA 即该类 recall；各组 BA 只平均该组实际类别。完整逐 seed、85-pass 结果见 analysis/breakdown_by_seed.csv。", "",
                  "| 维度 | 组 | n | ΔOA | ΔBA | ΔNLL | ΔBrier |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in tables:
        if row["arm"] == "layer2" and row["dimension"] not in ("all", "pass_group"):
            lines.append(f"| {row['dimension']} | {row['group']} | {row['n']} | " + " | ".join(f"{row['delta_' + m + '_mean']:+.6f}" for m in METRICS) + " |")
    lines.extend(["", "## 原图代表案例", "", "选择规则及三 seed 预测、分差、参考 ID 见 analysis/cases/cases.json；原图来源、OBB 和 crop/pad 见 native_sources.json。",
                  "优先状态跨 seed 一致，再兼顾类别/港口差异及分差变化。不存在的状态如实省略；这些事后代表例不估计总体比例。",
                  "VV/VH 分开保留原生像素及模型 crop/pad，展示实际同类、混淆类参考，另含全部 16 条参考库。",
                  "面板生成不等于已人工回看；科学原因和最终判断由主代理完成。", ""])
    for case in cases:
        lines.append(f"- {case['id']}：{case['state']}，{case['class_name']} / {case['port']}，{case['outcome_counts']}，平均分差变化 {case['mean_margin_delta']:+.6f}。")
    lines.extend(["", "[查看原图案例](analysis/cases/index.html)", ""])
    (HERE / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    missing = [name for name in ("predictions.npz", "query_records.json") if not (HERE / name).is_file()]
    if missing:
        message = "等待输入：" + ", ".join(missing)
        report_path = HERE / "NUMERIC_REPORT_ZH.md"
        if not report_path.exists():
            report_path.write_text("# 冻结 shallow-M2 扩围分析待运行\n\n" + message +
                                  "\n\n输入到位后运行：\n\n```powershell\npython -B \"" + str(HERE / "analyze.py") +
                                  "\"\n```\n\n" + SCOPE + "\n\n将输出逐 seed、分组、配对差值、2000 次 pass bootstrap 和 8 个原图案例。\n", encoding="utf-8")
        print(message)
        return
    helper = helpers()
    ids, labels, scores, rows, refs = load_inputs()
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    values = sample_values(scores, labels, helper)
    full, tables = numeric_tables(values, labels, rows, helper, out)
    outcomes, _, _ = event_tables(ids, labels, values, rows, helper, out)
    bootstrap, draws = cluster_bootstrap(values, labels, rows)
    helper.write_csv(out / "bootstrap_replicates.csv",
                     [dict(replicate=i + 1, **dict(zip(METRICS, row))) for i, row in enumerate(draws)],
                     ["replicate", *METRICS])
    cases = select_cases(ids, labels, scores, values, rows, refs, helper)
    make_panels(out, cases, rows, refs, helper)
    layer = [row for row in full if row["arm"] == "layer2"]
    summary = dict(cohort=cohort_summary(rows, labels),
                   by_arm={arm: {m: mean_std([r[m] for r in full if r["arm"] == arm]) for m in METRICS} for arm in ARMS},
                   paired_deltas={m: mean_std([r[f"delta_{m}"] for r in layer]) for m in METRICS},
                   positive_delta_seeds={m: sum(r[f"delta_{m}"] > 0 for r in layer) for m in METRICS},
                   outcomes=outcomes, bootstrap=bootstrap, cases=[c["id"] for c in cases],
                   scope=SCOPE, score_ensemble=False, new_training=False,
                   visual_review="pending main-agent review", references_manifest=str(PACKAGE / "manifests/references.json"))
    helper.dump(out / "summary.json", summary)
    write_report(summary, full, tables, cases)
    print(json.dumps(dict(by_arm=summary["by_arm"], paired_deltas=summary["paired_deltas"],
                          outcomes=outcomes, bootstrap=bootstrap), ensure_ascii=False, indent=2))
    print(f"Report: {HERE / 'NUMERIC_REPORT_ZH.md'}")


if __name__ == "__main__":
    main()
