"""CPU-only analysis of saved shallow-M2 logits; never imports run.py.

python -B analyze.py --data-only
python -B analyze.py

Outputs stay beside this script. Scores and initial are unnormalized logits.
No model loading, fitting, checkpoint selection, or reference resampling occurs.
"""
import argparse
from collections import Counter
import csv
import html
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2] / "sar_m123_current_20261027"
ARMS = ("baseline", "layer1", "layer2", "both")
SEEDS = (17, 29, 43)
CLASSES = ("bulk_carrier", "fishing", "general_cargo", "chemical_tanker",
           "container", "crude_oil_tanker", "tug", "offshore")
METRICS = ("OA", "BA", "NLL", "Brier")
STATES = ("rescued", "harmed", "CC", "WW")
METRIC_FIELDS = ["arm", "seed", "n", *METRICS,
                 *[f"delta_{m}" for m in METRICS], *STATES,
                 "trainable", "seconds"]
CLASS_FIELDS = ["arm", "seed", "class_id", "class_name", "n", "ships",
                "recall", "baseline_recall", "delta_recall", "NLL", "Brier",
                "delta_NLL", "delta_Brier", *STATES]
SCOPE = ("136条源适配样本的小范围研发；160条反复复用的开发集不是独立确认集。"
         "三seed逐一匹配baseline，不将模型平均概率作为主结果；seed不是独立船只或独立数据集。"
         "OA/BA为百分数，差值为百分点；BA仅平均开发集中实际出现的7类。"
         "NLL为自然对数交叉熵，Brier为8类概率平方误差之和；后二者越低越好。"
         "参考库固定16条、每类2条，其中1条无VH；不据外观命名物理原因。")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifests(package):
    result = {}
    for split in ("source_adapt", "development", "references"):
        manifest = package / "manifests" / f"{split}.json"
        rows = read_json(manifest)
        for row in rows:
            row["id"] = str(row["id"])
            for pol in ("vv", "vh"):
                if row.get(pol):
                    path = Path(row[pol]["image"])
                    row[pol]["image"] = str(path if path.is_absolute() else manifest.parent / path)
        result[split] = rows
    return result


def native_module(package):
    spec = importlib.util.spec_from_file_location("shallow_analysis_native", package / "sar_m123/input.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data_summary(manifests):
    summary = {}
    for split, rows in manifests.items():
        summary[split] = dict(
            n=len(rows), ships=len({r["mmsi"] for r in rows}),
            acquisitions=len({r["pass_group"] for r in rows}),
            ports=dict(sorted(Counter(r["port"] for r in rows).items())),
            acquisitions_by_id=dict(sorted(Counter(r["pass_group"] for r in rows).items())),
            classes=[dict(class_id=c, class_name=name,
                          n=sum(r["label"] == c for r in rows),
                          ships=len({r["mmsi"] for r in rows if r["label"] == c}))
                     for c, name in enumerate(CLASSES)],
            polarization=dict(Counter("both" if r.get("vv") and r.get("vh") else
                                     "vv_only" if r.get("vv") else "vh_only" for r in rows)))
    summary["overlaps"] = {}
    for a, b in (("source_adapt", "development"), ("source_adapt", "references"),
                 ("development", "references")):
        summary["overlaps"][f"{a}__{b}"] = {
            field: sorted({str(r[field]) for r in manifests[a]} &
                          {str(r[field]) for r in manifests[b]})
            for field in ("id", "mmsi", "pass_group", "port")}
    return summary


def native_tile(row, pol, native, role=""):
    font = ImageFont.load_default(size=12)
    view = native.prepare_view(row.get(pol))
    raw_size = (0, 0) if view is None else (view["raw"].shape[1], view["raw"].shape[0])
    width = max(360, raw_size[0] + 180)
    height = max(160, raw_size[1]) + 90
    tile = Image.new("RGB", (width, height), "#f5f5f5")
    draw = ImageDraw.Draw(tile)
    draw.text((8, 5), f"{role} ID {row['id']} / {pol.upper()}", fill="black", font=font)
    draw.text((8, 23), f"{row['label']} {CLASSES[row['label']]} / {row['port']}", fill="black", font=font)
    receipt = dict(id=row["id"], pol=pol, role=role, label=row["label"],
                   mmsi=row["mmsi"], pass_group=row["pass_group"], port=row["port"],
                   source_path=row[pol]["image"] if row.get(pol) else None,
                   available=view is not None and view["valid_target"],
                   display="Native 8-bit values, 1 source pixel = 1 output pixel; no resize or contrast stretch")
    if view is None:
        draw.text((8, 65), "UNAVAILABLE (not a black observed chip)", fill="#a00000", font=font)
    else:
        tile.paste(Image.fromarray(view["raw"]).convert("RGB"), (8, 66))
        tile.paste(Image.fromarray(view["pixels"]).convert("RGB"), (width - 168, 66))
        draw.text((8, 47), f"raw {raw_size[0]}x{raw_size[1]}", fill="black", font=font)
        draw.text((width - 168, 47), "model crop/pad 160x160", fill="black", font=font)
        receipt.update(raw_width=raw_size[0], raw_height=raw_size[1],
                       placement=view["placement"], valid_target=view["valid_target"])
    return tile, receipt


def panel(rows, pol, path, native, roles=None, columns=4):
    tiles, receipts = [], []
    for i, row in enumerate(rows):
        tile, receipt = native_tile(row, pol, native, roles[i] if roles else "REF")
        tiles.append(tile)
        receipts.append(receipt)
    width, height = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (columns * width, ((len(tiles) + columns - 1) // columns) * height), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * width, (i // columns) * height))
    sheet.save(path)
    return receipts


def prepare_data_artifacts(out, package, manifests):
    native = native_module(package)
    coverage = data_summary(manifests)
    dump(out / "data_summary.json", coverage)
    folder = out / "data_preview"
    folder.mkdir(exist_ok=True)
    receipts = []
    for split, ids in (("source_adapt", ("294", "4256")), ("development", ("104", "3951"))):
        index = {r["id"]: r for r in manifests[split]}
        for case_id in ids:
            for pol in ("vv", "vh"):
                receipt = panel([index[case_id]], pol, folder / f"{split}_{case_id}_{pol}.png",
                                native, [split], columns=1)[0]
                receipt["selection"] = "Fixed before reading this experiment's predictions; historical native-input examples plus source 294"
                receipts.append(receipt)
    refs = manifests["references"]
    if len(refs) != 16 or Counter(r["label"] for r in refs) != Counter({c: 2 for c in range(8)}):
        raise ValueError("Expected actual fixed bank of 16 references, two per output class")
    for pol in ("vv", "vh"):
        receipts.extend(panel(refs, pol, folder / f"reference_bank_{pol}.png", native))
    dump(folder / "native_sources.json", receipts)
    return coverage, native


def per_sample(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probability = np.exp(log_prob)
    return dict(prediction=logits.argmax(axis=1),
                NLL=-log_prob[np.arange(len(labels)), labels],
                Brier=np.square(probability - np.eye(8)[labels]).sum(axis=1))


def metrics(logits, labels):
    values = per_sample(logits, labels)
    correct = values["prediction"] == labels
    return dict(OA=float(correct.mean() * 100),
                BA=float(np.mean([correct[labels == c].mean() for c in np.unique(labels)]) * 100),
                NLL=float(values["NLL"].mean()), Brier=float(values["Brier"].mean()))


def states(base, new, labels):
    bc, nc = base.argmax(axis=1) == labels, new.argmax(axis=1) == labels
    return dict(rescued=~bc & nc, harmed=bc & ~nc, CC=bc & nc, WW=~bc & ~nc)


def aggregate(rows, fields):
    return {field: dict(mean=float(np.mean([r[field] for r in rows])),
                        std=float(np.std([r[field] for r in rows], ddof=1))) for field in fields}


def load_predictions(path, manifests):
    with np.load(path, allow_pickle=False) as pack:
        ids, labels = pack["ids"].astype(str), pack["labels"]
        scores, initial = pack["scores"].astype(float), pack["initial"].astype(float)
        arms, seeds = tuple(pack["arms"].astype(str)), tuple(int(x) for x in pack["seeds"])
    if arms != ARMS or seeds != SEEDS:
        raise ValueError(f"Unexpected arm/seed axes: {arms}, {seeds}")
    n = len(ids)
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or scores.shape != (4, 3, n, 8) or initial.shape != (n, 8):
        raise ValueError("Expected labels[n], initial[n,8], scores[4,3,n,8]")
    if not np.isfinite(scores).all() or not np.isfinite(initial).all():
        raise ValueError("Predictions are incomplete or contain nonfinite logits; no results reported")
    dev = {r["id"]: r for r in manifests["development"]}
    if len(set(ids)) != n or set(ids) != set(dev):
        raise ValueError("Prediction IDs must cover the actual 160 development observations exactly once")
    rows = [dev[case_id] for case_id in ids]
    if not np.array_equal(labels, [r["label"] for r in rows]):
        raise ValueError("Prediction labels differ from the development manifest")
    return ids, labels, scores, initial, rows


def select_cases(ids, labels, scores, initial, rows, refs):
    baseline = scores[0, 0]
    competitor_logits = initial.copy()
    competitor_logits[np.arange(len(labels)), labels] = -np.inf
    rival = competitor_logits.argmax(axis=1)
    idx = np.arange(len(labels))
    margin0 = baseline[idx, labels] - baseline[idx, rival]
    cases, counts = [], {}
    for arm_index, arm in enumerate(ARMS[1:], 1):
        new = scores[arm_index, 0]
        margin1 = new[idx, labels] - new[idx, rival]
        group = states(baseline, new, labels)
        counts[arm] = {name: int(mask.sum()) for name, mask in group.items()}
        old_values, new_values = per_sample(baseline, labels), per_sample(new, labels)
        for state in STATES:
            candidates = np.flatnonzero(group[state])
            if not len(candidates):
                continue
            i = sorted(candidates, key=lambda j: (-abs(margin1[j] - margin0[j]), str(ids[j])))[0]
            row = rows[i]
            eligible = [r for r in refs if r["id"] != row["id"] and
                        not (r.get("mmsi") and row.get("mmsi") and r["mmsi"] == row["mmsi"])]
            cases.append(dict(arm=arm, seed=17, state=state, id=str(ids[i]),
                label=int(labels[i]), class_name=CLASSES[labels[i]], port=row["port"],
                mmsi=row["mmsi"], pass_group=row["pass_group"],
                original_initial_prediction=int(initial[i].argmax()),
                baseline_prediction=int(baseline[i].argmax()), prediction=int(new[i].argmax()),
                fixed_rival=int(rival[i]), fixed_rival_name=CLASSES[rival[i]],
                original_initial_fixed_margin=float(initial[i, labels[i]] - initial[i, rival[i]]),
                baseline_fixed_margin=float(margin0[i]), fixed_margin=float(margin1[i]),
                delta_fixed_margin=float(margin1[i] - margin0[i]),
                baseline_NLL=float(old_values["NLL"][i]), NLL=float(new_values["NLL"][i]),
                baseline_Brier=float(old_values["Brier"][i]), Brier=float(new_values["Brier"][i]),
                baseline_logits=baseline[i].tolist(), logits=new[i].tolist(),
                eligible_reference_ids=[r["id"] for r in eligible],
                true_reference_ids=[r["id"] for r in eligible if r["label"] == labels[i]],
                fixed_rival_reference_ids=[r["id"] for r in eligible if r["label"] == rival[i]],
                selection="seed17 only; within arm/state largest absolute fixed-rival margin change, then lexical ID; posthoc representative, not prevalence estimate",
                fixed_rival_rule="Strongest nontruth class in original initial logits; fixed across all arms, including WW",
                visual_status="Awaiting main-agent review; generated panels are not evidence of having viewed them"))
    return cases, counts


def case_artifacts(out, cases, rows, refs, native):
    folder = out / "CASES"
    folder.mkdir(exist_ok=True)
    queries, ref_index = {r["id"]: r for r in rows}, {r["id"]: r for r in refs}
    receipts, blocks = [], []
    notes = ["# 原图案例草案", "", "以下案例由实际seed17预测选择；图像解释由主代理查看后补充。",
             "分组比较matched baseline(seed17)；竞争类固定为initial原基线最强非真值类。CC/WW仍分析证据变化。",
             "代表例按固定竞争类分差变化绝对值选取，不代表总体发生率；各组数量见summary.json。", ""]
    for case in cases:
        name = f"{case['arm']}_{case['state']}_{case['id']}"
        query = queries[case["id"]]
        own = [ref_index[x] for x in case["true_reference_ids"]]
        rival = [ref_index[x] for x in case["fixed_rival_reference_ids"]]
        group = [query, *own, *rival]
        roles = ["QUERY", *(["TRUE REF"] * len(own)), *(["FIXED RIVAL REF"] * len(rival))]
        heading = (f"{case['arm']} / {case['state']} / ID {case['id']} / seed17 / "
                   f"true {case['class_name']} / fixed rival {case['fixed_rival_name']}")
        blocks.append(f"<section><h2>{html.escape(heading)}</h2><p>"
                      f"baseline → arm: {CLASSES[case['baseline_prediction']]} → {CLASSES[case['prediction']]} ; "
                      f"fixed margin {case['baseline_fixed_margin']:.6f} → {case['fixed_margin']:.6f}; "
                      f"NLL {case['baseline_NLL']:.6f} → {case['NLL']:.6f}</p>")
        notes.extend([f"## {heading}", "",
            f"预测：{CLASSES[case['baseline_prediction']]} → {CLASSES[case['prediction']]}；"
            f"固定竞争类分差 {case['baseline_fixed_margin']:.6f} → {case['fixed_margin']:.6f}；"
            f"NLL {case['baseline_NLL']:.6f} → {case['NLL']:.6f}。",
            f"MMSI {case['mmsi']}；采集 {case['pass_group']}；港口 {case['port']}。",
            f"同类参考 {case['true_reference_ids']}；固定竞争类参考 {case['fixed_rival_reference_ids']}。",
            "图像可见事实：待主代理逐极化查看。模型证据：上述实际预测与分差。机制解释：尚未证明。", ""])
        for pol in ("vv", "vh"):
            filename = f"{name}_{pol}.png"
            receipts.extend(panel(group, pol, folder / filename, native, roles, columns=len(group)))
            blocks.append(f'<h3>{pol.upper()}</h3><div class="image"><img src="{filename}" alt="{html.escape(heading)}"></div>')
            notes.append(f"![{pol.upper()}]({filename})")
        blocks.append("</section>")
        notes.append("")
    banks = "".join(f'<h2>Actual 16 references / {p.upper()}</h2><div class="image"><img src="../data_preview/reference_bank_{p}.png" alt="16 actual refs {p}"></div>' for p in ("vv", "vh"))
    document = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>Shallow M2 actual cases</title>"
                "<style>body{font:15px Arial,sans-serif;margin:24px;color:#171717}section{border-top:1px solid #bbb;margin-top:28px}"
                "h2{font-size:18px}h3{font-size:15px}.image{overflow:auto}img{max-width:none;image-rendering:pixelated}</style>"
                f"<h1>Shallow M2 / seed17 实际案例</h1><p>{html.escape(SCOPE)}</p>"
                "<p>每张图左侧为原chip，右侧为模型160像素裁剪/填充；均1:1像素，不拉伸、不归一化显示。"
                "缺失VH明确标记。参考类别固定自initial原基线，未随新预测替换。图像观察与解释待主代理完成。</p>"
                + "".join(blocks) + banks + "</html>")
    (folder / "index.html").write_text(document, encoding="utf-8")
    (folder / "CASE_NOTES_ZH.md").write_text("\n".join(notes), encoding="utf-8")
    dump(folder / "native_sources.json", receipts)
    dump(folder / "cases.json", cases)


def report(out, summary, metric_rows):
    lines = ["# Shallow M2 结果草案", "", SCOPE, "",
             "本文件为自动数值草案；不替代训练后的原图复核与最终判断。", "",
             "## 各arm三seed", "",
             "| arm | seed | OA (%) | BA (%) | NLL | Brier | ΔOA (pp) | ΔBA (pp) | ΔNLL | ΔBrier |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        values = [row[m] for m in METRICS] + [row[f"delta_{m}"] for m in METRICS]
        lines.append(f"| {row['arm']} | {row['seed']} | " + " | ".join(f"{v:.6f}" for v in values) + " |")
    lines.extend(["", "## 三seed均值与样本标准差", "",
                  "每个seed单独计算指标，再对三项指标取均值和ddof=1标准差；差值在匹配seed内先算。", "",
                  "| arm | OA | BA | NLL | Brier | ΔOA | ΔBA | ΔNLL | ΔBrier |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in ARMS:
        numbers = summary["by_arm"][arm]
        cells = [numbers["metrics"][m] for m in METRICS] + [numbers["paired"][f"delta_{m}"] for m in METRICS]
        lines.append(f"| {arm} | " + " | ".join(f"{x['mean']:.6f} ± {x['std']:.6f}" for x in cells) + " |")
    lines.extend(["", "## 初始模型与匹配基线", "",
                  "initial仅记录训练前原基线并固定案例竞争类；主对照为scores中相同seed的baseline arm。",
                  "initial指标：" + "；".join(f"{m}={summary['initial_metrics'][m]:.6f}" for m in METRICS) + "。",
                  "捕鱼类在开发集n=0，classes.csv中召回留空，不以0计入BA。", "",
                  "## seed17判对/判错案例", "",
                  "| arm | rescued | harmed | CC | WW |", "|---|---:|---:|---:|---:|"])
    for arm, counts in summary["case_group_counts_seed17"].items():
        lines.append(f"| {arm} | " + " | ".join(str(counts[s]) for s in STATES) + " |")
    lines.extend(["", "每个arm每组最多一例，按固定竞争类分差变化绝对值选取；空组不以其他seed或比较对象补齐。",
                  "[原图与同类/固定竞争类参考](CASES/index.html)；页面末尾含实际16refs的VV/VH面板。",
                  "CASE_NOTES_ZH.md中的图像观察由主代理回看后填写；生成图片不等于已读图。", "",
                  "## 结构与运行记录", "",
                  "diagnostics.json原样纳入summary.json的diagnostics字段；本脚本不从文件存在推断结构自检通过。",
                  f"runs.json记录{len(summary['runs'])}项；缺失arm/seed：{summary['missing_run_records']}。",
                  "trainable/seconds保留于metrics.csv，curve原样保留于summary.json；不据训练曲线选择开发集最优checkpoint。", "",
                  "## 当前判断边界", "",
                  "数据层面仅有3个开发采集组且化学品船占70/160；应结合classes.csv和summary.json分港/采集结果解释变化。",
                  "layer1/layer2/both与matched baseline的差值衡量本轮完整模型的实际收益或退化。",
                  "若both超过单层，也不能仅据此声称互补机制；仍需接口自检与原图证据解释。",
                  "浅层细网格区域读取是否有效、哪些群体被救回或误伤、是否值得保留，待主代理结合数字、diagnostics和原图收尾。",
                  "本草案不声称物理机制、独立泛化确认或新颖性已成立。", ""])
    (out / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(out=HERE, package=PACKAGE, data_only=False):
    out, package = Path(out), Path(package)
    out.mkdir(parents=True, exist_ok=True)
    manifests = load_manifests(package)
    coverage, native = prepare_data_artifacts(out, package, manifests)
    if data_only:
        if not (out / "predictions.npz").exists():
            write_csv(out / "metrics.csv", [], METRIC_FIELDS)
            write_csv(out / "classes.csv", [], CLASS_FIELDS)
            dump(out / "summary.json", dict(status="pending_predictions", scope=SCOPE,
                coverage=coverage, by_arm={}, cases=[],
                missing_inputs=[name for name in ("predictions.npz", "runs.json", "diagnostics.json")
                                if not (out / name).exists()]))
            (out / "FINDINGS_ZH.md").write_text(
                "# Shallow M2 结果草案\n\n" + SCOPE + "\n\n"
                "当前尚无本轮完整predictions.npz，不报告任何arm收益、排名或案例翻转。\n"
                "metrics.csv/classes.csv仅包含表头；summary.json状态为pending_predictions。\n\n"
                "数据覆盖与4条原图依据见DATA_NOTES_ZH.md，原生图及16refs面板见data_preview/。\n"
                "主代理完成训练后运行 `python -B analyze.py`，产生三seed指标、匹配差值、逐类表及CASES/index.html。\n"
                "图像可见事实与机制解释需主代理训练后回看，不能将自动选例与生成面板记为已读图。\n",
                encoding="utf-8")
        print("Data coverage and native panels written; no prediction results claimed.")
        return
    required = ("predictions.npz", "runs.json", "diagnostics.json")
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(f"Results are pending: {missing}. Use --data-only before training completes.")
    ids, labels, scores, initial, rows = load_predictions(out / "predictions.npz", manifests)
    runs, diagnostics = read_json(out / "runs.json"), read_json(out / "diagnostics.json")
    bypass_scores = np.load(out / "predictions.npz")["bypasses"]
    bypass_rows = []
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            full, off = metrics(scores[a, s], labels), metrics(bypass_scores[a, s], labels)
            change = states(bypass_scores[a, s], scores[a, s], labels)
            bypass_rows.append(dict(arm=arm, seed=seed, **{f"full_{m}":full[m] for m in METRICS},
                **{f"off_{m}":off[m] for m in METRICS}, **{f"delta_{m}":full[m]-off[m] for m in METRICS},
                **{k:int(v.sum()) for k,v in change.items()}))
    write_csv(out / "bypass.csv", bypass_rows, list(bypass_rows[0]))
    run_index = {(r["arm"], int(r["seed"])): r for r in runs}
    metric_rows, class_rows, grouped = [], [], []
    for a, arm in enumerate(ARMS):
        for s, seed in enumerate(SEEDS):
            new, base = scores[a, s], scores[0, s]
            values, base_values = metrics(new, labels), metrics(base, labels)
            transitions = states(base, new, labels)
            run = run_index.get((arm, seed), {})
            metric_rows.append(dict(arm=arm, seed=seed, n=len(labels), **values,
                **{f"delta_{m}": values[m] - base_values[m] for m in METRICS},
                **{k: int(v.sum()) for k, v in transitions.items()},
                trainable=run.get("trainable"), seconds=run.get("seconds")))
            pv, bv = per_sample(new, labels), per_sample(base, labels)
            for c, name in enumerate(CLASSES):
                mask = labels == c
                row = dict(arm=arm, seed=seed, class_id=c, class_name=name, n=int(mask.sum()),
                           ships=len({r["mmsi"] for i, r in enumerate(rows) if mask[i]}),
                           **{k: int(v[mask].sum()) for k, v in transitions.items()})
                row.update({k: None for k in ("recall", "baseline_recall", "delta_recall", "NLL", "Brier", "delta_NLL", "delta_Brier")})
                if mask.any():
                    row["recall"] = float((pv["prediction"][mask] == c).mean() * 100)
                    row["baseline_recall"] = float((bv["prediction"][mask] == c).mean() * 100)
                    row["delta_recall"] = row["recall"] - row["baseline_recall"]
                    for m in ("NLL", "Brier"):
                        row[m] = float(pv[m][mask].mean())
                        row[f"delta_{m}"] = float((pv[m][mask] - bv[m][mask]).mean())
                class_rows.append(row)
            for field in ("port", "pass_group", "polarization"):
                groups = np.array([("both" if r.get("vv") and r.get("vh") else "vv_only")
                                   if field == "polarization" else r[field] for r in rows])
                for value in sorted(set(groups)):
                    mask = groups == value
                    mv, mb = metrics(new[mask], labels[mask]), metrics(base[mask], labels[mask])
                    grouped.append(dict(arm=arm, seed=seed, dimension=field, group=value,
                        n=int(mask.sum()), ships=len({r["mmsi"] for i, r in enumerate(rows) if mask[i]}),
                        **mv, **{f"delta_{m}": mv[m] - mb[m] for m in METRICS},
                        **{k: int(v[mask].sum()) for k, v in transitions.items()}))
    cases, counts = select_cases(ids, labels, scores, initial, rows, manifests["references"])
    summary = dict(status="numeric_analysis_complete_visual_review_pending", scope=SCOPE,
        score_semantics="float64 stable log-softmax of unnormalized logits", seeds=list(SEEDS),
        arms=list(ARMS), coverage=coverage, initial_metrics=metrics(initial, labels),
        by_arm={}, grouped_metrics=grouped, case_group_counts_seed17=counts, cases=cases,
        runs=runs, diagnostics=diagnostics, same_weights_early_bypass=bypass_rows,
        missing_run_records=[dict(arm=a, seed=s) for a in ARMS for s in SEEDS if (a, s) not in run_index])
    for arm in ARMS:
        selected = [r for r in metric_rows if r["arm"] == arm]
        summary["by_arm"][arm] = dict(metrics=aggregate(selected, METRICS),
            paired=aggregate(selected, [f"delta_{m}" for m in METRICS]),
            positive_delta_seeds={m: sum(r[f"delta_{m}"] > 0 for r in selected) for m in METRICS},
            favorable_delta_seeds={m: sum((r[f"delta_{m}"] > 0 if m in ("OA", "BA") else r[f"delta_{m}"] < 0)
                                         for r in selected) for m in METRICS})
    write_csv(out / "metrics.csv", metric_rows, METRIC_FIELDS)
    write_csv(out / "classes.csv", class_rows, CLASS_FIELDS)
    dump(out / "summary.json", summary)
    case_artifacts(out, cases, rows, manifests["references"], native)
    report(out, summary, metric_rows)
    print(f"Wrote 12 model/seed metric rows, 96 class rows, {len(cases)} seed17 cases: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    analyze(data_only=args.data_only)
