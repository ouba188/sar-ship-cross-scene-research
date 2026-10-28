"""CPU-only five-arm layer2 depth analysis, reusing prior local helpers.

python -B analyze.py --data-only
python -B analyze.py
No runner, model, checkpoint, training or GPU imports. Outputs stay in HERE.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "m2_residual_placement_20261028/analyze.py"
spec = importlib.util.spec_from_file_location("depth_analysis_helpers", HELPER)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
p, h = r.p, r.h
PACKAGE, CLASSES, METRICS, STATES = r.PACKAGE, r.CLASSES, r.METRICS, r.STATES
ARMS = ("baseline", "stage", "block1", "block2", "block3")
SEEDS = (17, 29, 43)
SCOPE = ("同一layer2内的受控开发位置比较：136条源适配、160条历史开发查询、16条固定参考。"
         "开发集反复使用，不声称独立源验证选择，更不是未见目标盲测；3075扩围不参加本轮。"
         "三seed是匹配训练随机条件，不是三份独立数据；不平均概率作为主结果。"
         "OA/BA以百分数表示，差值为百分点；BA只平均本开发集实际出现的7类。"
         "NLL为自然对数交叉熵，Brier为8类概率平方误差之和，二者越低越好。")
DESIGN = [
    "已有原图结论来自上一轮CASE_NOTES_ZH.md，本支线读取的是记录，未重新查看图片：30206的救回依赖双极化及侧向路径，34278的误伤可在关闭VV回写后恢复；同一回写对不同查询的影响方向不同。",
    "7602在stage正确、末块preadd错误，普通货仅12条，其单例召回变化足以改变BA；因此本轮同时看逐类和救回/误伤，不能只看OA。持续判对35906及持续判错26255也有分差变化，不能把无翻转视为无作用。",
    "沿用已核实的LITERATURE_DESIGN_ZH.md：SE扩展版区分残差单元位置与内部3x3位置，GCNet比较最后1x1后与相加后，Non-local区分位置与覆盖次数。它们提供控制变量的组织依据，未给出本任务layer2内最优深度。未做新文献检索。",
    "本轮固定每极化1个新增M2、512通道20x20、每臂额外197328参数及训练预算，只移动同一layer2内的preadd位置。block1/2/3分别为零基layer2[1]/[2]/[3]的bn3后、identity相加前；stage在layer2最终ReLU后。",
    "可检验假设：更早写回改变后续残差块处理区域证据的机会，可能改善或损害类别竞争。深度同时改变输入表征与后续传播，不能仅凭指标证明某个卷积或图像物理机制。路径诊断由主代理完成。",
]


def comparison_pairs(arms):
    return ([(arm, "baseline") for arm in arms]
            + [(arm, ref) for arm in ("block1", "block2") for ref in ("block3", "stage")]
            + [("block2", "block1")])


# Overrides affect only these privately imported modules, not previous files.
p.comparison_pairs = r.comparison_pairs = comparison_pairs
p.PINNED = {}
h.SCOPE = SCOPE


def load_manifests(package):
    manifests = h.load_manifests(package)
    if [len(manifests[s]) for s in ("source_adapt", "development", "references")] != [136, 160, 16]:
        raise ValueError("Expected frozen 136/160/16 development protocol")
    return manifests


def polarization(row):
    return "both" if row.get("vv") and row.get("vh") else "vv_only" if row.get("vv") else "vh_only"


def prepare_data(out, package, manifests):
    coverage, native = h.prepare_data_artifacts(out, package, manifests)
    table = []
    for split, rows in manifests.items():
        for field in ("label", "port", "pass_group", "polarization"):
            groups = {}
            for row in rows:
                key = polarization(row) if field == "polarization" else row[field]
                groups.setdefault(key, []).append(row)
            for key, group in sorted(groups.items()):
                table.append(dict(split=split, dimension=field, group=str(key),
                    name=CLASSES[key] if field == "label" else str(key), n=len(group),
                    observations=len({x["id"] for x in group}), ships=len({str(x["mmsi"]) for x in group}),
                    passes=len({x["pass_group"] for x in group}),
                    both=sum(polarization(x) == "both" for x in group),
                    vv_only=sum(polarization(x) == "vv_only" for x in group)))
    h.write_csv(out / "data_groups.csv", table, list(table[0]))
    refs = manifests["references"]
    support = []
    for query in manifests["development"]:
        eligible = [x for x in refs if x["id"] != query["id"] and str(x["mmsi"]) != str(query["mmsi"])]
        support.append(dict(id=query["id"], mmsi=query["mmsi"], label=query["label"],
            eligible_ids=";".join(x["id"] for x in eligible),
            true_ids=";".join(x["id"] for x in eligible if x["label"] == query["label"]),
            eligible_n=len(eligible), true_n=sum(x["label"] == query["label"] for x in eligible)))
    h.write_csv(out / "reference_eligibility.csv", support, list(support[0]))
    lines = ["# 本轮数据摘要", "", SCOPE, "", "| split | 观测 | 船只 | 过境 | 港口 | 双极化 | 仅VV |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for split in manifests:
        c = coverage[split]
        lines.append(f"| {split} | {c['n']} | {c['ships']} | {c['acquisitions']} | {len(c['ports'])} | {c['polarization'].get('both', 0)} | {c['polarization'].get('vv_only', 0)} |")
    lines += ["", "| 类别 | source_adapt | development | references |", "|---|---:|---:|---:|"]
    for c, name in enumerate(CLASSES):
        lines.append(f"| {name} | " + " | ".join(str(coverage[s]["classes"][c]["n"]) for s in manifests) + " |")
    lines += ["", "开发港口：" + "；".join(f"{k}={v}" for k, v in coverage["development"]["ports"].items()) + "。",
              "", "| split pair | 重叠ID | 重叠船只 | 重叠过境 | 重叠港口 |", "|---|---:|---:|---:|---:|"]
    for pair, overlap in coverage["overlaps"].items():
        lines.append(f"| {pair} | " + " | ".join(str(len(overlap[k])) for k in ("id", "mmsi", "pass_group", "port")) + " |")
    lines += ["", "这些是不交叉/覆盖的实际统计，不改变历史开发集被反复使用的事实。",
        "reference_eligibility.csv记录每查询经ID/MMSI排除后的实际参考及同类参考。",
        "data_groups.csv保留分割内类别、港口、过境及manifest极化分组；极化分组不等于单独VV/VH推理消融。",
        "data_preview内为4个固定输入例的VV/VH及16条实际参考库；原始8bit显示chip与模型160裁剪/补零并列，1:1像素、不伸缩、不拉伸对比度，不是原始SAR辐射值。",
        "native_sources.json记录路径、有效性及裁剪来源。图库已生成但本支线未人工查看；可见事实与路径解释待主代理补充。", ""]
    (out / "DATA_SUMMARY_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    return coverage, native


def aggregate_strata(rows, keys, fields):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    output = []
    for key, group in groups.items():
        item = dict(zip(keys, key))
        item.update(n=group[0]["n"], ships=group[0]["ships"], seeds=len(group))
        for field in fields:
            values = [x[field] for x in group if x[field] is not None]
            item[field + "_mean"] = float(np.mean(values)) if values else None
            item[field + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        output.append(item)
    return output


def transition_totals(paired):
    totals = {}
    for arm, reference in comparison_pairs(ARMS):
        if arm == reference:
            continue
        group = [x for x in paired if x["arm"] == arm and x["reference"] == reference]
        total, id_sets, ship_sets = {}, {}, {}
        for state in STATES:
            events = [x for x in group if x["state"] == state]
            id_sets[state] = {x["id"] for x in events}
            ship_sets[state] = {str(x["mmsi"]) for x in events}
            total[state] = dict(events=len(events), ids=len(id_sets[state]), ships=len(ship_sets[state]),
                                observation_ids=sorted(id_sets[state]), mmsis=sorted(ship_sets[state]))
        total["rescued_and_harmed_ids"] = sorted(id_sets["rescued"] & id_sets["harmed"])
        total["rescued_and_harmed_ships"] = sorted(ship_sets["rescued"] & ship_sets["harmed"])
        totals[f"{arm}__vs__{reference}"] = total
    return totals


def select_cases(ids, labels, scores, initial, arms, seeds, rows, refs):
    cases, counts = p.select_cases(ids, labels, scores, initial, arms, seeds, rows, refs)
    # Include real later-seed flips absent from the seed17 representatives.
    for s, seed in enumerate(seeds):
        if seed == 17:
            continue
        extra, _ = p.select_cases(ids, labels, scores[:, s:s+1], initial, arms, [seed], rows, refs)
        for case in extra:
            if case["state"] not in ("rescued", "harmed"):
                continue
            key = tuple(case[k] for k in ("arm", "reference_arm", "state", "id"))
            if any(tuple(c[k] for k in ("arm", "reference_arm", "state", "id")) == key for c in cases):
                continue
            i, rival = list(ids).index(case["id"]), case["fixed_rival"]
            a, b = arms.index(case["arm"]), arms.index(case["reference_arm"])
            case["per_seed"] = [dict(seed=order,
                reference_prediction=int(scores[b, ss, i].argmax()), prediction=int(scores[a, ss, i].argmax()),
                reference_fixed_margin=float(scores[b, ss, i, labels[i]] - scores[b, ss, i, rival]),
                fixed_margin=float(scores[a, ss, i, labels[i]] - scores[a, ss, i, rival]))
                for ss, order in enumerate(seeds)]
            case["selection"] = "Supplementary later-seed flip; not counted in seed17 groups; largest absolute initial-fixed-rival margin change"
            cases.append(case)
    return cases, counts


def include_path_cases(cases, diagnosis, ids, labels, scores, initial, arms, seeds, rows, refs):
    """Attach only supplied representative diagnostics, preserving their own seed."""
    for diagnostic in diagnosis["cases"]:
        arm, case_id, seed = diagnostic["arm"], str(diagnostic["id"]), int(diagnostic["seed"])
        i, s = list(ids).index(case_id), seeds.index(seed)
        if int(diagnostic["label"]) != int(labels[i]):
            raise ValueError(f"Path diagnostic label disagrees with predictions: {arm}/{case_id}")
        selected = [c for c in cases if c["arm"] == arm and c["reference_arm"] == "baseline" and c["id"] == case_id]
        if selected:
            case = next((c for c in selected if c["seed"] == seed), selected[0])
        else:
            candidates, _ = p.select_cases(ids[i:i+1], labels[i:i+1], scores[:, s:s+1, i:i+1],
                initial[i:i+1], arms, [seed], rows[i:i+1], refs)
            case = next(c for c in candidates if c["arm"] == arm and c["reference_arm"] == "baseline")
            a, b, rival = arms.index(arm), arms.index("baseline"), case["fixed_rival"]
            case["per_seed"] = [dict(seed=order,
                reference_prediction=int(scores[b, ss, i].argmax()), prediction=int(scores[a, ss, i].argmax()),
                reference_fixed_margin=float(scores[b, ss, i, labels[i]] - scores[b, ss, i, rival]),
                fixed_margin=float(scores[a, ss, i, labels[i]] - scores[a, ss, i, rival]))
                for ss, order in enumerate(seeds)]
            case["selection"] = "Main-agent representative path case; seed and kind retained separately; not added to seed17 group counts"
            cases.append(case)
        case.setdefault("path_diagnoses", []).append(diagnostic)
        case.setdefault("path_diagnosis", diagnostic)
        case["path_diagnosis_scope"] = diagnosis.get("scope", "Representative cases only; scope supplied by main agent")
        case["path_fixed_rival_rule"] = "Use rival supplied by path_diagnosis; separate from initial-fixed gallery rival"
        eligible = set(case["eligible_reference_ids"])
        added = [x["id"] for x in refs if x["id"] in eligible and x["label"] == int(diagnostic["rival"])]
        case["path_fixed_rival_reference_ids"] = list(dict.fromkeys(case.get("path_fixed_rival_reference_ids", []) + added))
    return dict(cases=len(diagnosis["cases"]), ids=sorted({str(d["id"]) for d in diagnosis["cases"]}),
                scope=diagnosis.get("scope"), visual_status="pending_main_review",
                queue_evaluation="not_requested_or_performed")


def path_artifacts(out, cases, rows, refs, native):
    """Separate panels keep diagnostic rivals distinct from predicted/initial rivals."""
    queries, references = {x["id"]: x for x in rows}, {x["id"]: x for x in refs}
    records, receipts, sections = [], [], []
    for case in cases:
        for diagnostic in case.get("path_diagnoses", []):
            key = f"path_{case['arm']}_seed{diagnostic['seed']}_{case['id']}_{diagnostic['kind']}"
            own = case["true_reference_ids"]
            rival = [x["id"] for x in refs if x["id"] in case["eligible_reference_ids"] and x["label"] == diagnostic["rival"]]
            group = [queries[case["id"]], *[references[x] for x in own + rival]]
            roles = ["QUERY", *(["TRUE REF"] * len(own)), *(["PATH RIVAL REF"] * len(rival))]
            panels = {}
            sections.append(f"<section><h2>{key}</h2><p>Diagnostic rival={diagnostic['rival']}; paths retained in path_cases.csv. NOT visually reviewed.</p>")
            for pol in ("vv", "vh"):
                filename = f"{key}_{pol}.png"
                receipt = h.panel(group, pol, out / "CASES" / filename, native, roles, columns=len(group))
                for item in receipt:
                    item.update(case_key=key, panel=filename,
                                allowed_for_query_polarization=bool(receipt[0]["available"] and item["available"]))
                receipts.extend(receipt)
                panels[pol] = str(out / "CASES" / filename)
                sections.append(f"<h3>{pol.upper()}</h3><div class='image'><img src='{filename}'></div>")
            sections.append("</section>")
            case.setdefault("path_panels", []).append(dict(seed=diagnostic["seed"], kind=diagnostic["kind"],
                rival=diagnostic["rival"], true_reference_ids=own, rival_reference_ids=rival, panels=panels))
            for path, value in diagnostic["paths"].items():
                records.append(dict(arm=diagnostic["arm"], seed=diagnostic["seed"], kind=diagnostic["kind"],
                    id=str(diagnostic["id"]), label=diagnostic["label"], rival=diagnostic["rival"], path=path, **value))
    if records:
        fields = list(dict.fromkeys(k for row in records for k in row))
        h.write_csv(out / "path_cases.csv", records, fields)
        h.dump(out / "CASES/path_native_sources.json", receipts)
        index = out / "CASES/index.html"
        index.write_text(index.read_text(encoding="utf-8").replace("</html>", "".join(sections) + "</html>"), encoding="utf-8")
        h.dump(out / "CASES/cases.json", cases)


def interpretation_lines(summary, paired, class_summary):
    lines = ["## 前中末的具体差异", "",
        "本历史开发比较中，中段block2的平均OA和BA高于末段block3，前段block1退化。"
        "block2不是全面胜出：相对block3其NLL三个seed均更差，OA在seed43下降；stage仍有最高平均BA。", "",
        "### block2相对block3：三seed的类别贡献", "",
        "| seed | dOA | dBA | 救回 | 误伤 |", "|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
        events = [x for x in paired if x["arm"] == "block2" and x["reference"] == "block3" and x["seed"] == seed]
        rescue, harm = (sum(x["state"] == state for x in events) for state in ("rescued", "harmed"))
        delta_ba = np.mean([sum((x["state"] == "rescued") - (x["state"] == "harmed") for x in events if x["label"] == c)
                           / sum(x["label"] == c for x in events) * 100 for c in sorted({x["label"] for x in events})])
        lines.append(f"| {seed} | {(rescue-harm)/160*100:+.6f} | {delta_ba:+.6f} | {rescue} | {harm} |")
    lines += ["", "| 类别 | n | 三seed平均dRecall | 对dBA的贡献 |", "|---|---:|---:|---:|"]
    selected = [x for x in class_summary if x["arm"] == "block2" and x["reference"] == "block3"]
    active_classes = sum(x["n"] > 0 for x in selected)
    for row in selected:
        change = row["delta_recall_mean"]
        if change is not None and abs(change) > 1e-10:
            lines.append(f"| {row['class_name']} | {row['n']} | {change:+.6f} | {change/active_classes:+.6f} |")
    lines += ["", "三seed平均散货贡献+0.761905个百分点BA，化学品-0.068027、集装箱-0.216450，合计+0.477427。其余实际出现类召回不变；不代表其概率完全不变。", "",
        "| 查询 | 类别 | 结局 | seed | block3预测 → block2预测 | 港口 |", "|---|---|---|---|---|---|"]
    flips = {}
    for x in paired:
        if x["arm"] == "block2" and x["reference"] == "block3" and x["state"] in ("rescued", "harmed"):
            key = (x["id"], x["class_name"], x["state"], x["reference_prediction"], x["prediction"], x["port"])
            flips.setdefault(key, []).append(x["seed"])
    for (case_id, name, state, before, after, port), seeds in flips.items():
        lines.append(f"| {case_id} | {name} | {state} | {','.join(map(str,seeds))} | {CLASSES[before]} → {CLASSES[after]} | {port} |")
    lines += ["", "12457是稳定新增救回：三个seed均由化学品改判正确散货。26652仅seed17增加散货救回；21161仅seed29增加化学品救回。30167仅seed17新增集装箱误伤；20791、30206仅seed43新增化学品误伤。",
        "seed43因此是散货+1、化学品-2，总正确数-1，OA下降0.625个百分点；但BA按类等权，(1/25-2/70)/7×100=+0.163265个百分点。它解释了此seed的OA/BA反向，不是三seed都全面改善。",
        "地理上block2相对block3在Jebel Ali的OA+1.449275个百分点，Port Klang为-1.190476，Singapore为0；本数据一港对应一次过境，不能将港口效应与过境效应独立解释。", "",
        "### 对stage及前段的边界", "",
        "block2相对stage平均OA+0.625000、BA-0.440940个百分点。普通货7602三个seed都从正确变为化学品，单例导致该类召回-8.333333个百分点、BA贡献-1.190476；散货+5.333333、化学品+1.428571、集装箱-1.515152个百分点召回变化合计只抵消一部分。",
        "block1相对baseline平均OA-1.666667、BA-0.218769个百分点；化学品召回-2.857143、集装箱-9.090909，虽普通货+8.333333和原油轮+2.083333，仍不足抵消。20791/31779/35728三个seed均从正确化学品变为普通货，30167/34278均从正确集装箱变为散货；37321普通货与21161化学品三个seed获救。",
        "block2相对block1三个seed均净增加6条正确查询，OA均+3.75个百分点，合计18次救回、0次误伤（10条唯一观测/10艘船）。该结论只针对正确/错误翻转，持续错误仍可能改换错误类别。", ""]
    if summary.get("path_case_coverage"):
        lines += ["### 代表案例路径证据", "",
            "仅接入主线11个案例、10条唯一查询的路径记录，未另做整队列干预。以下均为seed17同权重、同时作用于查询和参考的诊断；路径名沿用主线定义。",
            "12457：block2 full正确散货，真类对竞争类margin+0.331693；backbone_only仍正确(+0.647423)，readout_only错误(-0.333770)，off_VV错误(-0.347150)。支持此新增救回依赖VV回写经骨干传播；不能证明某种图像物理线索。",
            "30167：block2新增误伤为散货，margin-0.016629；off_VV恢复正确(+0.547821)，off_side恢复(+0.382818)，off_all恢复(+0.115921)。这是同一新增位置也会推错类别竞争的实例，侧向路径不等于纯背景像素。",
            "30206：block2 full正确，top1_margin+0.617715；backbone_only正确，而readout_only、off_VV、off_VH、off_side均错。其诊断rival为集装箱，off_VH后fixed_margin仍+0.339312，但top1_margin为-0.236735、误判原油轮，故固定竞争类分差变正不等于已判对。seed17路径不能用来解释seed43该查询的新增误伤；后者仍缺该seed路径证据。",
            "block1的20791：off_all仍错为普通货，margin-0.201867，说明不能把全部误伤归于当前一次推理回写；训练后其他可训练权重的变化也未被该关闭操作撤销，具体优化成因未证明。",
            "持续结果亦有证据变化：block1的26255仍错，但full margin-7.192376优于off_all的-10.139714；block2的28385持续正确但full margin2.657938低于off_all的3.680148，31141持续错误且full -11.784904比off_all -9.066288更差。它们不计入救回/误伤。",
            "当前可确认的是分类变化的数据组成及所选seed17案例的路径依赖。前中末差异是否来自后续卷积更充分利用区域证据、训练协调或其他机制，仍不能仅由这些诊断决定。原图可见事实由主代理人工回看补充。", ""]
    return lines


def report(out, summary=None, metrics=None, paired=None, class_summary=None):
    lines = ["# M2同stage深度：CPU数值报告", "", SCOPE, "", "## 设计依据", ""]
    lines += [paragraph + "\n" for paragraph in DESIGN]
    lines += ["依据记录：../m2_residual_placement_20261028/CASE_NOTES_ZH.md；../m2_block_placement_20261028/LITERATURE_DESIGN_ZH.md。",
        "", "## 固定协议", "", "臂顺序baseline, stage, block1, block2, block3；seed17/29/43。",
        "6次新fit为block1/2各3seed；9个旧模型为baseline/stage/block3各3seed，主线复用并重放。固定epoch6、54步。这里描述计划，是否完成以主线执行记录为准。",
        "block1/2与block3、stage及彼此比较；各非基线臂另与baseline比较，共9组有效比较。",
        "支持主线path_diagnosis.json的代表案例字段arm/seed/kind/id/label/rival/paths；只接入给定案例，不要求整队列路径推理。诊断seed及rival独立保留，不混同图库的选例seed及initial固定竞争类。", ""]
    if summary is None:
        lines += ["## 当前状态", "", "脚本与数据摘要已准备；本轮预测尚未接入分析，没有本轮指标、优劣或救回/误伤结论。",
            "data_preview已生成原生VV/VH与实际参考预览，本支线未人工查看。预测完成后运行python -B analyze.py。", ""]
    else:
        lines += ["主线complete.json记录实际完成6次新fit、9次复用、每fit54步、固定epoch6；未调参、未更换默认模型。", ""]
        lines += ["## 每臂均值与样本SD", "", "| arm | OA | BA | NLL | Brier |", "|---|---:|---:|---:|---:|"]
        for arm, value in summary["by_arm"].items():
            lines.append(f"| {arm} | " + " | ".join(f"{value['metrics'][m]['mean']:.6f} +/- {value['metrics'][m]['std']:.6f}" for m in METRICS) + " |")
        lines += ["", "## 逐seed结果", "", "| arm | seed | OA | BA | NLL | Brier |", "|---|---:|---:|---:|---:|---:|"]
        for row in metrics:
            if row["reference"] == "baseline":
                lines.append(f"| {row['arm']} | {row['seed']} | " + " | ".join(f"{row[m]:.6f}" for m in METRICS) + " |")
        lines += ["", "## 同seed配对差值", "", "逐seed算差后取均值和样本SD(ddof=1)；480个预测事件仍对应160条查询。", "",
            "| comparison | dOA | dBA | dNLL | dBrier | 有利seed OA/BA/NLL/Brier |", "|---|---:|---:|---:|---:|---|"]
        for pair, value in summary["comparisons"].items():
            if pair == "baseline__vs__baseline":
                continue
            lines.append(f"| {pair} | " + " | ".join(f"{value['paired']['delta_' + m]['mean']:+.6f} +/- {value['paired']['delta_' + m]['std']:.6f}" for m in METRICS) + " | " + "/".join(str(value["favorable_delta_seeds"][m]) for m in METRICS) + " |")
        lines += ["", "## 救回与误伤", "", "| comparison | 救回事件/唯一观测/船 | 误伤事件/唯一观测/船 | 两组交集观测/船 |", "|---|---|---|---|"]
        for pair, value in summary["transition_totals"].items():
            lines.append(f"| {pair} | " + " | ".join(f"{value[s]['events']}/{value[s]['ids']}/{value[s]['ships']}" for s in ("rescued", "harmed")) + f" | {len(value['rescued_and_harmed_ids'])}/{len(value['rescued_and_harmed_ships'])} |")
        lines += ["", "同一观测/船可能在不同seed分别救回和误伤；两组不可当互斥独立样本累加。", "",
            "classes.csv/grouped.csv保留逐seed类别、港口、过境、极化分层；classes_summary.csv/grouped_summary.csv提供对应均值和样本SD。空类为NA，不作零进入BA。",
            "paired_predictions.csv保留全部4320条逐样本配对事件及固定竞争类分差。极化分组按manifest可用性，不是独立极化推理消融。",
            "[案例图库](CASES/index.html)包含seed17每组CC/WW/救回/误伤代表例及其他seed新增翻转例，空组保留0。选择按initial固定非真类分差绝对变化，ID字典序打破平局。代表例不是总体比例。",
            "cases.json、case_index.csv与native_sources.json记录实际参考、各seed预测、原生路径与缺失极化。所有图库均标记未人工查看；本支线不填写图像可见事实或物理成因。",
            "bypass.csv为同权重关闭新增回写，不能替代独立重训。主代理负责人工原图回看、模型路径诊断和最终研究判断。",
            "若主线提供path_diagnosis.json，path_cases.csv原样展开其代表案例路径结果，图库额外提供PATH RIVAL REF面板；不把这些案例推断成整队列效果。", ""]
        lines += interpretation_lines(summary, paired, class_summary)
    if (out / "ANALYSIS_VALIDATION.json").exists():
        lines += ["## 脚本回放验证", "", "ANALYSIS_VALIDATION.json记录CPU回放：旧baseline/stage/preadd指标与归档最大误差0；九组比较、OA净变化与救回减误伤、唯一观测/船数、空类NA及实际参考选例通过。五臂验证仅在内存中别名映射旧分数，未保存为本轮预测或实验结果；未导入torch。", ""]
        validation = h.read_json(out / "ANALYSIS_VALIDATION.json").get("current_experiment")
        if validation:
            lines += [f"正式结果校验：{validation['metric_rows']}条指标、{validation['paired_rows']}条配对事件、"
                f"{validation['cases']}个案例记录（{validation['unique_queries']}条唯一查询）；"
                f"{validation['supplied_path_cases']}个主线路径案例、{validation['path_rows']}条路径记录全部接入。"
                f"full路径固定竞争类分差与正式预测最大误差{validation['max_full_path_fixed_margin_error']:.9g}。"
                f"OA救回减误伤、BA逐类贡献、唯一观测/船数均核对通过；{validation['case_png_files']}张案例PNG解码及尺寸检查通过，但未人工看图。", ""]
    lines += ["## 已做与未做", "", ("已做：CPU正式预测分析、逐seed及分层表、救回误伤去重、实际参考图库及主线代表案例路径接入。" if summary else "已做：CPU脚本、实际manifest数据摘要和原生预览；尚未分析本轮预测。"),
        "未做：GPU计算、训练、模型修改、独立源验证、未见目标盲测、3075扩围、人工看图、路径干预、Git或记忆更新。",
        "训练/重放状态只引用主线记录，不以日志中的训练秒数推断端到端效率；主线cost为共同stage1入口的局部计时，不能当作端到端成本。", ""]
    (out / "NUMERIC_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(predictions, out, package):
    manifests = load_manifests(package)
    ids, labels, scores, bypasses, initial, arms, seeds, rows = p.load_predictions(predictions, manifests)
    if tuple(arms) != ARMS or tuple(seeds) != SEEDS:
        raise ValueError(f"Expected ordered arms {ARMS}, seeds {SEEDS}; got {arms}, {seeds}")
    source = predictions.parent
    runs = h.read_json(source / "runs.json") if (source / "runs.json").exists() else []
    coverage, native = prepare_data(out, package, manifests)
    metrics, classes, grouped = p.comparison_rows(arms, seeds, scores, labels, rows, runs)
    paired = r.prediction_rows(ids, labels, scores, initial, arms, seeds, rows)
    bypass = []
    for a, arm in enumerate(arms):
        for s, seed in enumerate(seeds):
            full, off = h.metrics(scores[a, s], labels), h.metrics(bypasses[a, s], labels)
            bypass.append(dict(arm=arm, seed=seed, **{f"full_{m}": full[m] for m in METRICS},
                **{f"off_{m}": off[m] for m in METRICS}, **{f"delta_{m}": full[m]-off[m] for m in METRICS},
                **{k: int(v.sum()) for k, v in h.states(bypasses[a, s], scores[a, s], labels).items()}))
    cases, counts = select_cases(ids, labels, scores, initial, arms, seeds, rows, manifests["references"])
    path_file = source / "path_diagnosis.json"
    path_coverage = None
    if path_file.exists():
        path_coverage = include_path_cases(cases, h.read_json(path_file), ids, labels, scores, initial,
            arms, seeds, rows, manifests["references"])
        path_coverage["source"] = str(path_file)
    p.gallery(out, cases, rows, manifests["references"], native)
    path_artifacts(out, cases, rows, manifests["references"], native)
    notes = out / "CASES/CASE_NOTES_DRAFT_ZH.md"
    notes.write_text(notes.read_text(encoding="utf-8").replace("分组相对reference_arm，seed17；", "分组相对reference_arm；主体seed17，补充翻转例以标题seed为准；"), encoding="utf-8")
    # Avoid a link to a manual report which this branch is not authorized to create.
    index = out / "CASES/index.html"
    index.write_text(index.read_text(encoding="utf-8").replace("../CASE_NOTES_ZH.md", "CASE_NOTES_DRAFT_ZH.md").replace("人工原图与路径分析", "原图待复核草案（未人工查看）"), encoding="utf-8")
    h.write_csv(out / "case_index.csv", [dict(arm=c["arm"], reference=c["reference_arm"], seed=c["seed"],
        state=c["state"], id=c["id"], label=c["label"], fixed_rival=c["fixed_rival"],
        path_seeds=";".join(str(d["seed"]) for d in c.get("path_diagnoses", [])),
        path_kinds=";".join(d["kind"] for d in c.get("path_diagnoses", [])),
        path_rivals=";".join(str(d["rival"]) for d in c.get("path_diagnoses", [])),
        vv=c["panels"]["vv"], vh=c["panels"]["vh"], visual_status=c["visual_status"]) for c in cases],
        ["arm", "reference", "seed", "state", "id", "label", "fixed_rival", "path_seeds", "path_kinds", "path_rivals", "vv", "vh", "visual_status"])
    summary = dict(status="numeric_complete_visual_review_pending", scope=SCOPE, arms=arms, seeds=seeds,
        predictions=str(predictions), helper=str(HELPER), coverage=coverage,
        planned_new_fits=6, planned_reused_models=9, fixed_epoch=6, fixed_steps=54,
        capacity_note="stage/block1/block2/block3: one M2 per polarization; 512x20x20; extra197328 parameters per arm",
        score_semantics="unnormalized logits; float64 stable log-softmax", initial_metrics=h.metrics(initial, labels),
        by_arm={}, comparisons={}, transition_totals=transition_totals(paired),
        case_group_counts_seed17=counts, cases=cases, runs=runs, path_case_coverage=path_coverage,
        diagnostics=h.read_json(source / "diagnostics.json") if (source / "diagnostics.json").exists() else None,
        execution=h.read_json(source / "complete.json") if (source / "complete.json").exists() else None)
    for arm, reference in comparison_pairs(arms):
        selected = [x for x in metrics if x["arm"] == arm and x["reference"] == reference]
        value = dict(metrics=h.aggregate(selected, METRICS), paired=h.aggregate(selected, [f"delta_{m}" for m in METRICS]),
            favorable_delta_seeds={m: sum((x[f"delta_{m}"] > 0 if m in ("OA", "BA") else x[f"delta_{m}"] < 0) for x in selected) for m in METRICS})
        summary["comparisons"][f"{arm}__vs__{reference}"] = value
        if reference == "baseline":
            summary["by_arm"][arm] = value
    class_summary = aggregate_strata(classes, ("arm", "reference", "class_id", "class_name"),
        ("recall", "reference_recall", "delta_recall", "NLL", "Brier", "delta_NLL", "delta_Brier"))
    group_summary = aggregate_strata(grouped, ("arm", "reference", "dimension", "group"),
        (*METRICS, *[f"delta_{m}" for m in METRICS]))
    for name, table in (("metrics", metrics), ("classes", classes), ("grouped", grouped), ("bypass", bypass),
                        ("paired_predictions", paired), ("classes_summary", class_summary), ("grouped_summary", group_summary)):
        h.write_csv(out / f"{name}.csv", table, list(table[0]))
    h.dump(out / "summary.json", summary)
    report(out, summary, metrics, paired, class_summary)
    print(f"CPU analysis complete: {len(metrics)} metric rows, {len(paired)} paired rows, {len(cases)} cases; NOT visually reviewed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=HERE / "predictions.npz")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(HERE):
        raise ValueError("This CPU branch may write only inside its assigned directory")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.data_only:
        prepare_data(args.out, args.package.resolve(), load_manifests(args.package.resolve()))
        if not (args.out / "summary.json").exists():
            report(args.out)
        print("Data summaries and native previews prepared; images NOT manually reviewed.")
    else:
        analyze(args.predictions.resolve(), args.out, args.package.resolve())
