"""CPU-only pilot diagnosis and native panels; never import training code."""
import html
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[2] / "sar_m123_current_20261027"
SEEDS = (17, 29, 43)
ARMS = ("total", "delta", "control")
STATES = ("rescued", "harmed", "CC", "WW")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


repair = module("delta_pilot_native_helpers", HERE.parent / "m3_relation_repair_20261029/analyze.py")
base, native = repair.base, repair.native


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(lines, headers, rows):
    lines.extend(["", "| " + " | ".join(headers) + " |",
                  "|" + "|".join(["---"] * len(headers)) + "|"])
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)


def cm(score, y):
    result = np.zeros((8, 8), dtype=int)
    np.add.at(result, (y, score.argmax(1)), 1)
    return result


def events(old, new, rows, seed):
    result = []
    for i, row in enumerate(rows):
        y = row["label"]
        rival_scores = old[i].copy()
        rival_scores[y] = -np.inf
        rival = int(rival_scores.argmax())
        a, b = int(old[i].argmax()), int(new[i].argmax())
        result.append(dict(id=str(row["id"]), label=y, seed=seed,
            state=base.state(a, b, y), old_prediction=a, new_prediction=b,
            changed=a != b, rival=rival,
            delta_margin=float(new[i, y] - new[i, rival] - old[i, y] + old[i, rival])))
    return result


def load(rows):
    scores, baselines, receipts = {}, {}, {}
    expected = [str(r["id"]) for r in rows]
    y = np.array([r["label"] for r in rows])
    assert len(rows) == len(set(expected)) == 160 and len(np.unique(y)) == 7
    for stage, folder in (("fit", HERE), ("adapt", HERE / "adapt_pilot")):
        for arm in ARMS:
            seeds = SEEDS if stage == "adapt" and arm != "delta" else (17,)
            for seed in seeds:
                path = folder / "predictions" / f"{arm}_s{seed}.npz"
                receipt = read(folder / "training" / f"{arm}_s{seed}" / "training.json")
                assert receipt["arm"] == arm and receipt["seed"] == seed
                assert receipt["steps"] == (153 if stage == "fit" else 36)
                assert receipt["source_count"] == (1205 if stage == "fit" else 136)
                assert receipt["nonselected_parameters_unchanged"]
                assert receipt["initial_error"] < 1e-5
                assert receipt["native_error"] < 1e-4 and receipt["replay_error"] < 1e-4
                with np.load(path, allow_pickle=False) as pack:
                    ids = pack["ids"].astype(str).tolist()
                    assert len(ids) == len(set(ids)) == 160 and set(ids) == set(expected)
                    lookup = {key: i for i, key in enumerate(ids)}
                    order = [lookup[key] for key in expected]
                    assert np.array_equal(pack["y"][order], y)
                    score = pack["score"][order].astype(float)
                    old = pack["baseline"][order].astype(float)
                assert score.shape == old.shape == (160, 8)
                assert np.isfinite(score).all() and np.isfinite(old).all()
                if seed in baselines:
                    assert np.array_equal(old, baselines[seed]), (stage, arm, seed)
                baselines[seed] = old
                for endpoint, values in (("before", old), ("after", score)):
                    actual = base.metrics(values, y)
                    assert all(abs(actual[k] - receipt[endpoint][k.lower()]) < 1e-6
                               for k in base.METRICS)
                scores[stage, arm, seed] = score
                receipts[stage, arm, seed] = receipt
    return y, baselines, scores, receipts


def calibration(receipts):
    output = {}
    for key, receipt in receipts.items():
        if receipt["diagnostic"] is None:
            continue
        pols = []
        for pol, values in zip(("VV", "VH"), receipt["diagnostic"]):
            records = values["records"]
            t = np.array([r["total_rms"] for r in records])
            d = np.array([r["delta_rms"] for r in records])
            common = np.array([r["common_rms"] for r in records])
            labels = np.array([r["label"] for r in records])
            live = t > 0
            assert len(records) == 64 and all(np.sum(labels == c) == 8 for c in range(8))
            assert np.all(d[~live] == 0) and np.all(common[~live] == 0)
            ratio = d[live] / t[live]
            pols.append(dict(pol=pol, n=64, nonzero=int(live.sum()),
                total_scale=float(np.sqrt(np.mean(t*t))), delta_scale=float(np.sqrt(np.mean(d*d))),
                ratio=float(np.sqrt(np.sum(d*d)/np.sum(t*t))),
                common_ratio=float(np.sqrt(np.sum(common*common)/np.sum(t*t))),
                mean=float(ratio.mean()), quantiles=np.quantile(ratio, [0, .1, .5, .9, 1]).tolist(),
                zero_ids=[r["id"] for r, v in zip(records, live) if not v],
                by_class=[dict(c=c, live=int(((labels == c)&live).sum()),
                    ratio=float(np.sqrt(np.sum(d[labels == c]**2)/np.sum(t[labels == c]**2))))
                    for c in range(8)]))
            selected = d if key[1] == "delta" else t
            assert abs(values["scale"] - np.sqrt(np.mean(selected**2))) < 1e-6
        output[key] = pols
    for stage in ("fit", "adapt"):
        assert receipts[stage, "total", 17]["diagnostic"][0]["records"] == receipts[stage, "delta", 17]["diagnostic"][0]["records"]
        assert receipts[stage, "total", 17]["diagnostic"][1]["records"] == receipts[stage, "delta", 17]["diagnostic"][1]["records"]
    return output


def make_cases(rows, refs, baselines, scores):
    folder = HERE / "analysis/cases"
    folder.mkdir(parents=True, exist_ok=True)
    cases, sources, blocks, used = [], [], [], set()
    lookup = {str(r["id"]): i for i, r in enumerate(rows)}
    for stage, arm, seeds in (("fit", "delta", (17,)), ("adapt", "total", SEEDS)):
        all_events = [e for seed in seeds for e in events(baselines[seed], scores[stage, arm, seed], rows, seed)]
        for state in STATES:
            pool = [e for e in all_events if e["state"] == state]
            frequency = Counter(e["id"] for e in pool)
            ranked = sorted(pool, key=lambda e: (-frequency[e["id"]], -abs(e["delta_margin"]), e["id"], e["seed"]))
            pick = next((e for e in ranked if e["id"] not in used), None)
            assert pick is not None, (stage, state, "No distinct representative")
            case = dict(pick, comparison=f"{stage}_{arm}_minus_original", stage=stage,
                state_seed_count=frequency[pick["id"]], selected_seed=pick["seed"],
                selection_rule="State frequency across available orders, then absolute fixed-baseline-rival margin change, lexical ID and seed; globally distinct IDs.",
                visual_status="not_reviewed", interpretation="Outcome-selected case, not prevalence or causal proof.")
            used.add(pick["id"])
            i = lookup[pick["id"]]
            query = rows[i]
            case.update(mmsi=query.get("mmsi"), port=query.get("port"), pass_group=query.get("pass_group"))
            case["orders"] = []
            rivals = {pick["rival"]}
            for seed in seeds:
                old = baselines[seed][i]
                item = dict(seed=seed, baseline_logits=old.tolist(), baseline_prediction=int(old.argmax()))
                for name in ARMS:
                    key = stage, name, seed
                    if key in scores:
                        z = scores[key][i]
                        pred = int(z.argmax())
                        rr = z.copy(); rr[query["label"]] = -np.inf
                        rivals.add(int(rr.argmax()))
                        if pred != query["label"]:
                            rivals.add(pred)
                        item[name] = dict(logits=z.tolist(), prediction=pred,
                            state=base.state(int(old.argmax()), pred, query["label"]),
                            true_vs_fixed_rival_margin=float(z[query["label"]] - z[pick["rival"]]))
                rr = old.copy(); rr[query["label"]] = -np.inf
                rivals.add(int(rr.argmax()))
                item["baseline_true_vs_fixed_rival_margin"] = float(old[query["label"]] - old[pick["rival"]])
                case["orders"].append(item)
            eligible = [r for r in refs if base.eligible(query, r)]
            selected = [r for r in eligible if r["label"] in {query["label"], *rivals}]
            case["eligible_reference_ids"] = [str(r["id"]) for r in eligible]
            case["display_candidate_reference_ids"] = [str(r["id"]) for r in selected]
            case["reference_labels_used_for_display_only"] = True
            case["panels"] = {}
            for pol in ("vv", "vh"):
                qview = native.prepare_view(query.get(pol))
                qlive = qview is not None and qview["valid_target"]
                legal = []
                if qlive:
                    for r in selected:
                        view = native.prepare_view(r.get(pol))
                        if view is not None and view["valid_target"]:
                            legal.append(r)
                group = [query, *legal]
                roles = ["QUERY"] + ["TRUE REF" if r["label"] == query["label"] else "RIVAL REF" for r in legal]
                name = f"{stage}_{arm}_{state}_{query['id']}_{pol}.png"
                path = folder / name
                records = base.panel(group, roles, pol, path, native)
                # Verify one-to-one original and crop pixels in every rendered tile.
                width = max(max(r["raw_size"][0], 160) + 352 for r in records)
                height = max(max(r["raw_size"][1], 160) + 83 for r in records)
                columns = min(2, len(records))
                with Image.open(path) as canvas:
                    assert canvas.size == (width*columns, height*((len(records)+columns-1)//columns))
                    for n, (r, source) in enumerate(zip(group, records)):
                        view = native.prepare_view(r.get(pol))
                        if view is not None:
                            left, top = (n % columns)*width, (n // columns)*height
                            h, w = view["raw"].shape
                            got = np.asarray(canvas.crop((left+8, top+63, left+8+w, top+63+h)))
                            assert np.array_equal(got, np.repeat(view["raw"][..., None], 3, 2))
                            x = left + max(w, 160) + 16
                            got = np.asarray(canvas.crop((x, top+63, x+160, top+223)))
                            assert np.array_equal(got, np.repeat(view["pixels"][..., None], 3, 2))
                        source.update(panel_file=name, case_id=case["id"], comparison=case["comparison"],
                            pair_legal=source["role"] != "QUERY", source_class=r["label"],
                            reference_bank_index=None if source["role"] == "QUERY" else next(j for j, ref in enumerate(refs) if str(ref["id"]) == str(r["id"])),
                            pixel_equality_checked=view is not None)
                sources.extend(records)
                case["panels"][pol] = dict(file=name, query_available=bool(qlive),
                    legal_display_reference_ids=[str(r["id"]) for r in legal])
            cases.append(case)
            blocks.append("<h2>" + html.escape(f"{case['comparison']} / {state} / ID {case['id']}") + "</h2><pre>" + html.escape(json.dumps(case, ensure_ascii=False, indent=2)) + "</pre>")
            blocks.extend(f'<h3>{p.upper()}</h3><img src="{case["panels"][p]["file"]}">' for p in ("vv", "vh"))
    assert len(cases) == 8 and len({c["id"] for c in cases}) == 8
    base.dump(folder / "cases.json", cases)
    base.dump(folder / "native_sources.json", sources)
    (folder / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>M3 pilot cases</title><style>body{font:14px Arial;margin:24px}img{image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>M3 pilot cases: not reviewed</h1><p>Native pixels and actual legal source references. Displayed references are not claimed to have the highest model weights. Outcome-selected examples; all order logits are recorded.</p>' + "".join(blocks), encoding="utf-8")
    return cases, sources


def main():
    rows = read(RELEASE / "manifests/development.json")
    refs = read(RELEASE / "manifests/references.json")
    for record in rows + refs:
        for pol in ("vv", "vh"):
            if record.get(pol):
                path = Path(record[pol]["image"])
                if not path.is_absolute():
                    record[pol]["image"] = str((RELEASE / "manifests" / path).resolve())
    assert len(refs) == 16
    y, baselines, scores, receipts = load(rows)
    cal = calibration(receipts)
    cases, sources = make_cases(rows, refs, baselines, scores)
    lines = ["# M3 pilot CPU 诊断", "", "2026-10-30。仅分析本目录 fit seed17 与 adapt_pilot 的 total/control 17、29、43 及 delta17 固定强度1端点。所有 NPZ 均为历史 development160；训练分别用 source_fit1205 或 source_adapt136，不是训练集预测。未读 strength 结果、expanded 或公开测试，未加载模型、导入 torch、训练或使用 GPU。", "", "## 结论与证据边界", "",
        "fit 三臂的主要变化均是普通货船改善而化学品船退化；delta 没有优于同预算 control。adapt固定强度1下，total三顺序平均相对原始baseline为 OA -1.041667 pp / BA -0.125850 pp，control为 OA -0.625000 pp / BA +0.025510 pp；total相对control为 -0.416667/-0.151361 pp。total仅seed17双指标改善，29/43双指标下降，当前完整端点不能认定修复成功。强度0/.25/.5/1选择由主代理处理，本报告不读取该结果。", "",
        "本分析确认预测/标签/ID对应、同 seed 原始 baseline 完全一致、10份训练收据完成，NumPy重算指标与收据误差小于1e-6。BA只平均出现的7类，fishing真实样本数为0。重复顺序不是独立样本，未计算伪独立显著性。"]
    metric_rows = []
    for seed in SEEDS:
        m = base.metrics(baselines[seed], y)
        metric_rows.append(["original", seed, *[f"{m[k]:.6f}" for k in base.METRICS], "-", "-"])
    for (stage, arm, seed), score in scores.items():
        m, old = base.metrics(score, y), base.metrics(baselines[seed], y)
        metric_rows.append([f"{stage}/{arm}", seed, *[f"{m[k]:.6f}" for k in base.METRICS], f"{m['OA']-old['OA']:+.6f}", f"{m['BA']-old['BA']:+.6f}"])
    table(lines, ["端点", "seed", "OA%", "BA%", "NLL", "Brier", "ΔOA pp", "ΔBA pp"], metric_rows)
    lines += ["", "### adapt 三顺序配对", "", "下表为每个顺序先作差，再计算均值与样本SD；不是集成预测。NLL/Brier差值越低越好。"]
    for arm, old_arm in (("total", None), ("control", None), ("total", "control")):
        differences = []
        es = []
        for seed in SEEDS:
            old = baselines[seed] if old_arm is None else scores["adapt", old_arm, seed]
            new = scores["adapt", arm, seed]
            ma, mb = base.metrics(old, y), base.metrics(new, y)
            differences.append([mb[k]-ma[k] for k in base.METRICS])
            es.extend(events(old, new, rows, seed))
        x = np.array(differences)
        label = f"{arm} minus {old_arm or 'original'}"
        table(lines, [label, *base.METRICS], [["均值 +/- SD", *[f"{m:+.6f} +/- {sd:.6f}" for m, sd in zip(x.mean(0), x.std(0, ddof=1))]],])
        rescued = {e["id"] for e in es if e["state"] == "rescued"}
        harmed = {e["id"] for e in es if e["state"] == "harmed"}
        lines.append(f"\n{label}: OA正向 {int((x[:,0]>0).sum())}/3，BA正向 {int((x[:,1]>0).sum())}/3；救回 {sum(e['state']=='rescued' for e in es)} 顺序事件/{len(rescued)} 唯一样本，误伤 {sum(e['state']=='harmed' for e in es)} 顺序事件/{len(harmed)} 唯一样本；跨顺序既救回又误伤 {len(rescued & harmed)} 个。")
    lines += ["", "## 每类正确数与混淆", "", "类别编号：" + "; ".join(f"{i}={name}" for i, name in enumerate(base.CLASSES)) + "。", "", "每格为正确数/观测数；adapt合计为三顺序事件，分母是原样本数的3倍。"]
    class_rows = []
    for c in range(8):
        mask = y == c; n = int(mask.sum())
        count = lambda z: int(((z.argmax(1) == y)&mask).sum())
        class_rows.append([f"{c} {base.CLASSES[c]}", f"{count(baselines[17])}/{n}", *[f"{count(scores['fit',arm,17])}/{n}" for arm in ARMS],
            f"{sum(count(baselines[s]) for s in SEEDS)}/{n*3}", *[f"{sum(count(scores['adapt',arm,s]) for s in SEEDS)}/{n*3}" for arm in ("total","control")], f"{count(scores['adapt','delta',17])}/{n}"])
    table(lines, ["真类", "baseline17", "fit total", "fit delta", "fit control", "baseline 3序", "adapt total 3序", "adapt control 3序", "adapt delta17"], class_rows)
    detail_rows = []
    for c in range(8):
        mask = y == c
        values = []
        for seed in SEEDS:
            endpoints = [baselines[seed], scores['adapt','total',seed], scores['adapt','control',seed]]
            values.append('/'.join(str(int(((z.argmax(1)==y)&mask).sum())) for z in endpoints))
        detail_rows.append([c, int(mask.sum()), *values])
    table(lines, ["真类", "每序样本数", "s17 baseline/total/control正确数", "s29 baseline/total/control正确数", "s43 baseline/total/control正确数"], detail_rows)
    lines += ["", "adapt total相对baseline的净正确事件为散货-2、化学品船-5、原油船+2，其他类别净0；control为化学品船-4、原油船+1，其他类别净0。total的化学品船误伤7事件中，5次转判拖船、2次转判普通货船；因此adapt的主要误伤转向不等同于fit的化学品船→普通货船模式。类别净0不等于样本完全不变，见后续救回/误伤及WW变化。"]
    lines += ["", "完整混淆矩阵：行是真类，列为预测0..7；fit逐端点160条，adapt三顺序合计480个事件，delta17为160条。"]
    matrices = [("baseline17", cm(baselines[17],y))] + [(f"fit {a}17", cm(scores['fit',a,17],y)) for a in ARMS]
    matrices += [("baseline three orders", sum(cm(baselines[s],y) for s in SEEDS))]
    matrices += [(f"adapt {a} three orders", sum(cm(scores['adapt',a,s],y) for s in SEEDS)) for a in ("total","control")]
    matrices += [("adapt delta17", cm(scores['adapt','delta',17],y))]
    for name, values in matrices:
        lines += ["", f"**{name}**", "```text", "true \\ pred    0   1   2   3   4   5   6   7"]
        lines.extend(f"{c:>11} " + " ".join(f"{v:3d}" for v in row) for c,row in enumerate(values))
        lines.append("```")
    lines += ["", "## 变化分布", "", "CC=持续正确；WW同/变=持续错误且预测类不变/改变。救回、误伤、WW变之和为所有top1翻转。"]
    event_rows, by_class_rows = [], []
    for (stage, arm, seed), score in scores.items():
        es = events(baselines[seed],score,rows,seed)
        counts = Counter(e["state"] for e in es)
        wwchange = sum(e["state"] == "WW" and e["changed"] for e in es)
        event_rows.append([stage,arm,seed,counts["CC"],counts["rescued"],counts["harmed"],counts["WW"]-wwchange,wwchange,sum(e["changed"] for e in es)])
    table(lines,["训练来源","臂","seed","CC","救回","误伤","WW同","WW变","翻转"],event_rows)
    for stage, arm, seeds in [("fit",a,(17,)) for a in ARMS]+[("adapt",a,SEEDS if a!="delta" else (17,)) for a in ARMS]:
        es=[e for s in seeds for e in events(baselines[s],scores[stage,arm,s],rows,s)]
        by_class_rows.append([f"{stage}/{arm}", *["/".join(str(sum(e['label']==c and (e['state']==state if state!='WW变' else e['state']=='WW' and e['changed']) for e in es)) for state in ('rescued','harmed','WW变')) for c in range(8)]])
    table(lines,["救回/误伤/WW变事件",*[str(c) for c in range(8)]],by_class_rows)
    lines += ["", "fit 三臂共同将11条原本判对的化学品船转判普通货船；true=general_cargo时化学品船→普通货船的救回为 total4/delta3/control4。全体预测为普通货船的数量由baseline22增至37/36/37，化学品船由53降至38/37/37。因12条普通货船和70条化学品船的类别分母差异，BA可上涨而OA下降；这是已确认的决策再分配，不是已证明的物理机制或训练饱和根因。"]
    lines += ["", "### adapt 的主要配对转向", "", "以下按真类和旧→新预测联合计数，仅列实际翻转；三顺序合计。"]
    for arm in ("total","control"):
        es=[e for seed in SEEDS for e in events(baselines[seed],scores['adapt',arm,seed],rows,seed)]
        transitions=Counter((e['label'],e['old_prediction'],e['new_prediction']) for e in es if e['changed'])
        table(lines,[arm,"旧→新","事件数"],[[t,f"{old}→{new}",n] for (t,old,new),n in sorted(transitions.items(),key=lambda kv:(-kv[1],kv[0]))])
    lines += ["", "## Source calibration 的 delta/total RMS", "", "run.calibrate 在训练开始前从 source_fit 每类取清单最前8条，共64条；adapt运行也仍用这批fit校准样本，并非source_adapt136。无随机抽样种子。每条RMS覆盖32通道及10x10空间，包含空位置；下面合并比例为 sqrt(sum(delta_rms²)/sum(total_rms²))，不是逐样本比例均值。total/delta同seed的校准records逐项完全一致，故合并展示；control没有这组诊断。", "", "比例反映新增output之前的32通道信号，不是最终1024通道回写、分类贡献或正交能量占比。common与delta不正交，不能把(1-RMS比例)解释为共同信息占比。"]
    cal_rows=[]
    for key, pols in cal.items():
        if key[1]!='total':
            continue
        for p in pols:
            cal_rows.append([f"{key[0]} s{key[2]}",p['pol'],f"{p['nonzero']}/64",f"{p['total_scale']:.6f}",f"{p['delta_scale']:.6f}",f"{100*p['ratio']:.4f}%",f"{100*p['quantiles'][2]:.4f}%",f"{100*p['quantiles'][1]:.4f}%..{100*p['quantiles'][3]:.4f}%"])
    table(lines,["校准端点","极化","total非零","total RMS","delta RMS","合并比例","逐条中位数","逐条P10..P90"],cal_rows)
    p17=cal['fit','total',17]
    lines += ["", f"fit seed17 common/total合并RMS为 VV {p17[0]['common_ratio']*100:.4f}%、VH {p17[1]['common_ratio']*100:.4f}%。delta分支采用自己的源端RMS除法，归一化系数相对total分别大 {p17[0]['total_scale']/p17[0]['delta_scale']:.2f}、{p17[1]['total_scale']/p17[1]['delta_scale']:.2f} 倍；fit delta首批extra_output梯度范数VV/VH为0.066414/0.089039，记录显示分支确实收到梯度，不能解释成‘delta太小所以没有进入优化’；但也不能由此断定放大噪声。三顺序delta/total并非常数，尤其seed43的VV为6.8598%，不能将seed17约2.8%推广为所有端点固定比例。"]
    table(lines,["source_fit校准类","VV非零/8","VV比例","VH非零/8","VH比例"],[[c,p17[0]['by_class'][c]['live'],f"{p17[0]['by_class'][c]['ratio']*100:.4f}%",p17[1]['by_class'][c]['live'],f"{p17[1]['by_class'][c]['ratio']*100:.4f}%"] for c in range(8)])
    lines += ["", "seed17 VH total/common/delta三者均零的ID：" + ", ".join(map(str,p17[1]['zero_ids'])) + "。逐条比例排除这些0/0；合并RMS尺度保留全部64条，与训练校准一致。记录不足以单独区分缺极化、无合法区域/参考等零值原因，本次不读取额外缓存推断。", "", "## 训练材料与归因边界"]
    table(lines,["端点","source_count","baseline训练OA","baseline训练BA","baseline训练NLL","首轮CE","末轮CE"],[[f"{stage}/{arm}/s{seed}",r['source_count'],*[f"{r['train_baseline'][k]:.6f}" if 'train_baseline' in r else '未保存' for k in ('oa','ba','nll')],f"{r['curve'][0]['ce']:.6f}",f"{r['curve'][-1]['ce']:.6f}"] for (stage,arm,seed),r in receipts.items()])
    lines += ["", "fit 的低加权CE约0.04与目标接近饱和相容，但原fit收据未保存train_baseline，且本次白名单输入没有训练集逐样本预测，不能据开发集或低loss确认饱和。adapt收据已有train_baseline：17/29/43在136条上分别判对84/72/87条，仍错52/64/49条，确认这些样本尚有纠错空间，但不证明本修法可以利用。不要把不同样本、权重归约与训练阶段的CE绝对值当作同一难度标尺。", "", "从fit到adapt同时改变样本(1205→136)、LR(.001→.0001)、轮数/步数(3/153→6/36)，因此差异不能单独归因于换数据。total/delta新增65536参数，control续训原M3共131446参数；同阶段具有相同步数/目标，但容量和更新对象不同。只有单个delta顺序，不作三顺序结论。", "", "## 原图案例交接", "", "已生成8个不同查询、16张VV/VH面板；fit delta17与adapt total三顺序各取救回/误伤/持续对/持续错1个。选择先按同状态出现的顺序数，再按固定baseline竞争类margin变化绝对值、ID、seed排序；所选seed用于明确代表状态，cases.json保留全部适用顺序及total/control/delta可用端点。adapt的救回代表只在1/3顺序被救回，误伤代表在2/3顺序被误伤，不能将代表seed状态当作三顺序共同状态。案例不代表全体比例。"]
    table(lines,["比较","状态","ID","代表seed","该状态顺序数","VV合法展示参考","VH合法展示参考"],[[c['comparison'],c['state'],c['id'],c['selected_seed'],c['state_seed_count'],len(c['panels']['vv']['legal_display_reference_ids']),len(c['panels']['vh']['legal_display_reference_ids'])] for c in cases])
    lines += ["", "输出在 analysis/cases/index.html、cases.json、native_sources.json及16张PNG。复用 m3_relation_repair_20261029/analyze.py 的base.panel/native接口，展示原始uint8像素、native160 crop/pad和独立OBB覆盖。参考来自实际固定16条source库，排除同ID/MMSI，并逐极化保留查询和参考均有效者；按真实类/各端点竞争类挑展示子集，不声称是最高权重参考。标签仅用于事后展示。", "", "每个可用tile的原图和crop区域已逐像素与prepare_view结果比对，无resize/拉伸；缺极化标为UNAVAILABLE，不用黑图替代观测。图片及来源记录均visual_status=not_reviewed，由主代理回看；本次不从图像外观推断物理原因。", "", "复跑：`D:/Program_documents/Anaconda_envs/envs/ClearSAR/python.exe -B -X utf8 analyze.py`。明确白名单读取本轮10份NPZ和10份收据、development/reference清单，以及选中案例的原图；不会读取strength或最终测试。"]
    assert 'torch' not in sys.modules
    (HERE / "PILOT_DIAGNOSIS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status="complete_cpu_only",prediction_archives=len(scores),cases=len(cases),panels=2*len(cases),source_entries=len(sources),torch_imported=False,
        selected=[dict(comparison=c['comparison'],state=c['state'],id=c['id'],seed=c['selected_seed'],state_seed_count=c['state_seed_count']) for c in cases]),ensure_ascii=False))


if __name__ == "__main__":
    main()
