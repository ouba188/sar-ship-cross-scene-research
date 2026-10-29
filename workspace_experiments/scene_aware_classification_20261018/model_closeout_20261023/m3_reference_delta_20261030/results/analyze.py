"""Final CPU-only tables and new cases from completed, frozen predictions."""
import csv
import html
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "results"
PUB = HERE.parent / "final_oss2_replication_20261029"
RELEASE = HERE.parents[2] / "sar_m123_current_20261027"
SEEDS = (17, 29, 43)
METRICS = ("OA", "BA", "NLL", "Brier")
PRIVATE = ("bulk_carrier", "fishing", "general_cargo", "chemical_tanker",
           "container", "crude_oil_tanker", "tug", "offshore")
PUBLIC = ("bulk_carrier", "general_cargo", "product_chemical_tanker", "container_ship")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path):
    result = read(path)
    for row in result:
        row["id"] = str(row["id"])
        for pol in ("vv", "vh"):
            if row.get(pol):
                p = Path(row[pol]["image"])
                if not p.is_absolute():
                    row[pol]["image"] = str((path.parent / p).resolve())
    return result


def metric(z, y):
    shifted = z - z.max(1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    correct = z.argmax(1) == y
    return dict(OA=float(100*correct.mean()),
        BA=float(100*np.mean([correct[y == c].mean() for c in np.unique(y)])),
        NLL=float(-logp[np.arange(len(y)), y].mean()),
        Brier=float(((np.exp(logp)-np.eye(z.shape[1])[y])**2).sum(1).mean()))


def state(old, new, y):
    return "CC" if old == y and new == y else "rescued" if new == y else "harmed" if old == y else "WW"


def events(old, new, group, seed):
    result = []
    for i, row in enumerate(group):
        truth = int(row["label"])
        a, b = int(old[i].argmax()), int(new[i].argmax())
        other = old[i].copy(); other[truth] = -np.inf
        rival = int(other.argmax())
        result.append(dict(id=row["id"], label=truth, seed=seed, state=state(a,b,truth),
            old=a, new=b, changed=a != b, rival=rival,
            margin_change=float(new[i,truth]-new[i,rival]-old[i,truth]+old[i,rival])))
    return result


def csv_file(path, records):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def table(lines, header, values):
    lines += ["", "| " + " | ".join(header) + " |", "|" + "|".join(["---"]*len(header)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in values)


def load_all():
    specs = []
    def add(scope, folder, arms, seeds, manifest, classes, steps, prefix=""):
        for seed in seeds:
            for arm in arms:
                specs.append(dict(scope=scope, arm=arm, seed=seed, classes=classes, manifest=manifest,
                    prediction=HERE/folder/"predictions"/f"{prefix}{arm}_s{seed}.npz",
                    receipt=HERE/folder/"training"/f"{arm}_s{seed}"/"training.json", steps=steps))
    dev = RELEASE/"manifests/development.json"
    add("semantic", "semantic_pilot", ("label","label_shuffled"), (17,), dev, PRIVATE, 36)
    add("semantic_live", "semantic_live_pilot", ("label","label_live","label_live_shuffled"), (17,), dev, PRIVATE, 36)
    add("oss2_validation", "oss2_validation", ("total","control"), (17,), PUB/"manifests/validation.json", PUBLIC, 66)
    add("joint_dev", "joint_adaptation", ("control","joint_low","joint_high"), SEEDS, dev, PRIVATE, 54)
    add("oss2_evaluation", "oss2_joint", ("control","joint_low"), SEEDS, PUB/"manifests/evaluation.json", PUBLIC, 96, "evaluation_")
    # Completion receipts are written after save/replay checks, not at model-save time.
    for s in specs:
        assert s["receipt"].is_file(), f"Waiting for completed receipt: {s['receipt']}"
    expanded = read(HERE/"joint_adaptation/expanded/results.json")
    assert expanded["query_count"] == 3075 and len(expanded["records"]) == 6
    for r in expanded["records"]:
        specs.append(dict(scope="expanded",arm=r["arm"],seed=r["seed"],classes=PRIVATE,
            manifest=HERE.parent/"m2_preadd_expansion_20261028/query_records.json",
            prediction=HERE/"joint_adaptation/expanded"/f"{r['arm']}_s{r['seed']}.npz",
            receipt=None, expanded_receipt=r))
    datasets, cache = {}, {}
    for s in specs:
        scope, arm, seed = s["scope"], s["arm"], s["seed"]
        if s["manifest"] not in cache:
            cache[s["manifest"]] = rows(s["manifest"])
        group = cache[s["manifest"]]
        expected = [r["id"] for r in group]
        y = np.array([r["label"] for r in group])
        rec = read(s["receipt"]) if s["receipt"] else s["expanded_receipt"]
        if s["receipt"]:
            assert rec["steps"] == s["steps"] and rec["seed"] == seed and rec["arm"] == arm
            assert rec["replay_error"] < 1e-4
        with np.load(s["prediction"],allow_pickle=False) as pack:
            ids = pack["ids"].astype(str).tolist()
            assert len(ids) == len(set(ids)) == len(expected) and set(ids) == set(expected)
            index = {key:i for i,key in enumerate(ids)}
            order = [index[key] for key in expected]
            assert np.array_equal(pack["y"][order],y)
            score, old = pack["score"][order].astype(float), pack["baseline"][order].astype(float)
        assert score.shape == old.shape == (len(group),len(s["classes"]))
        assert np.isfinite(score).all() and np.isfinite(old).all()
        d = datasets.setdefault(scope,dict(rows=group,y=y,classes=s["classes"],scores={},receipts={},paths=[]))
        if ("saved",seed) in d["scores"]:
            assert np.array_equal(old,d["scores"]["saved",seed])
        d["scores"]["saved",seed] = old
        d["scores"][arm,seed] = score
        d["receipts"][arm,seed] = rec
        d["paths"].append(str(s["prediction"]))
        expected_metrics = rec["evaluation"] if scope == "oss2_evaluation" else rec["after"]
        assert all(abs(metric(score,y)[k]-expected_metrics[k.lower()]) < 1e-6 for k in METRICS)
    return datasets


def make_cases(data):
    folder = OUT/"cases"
    folder.mkdir(parents=True,exist_ok=True)
    native = module("final_native_input",HERE.parents[2]/"sar_m123_standalone/sar_m123/input.py")
    cases, sources, absent, blocks = [], [], [], []
    for scope in ("expanded","oss2_evaluation"):
        d = data[scope]; group = d["rows"]; scores = d["scores"]
        helper = module(f"final_panel_{scope}",HERE.parent/"spatial_m12_refit_20261029/analysis.py")
        helper.CLASSES = tuple(d["classes"])
        refs = rows(PUB/"manifests/references.json" if scope == "oss2_evaluation" else RELEASE/"manifests/references.json")
        es = [e for seed in SEEDS for e in events(scores["saved",seed],scores["joint_low",seed],group,seed)]
        lookup = {r["id"]:i for i,r in enumerate(group)}
        used = set()
        for category in ("rescued","harmed","CC","WW"):
            pool = [e for e in es if e["state"] == category]
            frequencies = Counter(e["id"] for e in pool)
            pool.sort(key=lambda e:(-frequencies[e["id"]],-abs(e["margin_change"]),e["id"],e["seed"]))
            selected = next((e for e in pool if e["id"] not in used),None)
            if selected is None:
                absent.append(dict(scope=scope,state=category,reason="No distinct query in this state against saved original across completed orders.",state_events=len(pool)))
                continue
            used.add(selected["id"])
            i = lookup[selected["id"]]; q=group[i]; truth=q["label"]
            case = dict(selected,scope=scope,comparison="joint_low_minus_saved",selected_seed=selected["seed"],
                state_seed_count=frequencies[selected["id"]],classes=list(d["classes"]),class_name=d["classes"][truth],
                port=q.get("port"),pass_group=q.get("pass_group"),mmsi=q.get("mmsi"),
                visual_status="not_reviewed",orders=[],panels={},reference_labels_for_display_only=True)
            rivals={selected["rival"]}
            for seed in SEEDS:
                item=dict(seed=seed)
                for arm in ("saved","control","joint_low"):
                    z=scores[arm,seed][i]; p=int(z.argmax()); other=z.copy();other[truth]=-np.inf
                    rivals.add(int(other.argmax()))
                    if p != truth:rivals.add(p)
                    item[arm]=dict(logits=z.tolist(),prediction=p,class_name=d["classes"][p],
                        truth_vs_fixed_rival_margin=float(z[truth]-z[selected["rival"]]))
                item["state_vs_saved"]=state(item["saved"]["prediction"],item["joint_low"]["prediction"],truth)
                item["state_vs_control"]=state(item["control"]["prediction"],item["joint_low"]["prediction"],truth)
                case["orders"].append(item)
            legal=[r for r in refs if helper.eligible(q,r)]
            displayed=[r for r in legal if r["label"] in {truth,*rivals}]
            case["eligible_reference_ids"]=[r["id"] for r in legal]
            case["display_candidate_reference_ids"]=[r["id"] for r in displayed]
            for pol in ("vv","vh"):
                v=native.prepare_view(q.get(pol));active=v is not None and v["valid_target"]
                shown=[]
                if active:
                    for r in displayed:
                        rv=native.prepare_view(r.get(pol))
                        if rv is not None and rv["valid_target"]:shown.append(r)
                name=f"{scope}_{category}_{q['id']}_{pol}.png"
                members=[q,*shown]
                roles=["QUERY"]+["TRUE REF" if r["label"]==truth else "RIVAL REF" for r in shown]
                entries=helper.panel(members,roles,pol,folder/name,native)
                for r,entry in zip(members,entries):
                    entry.update(scope=scope,case_id=q["id"],panel_file=name,label=r["label"],
                        class_name=d["classes"][r["label"]],pair_legal=entry["role"]!="QUERY",
                        reference_bank_index=None if entry["role"]=="QUERY" else next(j for j,x in enumerate(refs) if x["id"]==r["id"]))
                sources.extend(entries)
                case["panels"][pol]=dict(file=name,query_available=bool(active),legal_reference_ids=[r["id"] for r in shown])
            cases.append(case)
            blocks += ["<h2>"+html.escape(f"{scope} / {category} / {q['id']}")+"</h2><pre>"+html.escape(json.dumps(case,ensure_ascii=False,indent=2))+"</pre>"]
            blocks.extend(f'<h3>{pol.upper()}</h3><img src="{case["panels"][pol]["file"]}">' for pol in ("vv","vh"))
    dump(folder/"cases.json",cases)
    dump(folder/"native_sources.json",sources)
    dump(folder/"selection.json",dict(comparison="joint_low_minus_saved",states=["rescued","harmed","CC","WW"],
        rule="No random draw: state frequency across three orders, absolute selected-seed baseline-fixed-rival margin change, lexical ID, seed; distinct within scope.",
        absent=absent,visual_status="not_reviewed",note="Selected outcomes are not prevalence. Actual eligible references, not claimed top-attention references."))
    (folder/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Frozen joint final cases</title><style>body{font:14px Arial;margin:24px}img{image-rendering:pixelated}pre{white-space:pre-wrap}</style><h1>Final cases: not reviewed</h1><p>Original uint8 pixels, native160 crop/pad and separate OBB overlay. VV/VH separately; source references exclude same ID/MMSI and unavailable polarizations. Public labels use the four-class taxonomy.</p><pre>'+html.escape(json.dumps(absent,ensure_ascii=False,indent=2))+"</pre>"+"".join(blocks),encoding="utf-8")
    return cases,absent


def main():
    data=load_all()
    endpoint_rows, paired, means, drift, class_rows=[],[],[],[],[]
    for scope,d in data.items():
        y=d["y"]; group=d["rows"]; scores=d["scores"]
        for (arm,seed),z in scores.items():
            endpoint_rows.append(dict(scope=scope,arm=arm,seed=seed,n=len(y),**metric(z,y)))
        comparisons=[(a,"saved") for a in dict.fromkeys(a for a,s in scores if a!="saved")]
        if scope in ("joint_dev","expanded","oss2_evaluation"):
            comparisons += [(a,"control") for a in ("joint_low","joint_high") if any(k[0]==a for k in scores)]
        if scope=="oss2_validation":comparisons += [("total","control")]
        if scope=="semantic":comparisons += [("label","label_shuffled")]
        if scope=="semantic_live":comparisons += [("label_live","label"),("label_live","label_live_shuffled")]
        for arm,reference in comparisons:
            seeds=sorted(s for a,s in scores if a==arm)
            values=[];all_events=[]
            for seed in seeds:
                old,new=scores[reference,seed],scores[arm,seed]
                before,after=metric(old,y),metric(new,y)
                es=events(old,new,group,seed);counts=Counter(e["state"] for e in es)
                delta={"delta_"+k:after[k]-before[k] for k in METRICS}
                row=dict(scope=scope,arm=arm,reference=reference,seed=seed,n=len(y),**delta,
                    rescued=counts["rescued"],harmed=counts["harmed"],CC=counts["CC"],WW=counts["WW"],
                    flips=sum(e["changed"] for e in es),max_abs_logit=float(np.abs(new-old).max()))
                paired.append(row);values.append(row);all_events.extend(es)
                if arm=="control" and scope in ("joint_dev","expanded","oss2_evaluation"):
                    drift.append(dict(scope=scope,seed=seed,max_abs_logit=row["max_abs_logit"],
                        changes=[dict(e,old_logits=old[i].tolist(),new_logits=new[i].tolist()) for i,e in enumerate(es) if e["changed"]]))
            mean=dict(scope=scope,arm=arm,reference=reference,seeds=seeds,n=len(y))
            for k in METRICS:
                x=[v["delta_"+k] for v in values]
                mean["delta_"+k]=float(np.mean(x))
                mean["sd_"+k]=float(np.std(x,ddof=1)) if len(x)>1 else None
            for category in ("rescued","harmed","CC","WW"):
                matched=[e for e in all_events if e["state"]==category]
                mean[category+"_events"]=len(matched)
                mean[category+"_unique_ids"]=len({e["id"] for e in matched})
            means.append(mean)
            if scope in ("joint_dev","expanded","oss2_evaluation","oss2_validation"):
                for c,name in enumerate(d["classes"]):
                    es=[e for e in all_events if e["label"]==c]
                    class_rows.append(dict(scope=scope,arm=arm,reference=reference,label=c,class_name=name,
                        n=int((y==c).sum()),orders=len(seeds),before_correct=sum(e["old"]==c for e in es),
                        after_correct=sum(e["new"]==c for e in es),rescued=sum(e["state"]=="rescued" for e in es),
                        harmed=sum(e["state"]=="harmed" for e in es),WW_changed=sum(e["state"]=="WW" and e["changed"] for e in es)))
    cases,absent=make_cases(data)
    csv_file(OUT/"metrics.csv",endpoint_rows)
    csv_file(OUT/"actual_deltas.csv",paired)
    csv_file(OUT/"class_changes.csv",class_rows)
    dump(OUT/"baseline_reproduction.json",drift)
    summary=dict(status="complete_cpu_only",goal_achieved=False,frozen_candidate="joint_low, M3 output LR3e-5; no further tuning",
        mean_paired_deltas=means,case_count=len(cases),panel_count=len(cases)*2,missing_case_states=absent,
        scopes={k:dict(n=len(d["rows"]),classes=d["classes"],ships=len({r.get('mmsi') for r in d['rows']}),
            passes=len({r.get('pass_group') for r in d['rows']}),paths=d['paths']) for k,d in data.items()},
        visual_status="not_reviewed",old_panels="Unchanged; main-agent review recorded in CASE_NOTES_ZH.md")
    dump(OUT/"summary.json",summary)
    lines=["# 冻结 M3 联合适配最终结果", "", "2026-10-30。CPU只读分析已完成的预测/收据，无模型加载、训练或GPU调用。joint_low已冻结为原适配流程加M3 output LR3e-5；本轮目标尚未达成，不再调整该候选。原analysis/cases的16张图及CASE_NOTES_ZH.md不改动。", "", "## 保存原版、新control与候选必须分开", "",
        "saved是NPZ内保存原版端点；control是本轮重新执行原训练策略得到的模型。joint_dev的control s17相对saved最大logit差0.0666008，两条原正确样本31779(化学品→普通货)、34278(集装箱→散货)被误伤。joint_low s17相对新control的2条救回均只是恢复保存原版正确性，不能记为超越保存原版的M3收益。s29的新control又比saved多判对12478，s43正确性无变化。具体数值误差根因未证实，不把训练复现差异命名为新模块效果。", "",
        "所有actual delta均为候选减对应参照；OA/BA为百分点，NLL/Brier越低越好。均值先按顺序配对，再对三个顺序平均，不是模型集成。SD是顺序间样本SD，不是独立样本置信区间。"]
    table(lines,["范围","候选","参照","顺序","ΔOA","ΔBA","ΔNLL","ΔBrier"],
        [[m['scope'],m['arm'],m['reference'],','.join(map(str,m['seeds'])),*[f"{m['delta_'+k]:+.6f}" for k in METRICS]] for m in means if m['scope'] in ('joint_dev','expanded','oss2_evaluation')])
    lines += ["", "joint_low在development三个顺序与saved的OA/BA完全相同，只是风险指标微降；在expanded3075三个顺序OA均下降，未达到冻结扩围目标。以下同时保留新control参照，避免把重放漂移混入候选净收益。OSS2结论仅适用于历史四类检测后evaluation378，不能推广为全库或全新盲测。"]
    lines += ["", "## 各顺序 actual delta"]
    table(lines,["范围","候选","参照","seed","ΔOA","ΔBA","ΔNLL","ΔBrier","救回/误伤","翻转"],
        [[r['scope'],r['arm'],r['reference'],r['seed'],*[f"{r['delta_'+k]:+.6f}" for k in METRICS],f"{r['rescued']}/{r['harmed']}",r['flips']] for r in paired if r['scope'] in ('joint_dev','expanded','oss2_evaluation')])
    lines += ["", "## 新control复现差异"]
    table(lines,["范围","seed","max|logit差|","救回/误伤","top1改变"],
        [[r['scope'],r['seed'],f"{r['max_abs_logit']:.8f}",f"{r['rescued']}/{r['harmed']}",r['flips']] for r in paired if r['arm']=='control' and r['reference']=='saved' and r['scope'] in ('joint_dev','expanded','oss2_evaluation')])
    lines += ["", "所有改变ID、前后logits、类别与正确性见baseline_reproduction.json。极小输入/保存重放误差不能保证训练轨迹与旧训练位级一致；本分析不新增根因推导。", "", "## 已完成的小验证与源验证", "", "semantic与semantic_live仅seed17，学习率分别1e-4与1e-3；live阶段的label是同LR的对照。OSS2源验证为validation228、source_adapt241训练66步，和最后OSS2联合适配96步不是同一实验。"]
    table(lines,["范围","端点","OA","BA","NLL","Brier"],
        [[r['scope'],r['arm'],*[f"{r[k]:.6f}" for k in METRICS]] for r in endpoint_rows if r['scope'] in ('semantic','semantic_live','oss2_validation')])
    table(lines,["范围","候选","参照","ΔOA","ΔBA","ΔNLL","ΔBrier"],
        [[m['scope'],m['arm'],m['reference'],*[f"{m['delta_'+k]:+.6f}" for k in METRICS]] for m in means if m['scope'] in ('semantic','semantic_live','oss2_validation')])
    lines += ["", "semantic正确/打乱标签均未优于原版；semantic_live三个端点均低于原版，当前数据不支持语义分支带来分类收益。源验证total/control均OA57.0175、BA48.3994，均低于该源验证baseline57.8947/50.1235；概率风险改善不等于分类目标达成。", "", "## 最终类别变化与状态", "", "计数是三个顺序的事件；不是三倍独立样本。公开四类名称单独定义，未沿用私有八类索引。"]
    table(lines,["范围","参照","类","每序n","原正确事件","joint_low正确事件","救回","误伤","WW改类"],
        [[r['scope'],r['reference'],r['class_name'],r['n'],r['before_correct'],r['after_correct'],r['rescued'],r['harmed'],r['WW_changed']] for r in class_rows if r['arm']=='joint_low' and r['scope'] in ('expanded','oss2_evaluation')])
    table(lines,["范围","参照","救回事件/唯一样本","误伤事件/唯一样本"],
        [[m['scope'],m['reference'],f"{m['rescued_events']}/{m['rescued_unique_ids']}",f"{m['harmed_events']}/{m['harmed_unique_ids']}"] for m in means if m['arm']=='joint_low' and m['scope'] in ('expanded','oss2_evaluation')])
    lines += ["", "## 新增原像素面板", "", "仅为expanded3075与OSS2 evaluation378生成joint_low相对saved的救回、误伤、持续对、持续错各1；每例同时记录三个顺序的saved/control/joint_low分数与两套参照状态。不存在的状态明确缺失，不换参照凑数。优先多顺序同状态，其次固定baseline竞争类margin变化绝对值，最后ID/seed；无随机抽样。", ""]
    table(lines,["范围","状态","ID","代表seed","同状态顺序数"],
        [[c['scope'],c['state'],c['id'],c['selected_seed'],c['state_seed_count']] for c in cases])
    for a in absent:lines.append(f"\n缺失：{a['scope']} / {a['state']}，实际对应事件数 {a['state_events']}。")
    lines += ["", "入口cases/index.html；cases.json记录真实类别名称、模型分数、参考ID/顺序及缺极化状态，native_sources.json记录绝对原图路径、原始尺寸、crop/pad和OBB来源。复用既有panel/native helper；VV/VH分开，原始像元1:1、无resize或对比度拉伸。参考取实际源库并排除同ID/MMSI及不可用极化；展示同类与实际竞争类子集，不宣称这些是最高权重参考。", "", "新面板均not_reviewed，主代理回看后写FINDINGS；本支线不依据外观编写物理解释。原16张图不重生成。", "", "## 产物", "", "metrics.csv：各端点实际指标；actual_deltas.csv：逐顺序对saved/new control的真实差值及翻转；class_changes.csv：逐类正确/救回/误伤；summary.json：均值、SD、范围及完成状态；baseline_reproduction.json：control重放改变的逐例证据；cases/：最终病例。复跑本目录analyze.py只重建这些结果，不触发训练或调参。"]
    (OUT/"RESULTS_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    assert 'torch' not in sys.modules
    print(json.dumps(dict(status="complete_cpu_only",cases=len(cases),panels=2*len(cases),absent=absent,
        final=[m for m in means if m['arm']=='joint_low' and m['scope'] in ('expanded','oss2_evaluation')]),ensure_ascii=False))


if __name__ == "__main__":
    main()
