"""Paired full-system evidence and geometry coverage, without target selection."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "prepool_local_probe_20261015"
ROOT = OLD / "body_context_expansion"
METRICS = ["oa", "ba", "macro_nll", "macro_brier"]
KEY = ["port", "fold", "budget", "origin", "draw", "sample_id"]


def coverage():
    pools = json.loads((ROOT / "ring_relaxed/query_coverage/frozen.json").read_text(encoding="utf-8"))["pools"]
    geo = pd.read_csv(HERE / "geometry_metadata.csv").set_index("global_row")
    records, grid = [], []
    for pool in pools:
        port, slug = pool["pool"], pool["pool"].replace(" ", "_")
        for split in ("fit", "selection", "target"):
            folder = (ROOT if split == "target" else OLD / "body_context") / slug
            meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
            tasks = json.loads((folder / "tasks.json").read_text()) if split == "target" else [t for t in pool["splits"][split] if t["budget"] == 32]
            for role in ("q", "s"):
                needed = sorted({i for t in tasks for i in t[role]})
                m = meta.iloc[needed]
                g = geo.loc[m.global_row]
                joint = g[["ship_valid", "radar_valid", "incidence_valid"]].astype(bool).all(axis=1)
                records.append(dict(pool=port, split=split, role=role, samples=len(m), ships=m.mmsi.nunique(),
                    scenes=m.acquisition.nunique(), classes=m.class_id.nunique(), tasks=len(tasks),
                    ship_valid=int(g.ship_valid.sum()), radar_valid=int(g.radar_valid.sum()), incidence_valid=int(g.incidence_valid.sum()),
                    joint_valid=int(joint.sum()), joint_fraction=float(joint.mean())))
        with np.load(HERE / "spatial_cache" / (slug + ".npz")) as z:
            for r, region in enumerate(("body", "touch", "near", "far")):
                count = (z["areas"][:, r] > 0).sum(1)
                grid.append(dict(pool=port, region=region, samples=len(count), nonempty_min=int(count.min()),
                    nonempty_median=float(np.median(count)), nonempty_max=int(count.max()),
                    region_empty=int((count == 0).sum()), outer_edge_fraction=float(z["edge"].mean())))
    pd.DataFrame(records).to_csv(HERE / "task_geometry_coverage.csv", index=False)
    pd.DataFrame(grid).to_csv(HERE / "token_grid_coverage.csv", index=False)
    return pd.DataFrame(records)


def summarize_details(frame):
    result = []
    unit = ["port", "fold", "budget", "origin", "draw", "arm", "seed"]
    for key, g in frame.groupby(unit, sort=False):
        byclass = g.groupby("truth")[["correct", "nll", "brier", "ap", "class_kernel_margin"]].mean()
        result.append(dict(zip(unit, key), n=len(g), oa=100 * g.correct.mean(), ba=100 * byclass.correct.mean(),
            macro_nll=byclass.nll.mean(), macro_brier=byclass.brier.mean(), map_r=g.ap.mean(), macro_map_r=byclass.ap.mean(),
            class_kernel_margin=byclass.class_kernel_margin.mean(), retrieval_coverage=g.ap.notna().mean()))
    return pd.DataFrame(result)


def main():
    cov = coverage()
    allmetrics, detail, training = [], [], []
    for folder in sorted((HERE / "evaluation").iterdir()):
        for file in folder.glob("*_metrics.csv"):
            allmetrics.append(pd.read_csv(file))
        for file in folder.glob("*_predictions.csv"):
            detail.append(pd.read_csv(file, dtype={"mmsi": str}))
    metrics, preds = pd.concat(allmetrics, ignore_index=True), pd.concat(detail, ignore_index=True)
    assert metrics.groupby(["arm", "seed"]).size().eq(490).all()
    archived = pd.read_csv(ROOT / "ring_region_spatial/metrics.csv")
    replay = metrics.loc[metrics.arm.eq("body")].merge(archived.loc[archived.arm.eq("body")], on=["port", "fold", "budget", "origin", "draw"], suffixes=("", "_old"), validate="one_to_one")
    for key in METRICS:
        np.testing.assert_allclose(replay[key], replay[key + "_old"], atol=1e-9, rtol=1e-9)
    perport = metrics.groupby(["port", "budget", "origin", "arm", "seed"])[METRICS].mean().reset_index()
    old = pd.read_csv(ROOT / "ring_region_spatial/per_port.csv")
    old = old.loc[old.arm.eq("multiring")].copy()
    old["seed"] = -1
    half = pd.read_csv(ROOT / "ring_predictability/target_per_port.csv")
    half = half.loc[half.arm.eq("half")].rename(columns={"pool": "port"}).copy()
    half["seed"] = -1
    perport = pd.concat([perport, old, half[perport.columns]], ignore_index=True)
    perport.to_csv(HERE / "target_per_port.csv", index=False)
    seedmeans = perport.groupby(["budget", "origin", "arm", "seed"])[METRICS].mean().reset_index()
    seedmeans.to_csv(HERE / "target_seed_means.csv", index=False)
    summary = seedmeans.groupby(["budget", "origin", "arm"])[METRICS].agg(["mean", "std"])
    summary.to_csv(HERE / "target_summary.csv")
    contrasts = []
    for reference in ("body", "multiring", "half", "A", "B"):
        c = perport.loc[perport.arm.eq("C")]
        ref = perport.loc[perport.arm.eq(reference)]
        join = ["port", "budget", "origin"] + (["seed"] if reference in ("A", "B") else [])
        paired = c.merge(ref, on=join, suffixes=("", "_ref"), validate="one_to_one" if "seed" in join else "many_to_one")
        for key in METRICS:
            paired["delta_" + key] = paired[key] - paired[key + "_ref"]
        paired["reference"] = reference
        contrasts.append(paired[["port", "budget", "origin", "seed", "reference"] + ["delta_" + k for k in METRICS]])
    pd.concat(contrasts).to_csv(HERE / "paired_contrasts.csv", index=False)
    diagnostic = []
    for name, mask in (("full", np.ones(len(preds), bool)), ("geometry_valid", preds.geometry_valid),
                       ("geometry_missing", ~preds.geometry_valid), ("edge", preds.edge), ("not_edge", ~preds.edge)):
        table = summarize_details(preds.loc[mask])
        if not len(table):
            continue
        table["scope"] = name
        diagnostic.append(table)
    pd.concat(diagnostic).to_csv(HERE / "diagnostic_metrics.csv", index=False)
    events = []
    for ref in ("body", "A", "B"):
        c = preds.loc[preds.arm.eq("C")]
        r = preds.loc[preds.arm.eq(ref)]
        join = KEY + (["seed"] if ref != "body" else [])
        paired = c.merge(r[join + ["correct", "pred", "brier"]], on=join, suffixes=("", "_ref"), validate="one_to_one" if ref != "body" else "many_to_one")
        paired["rescue"] = paired.correct & ~paired.correct_ref
        paired["harm"] = ~paired.correct & paired.correct_ref
        paired["reference"] = ref
        events.append(paired)
    events = pd.concat(events, ignore_index=True)
    events.loc[events.rescue | events.harm].to_csv(HERE / "rescue_harm_cases.csv", index=False)
    events.groupby(["port", "budget", "origin", "seed", "reference", "truth"])[["rescue", "harm"]].sum().to_csv(HERE / "class_rescue_harm.csv")
    for file in (HERE / "trained").glob("*/*_seed?.pt"):
        saved = torch.load(file, weights_only=False, map_location="cpu")
        training.append(dict(pool=file.parent.name, arm=saved["mode"], seed=saved["seed"], best_step=saved["best_step"],
                             selection_ba=saved["selection_ba"], seconds=saved["seconds"], parameters=saved["parameters"]))
    train = pd.DataFrame(training)
    train.to_csv(HERE / "training_summary.csv", index=False)
    cross = seedmeans.loc[seedmeans.origin.eq("cross")].groupby(["budget", "arm"])[METRICS].mean().reset_index()
    lines = ["# 船体—雷达局部读取实验", "", "2026-10-16。完整分类流程结果，历史研发队列；不称独立确认。", "",
             "## 跨采集完整模型", "", "三训练种子先各自等港汇总，再取平均；固定基线只有一套参数。单位为百分比，差值单位百分点。", "",
             "| 支持数 | 方法 | OA | BA | 宏NLL | 宏Brier |", "|---|---|---|---|---|---|"]
    for row in cross.itertuples():
        lines.append(f"| {row.budget} | {row.arm} | {row.oa:.4f} | {row.ba:.4f} | {row.macro_nll:.5f} | {row.macro_brier:.5f} |")
    lines += ["", "## 实际口径", "", "- A：普通局部内容与图像位置；B：聚合后加入同样角度；C：几何与局部内容关系交互。",
              "- 每个留港源池单独训练，只用合法B32源任务；B16/B64是预算迁移。查询按源端抽样，目标标签不参与选择。",
              "- 船轴是无向散射轴，雷达轴是局部地距代理。不是AIS真艏向，不宣称纯因果或严格旋转不变。",
              "- body全任务结果已与原缓存重放一致；局部更新在完整SS/QS支持求解内发生。",
              f"- 实际训练{len(train)}个模型；源选择停在初始化的有{int(train.best_step.eq(0).sum())}个。训练计时合计{train.seconds.sum()/60:.2f}分钟，不含缓存抽取。",
              "- 三个训练种子不是五次支持抽样；同船重复查询不算独立样本。几何有效子集与完整队列分别记录。",
              "", "## 证据文件", "", "target_per_port.csv / target_seed_means.csv：逐港与种子。",
              "paired_contrasts.csv：C对A/B/body/multiring/半剂量的同范围差值。",
              "diagnostic_metrics.csv：几何有效、缺失、截断等范围的概率风险与类别检索。",
              "rescue_harm_cases.csv / class_rescue_harm.csv：正反例与类别贡献，不把注意力图当因果证据。",
              "task_geometry_coverage.csv / token_grid_coverage.csv：实际监督与空间覆盖。",
              "", "结果解释与下一步判断见本轮结论补充，不能仅凭表中最佳均值选择部署方法。"]
    (HERE / "RESULT_TABLES_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "DONE.json").write_text(json.dumps(dict(training_models=len(train), task_units=len(metrics), prediction_rows=len(preds),
        body_replay=True, source_only_selection=True, target_tasks=490), indent=2), encoding="utf-8")
    print(cross.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
