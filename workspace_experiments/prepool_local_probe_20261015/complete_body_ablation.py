"""Complete the frozen expansion with body-only and paired component effects."""

import json
import os
from pathlib import Path
import time

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from run_body_context import FIELDS, evaluate, packed_task

ROOT = Path(__file__).resolve().parent / "body_context_expansion"
OUT = ROOT / "explainability"
METRICS = ["oa", "ba", "macro_nll", "macro_brier", "macro_map_r"]
TASK_KEYS = ["port", "fold", "budget", "origin", "draw"]
PRED_KEYS = TASK_KEYS + ["sample_id"]


def aggregate(table, extras=()):
    keys = ["budget", "origin", "arm", *extras]
    scene = table.groupby(["port", "fold", *keys])[METRICS].mean()
    port = scene.groupby(["port", *keys]).mean()
    return scene, port, port.groupby(keys).mean()


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    frozen = json.loads((ROOT / "frozen.json").read_text(encoding="utf-8"))
    common = {e["fold"] for e in frozen["episodes"] if 64 in e["budgets"]}
    tables, predictions = [], []
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            slug = pool["port"].replace(" ", "_")
            folder = ROOT / slug
            with np.load(folder / "readout.npz") as cache:
                arrays = {k: cache[k] for k in (*FIELDS, "weights")}
            meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
            all_tasks = json.loads((folder / "tasks.json").read_text(encoding="utf-8"))
            for budget in sorted({t["budget"] for t in all_tasks}):
                tasks = [t for t in all_tasks if t["budget"] == budget]
                packed = [packed_task(arrays, t) for t in tasks]
                metric, pred = evaluate(arrays, tasks, packed, meta, "ship", predictions=True)
                metric["budget"] = budget
                pred["budget"] = budget
                metric.to_csv(OUT / f"{slug}_ship_B{budget}_metrics.csv", index=False)
                pred.to_csv(OUT / f"{slug}_ship_B{budget}_predictions.csv", index=False)
                old = pd.read_csv(folder / f"metrics_B{budget}.csv")
                old = old.loc[old.arm.isin(["original", "ring", "additive"])]
                oldpred = pd.read_csv(folder / f"predictions_B{budget}.csv", dtype={"mmsi": str})
                oldpred = oldpred.loc[oldpred.arm.isin(["original", "ring", "additive"])]
                tables.extend([old, metric])
                predictions.extend([oldpred, pred])
                print(slug, budget, "completed", len(tasks), flush=True)
    table = pd.concat(tables, ignore_index=True)
    pred = pd.concat(predictions, ignore_index=True)
    # A row mismatch here would make the ablation unpaired, even with equal totals.
    counts = pred.groupby(PRED_KEYS).arm.nunique()
    assert counts.eq(4).all()
    assert not pred.duplicated(PRED_KEYS + ["arm"]).any()
    label_counts = pred.groupby(PRED_KEYS).truth.nunique()
    assert label_counts.eq(1).all()
    pred["correct"] = pred.pred.eq(pred.truth)
    check_keys = TASK_KEYS + ["arm"]
    oa = pred.groupby(check_keys).correct.mean().mul(100).rename("oa_actual")
    ba = pred.groupby(check_keys + ["truth"]).correct.mean().groupby(check_keys).mean().mul(100).rename("ba_actual")
    checked = table.merge(oa, on=check_keys, validate="one_to_one").merge(ba, on=check_keys, validate="one_to_one")
    np.testing.assert_allclose(checked.oa, checked.oa_actual, atol=1e-10)
    np.testing.assert_allclose(checked.ba, checked.ba_actual, atol=1e-10)
    for scope, subset in (("all", table), ("common", table.loc[table.fold.isin(common)])):
        scene, port, summary = aggregate(subset)
        scene.to_csv(OUT / f"{scope}_per_scene.csv")
        port.to_csv(OUT / f"{scope}_per_port.csv")
        summary.to_csv(OUT / f"{scope}_summary.csv")
        _, _, draws = aggregate(subset, ["draw"])
        draws.to_csv(OUT / f"{scope}_by_draw.csv")
        for reference in ("original", "ship", "ring"):
            frame = summary.reset_index().merge(summary.xs(reference, level="arm").reset_index(),
                on=["budget", "origin"], suffixes=("", "_reference"))
            for name in METRICS:
                frame["delta_" + name] = frame[name] - frame[name + "_reference"]
            frame.to_csv(OUT / f"{scope}_vs_{reference}.csv", index=False)
    changed = []
    for reference in ("original", "ship", "ring"):
        ref = pred.loc[pred.arm.eq(reference), PRED_KEYS + ["pred"]].rename(columns={"pred": "reference_pred"})
        comp = pred.loc[pred.arm.eq("additive")].merge(ref, on=PRED_KEYS, validate="one_to_one")
        comp["rescued"] = comp.correct & comp.reference_pred.ne(comp.truth)
        comp["harmed"] = ~comp.correct & comp.reference_pred.eq(comp.truth)
        comp["reference"] = reference
        changed.append(comp.loc[comp.rescued | comp.harmed])
    pd.concat(changed).to_csv(OUT / "additive_changed_cases.csv", index=False)
    (OUT / "body_ablation_done.json").write_text(json.dumps(dict(
        evaluated_body_tasks=int(table.arm.eq("ship").sum()), checked_metric_rows=len(checked),
        matched_prediction_rows=len(pred), seconds=time.perf_counter()-started,
        protocol="frozen same queries/supports/priors/readout; no training or model selection",
        counts_are_repeated_observations=True), indent=2), encoding="utf-8")
    print("COMMON\n", pd.read_csv(OUT / "common_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
