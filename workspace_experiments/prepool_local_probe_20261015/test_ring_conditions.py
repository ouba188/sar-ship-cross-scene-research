"""Paired explicit-ring shuffles and half-body replacement, without training."""

import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from run_body_context import FIELDS, packed_task, predict, shuffle_gate_input

ROOT = Path(__file__).resolve().parent / "body_context_expansion"
OUT = ROOT / "ring_conditions"
KEYS = ["port", "fold", "budget", "origin", "draw", "sample_id"]
METRICS = ["oa", "ba", "macro_nll", "macro_brier"]


def summary(frame):
    keys = ["port", "fold", "budget", "origin", "draw", "arm"]
    a = frame.groupby(keys).correct.mean().mul(100).rename("oa")
    b = frame.groupby(keys + ["truth"])[["correct", "nll", "brier"]].mean().groupby(keys).mean()
    result = a.to_frame().join(b.rename(columns={"correct": "ba", "nll": "macro_nll", "brier": "macro_brier"}))
    result.ba *= 100
    return result.reset_index()


def class_support_scores(q, s, labels):
    sim = q @ s.T
    scores = np.full((len(q), 8), -np.inf)
    for cls in np.unique(labels):
        values = np.sort(sim[:, labels == cls], axis=1)
        scores[:, cls] = values[:, -min(3, values.shape[1]):].mean(axis=1)
    return scores


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    frozen = json.loads((ROOT / "frozen.json").read_text(encoding="utf-8"))
    all_frames, features, replay, movement = [], [], [], []
    with threadpool_limits(limits=2):
        for pool in frozen["pools"]:
            if not pool["selected"]:
                continue
            slug = pool["port"].replace(" ", "_")
            folder = ROOT / slug
            with np.load(folder / "readout.npz") as cache:
                arrays = {k: cache[k] for k in (*FIELDS, "weights")}
            meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
            tasks_all = json.loads((folder / "tasks.json").read_text(encoding="utf-8"))
            for budget in sorted({t["budget"] for t in tasks_all}):
                tasks = [t for t in tasks_all if t["budget"] == budget]
                old = pd.read_csv(folder / f"predictions_B{budget}.csv", dtype={"mmsi": str})
                old = old.loc[old.arm.isin(["original", "additive"])]
                ship = pd.read_csv(ROOT / "explainability" / f"{slug}_ship_B{budget}_predictions.csv", dtype={"mmsi": str})
                all_frames.extend([old, ship])
                for ti, info in enumerate(tasks):
                    t = packed_task(arrays, info)
                    q, s = t["q"].numpy(), t["s"].numpy()
                    y, ys = t["labels"][q].numpy(), t["labels"][s].numpy()
                    records = ship.loc[ship.task.eq(ti)].copy()
                    np.testing.assert_array_equal(records.sample_id, meta.iloc[info["q"]].sample_id)
                    local_half = torch.cat((np.sqrt(.5) * t["ship"], np.sqrt(.5) * t["expert"][:, 128:256] / 4), dim=1)
                    local_real = torch.cat((np.sqrt(.5) * t["ship"], np.sqrt(.5) * t["ring"]), dim=1)
                    locals_ = {"body_half": local_half}
                    for seed in range(3):
                        shuffled, _ = shuffle_gate_input(info, meta, arrays, 20261016 + seed)
                        locals_[f"ring_shuffle_{seed}"] = torch.cat((np.sqrt(.5) * t["ship"], np.sqrt(.5) * shuffled), dim=1)
                        movement.append(dict(port=pool["port"], budget=budget, task=ti, origin=info["origin"], seed=seed,
                            changed_fraction=float(torch.any(shuffled != t["ring"], dim=1).double().mean())))
                    with torch.no_grad():
                        if ti == 0:
                            actual = predict(t, local_real).numpy()
                            expected = old.loc[old.arm.eq("additive") & old.task.eq(ti), [f"p{c}" for c in range(8)]].to_numpy()
                            np.testing.assert_allclose(actual, expected, atol=1e-10)
                            # The control restores half of the replaced D18 component exactly.
                            local_ss = 16 * (local_half[s] @ local_half[s].T)
                            target_ss = .5 * (16 * t["ship"][s] @ t["ship"][s].T + t["expert"][s, 128:256] @ t["expert"][s, 128:256].T)
                            np.testing.assert_allclose(local_ss, target_ss, atol=1e-10)
                            replay.append(dict(port=pool["port"], budget=budget, error=float(abs(actual-expected).max())))
                        for arm, local in locals_.items():
                            p = predict(t, local).numpy()
                            result = records.copy()
                            result["arm"] = arm
                            result["pred"] = p.argmax(1)
                            result["correct"] = result.pred.eq(result.truth)
                            result["nll"] = -np.log(p[np.arange(len(y)), y].clip(1e-12))
                            result["brier"] = ((p-np.eye(8)[y])**2).sum(1)
                            competitors = p.copy()
                            competitors[np.arange(len(y)), y] = -np.inf
                            result["margin"] = p[np.arange(len(y)), y] - competitors.max(1)
                            result["ap"] = np.nan
                            for cls in range(8):
                                result[f"p{cls}"] = p[:, cls]
                            all_frames.append(result)
                    f = records[KEYS + ["mmsi", "truth"]].copy()
                    bscore = class_support_scores(t["ship"][q].numpy(), t["ship"][s].numpy(), ys)
                    rscore = class_support_scores(t["ring"][q].numpy(), t["ring"][s].numpy(), ys)
                    probs = records[[f"p{c}" for c in range(8)]].to_numpy()
                    sorted_p = np.sort(probs, axis=1)
                    f["body_top2_gap"] = sorted_p[:, -1] - sorted_p[:, -2]
                    f["body_support_class"] = bscore.argmax(1)
                    f["ring_support_class"] = rscore.argmax(1)
                    f["support_class_agreement"] = f.body_support_class.eq(f.ring_support_class)
                    f["ring_support_match"] = rscore.max(1)
                    f["body_support_gap"] = np.sort(bscore, axis=1)[:, -1] - np.sort(bscore, axis=1)[:, -2]
                    f["true_class_support_count"] = [(ys == cls).sum() for cls in y]
                    for name, score in (("body", bscore), ("ring", rscore)):
                        competitor = score.copy()
                        competitor[np.arange(len(y)), y] = -np.inf
                        value = score[np.arange(len(y)), y] - competitor.max(1)
                        f[f"{name}_true_support_margin_diagnostic"] = np.where(np.isfinite(value), value, np.nan)
                    features.append(f)
                print(slug, budget, "controls complete", flush=True)
    frame = pd.concat(all_frames, ignore_index=True)
    assert not frame.duplicated(KEYS + ["arm"]).any()
    assert frame.groupby(KEYS).arm.nunique().eq(7).all()
    frame.to_csv(OUT / "predictions.csv", index=False)
    pd.concat(features).to_csv(OUT / "support_conditions.csv", index=False)
    metric = summary(frame)
    metric.to_csv(OUT / "metrics.csv", index=False)
    mean = metric.groupby(["port", "fold", "budget", "origin", "arm"])[METRICS].mean().groupby(
        ["port", "budget", "origin", "arm"]).mean().groupby(["budget", "origin", "arm"]).mean()
    mean.to_csv(OUT / "summary.csv")
    pd.DataFrame(movement).to_csv(OUT / "shuffle_movement.csv", index=False)
    (OUT / "controls_done.json").write_text(json.dumps(dict(replay=replay, metric_rows=len(metric),
        prediction_rows=len(frame), task_units=int(metric.arm.eq("body_half").sum()),
        half_body_control="12.5% body replacement, remaining 12.5% restored original D18",
        explicit_ring_shuffle_seeds=[20261016,20261017,20261018]), indent=2), encoding="utf-8")
    print(mean.to_string())


if __name__ == "__main__":
    main()
