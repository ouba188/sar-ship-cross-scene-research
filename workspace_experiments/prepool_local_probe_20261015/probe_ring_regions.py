"""Frozen region and position-preserving statistics in the full cached classifier."""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from full_readout import source_gallery
from run_body_context import FIELDS, packed_task, predict
from test_ring_conditions import summary

HERE = Path(__file__).resolve().parent
ROOT = HERE / "body_context_expansion"
OUT = ROOT / "ring_region_spatial"
ARMS = ("body", "near", "touch", "far", "wide", "multiring", "near_position", "near_bag")


def normalized(raw, fit, selected):
    pca = PCA(n_components=raw.shape[1], svd_solver="randomized", random_state=20261015)
    pca.fit(raw[fit])
    z = pca.transform(raw[selected]).astype(np.float64)
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    return z


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    frozen = json.loads((ROOT / "frozen.json").read_text(encoding="utf-8"))
    pools = [p for p in frozen["pools"] if p["selected"]]
    inventory, galleries = [], {}
    for p in pools:
        folder = Path(p["source_folder"])
        train = pd.read_csv(folder / "source_train.csv")
        val = pd.read_csv(folder / "source_validation.csv")
        train = train.loc[~train.acquisition.isin(val.acquisition)]
        gallery = source_gallery(folder)
        gallery = gallery[np.isin(gallery, train.row_index)]
        galleries[p["port"]] = gallery
        target = pd.read_csv(ROOT / p["port"].replace(" ", "_") / "metadata.csv")
        tasks = json.loads((ROOT / p["port"].replace(" ", "_") / "tasks.json").read_text())
        needed = sorted({x for t in tasks for x in t["q"] + t["s"]})
        inventory.extend(gallery.tolist())
        inventory.extend(target.iloc[needed].global_row.tolist())
    rows = np.unique(inventory).astype(int)
    np.save(OUT / "required_rows.npy", rows)
    if not (OUT / "raw_regions.npz").exists():
        subprocess.run([sys.executable, str(HERE / "extract_ring_regions.py")], check=True)
    with np.load(OUT / "raw_regions.npz") as cache:
        np.testing.assert_array_equal(rows, cache["rows"])
        raw = {k: cache[k] for k in cache.files if k != "rows"}
    coverage = pd.read_csv(OUT / "region_coverage.csv")
    missing = {k: int((~np.isfinite(v).all(axis=1)).sum()) for k, v in raw.items()}
    print("COVERAGE", len(rows), missing, flush=True)
    if any(missing.values()):
        raise RuntimeError("Empty regions detected; inspect coverage before defining any fallback")
    frames, replays = [], []
    with threadpool_limits(limits=2):
        for p in pools:
            port = p["port"]
            slug = port.replace(" ", "_")
            folder = ROOT / slug
            with np.load(folder / "readout.npz") as cache:
                arrays = {k: cache[k] for k in (*FIELDS, "weights")}
            meta = pd.read_csv(folder / "metadata.csv", dtype={"mmsi": str})
            tasks = json.loads((folder / "tasks.json").read_text())
            needed = np.asarray(sorted({x for t in tasks for x in t["q"] + t["s"]}))
            selected = np.searchsorted(rows, meta.iloc[needed].global_row.to_numpy())
            np.testing.assert_array_equal(rows[selected], meta.iloc[needed].global_row)
            fit = np.searchsorted(rows, galleries[port])
            np.testing.assert_array_equal(rows[fit], galleries[port])
            z = {}
            for name, value in raw.items():
                coordinate = normalized(value, fit, selected)
                z[name] = np.zeros((len(meta), coordinate.shape[1]))
                z[name][needed] = coordinate
            z["multiring"] = np.concatenate([z[name] / np.sqrt(3) for name in ("touch", "near", "far")], axis=1)
            old = pd.concat([pd.read_csv(folder / f"predictions_B{b}.csv", dtype={"mmsi": str})
                             for b in sorted({t["budget"] for t in tasks})], ignore_index=True)
            for ti, info in enumerate(tasks):
                task = packed_task(arrays, info)
                localrows = np.r_[info["s"], info["q"]]
                qs = np.asarray(info["q"])
                query = meta.iloc[qs]
                y = arrays["labels"][qs]
                base = query[["sample_id", "mmsi"]].copy().reset_index(drop=True)
                for k in ("port", "fold", "budget", "origin", "draw"):
                    base[k] = info[k]
                base["task"], base["truth"] = ti, y
                locals_ = {"body": task["ship"], "near": torch.cat([np.sqrt(.5)*task["ship"], np.sqrt(.5)*task["ring"]], dim=1)}
                for name in ARMS[2:]:
                    locals_[name] = torch.cat([np.sqrt(.5)*task["ship"],
                        np.sqrt(.5)*torch.from_numpy(z[name][localrows])], dim=1)
                with torch.no_grad():
                    if ti == 0:
                        test_local = torch.cat([np.sqrt(.5)*task["ship"],
                            np.sqrt(.5)*torch.from_numpy(z["near"][localrows])], dim=1)
                        original = predict(task, locals_["near"]).numpy()
                        recomputed = predict(task, test_local).numpy()
                        delta = float(np.abs(original - recomputed).max())
                        np.testing.assert_allclose(original, recomputed, atol=1e-6, rtol=1e-6)
                        replays.append(dict(port=port, kind="recomputed_near_pixels", error=delta))
                    for name, local in locals_.items():
                        prob = predict(task, local).numpy()
                        if name == "near":
                            ref = old.loc[old.arm.eq("additive") & old.fold.eq(info["fold"]) &
                                old.budget.eq(info["budget"]) & old.origin.eq(info["origin"]) & old.draw.eq(info["draw"])]
                            ref = ref.set_index("sample_id").loc[query.sample_id]
                            np.testing.assert_allclose(prob, ref[[f"p{c}" for c in range(8)]], atol=1e-10)
                        frame = base.copy()
                        frame["arm"], frame["pred"] = name, prob.argmax(axis=1)
                        frame["correct"] = frame.pred.eq(frame.truth)
                        frame["nll"] = -np.log(prob[np.arange(len(y)), y].clip(1e-12))
                        frame["brier"] = ((prob - np.eye(8)[y]) ** 2).sum(axis=1)
                        for c in range(8):
                            frame[f"p{c}"] = prob[:, c]
                        frames.append(frame)
            print("PORT", port, "tasks", len(tasks), "done", flush=True)
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    metrics = summary(predictions)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    per_port = metrics.groupby(["port", "budget", "origin", "arm"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().reset_index()
    per_port.to_csv(OUT / "per_port.csv", index=False)
    per_port.groupby(["budget", "origin", "arm"])[["oa", "ba", "macro_nll", "macro_brier"]].mean().to_csv(OUT / "summary.csv")
    (OUT / "DONE.json").write_text(json.dumps(dict(seconds=time.perf_counter()-started, samples=len(rows),
        raw_missing=missing, archived_near_all_task_replay=True, near_recomputed_checks=replays,
        arms=list(ARMS), prediction_rows=len(predictions)), indent=2), encoding="utf-8")
    print("DONE", time.perf_counter()-started, flush=True)


if __name__ == "__main__":
    main()
