"""Evaluate one strict 24-port source encoder on frozen F_0921/OSS2 episodes."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
EXTERNAL = ROOT.parent / "external_scene_transfer_20261013"
sys.path.insert(0, str(ROOT / "f0921_split"))
from evaluate_pilot import backbone, encode, support_score  # noqa: E402
from general_domain_readout import fit_head, metrics  # noqa: E402


SOURCE_META = ROOT.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"
DATA = {
    "f0921": (EXTERNAL / "f0921_metadata.csv",
              Path("E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/inputs128.npy")),
    "oss2": (EXTERNAL / "oss2_metadata.csv",
             Path("D:/SAR_completion_cache_20261010/oss2_inputs128.npy")),
}
SOURCE_PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
MODES = ("G50", "G18", "D18", "G50_G18", "G50_D18",
         "fused_G50_G18", "fused_G50_D18")


def episode_rows(plans, name):
    selected = [p for p in plans if p["study"] == "dataset_adaptation" and p["target"] == name]
    assert sorted(p["budget"] for p in selected) == [16, 64]
    assert selected[0]["query"] == selected[1]["query"]
    rows = [selected[0]["query"]]
    for episode in selected:
        assert len(episode["support_rows"]) == 5
        for draw in episode["support_rows"]:
            rows.append(draw[0])
    return selected, np.unique(np.concatenate(rows).astype(np.int64))


def raw_features(name, pixel_path, rows, checkpoint):
    cache = OUT / f"{name}_raw.npz"
    if cache.exists():
        with np.load(cache) as z:
            np.testing.assert_array_equal(z["rows"], rows)
            return {key: z[key] for key in ("G50", "G18", "D18")}
    pixels = np.load(pixel_path, mmap_mode="r")
    assert rows.max() < len(pixels)
    arrays = {}
    for key in ("G50", "G18", "D18"):
        model, width = backbone("r50" if key == "G50" else "r18")
        if key == "D18":
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state["backbone"])
        arrays[key] = encode(model, pixels, rows, width)
        print(f"{name} encoded {key}: {arrays[key].shape}", flush=True)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(cache, rows=rows, **arrays)
    return arrays


def evaluate(name, plans, source, source_raw, target_meta, target_rows, target_raw):
    source_labels = pd.read_csv(SOURCE_META, usecols=["class_id"]).class_id.to_numpy(int)[source]
    target_labels = target_meta.class_id.to_numpy(int)[target_rows]
    labels = np.r_[source_labels, target_labels]
    n_source = len(source)
    source_index = np.arange(n_source)
    projected = {}
    for key in ("G50", "G18", "D18"):
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        pca.fit(source_raw[key])
        projected[key] = np.vstack((pca.transform(source_raw[key]),
                                    pca.transform(target_raw[key]))).astype(np.float32)
    branches = {
        "G50": projected["G50"], "G18": projected["G18"], "D18": projected["D18"],
        "G50_G18": (projected["G50"], projected["G18"]),
        "G50_D18": (projected["G50"], projected["D18"]),
    }
    heads = {key: fit_head(part, labels, source_index) for key, part in branches.items()}
    external_local = lambda rows: n_source + np.searchsorted(target_rows, np.asarray(rows))
    query_original = np.asarray(plans[0]["query"], dtype=np.int64)
    query = external_local(query_original)
    truth = target_meta.class_id.to_numpy(int)[query_original]
    assert len(query_original) == len(np.unique(target_meta.identity.iloc[query_original]))
    conditions = [(0, -1, None)]
    for plan in plans:
        for draw_idx, draw in enumerate(plan["support_rows"]):
            support_original = np.asarray(draw[0], dtype=np.int64)
            assert len(support_original) == plan["budget"]
            assert not set(target_meta.identity.iloc[query_original]) & set(
                target_meta.identity.iloc[support_original])
            conditions.append((plan["budget"], draw_idx, external_local(support_original)))

    records, predictions = [], []
    for budget, draw_idx, support in conditions:
        score = {key: support_score(part, heads[key], query, support, labels)
                 for key, part in branches.items()}
        for key in ("G18", "D18"):
            score[f"fused_G50_{key}"] = .5 * softmax(score["G50"], axis=1) + \
                                        .5 * softmax(score[f"G50_{key}"], axis=1)
        for mode in MODES:
            pred = score[mode].argmax(1)
            records.append({"dataset": name, "budget": budget, "draw": draw_idx,
                            "model": mode, "query_n": len(query),
                            "query_classes": len(np.unique(truth)), **metrics(truth, pred)})
            predictions.extend({"dataset": name, "budget": budget, "draw": draw_idx,
                                "model": mode, "sample_id": str(target_meta.sample_id.iloc[row]),
                                "truth": int(y), "prediction": int(p)}
                               for row, y, p in zip(query_original, truth, pred))
    return records, predictions


def main():
    if not (OUT / "training_complete.json").exists():
        raise RuntimeError("Train the 24-port source-only R18 before evaluation")
    with np.load(OUT / "source_split.npz") as split:
        source = split["source_fit"]
    plans = json.loads((EXTERNAL / "ENROLLMENT.json").read_text(encoding="utf-8"))
    tasks = {}
    for name, (meta_path, pixel_path) in DATA.items():
        selected, rows = episode_rows(plans, name)
        meta = pd.read_csv(meta_path, usecols=["sample_id", "identity", "class_id"],
                           dtype={"sample_id": str, "identity": str})
        assert rows.max() < len(meta)
        tasks[name] = (selected, rows, meta, pixel_path)

    checkpoint = OUT / "source_only_r18_best.pt"
    source_raw = raw_features("source24", SOURCE_PIXELS, source, checkpoint)
    records, predictions = [], []
    for name, (selected, rows, meta, pixel_path) in tasks.items():
        target_raw = raw_features(name, pixel_path, rows, checkpoint)
        part_records, part_predictions = evaluate(name, selected, source, source_raw,
                                                   meta, rows, target_raw)
        records.extend(part_records)
        predictions.extend(part_predictions)
        print(f"{name} evaluated {len(part_records)} conditions", flush=True)
    result = pd.DataFrame(records)
    result.to_csv(OUT / "external_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(OUT / "external_predictions.csv", index=False)
    summary = result.groupby(["dataset", "budget", "model"], sort=False)[["oa", "ba"]].mean()
    summary.to_csv(OUT / "external_summary.csv")
    (OUT / "readout_protocol.json").write_text(json.dumps({
        "source_only_ssl": True, "source_fit_rows": len(source),
        "target_query_labels_used_for_selection": False,
        "episodes": "external_scene_transfer_20261013/ENROLLMENT.json dataset_adaptation only",
        "source_pca_only": True, "budgets": [0, 16, 64], "draws_nonzero": 5,
        "model_scope": "compact RGB/PCA readout; no C4 or learned class-contrast",
    }, indent=2), encoding="utf-8")
    print(summary.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
