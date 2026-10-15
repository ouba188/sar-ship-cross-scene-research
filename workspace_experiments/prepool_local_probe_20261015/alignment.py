"""Pair B16 retrieval gains with errors of the unchanged full classifier."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from full_readout import CONTEXT, HERE, LOCAL, source_gallery


def normalized_pca(values, gallery, width):
    with threadpool_limits(limits=2):
        pca = PCA(n_components=min(width, values.shape[1]), svd_solver="randomized",
                  random_state=20261015)
        transformed = pca.fit(values[gallery]).transform(values).astype(np.float32)
    return transformed / np.maximum(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12)


def ap_per_query(similarity, qlabel, slabel):
    result = []
    for i, cls in enumerate(qlabel):
        relevant = slabel == cls
        count = int(relevant.sum())
        if not count:
            result.append(np.nan)
            continue
        hits = relevant[np.argsort(-similarity[i])[:count]]
        result.append(float(np.sum(np.cumsum(hits) /
                                   np.arange(1, count + 1) * hits) / count))
    return np.asarray(result)


def main():
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(HERE / "full_readout/predictions.csv")
    retrieval = []
    for episode in episodes:
        fold = episode["fold"]
        folder = Path(episode["folder"])
        with np.load(LOCAL / f"features_{fold}.npz") as saved:
            rows = saved["rows"].astype(int)
            gallery = np.searchsorted(rows, source_gallery(folder))
            features = {
                "gap": normalized_pca(saved["gap"], gallery, 256),
                "full": normalized_pca(saved["full_peak"], gallery, 64),
                "ship": normalized_pca(saved["ship_peak"], gallery, 64),
                "background": normalized_pca(saved["background_peak_1"], gallery, 64),
                "layer2": normalized_pca(saved["layer2_peak"], gallery, 64),
                "layer2_ship": normalized_pca(saved["layer2_ship_peak"], gallery, 64),
                "layer2_background": normalized_pca(
                    saved["layer2_background_peak_1"], gallery, 64),
                "near_ring": normalized_pca(saved["near_ring_mean"], gallery, 64),
                "near_contrast": normalized_pca(saved["near_contrast_peak"], gallery, 64),
                "raw_ship": normalized_pca(saved["raw_ship"], gallery, 64),
                "raw_ring": normalized_pca(saved["raw_ring"], gallery, 64),
                "raw_contrast": normalized_pca(saved["raw_contrast"], gallery, 64),
            }
        query = pd.read_csv(folder / "query.csv").set_index("sample_id").loc[episode["query"]]
        labels = pd.read_csv(folder / "query_labels.csv").set_index("sample_id")
        qlabel = labels.loc[episode["query"], "class_id"].to_numpy(int)
        candidates = pd.read_csv(folder / "support_candidates.csv").set_index("sample_id")
        qpos = np.searchsorted(rows, query.row_index.to_numpy(int))
        for draw_id, draw in enumerate(episode["draws"]):
            for origin in ("same", "cross"):
                support = candidates.loc[draw[origin]]
                spos = np.searchsorted(rows, support.row_index.to_numpy(int))
                slabel = support.class_id.to_numpy(int)
                base = features["gap"][qpos] @ features["gap"][spos].T
                measures = {"gap": ap_per_query(base, qlabel, slabel)}
                for name in ("full", "ship", "background", "layer2", "layer2_ship",
                             "layer2_background", "near_ring", "near_contrast",
                             "raw_ship", "raw_ring", "raw_contrast"):
                    local = features[name][qpos] @ features[name][spos].T
                    measures[name] = ap_per_query(.75 * base + .25 * local, qlabel, slabel)
                for i, sample_id in enumerate(episode["query"]):
                    retrieval.append(dict(fold=fold, draw=draw_id, origin=origin,
                                          sample_id=sample_id, truth=int(qlabel[i]),
                                          **{f"ap_{name}": values[i]
                                             for name, values in measures.items()}))
        print("retrieval aligned", fold, flush=True)
    frame = pd.DataFrame(retrieval)
    names = ["plain_original", "plain_full_peak_025", "plain_ship_peak_025",
             "plain_background_peak_1_025", "plain_layer2_peak_025",
             "plain_layer2_ship_peak_025", "plain_layer2_background_peak_1_025"]
    pred = predictions.loc[predictions.arm.isin(names)]
    pivot = pred.pivot(index=["fold", "draw", "origin", "sample_id"],
                       columns="arm", values=["correct", "true_probability"])
    pivot.columns = [f"{field}_{arm}" for field, arm in pivot.columns]
    frame = frame.join(pivot, on=["fold", "draw", "origin", "sample_id"], validate="one_to_one")
    assert frame.correct_plain_original.notna().all()
    archived = pd.concat([pd.read_csv(path) for path in sorted(LOCAL.glob("results_*.csv"))])
    for fold in frame.fold.unique():
        for origin in ("same", "cross"):
            chosen = frame.loc[frame.fold.eq(fold) & frame.origin.eq(origin)]
            for name, reference in (("gap", "gap"), ("full", "fusion_full_peak_w0.25")):
                expected = archived.loc[archived.fold.eq(fold) & archived.origin.eq(origin)
                                        & archived.representation.eq(reference)].iloc[0]
                draws = np.asarray(json.loads(expected.target_b16_draws))[:, 0]
                actual = chosen.groupby("draw")[f"ap_{name}"].mean().to_numpy()
                np.testing.assert_allclose(actual, draws, atol=1e-9, rtol=1e-9)
    out = HERE / "alignment"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "per_query.csv", index=False)
    valid = frame.loc[frame.origin.eq("cross") & frame.ap_gap.notna()].copy()
    valid["raw_contrast_minus_ship"] = valid.ap_raw_contrast - valid.ap_ship
    valid["raw_contrast_minus_raw_ship"] = valid.ap_raw_contrast - valid.ap_raw_ship
    print("raw contrast vs ship by fold and class", flush=True)
    print(valid.groupby(["fold", "truth"])[
        ["raw_contrast_minus_ship", "raw_contrast_minus_raw_ship"]]
          .agg(["mean", "size"]).round(4).to_string(), flush=True)
    valid["baseline_correct"] = valid.correct_plain_original.astype(bool)
    prediction_names = {"full": "plain_full_peak_025", "ship": "plain_ship_peak_025",
                        "background": "plain_background_peak_1_025",
                        "layer2": "plain_layer2_peak_025",
                        "layer2_ship": "plain_layer2_ship_peak_025",
                        "layer2_background": "plain_layer2_background_peak_1_025"}
    for name in prediction_names:
        valid[f"delta_{name}"] = valid[f"ap_{name}"] - valid.ap_gap
        valid[f"prob_delta_{name}"] = (valid[f"true_probability_{prediction_names[name]}"] -
                                        valid.true_probability_plain_original)
    summary = valid.groupby(["fold", "baseline_correct"])[
        [f"delta_{name}" for name in prediction_names]].agg(["mean", "size"])
    print(summary.round(4).to_string(), flush=True)
    print("cross by baseline correctness", flush=True)
    print(valid.groupby("baseline_correct")[
        [f"delta_{name}" for name in prediction_names]].mean().round(4).to_string(),
          flush=True)


if __name__ == "__main__":
    main()
