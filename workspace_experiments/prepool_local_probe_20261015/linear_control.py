"""Source-only equal-width linear readout for late and multilevel SAR-R18 features."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from full_readout import CONTEXT, HERE, LOCAL, source_gallery

sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_readout import design_block, fit_head  # noqa: E402


def project(raw, gallery, width):
    with threadpool_limits(limits=2):
        pca = PCA(n_components=width, svd_solver="randomized", random_state=20261015)
        pca.fit(raw[gallery])
        return pca.transform(raw).astype(np.float32)


def classification_scores(features, labels, train, test):
    head = fit_head(features, labels, train)
    scores = design_block(features, test) @ head
    return scores.argmax(1)


def metric(truth, pred):
    return (100. * np.mean(pred == truth),
            100. * np.mean([np.mean(pred[truth == cls] == cls)
                            for cls in np.unique(truth)]))


def main():
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    output = []
    for episode in episodes:
        folder = Path(episode["folder"])
        fold = episode["fold"]
        train_frame = pd.read_csv(folder / "source_train.csv")
        val_frame = pd.read_csv(folder / "source_validation.csv")
        query_frame = pd.read_csv(folder / "query.csv").set_index("sample_id").loc[episode["query"]]
        query_label = pd.read_csv(folder / "query_labels.csv").set_index("sample_id")
        with np.load(LOCAL / f"features_{fold}.npz") as saved:
            rows = saved["rows"].astype(int)
            gallery_rows = source_gallery(folder)
            gallery = np.searchsorted(rows, gallery_rows)
            val = np.searchsorted(rows, val_frame.row_index.to_numpy(int))
            query = np.searchsorted(rows, query_frame.row_index.to_numpy(int))
            np.testing.assert_array_equal(rows[gallery], gallery_rows)
            np.testing.assert_array_equal(rows[val], val_frame.row_index.to_numpy(int))
            np.testing.assert_array_equal(rows[query], query_frame.row_index.to_numpy(int))
            source_labels = train_frame.set_index("row_index").loc[gallery_rows, "class_id"].to_numpy(int)
            labels = np.zeros(len(rows), dtype=int)
            labels[gallery] = source_labels
            late128 = project(saved["gap"], gallery, 128)
            late64 = project(saved["gap"], gallery, 64)
            features = {
                "late128": late128,
                "late64": late64,
                "late64+late64": (late64, late64),
                "late64+stem_gap64": (late64, project(saved["stem_gap"], gallery, 64)),
                "late64+stem_peak64": (late64, project(saved["full_peak"], gallery, 64)),
                "late64+layer2_peak64": (late64, project(saved["layer2_peak"], gallery, 64)),
                "late64+layer2_ship64": (late64, project(saved["layer2_ship_peak"], gallery, 64)),
                "late64+layer2_bg64": (late64, project(saved["layer2_background_peak_1"],
                                                       gallery, 64)),
            }
        truth_val = val_frame.class_id.to_numpy(int)
        truth_query = query_label.loc[episode["query"], "class_id"].to_numpy(int)
        for arm, representation in features.items():
            pred_val = classification_scores(representation, labels, gallery, val)
            pred_query = classification_scores(representation, labels, gallery, query)
            for split, truth, pred in (("source_val", truth_val, pred_val),
                                       ("target_query", truth_query, pred_query)):
                oa, ba = metric(truth, pred)
                output.append(dict(fold=fold, arm=arm, split=split, n=len(truth),
                                   classes=len(np.unique(truth)), oa=oa, ba=ba))
        print("completed", fold, flush=True)
    pd.DataFrame(output).to_csv(HERE / "linear_control.csv", index=False)
    print(pd.DataFrame(output).groupby(["split", "arm"])[["oa", "ba"]].mean().round(3).to_string(),
          flush=True)


if __name__ == "__main__":
    main()
