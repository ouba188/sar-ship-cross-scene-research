"""Compare frozen R50 and SAR-SSL R18 routes across acquisition conditions."""
import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
OLD = PARENT / "external_scene_transfer_20261013"
SOURCE = PARENT / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916"
EXTRA = PARENT / ".." / "Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007/frozen_replay/source_extra.npy"
sys.path.insert(0, str(Path("E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")))
from unified_head_core import coefficients


def read_features(name, head):
    with np.load(OLD / f"{name}_encoded.npz") as saved:
        feature = {key: saved[key] for key in ("rows", "r", "t", "b", "pr", "pb", "pd")}
    feature["p18"] = head.decision_function(feature["t"][:, :64] * np.sqrt(384.0))
    return feature


def predict(feature, prior, query, support, labels):
    support_kernel = feature[support] @ feature[support].T + 1.0
    query_kernel = feature[query] @ feature[support].T + 1.0
    alpha = coefficients(support_kernel, prior[support], labels, 8.0)
    return softmax(prior[query] + query_kernel @ alpha, axis=1)


def score(probability, labels):
    predicted = probability.argmax(1)
    classes = np.unique(labels)
    return 100 * np.mean(predicted == labels), 100 * np.mean(
        [np.mean(predicted[labels == c] == c) for c in classes]
    )


def main():
    with threadpool_limits(limits=2):
        with (OLD / "fits/geometry.pkl").open("rb") as handle:
            model = pickle.load(handle)["model"]
        rows = np.load(OLD / "source_rows.npy")
        source_extra = np.load(EXTRA, mmap_mode="r")
        source_labels = pd.read_csv(SOURCE / "manifest.csv", usecols=["class_id"]).class_id.to_numpy(int)[rows]
        source_64 = model["pca_ssl"].transform(source_extra[rows]).astype(np.float32)
        source_64 = (source_64 - model["expert_center"][128:192]) / model["expert_scale"][128:192]
        head = RidgeClassifier(alpha=1.0, class_weight="balanced").fit(source_64, source_labels)
        np.testing.assert_array_equal(head.classes_, np.arange(8))

        plans = json.loads((OLD / "ENROLLMENT.json").read_text(encoding="utf-8"))
        plans = [p for p in plans if p["study"] == "acquisition_transfer"]
        features = {name: read_features(name, head) for name in ("f0921", "oss2")}
        truth = {
            name: pd.read_csv(OLD / f"{name}_metadata.csv", usecols=["class_id"]).class_id.to_numpy(int)
            for name in features
        }
        records = []
        replay_error = 0.0
        for plan in plans:
            name = plan["target"]
            feat = features[name]
            query_rows = np.asarray(plan["query"], int)
            query = np.searchsorted(feat["rows"], query_rows)
            y = truth[name][query_rows]
            with np.load(OLD / "predictions" / f"episode_{plan['episode']:03d}.npz") as saved:
                np.testing.assert_array_equal(query_rows, saved["query_rows"])
                old_predictions = saved["probabilities"]
            for draw, supports in enumerate(plan["support_rows"]):
                for arm_index, support_rows in enumerate(supports):
                    support_rows = np.asarray(support_rows, int)
                    support = np.searchsorted(feat["rows"], support_rows)
                    sy = truth[name][support_rows]
                    base = predict(feat["b"], feat["pb"], query, support, sy)
                    r50 = predict(feat["r"], feat["pr"], query, support, sy)
                    r18 = predict(feat["t"], feat["p18"], query, support, sy)
                    dual = predict(np.c_[feat["r"], feat["t"]], feat["pd"], query, support, sy)
                    outputs = {
                        "R50": r50,
                        "R18": r18,
                        "Dual": dual,
                        "Base+R50": 0.5 * (base + r50),
                        "Base+R18": 0.5 * (base + r18),
                        "Base+Dual": 0.5 * (base + dual),
                    }
                    for actual, archived in zip(
                        (base, outputs["Base+R50"], outputs["Base+Dual"]),
                        old_predictions[draw, arm_index],
                    ):
                        replay_error = max(replay_error, float(np.max(np.abs(actual - archived))))
                    for mode, probability in outputs.items():
                        oa, ba = score(probability, y)
                        records.append(dict(target=name, budget=plan["budget"], acquisition=plan["acquisition"],
                                            draw=draw, arm=plan["arms"][arm_index], mode=mode,
                                            n_query=len(y), oa=oa, ba=ba))

    assert replay_error < 1e-8
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    scene = metrics.groupby(["target", "budget", "acquisition", "arm", "mode"])[["oa", "ba"]].mean().reset_index()
    scene.to_csv(OUT / "scene_metrics.csv", index=False)
    summary = scene.groupby(["target", "budget", "arm", "mode"])[["oa", "ba"]].mean().reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    wide = scene.pivot(index=["target", "budget", "acquisition", "arm"], columns="mode", values="ba")
    effects = pd.DataFrame({
        "r18_minus_r50": wide["R18"] - wide["R50"],
        "dual_minus_r50": wide["Dual"] - wide["R50"],
        "base_r18_minus_base_r50": wide["Base+R18"] - wide["Base+R50"],
        "base_dual_minus_base_r50": wide["Base+Dual"] - wide["Base+R50"],
    }).reset_index()
    effects.to_csv(OUT / "scene_effects.csv", index=False)
    (OUT / "CHECKS.json").write_text(json.dumps(dict(source_n=len(rows), episodes=len(plans),
        metric_rows=len(metrics), archived_replay_max_error=replay_error, new_neural_training=0,
        target_labels_used_for_fitting=False), indent=2), encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("Archived replay max error:", replay_error)


if __name__ == "__main__":
    main()
