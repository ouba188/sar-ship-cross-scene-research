"""Matched OSS1 B0/B16 readout on frozen same/cross-acquisition draws."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "f0921_split"))
from evaluate_pilot import backbone, encode, support_score  # noqa: E402
from general_domain_readout import fit_head, metrics  # noqa: E402

SPLIT = ROOT / "public_split"
MODES = ("G50", "G18", "D18", "G50_G18", "G50_D18")


def main():
    if not (HERE / "training_complete.json").exists():
        raise RuntimeError("Source-only R18 training must complete first")
    index = pd.read_csv(HERE / "pixel_index.csv", dtype={"mmsi_or_ship_id": str})
    assert index.sample_id.is_unique
    positions = pd.Series(np.arange(len(index)), index=index.sample_id)
    source_ids = pd.concat((
        pd.read_csv(HERE / "source_train.csv", usecols=["sample_id"]),
        pd.read_csv(HERE / "source_validation.csv", usecols=["sample_id"]),
    )).sample_id
    query_frame = pd.read_csv(SPLIT / "oss1_query.csv", dtype={"mmsi": str})
    query_ids = query_frame.sample_id
    draws = pd.read_csv(SPLIT / "oss1_support_draws.csv", dtype={"mmsi": str})
    draws = draws[draws.budget.eq(16)]
    assert draws.draw.nunique() == 5
    source = positions.loc[source_ids].to_numpy(np.int64)
    query = positions.loc[query_ids].to_numpy(np.int64)
    labeled = pd.concat([
        pd.read_csv(path, usecols=["sample_id", "class_id"])
        for path in (HERE / "source_train.csv", HERE / "source_validation.csv",
                     SPLIT / "oss1_query_labels.csv", SPLIT / "oss1_support_candidates.csv")
    ], ignore_index=True)
    assert labeled.sample_id.is_unique
    labels = labeled.set_index("sample_id").loc[index.sample_id, "class_id"].to_numpy(int)
    pixels = np.load(HERE / "pixels128.npy", mmap_mode="r")

    cache = HERE / "pilot_features.npz"
    if cache.exists():
        with np.load(cache) as z:
            np.testing.assert_array_equal(z["sample_ids"], index.sample_id.to_numpy(str))
            raw = {key: z[key] for key in ("G50", "G18", "D18")}
    else:
        raw = {}
        rows = np.arange(len(index))
        for key in ("G50", "G18", "D18"):
            model, width = backbone("r50" if key == "G50" else "r18")
            if key == "D18":
                checkpoint = torch.load(HERE / "source_only_r18_best.pt",
                                        map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["backbone"])
            raw[key] = encode(model, pixels, rows, width)
            print("encoded", key, raw[key].shape, flush=True)
            del model
            torch.cuda.empty_cache()
        np.savez_compressed(cache, sample_ids=index.sample_id.to_numpy(str), **raw)

    feature = {}
    for key, array in raw.items():
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        feature[key] = pca.fit(array[source]).transform(array).astype(np.float32)
    branches = {
        "G50": feature["G50"], "G18": feature["G18"], "D18": feature["D18"],
        "G50_G18": (feature["G50"], feature["G18"]),
        "G50_D18": (feature["G50"], feature["D18"]),
    }
    heads = {name: fit_head(part, labels, source) for name, part in branches.items()}
    predicted = []
    for draw in (-1, 0, 1, 2, 3, 4):
        for origin in (("none",) if draw == -1 else ("same", "cross")):
            support = None
            if origin != "none":
                subset = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)]
                if len(subset) != 16 or subset.mmsi.nunique() != 16:
                    raise ValueError(f"Unexpected B16 draw {draw}/{origin}")
                support = positions.loc[subset.sample_id].to_numpy(np.int64)
            scores = {mode: support_score(branches[mode], heads[mode], query,
                                          support, labels) for mode in MODES}
            for mode, score in scores.items():
                predicted.append((draw, origin, mode, score))
            for suffix in ("G18", "D18"):
                predicted.append((draw, origin, f"fused_G50_{suffix}",
                                  .5 * softmax(scores["G50"], axis=1) +
                                  .5 * softmax(scores[f"G50_{suffix}"], axis=1)))

    truth = labels[query]
    records = []
    prediction_records = []
    for draw, origin, mode, score in predicted:
        prediction = score.argmax(1)
        chip = metrics(truth, prediction)
        ship_frame = pd.DataFrame({"ship": query_frame.mmsi,
                                   "truth": truth, "score": list(score)})
        vessel = ship_frame.groupby("ship", sort=False).agg(
            truth=("truth", "first"), score=("score", lambda x: np.stack(x).mean(0)))
        ship = metrics(vessel.truth.to_numpy(int), np.stack(vessel.score).argmax(1))
        records.append({"draw": draw, "support_origin": origin, "mode": mode,
                        "chips": len(query), "ships": len(vessel),
                        "chip_oa": chip["oa"], "chip_ba": chip["ba"],
                        "ship_oa": ship["oa"], "ship_ba": ship["ba"]})
        prediction_records.extend({
            "draw": draw, "support_origin": origin, "mode": mode,
            "sample_id": str(query_frame.sample_id.iloc[i]),
            "mmsi": str(query_frame.mmsi.iloc[i]), "truth": int(truth[i]),
            "prediction": int(prediction[i]),
        } for i in range(len(query)))
    result = pd.DataFrame(records)
    result.to_csv(HERE / "pilot_metrics.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(HERE / "pilot_predictions.csv", index=False)
    summary = result.groupby(["support_origin", "mode"], sort=False)[
        ["chip_oa", "chip_ba", "ship_oa", "ship_ba"]].mean().round(3)
    summary.to_csv(HERE / "pilot_summary.csv")
    (HERE / "pilot_readout.json").write_text(json.dumps({
        "source_fit_rows": len(source), "query_chip_rows": len(query),
        "query_ships": int(query_frame.mmsi.nunique()),
        "support_budget": 16, "draws": 5, "source_pca_only": True,
        "target_query_labels_used_for_selection": False,
        "method_scope": "matched RGB/PCA compact readout; no C4 or learned class-contrast",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
