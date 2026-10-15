"""Matched OSS2 B16 readout for the frozen acquisition-held pilot."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "f0921_split"))
from evaluate_pilot import backbone, encode, support_score  # noqa: E402
from general_domain_readout import fit_head, metrics  # noqa: E402


SPLIT = ROOT / "public_split"
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
PIXELS = Path("D:/SAR_completion_cache_20261010/oss2_inputs128.npy")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-fold", choices=("original", "balanced"), default="original")
    parser.add_argument("--episode-dir", type=Path)
    parser.add_argument("--dataset", choices=("oss1", "oss2"), default="oss2")
    args = parser.parse_args()
    if args.episode_dir and args.source_fold != "original":
        parser.error("--source-fold balanced is only available for the original pilot")
    out = (args.episode_dir / "fit" if args.episode_dir else
           ROOT / "strict_training" / ("oss2_v2" if args.source_fold == "balanced" else "oss2"))
    if not (out / "training_complete.json").exists():
        raise RuntimeError("Source-only R18 training must complete first")
    manifest_path = (Path(f"E:/Hermes/paper_tracking/sensitivity/opensarship{args.dataset[-1]}_transfer_20261007/target_manifest.csv")
                     if args.episode_dir else MANIFEST)
    pixel_path = (ROOT / "public_expansion" / "oss1_inputs128.npy"
                  if args.episode_dir and args.dataset == "oss1" else PIXELS)
    metadata = pd.read_csv(manifest_path, usecols=["sample_id", "class_id", "mmsi_or_ship_id", "scene_id"])
    index = pd.Series(np.arange(len(metadata)), index=metadata.sample_id)
    split = args.episode_dir if args.episode_dir else SPLIT
    source_folder = out if args.source_fold == "balanced" else split
    prefix = "" if args.episode_dir or args.source_fold == "balanced" else "oss2_"
    source_ids = pd.concat([
        pd.read_csv(source_folder / f"{prefix}source_train.csv", usecols=["sample_id"]),
        pd.read_csv(source_folder / f"{prefix}source_validation.csv", usecols=["sample_id"]),
    ]).sample_id
    query_ids = pd.read_csv(split / f"{'' if args.episode_dir else 'oss2_'}query.csv", usecols=["sample_id"]).sample_id
    draws = pd.read_csv(split / f"{'' if args.episode_dir else 'oss2_'}support_draws.csv")
    draws = draws[draws.budget.eq(16)]
    if draws.draw.nunique() != 5:
        raise ValueError("Expected five frozen B16 support draws")
    rows = np.unique(index.loc[pd.concat([source_ids, query_ids, draws.sample_id])].to_numpy(np.int64))
    local = lambda ids: np.searchsorted(rows, index.loc[ids].to_numpy(np.int64))
    source, query = local(source_ids), local(query_ids)
    labels = metadata.class_id.to_numpy(int)[rows]
    pixels = np.load(pixel_path, mmap_mode="r")

    cache = out / "pilot_features.npz"
    if cache.exists():
        with np.load(cache) as z:
            np.testing.assert_array_equal(z["rows"], rows)
            raw = {key: z[key] for key in ("G50", "G18", "D18")}
    else:
        raw = {}
        for key in ("G50", "G18", "D18"):
            model, width = backbone("r50" if key == "G50" else "r18")
            if key == "D18":
                checkpoint = torch.load(out / "source_only_r18_best.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint["backbone"])
            raw[key] = encode(model, pixels, rows, width)
            print("encoded", key, raw[key].shape, flush=True)
            del model
            torch.cuda.empty_cache()
        np.savez_compressed(cache, rows=rows, **raw)

    feature = {}
    for name, array in raw.items():
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        feature[name] = pca.fit(array[source]).transform(array).astype(np.float32)
    branches = {
        "G50": feature["G50"], "G18": feature["G18"], "D18": feature["D18"],
        "G50_G18": (feature["G50"], feature["G18"]),
        "G50_D18": (feature["G50"], feature["D18"]),
    }
    heads = {name: fit_head(part, labels, source) for name, part in branches.items()}
    query_meta = metadata.iloc[rows[query]].reset_index(drop=True)
    query_products = pd.read_csv(split / f"{'' if args.episode_dir else 'oss2_'}query.csv", usecols=["sample_id", "product"])
    query_products = query_products.set_index("sample_id").loc[query_meta.sample_id, "product"].to_numpy()
    predicted = []
    records = []
    product_records = []
    prediction_records = []
    for draw in (-1, 0, 1, 2, 3, 4):
        origins = ("none",) if draw == -1 else ("same", "cross")
        for origin in origins:
            support = None
            if origin != "none":
                subset = draws[draws.draw.eq(draw) & draws.support_origin.eq(origin)]
                if len(subset) != 16:
                    raise ValueError(f"Unexpected B16 draw {draw}/{origin}")
                support = local(subset.sample_id)
            for mode, part in branches.items():
                score = support_score(part, heads[mode], query, support, labels)
                predicted.append((draw, origin, mode, score))
            base = predicted[-5][3]
            for suffix in ("G18", "D18"):
                expert = next(s for d, o, m, s in predicted[-5:] if m == f"G50_{suffix}")
                predicted.append((draw, origin, f"fused_G50_{suffix}",
                                  .5 * softmax(base, axis=1) + .5 * softmax(expert, axis=1)))
    for draw, origin, mode, score in predicted:
        pred = score.argmax(1)
        chip = metrics(labels[query], pred)
        frame = pd.DataFrame({"ship": query_meta.mmsi_or_ship_id, "truth": labels[query],
                              "score": list(score)})
        vessel = frame.groupby("ship", sort=False).agg(
            truth=("truth", "first"), score=("score", lambda x: np.stack(x).mean(0)))
        vessel_pred = np.stack(vessel.score).argmax(1)
        ship = metrics(vessel.truth.to_numpy(int), vessel_pred)
        records.append({"draw": draw, "support_origin": origin, "mode": mode,
                        "chips": len(query), "ships": len(vessel),
                        "chip_oa": chip["oa"], "chip_ba": chip["ba"],
                        "ship_oa": ship["oa"], "ship_ba": ship["ba"]})
        prediction_records.extend({
            "draw": draw, "support_origin": origin, "mode": mode,
            "sample_id": str(query_meta.sample_id.iloc[i]),
            "mmsi": str(query_meta.mmsi_or_ship_id.iloc[i]),
            "product": str(query_products[i]), "truth": int(labels[query][i]),
            "prediction": int(pred[i]),
        } for i in range(len(query)))
        for product in ("GRDH", "SLC"):
            mask = query_products == product
            if mask.any():
                product_records.append({"draw": draw, "support_origin": origin,
                                        "mode": mode, "product": product,
                                        "chips": int(mask.sum()),
                                        **metrics(labels[query][mask], pred[mask])})
    result = pd.DataFrame(records)
    result.to_csv(out / "pilot_metrics.csv", index=False)
    pd.DataFrame(product_records).to_csv(out / "pilot_product_metrics.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(out / "pilot_predictions.csv", index=False)
    summary = result.groupby(["support_origin", "mode"], sort=False)[
        ["chip_oa", "chip_ba", "ship_oa", "ship_ba"]].mean().round(3)
    summary.to_csv(out / "pilot_summary.csv")
    (out / "pilot_readout.json").write_text(json.dumps({
        "source_fit_rows": len(source), "query_chip_rows": len(query),
        "query_ships": int(query_meta.mmsi_or_ship_id.nunique()),
        "support_budget": 16, "draws": 5, "source_pca_only": True,
        "target_query_labels_used_for_selection": False,
        "method_scope": "matched RGB/PCA compact readout; no C4 or learned class-contrast",
        "episode_dir": str(args.episode_dir) if args.episode_dir else None,
        "dataset": args.dataset,
        "product_note": "GRDH/SLC support composition not matched; report by product before attributing shift",
    }, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
