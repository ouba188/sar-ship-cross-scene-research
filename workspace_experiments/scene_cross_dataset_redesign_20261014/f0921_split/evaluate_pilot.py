"""Evaluate frozen F_0921 B16 episodes after source-only R18 SSL selection.

This matched RGB/PCA ridge pilot checks transfer, not the historical full model.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import backbone, normalize, rgb  # noqa: E402
from general_domain_readout import design_block, fit_head, metrics  # noqa: E402
from unified_head_core import coefficients  # noqa: E402


PIXELS = Path("E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/inputs128.npy")
METADATA = ROOT.parents[1] / "scene_port_recovery_20261014" / "f0921_water_regions.csv"


@torch.inference_mode()
def encode(model, pixels, rows, width):
    model.cuda().eval()
    out = np.empty((len(rows), width), dtype=np.float32)
    for lo in range(0, len(rows), 64):
        take = rows[lo:lo + 64]
        pair = torch.from_numpy(np.array(pixels[take], dtype=np.float32)).cuda() / 255.
        pair = torch.nn.functional.interpolate(pair, (112, 112), mode="bilinear",
                                                align_corners=False, antialias=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out[lo:lo + len(take)] = model(normalize(rgb(pair))).float().cpu().numpy()
    return out


def support_score(features, head, query, support, labels):
    q = design_block(features, query)
    prior = q @ head
    if support is None:
        return prior
    s = design_block(features, support)
    alpha = coefficients(s @ s.T, s @ head, labels[support], 8.)
    return prior + (q @ s.T) @ alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=("Gulf_of_Mexico", "California_Baja_Pacific"))
    parser.add_argument("--port", choices=("Rotterdam", "Singapore"))
    parser.add_argument("--repeated", action="store_true")
    args = parser.parse_args()
    if (args.region is None) == (args.port is None):
        parser.error("Choose exactly one of --region or --port")
    task = args.region or args.port
    folder = ROOT / task if args.region else ROOT.parent / "ports_split" / task
    pixel_path = PIXELS if args.region else Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
    if not (folder / "training_complete.json").exists():
        raise RuntimeError("Source-selected R18 training must finish first")
    with np.load(folder / "split.npz") as z:
        source = np.r_[z["ssl_train"], z["source_validation"]]
    with np.load(folder / "pilot_episode.npz") as z:
        episode = {k: z[k] for k in z.files}
    if args.repeated:
        with np.load(folder / "support_draws.npz") as z:
            np.testing.assert_array_equal(z["query"], episode["query"])
            np.testing.assert_array_equal(z["support_classes"], episode["support_classes"])
            support_same, support_other = z["same"], z["other"]
    else:
        support_same = episode["support_same"][None, :]
        support_other = episode["support_other"][None, :]
    if len(episode["support_same"]) < 16:
        raise ValueError("B16 support is required")
    rows = np.unique(np.r_[source, episode["query"], support_same.ravel(),
                           support_other.ravel()])
    local = lambda original: np.searchsorted(rows, original)
    source_idx = local(source)
    query = local(episode["query"])
    support_classes = episode["support_classes"]
    if len(support_classes) >= 64:
        by_class = {c: list(np.flatnonzero(support_classes == c))
                    for c in np.unique(support_classes)}
        chosen = []
        while len(chosen) < 16:
            for c in sorted(by_class):
                if by_class[c] and len(chosen) < 16:
                    chosen.append(by_class[c].pop(0))
        budgets = ((16, np.array(chosen, dtype=int)),
                   (64, np.arange(64, dtype=int)))
    else:
        budgets = ((16, np.arange(16, dtype=int)),)
    same = local(support_same)
    other = local(support_other)
    if args.region:
        data = pd.read_csv(METADATA, usecols=["sample_id", "class_id"], dtype={"sample_id": str})
    else:
        manifest = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")
        data = pd.read_csv(manifest, usecols=["instance_id", "class_id"])
        data = data.rename(columns={"instance_id": "sample_id"})
    labels = data.class_id.to_numpy(int)[rows]
    pixels = np.load(pixel_path, mmap_mode="r")

    feature_file = folder / ("pilot_features_repeated.npz" if args.repeated else "pilot_features.npz")
    if feature_file.exists():
        with np.load(feature_file) as z:
            np.testing.assert_array_equal(z["rows"], rows)
            raw = {name: z[name] for name in ("G50", "G18", "D18")}
    else:
        raw = {}
        for name in ("G50", "G18", "D18"):
            arch = "r50" if name == "G50" else "r18"
            model, width = backbone(arch)
            if name == "D18":
                state = torch.load(folder / "source_only_r18_best.pt", map_location="cpu", weights_only=False)
                model.load_state_dict(state["backbone"])
            raw[name] = encode(model, pixels, rows, width)
            print(f"encoded {name}: {raw[name].shape}", flush=True)
            del model
            torch.cuda.empty_cache()
        np.savez_compressed(feature_file, rows=rows, **raw)

    features = {}
    for name, array in raw.items():
        pca = PCA(n_components=128, svd_solver="randomized", random_state=20261014)
        features[name] = pca.fit(array[source_idx]).transform(array).astype(np.float32)
    heads = {
        "G50": fit_head(features["G50"], labels, source_idx),
        "G18": fit_head(features["G18"], labels, source_idx),
        "D18": fit_head(features["D18"], labels, source_idx),
        "G50_G18": fit_head((features["G50"], features["G18"]), labels, source_idx),
        "G50_D18": fit_head((features["G50"], features["D18"]), labels, source_idx),
    }
    branches = {
        "G50": features["G50"], "G18": features["G18"], "D18": features["D18"],
        "G50_G18": (features["G50"], features["G18"]),
        "G50_D18": (features["G50"], features["D18"]),
    }
    records = []
    predictions = []
    conditions = [("none", 0, -1, None)]
    for draw in range(len(same)):
        for budget, chosen in budgets:
            conditions.extend((("same_acquisition", budget, draw, same[draw, chosen]),
                               ("other_acquisition", budget, draw, other[draw, chosen])))
    for source_name, budget, draw, support in conditions:
        branch_scores = {
            name: support_score(branches[name], heads[name], query, support, labels)
            for name in branches
        }
        modes = dict(branch_scores)
        for suffix in ("G18", "D18"):
            modes[f"fused_G50_{suffix}"] = .5 * softmax(branch_scores["G50"], axis=1) + \
                .5 * softmax(branch_scores[f"G50_{suffix}"], axis=1)
        for mode, score in modes.items():
            pred = score.argmax(1)
            records.append({("region" if args.region else "port"): task,
                            "support_origin": source_name, "budget": budget, "draw": draw,
                            "model": mode, "query_n": len(query),
                            "query_classes": len(np.unique(labels[query])),
                            **metrics(labels[query], pred)})
            for original, p in zip(episode["query"], pred):
                predictions.append({"sample_id": data.sample_id.iloc[original],
                                    "support_origin": source_name, "budget": budget, "draw": draw,
                                    "model": mode,
                                    "truth": int(data.class_id.iloc[original]), "prediction": int(p)})
    suffix = "_repeated" if args.repeated else ""
    result = pd.DataFrame(records)
    result.to_csv(folder / f"pilot_metrics{suffix}.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / f"pilot_predictions{suffix}.csv", index=False)
    (folder / f"pilot_readout{suffix}.json").write_text(json.dumps({
        "source_fit_rows": len(source), "query_rows": len(query),
        "support_budgets": [int(budget) for budget, _ in budgets],
        "support_draws": len(same),
        "b16_selection": "class_round_robin" if len(budgets) > 1 else "all_16",
        "source_pca_only": True, "target_query_labels_used_for_selection": False,
        "method_scope": "matched RGB/PCA compact readout, not historical C4/full class-contrast model",
    }, indent=2), encoding="utf-8")
    if args.repeated:
        print(result.groupby(["support_origin", "budget", "model"])[["oa", "ba"]]
              .mean().round(3).to_string(), flush=True)
    else:
        print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
