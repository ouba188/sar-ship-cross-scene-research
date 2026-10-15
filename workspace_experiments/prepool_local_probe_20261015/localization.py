"""Locate frozen stem peak evidence using detector-box and equal-area background readouts."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from run import CONTEXT, HERE, PIXELS, cached_replay, evaluate, fixed_rows, unit_normalize

sys.path.insert(0, "E:/Hermes/paper_tracking/sensitivity/oof_reliability_check")
from general_domain_ssl import backbone, normalize, rgb  # noqa: E402

REGION = HERE.parent / "region_cue_replication_20261013/ports/frame.csv"
MANIFEST = (HERE.parent / "visual_reliable_baseline/artifacts/"
            "eight_class_adaptive_20260916/manifest.csv")


def spatial_masks(rows, size=56):
    identifiers = pd.read_csv(MANIFEST, usecols=["instance_id"]).instance_id.to_numpy(str)
    cols = ["sample_id"] + [f"{item}_{pol}" for pol in ("vv", "vh")
                             for item in ("native_h", "native_w", "bbox_x0", "bbox_y0",
                                          "bbox_x1", "bbox_y1")]
    metadata = pd.read_csv(REGION, usecols=cols).set_index("sample_id").loc[identifiers[rows]]
    ship = np.zeros((len(rows), size, size), dtype=bool)
    background = np.zeros((3, len(rows), size, size), dtype=bool)
    far_outside = np.zeros_like(ship)
    near_ring = np.zeros_like(ship)
    fallback = 0
    yy = (np.arange(size) + .5) / size
    xx = (np.arange(size) + .5) / size
    for i, (_, row) in enumerate(metadata.iterrows()):
        for pol in ("vv", "vh"):
            x0 = row[f"bbox_x0_{pol}"] / row[f"native_w_{pol}"]
            x1 = row[f"bbox_x1_{pol}"] / row[f"native_w_{pol}"]
            y0 = row[f"bbox_y0_{pol}"] / row[f"native_h_{pol}"]
            y1 = row[f"bbox_y1_{pol}"] / row[f"native_h_{pol}"]
            ship[i] |= ((yy[:, None] >= y0) & (yy[:, None] < y1) &
                        (xx[None, :] >= x0) & (xx[None, :] < x1))
        count = int(ship[i].sum())
        assert 0 < count < size * size / 2, (rows[i], count)
        far_outside[i] = ~binary_dilation(ship[i], iterations=2)
        assert far_outside[i].any()
        near_ring[i] = binary_dilation(ship[i], iterations=8) & far_outside[i]
        assert near_ring[i].any()
        pool = np.flatnonzero(far_outside[i])
        if len(pool) < count:
            fallback += 1
            pool = np.flatnonzero(~ship[i])
        rng = np.random.default_rng(20261015 + int(rows[i]))
        for draw in range(3):
            background[draw, i].flat[rng.choice(pool, count, replace=False)] = True
    return ship, background, far_outside, near_ring, fallback


@torch.inference_mode()
def encode_local(rows, checkpoint, ship, background, far_outside, near_ring,
                 layer2_ship, layer2_background):
    torch.set_num_threads(2)
    model, _ = backbone("r18")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["backbone"], strict=True)
    model.cuda().eval()
    pixels = np.load(PIXELS, mmap_mode="r")
    captured = {"stem": [], "layer2": []}
    hooks = [model.relu.register_forward_hook(
                 lambda _, __, feature: captured["stem"].append(feature)),
             model.layer2.register_forward_hook(
                 lambda _, __, feature: captured["layer2"].append(feature))]
    names = ["gap", "stem_gap", "layer2_peak", "layer2_ship_peak",
             "full_peak", "ship_peak", "ship_mean", "outside_peak", "outside_mean",
             "far_outside_peak", "far_outside_mean",
             "near_ring_mean", "near_ring_std", "near_contrast_peak",
             "near_contrast_mean", "far_contrast_peak",
             "raw_ship", "raw_ring", "raw_contrast",
             "layer2_ship_mean", "layer2_outside_peak", "layer2_outside_mean"] + [f"background_peak_{i}" for i in range(3)] + [
             f"layer2_background_peak_{i}" for i in range(3)]
    parts = {name: [] for name in names}
    hits, layer2_hits = [], []
    for start in range(0, len(rows), 32):
        stop = min(start + 32, len(rows))
        pair = torch.from_numpy(np.array(pixels[rows[start:stop]], dtype=np.float32)).cuda() / 255.
        image = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear",
                                            align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = model(image).float()
        stem = captured["stem"].pop().float()
        layer2 = captured["layer2"].pop().float()
        assert stem.shape[2:] == ship.shape[1:]
        assert layer2.shape[2:] == layer2_ship.shape[1:]
        mask = torch.from_numpy(ship[start:stop]).cuda()[:, None]
        far_mask = torch.from_numpy(far_outside[start:stop]).cuda()[:, None]
        ring_mask = torch.from_numpy(near_ring[start:stop]).cuda()[:, None]
        mid_mask = torch.from_numpy(layer2_ship[start:stop]).cuda()[:, None]
        outside = ~mask
        mid_outside = ~mid_mask
        ship_peak = stem.masked_fill(~mask, float("-inf")).amax((2, 3))
        ship_mean = (stem * mask).sum((2, 3)) / mask.sum((2, 3))
        far_mean = (stem * far_mask).sum((2, 3)) / far_mask.sum((2, 3))
        ring_mean = (stem * ring_mask).sum((2, 3)) / ring_mask.sum((2, 3))
        ring_second = (stem.square() * ring_mask).sum((2, 3)) / ring_mask.sum((2, 3))
        ring_std = (ring_second - ring_mean.square()).clamp_min(0).sqrt()
        raw_ship_mask = F.interpolate(mask.float(), size=pair.shape[-2:], mode="nearest")
        raw_ring_mask = F.interpolate(ring_mask.float(), size=pair.shape[-2:], mode="nearest")
        raw_ship_mean = (pair * raw_ship_mask).sum((2, 3)) / raw_ship_mask.sum((2, 3))
        raw_ring_mean = (pair * raw_ring_mask).sum((2, 3)) / raw_ring_mask.sum((2, 3))
        raw_ship_std = ((pair.square() * raw_ship_mask).sum((2, 3)) /
                        raw_ship_mask.sum((2, 3)) - raw_ship_mean.square()).clamp_min(0).sqrt()
        raw_ring_std = ((pair.square() * raw_ring_mask).sum((2, 3)) /
                        raw_ring_mask.sum((2, 3)) - raw_ring_mean.square()).clamp_min(0).sqrt()
        raw_ship = torch.cat((raw_ship_mean, raw_ship_std), dim=1)
        raw_ring = torch.cat((raw_ring_mean, raw_ring_std), dim=1)
        values = dict(gap=pooled, stem_gap=stem.mean((2, 3)),
                      layer2_peak=layer2.amax((2, 3)),
                      layer2_ship_peak=layer2.masked_fill(
                          ~mid_mask, float("-inf")).amax((2, 3)),
                      layer2_ship_mean=(layer2 * mid_mask).sum((2, 3)) /
                                       mid_mask.sum((2, 3)).clamp_min(1),
                      layer2_outside_peak=layer2.masked_fill(
                          mid_mask, float("-inf")).amax((2, 3)),
                      layer2_outside_mean=(layer2 * mid_outside).sum((2, 3)) /
                                          mid_outside.sum((2, 3)).clamp_min(1),
                      full_peak=stem.amax((2, 3)),
                      ship_peak=ship_peak,
                      ship_mean=ship_mean,
                      outside_peak=stem.masked_fill(mask, float("-inf")).amax((2, 3)),
                      outside_mean=(stem * outside).sum((2, 3)) /
                                   outside.sum((2, 3)).clamp_min(1),
                      far_outside_peak=stem.masked_fill(~far_mask, float("-inf")).amax((2, 3)),
                      far_outside_mean=far_mean,
                      near_ring_mean=ring_mean,
                      near_ring_std=ring_std,
                      near_contrast_peak=ship_peak - ring_mean,
                      near_contrast_mean=ship_mean - ring_mean,
                      far_contrast_peak=ship_peak - far_mean,
                      raw_ship=raw_ship,
                      raw_ring=raw_ring,
                      raw_contrast=raw_ship - raw_ring)
        peak_location = stem.flatten(2).argmax(2)
        hits.append(torch.gather(mask.flatten(2).expand(-1, stem.shape[1], -1),
                                 2, peak_location[:, :, None]).float().mean(1).cpu().numpy())
        mid_location = layer2.flatten(2).argmax(2)
        layer2_hits.append(torch.gather(
            mid_mask.flatten(2).expand(-1, layer2.shape[1], -1), 2,
            mid_location[:, :, None]).float().mean(1).cpu().numpy())
        for draw in range(3):
            bg = torch.from_numpy(background[draw, start:stop]).cuda()[:, None]
            values[f"background_peak_{draw}"] = stem.masked_fill(
                ~bg, float("-inf")).amax((2, 3))
            mid_bg = torch.from_numpy(layer2_background[draw, start:stop]).cuda()[:, None]
            values[f"layer2_background_peak_{draw}"] = layer2.masked_fill(
                ~mid_bg, float("-inf")).amax((2, 3))
        for name, value in values.items():
            parts[name].append(value.cpu().numpy().astype(np.float32))
    for hook in hooks:
        hook.remove()
    return ({name: np.concatenate(chunks) for name, chunks in parts.items()},
            np.concatenate(hits), np.concatenate(layer2_hits))


def projected(raw, gallery):
    result = {}
    with threadpool_limits(limits=2):
        for name, values in raw.items():
            width = min(256 if name == "gap" else 64, values.shape[1])
            pca = PCA(n_components=width, svd_solver="randomized", random_state=20261015)
            pca.fit(values[gallery])
            result[name] = unit_normalize(pca.transform(values))
    late = result["gap"]
    for name in list(result):
        if name == "gap":
            continue
        for weight in (.25, .50):
            result[f"fusion_{name}_w{weight:.2f}"] = np.concatenate(
                (np.sqrt(1 - weight) * late, np.sqrt(weight) * result[name]), axis=1)
    return result


def main(fold):
    episodes = json.loads((CONTEXT / "target_episodes.json").read_text(encoding="utf-8"))
    episode = next(item for item in episodes if item["fold"] == fold)
    folder = Path(episode["folder"])
    source = folder.parent / episode["port"].replace(" ", "_")
    rows, groups = fixed_rows(episode)
    groups["query"].attrs["folder"] = str(folder)
    ship, background, far_outside, near_ring, fallback = spatial_masks(rows)
    mid_ship, mid_background, _, _, mid_fallback = spatial_masks(rows, 14)
    raw, hits, mid_hits = encode_local(rows, source / "source_only_r18_best.pt",
                                       ship, background, far_outside, near_ring,
                                       mid_ship, mid_background)
    error, relative_rmse = cached_replay(rows, raw["gap"], source)
    gallery = np.searchsorted(rows, groups["gallery"].row_index.to_numpy(int))
    features = projected(raw, gallery)
    results = []
    for name, values in features.items():
        source_map, target, b16 = evaluate(values, groups, rows, episode)
        for origin, (micro, macro, by_class) in target.items():
            results.append(dict(fold=fold, representation=name, origin=origin,
                                source_port_map=source_map, target_map=micro,
                                target_macro_map=macro,
                                target_b16_map=b16[origin][0][0],
                                target_b16_macro_map=b16[origin][0][1],
                                target_b16_draws=json.dumps(b16[origin][1]),
                                target_by_class=json.dumps(by_class, sort_keys=True),
                                pooled_cache_max_error=error,
                                pooled_cache_relative_rmse=relative_rmse))
    out = HERE / "localization"
    out.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(out / f"results_{fold}.csv", index=False)
    np.savez_compressed(out / f"features_{fold}.npz", rows=rows, **raw)
    np.savez_compressed(out / f"masks_{fold}.npz", rows=rows,
                        ship_area=ship.mean((1, 2)), peak_ship_fraction=hits,
                        near_ring_area=near_ring.mean((1, 2)),
                        layer2_ship_area=mid_ship.mean((1, 2)),
                        layer2_peak_ship_fraction=mid_hits)
    print(fold, "rows", len(rows), "box area", ship.mean(),
          "peak inside", hits.mean(), "background fallbacks", fallback,
          "layer2 box area", mid_ship.mean(), "layer2 peak inside", mid_hits.mean(),
          "layer2 bg fallbacks", mid_fallback, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fold")
    main(parser.parse_args().fold)
