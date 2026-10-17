"""Extract raw geometry slots. Run as a separate process from the full runner.

Cache order: regions box/touch0-2/near2-8/far8-16; slots j*pi/4.
Areas are fractional feature-pixel counts, not normalized pooling weights.
Centers are bbox-union centroids in stem pixel-index coordinates (x, y).
No fitted normalization, labels, support centering, or geometry gating here.
"""

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "ship_radar_geometry_design_20261016"
SLOT_CENTERS = np.arange(4, dtype=np.float64) * np.pi / 4
KAPPA = 4.0


def angular_weights(shape, center, theta=0.0):
    """Return [4,H,W] soft assignments in x-right/y-down image coordinates."""
    yy, xx = np.indices(shape, dtype=np.float64)
    dx, dy = xx - center[0], yy - center[1]
    phi = np.arctan2(dy, dx)
    phi[(dx == 0) & (dy == 0)] = 0.0
    logits = KAPPA * np.cos(2 * (phi[None] - float(theta) % np.pi
                                - SLOT_CENTERS[:, None, None]))
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    return weights / weights.sum(axis=0, keepdims=True)


def regional_slots(box, theta=0.0):
    """Return mean-pooling masks [4,4,H,W], areas [4,4], edge, centroid."""
    box = np.asarray(box, dtype=bool)
    y, x = np.nonzero(box)
    if not len(x):
        raise ValueError("The localized bbox-union mask must be nonempty")
    center = np.array([x.mean(), y.mean()], dtype=np.float64)
    d2 = binary_dilation(box, iterations=2)
    d8 = binary_dilation(box, iterations=8)
    d16 = binary_dilation(box, iterations=16)
    regions = np.stack((box, d2 & ~box, d8 & ~d2, d16 & ~d8))
    mass = regions[:, None] * angular_weights(box.shape, center, theta)[None]
    areas = mass.sum(axis=(-2, -1))
    masks = np.divide(mass, areas[:, :, None, None], out=np.zeros_like(mass),
                      where=areas[:, :, None, None] > 0)
    edge = bool(d16[[0, -1]].any() or d16[:, [0, -1]].any())
    return masks.astype(np.float32), areas.astype(np.float32), edge, center.astype(np.float32)


def ship_angle(axis, valid):
    """Missing ship axes reuse image-coordinate slots; the runner owns gating."""
    return float(axis) % np.pi if valid and np.isfinite(axis) else 0.0


def load_legacy():
    # Only the standalone extraction process imports the old run/localization tree.
    spec = importlib.util.spec_from_file_location("legacy_spatial_extractor",
                                                OLD / "extract_spatial_tokens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract(port, rows, checkpoint, legacy, geometry):
    import torch
    from torch.nn import functional as F

    rows = np.asarray(rows, dtype=np.int64)
    output = HERE / "slots_cache" / (port.replace(" ", "_") + ".npz")
    if output.exists():
        with np.load(output) as cache:
            np.testing.assert_array_equal(cache["rows"], rows)
            expected = {"img_tokens": (len(rows), 4, 4, 64),
                        "ship_tokens": (len(rows), 4, 4, 64),
                        "img_areas": (len(rows), 4, 4),
                        "ship_areas": (len(rows), 4, 4), "edge": (len(rows),)}
            for key, shape in expected.items():
                if cache[key].shape != shape:
                    raise ValueError(f"Invalid cache shape: {key}: {cache[key].shape} != {shape}")
        print("CACHE", port, len(rows), "rows match", flush=True)
        return

    started = time.perf_counter()
    selected = geometry.loc[rows]
    angles = [ship_angle(a, v) for a, v in zip(selected.ship_axis_rad, selected.ship_valid)]
    box, *_ = legacy.spatial_masks(rows)
    model, _ = legacy.backbone("r18")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["backbone"],
                          strict=True)
    stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval()
    stem.requires_grad_(False)
    pixels = np.load(legacy.PIXELS, mmap_mode="r")
    tokens = {"img": [], "ship": []}
    areas = {"img": [], "ship": []}
    edges, centers = [], []
    print("START", port, len(rows), "checkpoint", str(checkpoint), flush=True)
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            stop = min(start + 32, len(rows))
            masks = {"img": [], "ship": []}
            for i in range(start, stop):
                image_slots = regional_slots(box[i], 0.0)
                ship_slots = image_slots if angles[i] == 0.0 else regional_slots(box[i], angles[i])
                for name, values in (("img", image_slots), ("ship", ship_slots)):
                    masks[name].append(values[0])
                    areas[name].append(values[1])
                edges.append(image_slots[2])
                centers.append(image_slots[3])
            pair = torch.from_numpy(np.array(pixels[rows[start:stop]], dtype=np.float32)).cuda() / 255.
            image = legacy.normalize(legacy.rgb(F.interpolate(
                pair, (112, 112), mode="bilinear", align_corners=False, antialias=True)))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature = stem(image)
            if tuple(feature.shape[1:]) != (64, 56, 56):
                raise ValueError(f"Unexpected stem shape: {tuple(feature.shape)}")
            feature = feature.float().flatten(2).transpose(1, 2)
            for name in tokens:
                mask = torch.from_numpy(np.stack(masks[name])).cuda().reshape(-1, 16, 56 * 56)
                values = torch.bmm(mask, feature).reshape(-1, 4, 4, 64)
                tokens[name].append(values.cpu().numpy())
            print("BATCH", port, f"{stop}/{len(rows)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, rows=rows,
                        img_tokens=np.concatenate(tokens["img"]),
                        ship_tokens=np.concatenate(tokens["ship"]),
                        img_areas=np.asarray(areas["img"], dtype=np.float32),
                        ship_areas=np.asarray(areas["ship"], dtype=np.float32),
                        edge=np.asarray(edges, dtype=bool),
                        centers=np.asarray(centers, dtype=np.float32))
    print("EXTRACTED", port, len(rows), "seconds", round(time.perf_counter() - started, 2),
          "output", str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Port name or underscore slug; omit for all inventory ports")
    args = parser.parse_args()
    import pandas as pd
    import torch

    torch.set_num_threads(2)
    legacy = load_legacy()
    entries = [entry for entry in legacy.inventory()
               if args.port is None or args.port.replace(" ", "_") == entry[0].replace(" ", "_")]
    if not entries:
        parser.error(f"Port not in legacy inventory: {args.port}")
    geometry = pd.read_csv(OLD / "geometry_metadata.csv",
                           usecols=["global_row", "ship_axis_rad", "ship_valid"]).set_index("global_row")
    for port, rows, checkpoint in entries:
        extract(port, rows, checkpoint, legacy, geometry)


if __name__ == "__main__":
    main()
