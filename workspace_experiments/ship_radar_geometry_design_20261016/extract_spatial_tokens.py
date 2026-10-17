"""Cache equal-content regional stem tokens with each legal source checkpoint."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / "prepool_local_probe_20261015"
ROOT = OLD / "body_context_expansion"
sys.path.insert(0, str(OLD))
# This extractor intentionally runs separately from full_readout's legacy imports.
from localization import PIXELS, spatial_masks, backbone, normalize, rgb


def regional_cells(box):
    """Partition each observed region into a 4x4 image-aligned local grid."""
    d2 = binary_dilation(box, iterations=2)
    d8 = binary_dilation(box, iterations=8)
    d16 = binary_dilation(box, iterations=16)
    regions = (box, d2 & ~box, d8 & ~d2, d16 & ~d8)
    masks = np.zeros((4, 16, 56, 56), np.float32)
    positions = np.zeros((4, 16, 2), np.float32)
    areas = np.zeros((4, 16), np.float32)
    for r, mask in enumerate(regions):
        y, x = np.nonzero(mask)
        if not len(y):
            continue
        by = np.minimum(3, (4 * (y - y.min()) // (y.max() - y.min() + 1)))
        bx = np.minimum(3, (4 * (x - x.min()) // (x.max() - x.min() + 1)))
        cells = by * 4 + bx
        for cell in range(16):
            keep = cells == cell
            count = int(keep.sum())
            if count:
                masks[r, cell, y[keep], x[keep]] = 1. / count
                positions[r, cell] = [(x[keep].mean() + .5) / 56,
                                      (y[keep].mean() + .5) / 56]
                areas[r, cell] = count
    return masks, positions, areas, bool(d16[[0, -1]].any() or d16[:, [0, -1]].any())


def inventory():
    frozen = json.loads((ROOT / "frozen.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "ring_relaxed/query_coverage/frozen.json").read_text(encoding="utf-8"))
    pools = {p["port"]: p for p in frozen["pools"] if p["selected"]}
    result = []
    for p in source["pools"]:
        port = p["pool"]
        slug = port.replace(" ", "_")
        rows = []
        for folder, tasks in ((OLD / "body_context" / slug,
                               [t for split in p["splits"].values() for t in split if t["budget"] == 32]),
                              (ROOT / slug, json.loads((ROOT / slug / "tasks.json").read_text()))):
            meta = pd.read_csv(folder / "metadata.csv")
            needed = sorted({i for t in tasks for i in t["s"] + t["q"]})
            rows.extend(meta.iloc[needed].global_row.to_list())
        checkpoint = Path(pools[port]["source_folder"]).parent / slug / "source_only_r18_best.pt"
        result.append((port, np.unique(rows), checkpoint))
    return result


@torch.inference_mode()
def extract(port, rows, checkpoint):
    output = HERE / "spatial_cache" / (port.replace(" ", "_") + ".npz")
    if output.exists():
        with np.load(output) as z:
            np.testing.assert_array_equal(z["rows"], rows)
        print("CACHE", port, len(rows), flush=True)
        return
    started = time.perf_counter()
    box, *_ = spatial_masks(rows)
    positions, areas, edges = [], [], []
    # Keep only a batch of dense masks on the GPU; cached tokens are compact.
    model, _ = backbone("r18")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["backbone"])
    stem = torch.nn.Sequential(model.conv1, model.bn1, model.relu).cuda().eval()
    pixels = np.load(PIXELS, mmap_mode="r")
    features, peaks = [], []
    for start in range(0, len(rows), 32):
        stop = min(start + 32, len(rows))
        masks = []
        for b in box[start:stop]:
            m, p, a, e = regional_cells(b)
            masks.append(m)
            positions.append(p)
            areas.append(a)
            edges.append(e)
        pair = torch.from_numpy(np.array(pixels[rows[start:stop]], dtype=np.float32)).cuda() / 255.
        image = normalize(rgb(F.interpolate(pair, (112, 112), mode="bilinear", align_corners=False, antialias=True)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feature = stem(image)
        feature = feature.float().flatten(2)
        m = torch.from_numpy(np.stack(masks)).cuda().flatten(1, 2).flatten(2)
        values = torch.bmm(m, feature.transpose(1, 2)).reshape(-1, 4, 16, 64)
        features.append(values.cpu().numpy())
        ship_mask = torch.from_numpy(box[start:stop]).cuda().flatten(1)
        peaks.append(feature.masked_fill(~ship_mask[:, None], -torch.inf).amax(2).cpu().numpy())
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(output, rows=rows, tokens=np.concatenate(features), positions=np.array(positions),
                        areas=np.array(areas), edge=np.array(edges), ship_peak=np.concatenate(peaks))
    print("EXTRACTED", port, len(rows), "seconds", round(time.perf_counter() - started, 2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for port, rows, checkpoint in inventory():
        if args.port is None or args.port == port.replace(" ", "_"):
            extract(port, rows, checkpoint)


if __name__ == "__main__":
    main()
