"""Extract pixel regions in an isolated process from legacy reader imports."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation

from localization import PIXELS, spatial_masks

OUT = Path(__file__).resolve().parent / "body_context_expansion/ring_region_spatial"


def moments(pair, mask):
    count = int(mask.sum())
    if not count:
        return np.full(4, np.nan), count
    values = pair[:, mask]
    return np.r_[values.mean(axis=1), values.std(axis=1)], count


def main():
    torch.set_num_threads(2)
    rows = np.load(OUT / "required_rows.npy")
    box, _, _, near, _ = spatial_masks(rows)
    pixels = np.load(PIXELS, mmap_mode="r")
    result = {name: [] for name in ("touch", "near", "far", "wide", "near_position", "near_bag")}
    coverage = []
    for i, row in enumerate(rows):
        d2 = binary_dilation(box[i], iterations=2)
        d8 = binary_dilation(box[i], iterations=8)
        d16 = binary_dilation(box[i], iterations=16)
        masks56 = dict(touch=d2 & ~box[i], near=near[i], far=d16 & ~d8, wide=d16 & ~box[i])
        assert np.array_equal(masks56["touch"] | masks56["near"] | masks56["far"], masks56["wide"])
        yy, xx = np.nonzero(box[i])
        cy, cx = (yy.min() + yy.max()) / 2., (xx.min() + xx.max()) / 2.
        gy, gx = np.indices(box[i].shape)
        quadrant = 2 * (gy > cy).astype(int) + (gx > cx).astype(int)
        for j in range(4):
            masks56[f"sector{j}"] = near[i] & (quadrant == j)
        names = list(masks56)
        masks = torch.nn.functional.interpolate(torch.from_numpy(np.stack(list(masks56.values())))[:, None].float(),
            (128, 128), mode="nearest")[:, 0].numpy().astype(bool)
        pair = pixels[row].astype(np.float64) / 255.
        record = dict(global_row=int(row), box_fraction=float(box[i].mean()),
            outer_reaches_chip_edge=bool(d16[[0, -1], :].any() or d16[:, [0, -1]].any()))
        values = {}
        for name, mask in zip(names, masks):
            values[name], record[name + "_pixels"] = moments(pair, mask)
        for name in ("touch", "near", "far", "wide"):
            result[name].append(values[name])
        tokens = np.stack([values[f"sector{j}"] for j in range(4)])
        result["near_position"].append(tokens.ravel())
        order = np.lexsort(tuple(tokens[:, j] for j in (3, 2, 1, 0)))
        result["near_bag"].append(tokens[order].ravel())
        coverage.append(record)
    pd.DataFrame(coverage).to_csv(OUT / "region_coverage.csv", index=False)
    raw = {name: np.asarray(values, dtype=np.float32) for name, values in result.items()}
    np.savez_compressed(OUT / "raw_regions.npz", rows=rows, **raw)
    print("EXTRACTED", len(rows), flush=True)


if __name__ == "__main__":
    main()
