"""Extract frozen-sample metadata and stretched-pixel box/ring conditions only."""

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from localization import MANIFEST, REGION, spatial_masks

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = HERE / "body_context_expansion"
OUT = BASE / "ring_conditions"
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
OBS = ROOT / "same_vessel_condition_study_20261010/observations.csv"


def main():
    torch.set_num_threads(2)
    required = pd.read_csv(BASE / "required_samples.csv", dtype={"mmsi": str})
    episodes = json.loads((BASE / "frozen.json").read_text(encoding="utf-8"))["episodes"]
    frozen_ids = {sid for e in episodes for sid in e["query"]}
    frozen_ids.update(sid for e in episodes for draw in e["draws"]
                      for origin in ("same", "cross") for sid in draw[origin])
    assert len(required) == required.sample_id.nunique() == 4289
    assert set(required.sample_id) == frozen_ids, "Frozen sample inventory differs"
    rows = required.row_index.to_numpy(int)
    manifest = pd.read_csv(MANIFEST, usecols=["instance_id"])
    np.testing.assert_array_equal(manifest.instance_id.to_numpy()[rows], required.sample_id)
    geometry_cols = [f"{key}_{pol}" for pol in ("vv", "vh") for key in
                     ("native_h", "native_w", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")]
    region = pd.read_csv(REGION, usecols=["sample_id", "class_name"] + geometry_cols)
    result = required.merge(region, on="sample_id", how="left", validate="one_to_one")
    assert result[["class_name"] + geometry_cols].notna().all().all()
    for pol in ("vv", "vh"):
        width = result[f"bbox_x1_{pol}"] - result[f"bbox_x0_{pol}"]
        height = result[f"bbox_y1_{pol}"] - result[f"bbox_y0_{pol}"]
        assert ((width > 0) & (height > 0)).all()
        result[f"box_occupancy_{pol}"] = width * height / (
            result[f"native_h_{pol}"] * result[f"native_w_{pol}"])
        result[f"box_aspect_{pol}"] = np.maximum(width, height) / np.minimum(width, height)

    ship, _, _, ring, fallback = spatial_masks(rows)
    masks = {name: F.interpolate(torch.from_numpy(mask[:, None]).float(),
                                size=(128, 128), mode="nearest")[:, 0].numpy().astype(bool)
             for name, mask in (("box", ship), ("ring", ring))}
    pixels = np.load(PIXELS, mmap_mode="r")
    assert pixels.dtype == np.uint8 and pixels.shape[1:] == (2, 128, 128)
    assert rows.min() >= 0 and rows.max() < len(pixels)
    stats = []
    for i, row_index in enumerate(rows):
        raw = pixels[row_index]
        pair = raw.astype(np.float64) / 255.0
        record = {}
        for name, mask_array in masks.items():
            mask = mask_array[i]
            assert mask.any()
            record[f"{name}_pixels_128"] = int(mask.sum())
            for channel, pol in enumerate(("vv", "vh")):
                values = pair[channel, mask]
                quantized = raw[channel, mask]
                record[f"raw_{name}_{pol}_mean"] = values.mean()
                record[f"raw_{name}_{pol}_std"] = values.std(ddof=0)
                record[f"raw_{name}_{pol}_saturation_fraction"] = (quantized == 255).mean()
                record[f"raw_{name}_{pol}_zero_fraction"] = (quantized == 0).mean()
        for pol in ("vv", "vh"):
            record[f"standardized_contrast_{pol}"] = (
                record[f"raw_box_{pol}_mean"] - record[f"raw_ring_{pol}_mean"]
            ) / max(record[f"raw_ring_{pol}_std"], 1 / 255)
        stats.append(record)
    result = pd.concat([result, pd.DataFrame(stats)], axis=1)
    assert np.isfinite(result[list(stats[0])].to_numpy(float)).all()

    observations = pd.read_csv(OBS, usecols=["instance_id", "status", "incidence_deg",
                                           "relative_status", "relative_signed_deg"])
    observations = observations.rename(columns={"instance_id": "sample_id",
                                                "status": "incidence_source_status",
                                                "relative_status": "relative_angle_source_status"})
    result = result.merge(observations, on="sample_id", how="left", validate="one_to_one")
    for value, status in (("incidence_deg", "incidence_source_status"),
                          ("relative_signed_deg", "relative_angle_source_status")):
        result[status] = result[status].fillna("sample_id_not_in_observations")
        result[value] = pd.to_numeric(result[value], errors="coerce")
        result.loc[~result[status].eq("ok") | ~np.isfinite(result[value]), value] = np.nan
    result["speed_knots"] = np.nan
    result["speed_source_status"] = "no_verified_sample_id_speed_join_in_bounded_search"
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "conditions.csv", index=False, float_format="%.10g")

    lines = ["# Ring condition sources", "",
             f"Coverage: {len(result)}/4289 unique required samples; {len(episodes)} frozen episodes.",
             "Frozen query/support union equals required_samples.sample_id; manifest row_index alignment checked.",
             "No predictions, correctness labels, performance outcomes, fitted thresholds or plots are used.", "",
             "## Sources and definitions", "",
             f"- Inventory: `{BASE / 'required_samples.csv'}` and `{BASE / 'frozen.json'}`.",
             f"- Class names and native bbox geometry: `{REGION}`; exact sample_id join.",
             f"- Cache row identity: `{MANIFEST}`, instance_id at row_index equals sample_id.",
             f"- Pixels: `{PIXELS}`, uint8 [N,2,128,128], channels VV then VH.",
             f"- Masks: `{HERE / 'localization.py'}` spatial_masks(rows), default 56x56.",
             "  Box is the union of normalized VV/VH native axis-aligned bboxes sampled at cell centers.",
             "  Ring is dilation(box,8) AND NOT dilation(box,2), using the existing scipy default structure.",
             "  Both masks are resized to 128x128 with torch nearest interpolation, exactly as the existing raw readout.",
             f"  Background sampling fallback count: {fallback}; sampled backgrounds are unused.",
             "- box_occupancy_vv/vh: native bbox width*height / native image width*height; fraction.",
             "- box_aspect_vv/vh: max(native bbox width,height)/min(width,height); dimensionless >=1.",
             "  These are detector-box image-space proxies, NOT true vessel length, physical aspect or segmentation occupancy.",
             "- native_h/w and bbox coordinates: native image pixels; box/ring_pixels_128: resized mask pixel counts.",
             "- raw_box/ring_vv/vh_mean/std: original stretched uint8 divided by 255, range [0,1]; population std (ddof=0).",
             "  No radiometric calibration: NOT sigma0, dB, physical sea state or a causal sea-condition measurement.",
             "- standardized_contrast_vv/vh: (box_mean-ring_mean)/max(ring_std,1/255); dimensionless.",
             "- saturation_fraction: fraction of mask pixels equal to uint8 255 (upper endpoint only).",
             "  zero_fraction separately records pixels equal to 0; endpoints alone do not establish sensor clipping.", "",
             "## Bounded optional-condition search", "",
             "Two candidate table headers inspected (limit three):",
             "- context_condition_study_20261015/geometry_by_sample.csv: no incidence, relative angle or speed fields.",
             "- same_vessel_condition_study_20261010/observations.csv: instance_id, incidence_deg, relative_signed_deg and validity statuses.",
             "  instance_id is joined exactly to required sample_id; only finite values with source status 'ok' are retained.",
             "  Angles are in degrees, relative_signed_deg preserves the source signed convention without reinterpretation.",
             "  Missing joins and unavailable/rejected source statuses are retained explicitly in the CSV.",
             "- Linked check_motion_teacher_feasibility.py uses motion_knowledge_20261011/attributes.npz",
             "  via fit/held_obs_ids and separate SOG validity masks, not a direct sample_id speed table.",
             "  A reliable target-sample speed join was not established within this bounded search; speed_knots is entirely missing.",
             "  No source-split speed is extrapolated to target samples. No network search performed.", "",
             "## Nonmissing coverage", "", "| Field | Available | Total |", "| --- | ---: | ---: |"]
    fields = ["class_name"] + [f"box_{item}_{pol}" for item in ("occupancy", "aspect")
                               for pol in ("vv", "vh")] + list(stats[0]) + [
                                   "incidence_deg", "relative_signed_deg", "speed_knots"]
    lines.extend(f"| {field} | {result[field].notna().sum()} | {len(result)} |" for field in fields)
    for status in ("incidence_source_status", "relative_angle_source_status"):
        lines += ["", f"{status}: " + "; ".join(
            f"{key}={count}" for key, count in result[status].value_counts().items())]
    (OUT / "CONDITION_SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(result)} unique samples to {OUT}", flush=True)
    print(result[["incidence_deg", "relative_signed_deg", "speed_knots"]].notna().sum().to_string())


if __name__ == "__main__":
    main()
