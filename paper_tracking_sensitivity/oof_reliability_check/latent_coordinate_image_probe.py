"""Bounded, paired-channel occlusion of two fixed Antwerp source coordinates."""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

ROOT = Path("E:/Hermes/paper_tracking/sensitivity")
BASE = Path("E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline")
DATA = BASE / "artifacts/eight_class_adaptive_20260916"
PRIOR = ROOT / "protected_drift_activation_20261008/latent_interpretation"
GEOMETRY = ROOT / "protected_drift_activation_20261008/ports/Antwerp-Bruges/geometry.pkl"
OUT = ROOT / "latent_clue_cards_20261008/image_probe"
os.environ["MPLCONFIGDIR"] = str(OUT / "mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
import torchvision
from torchvision.models import ResNet50_Weights

sys.path.insert(0, str(BASE))
from extract_full_six_spatial_features import build_encoder, read_one

SEED = 20261008
FILLS = ("globalmean", "gaussian_sigma4")
DIRECTIONS = ("S1", "D1")
REGIONS = {
    "center": (48, 80, 48, 80),
    "top_left": (0, 32, 0, 32),
    "top_right": (0, 32, 96, 128),
    "bottom_left": (96, 128, 0, 32),
    "bottom_right": (96, 128, 96, 128),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False),
                    encoding="utf-8")


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) and np.std(b) else None


def replacement_images(pixels):
    original = pixels.astype(np.float32)
    return (np.broadcast_to(original.mean((1, 2), keepdims=True), original.shape),
            gaussian_filter(original, sigma=(0, 4, 4), mode="reflect", truncate=4.0))


def input_perturbation_l2(pixels, patches):
    original = (pixels.astype(np.float32) / 255.0 - .5) / .25
    values = np.empty((2, len(patches)), dtype=np.float64)
    for fi, replacement in enumerate(replacement_images(pixels)):
        change = (replacement / 255.0 - .5) / .25 - original
        for pi, (y0, y1, x0, x1) in enumerate(patches):
            values[fi, pi] = np.linalg.norm(change[:, y0:y1, x0:x1].astype(np.float64))
    return values


class CoordinateProbe:
    def __init__(self, center, scale, directions, batch_size):
        self.center, self.scale, self.directions = center, scale, directions
        self.batch_size = batch_size
        self.checkpoint = (Path(torch.hub.get_dir()) / "checkpoints" /
                           ResNet50_Weights.IMAGENET1K_V2.url.rsplit("/", 1)[-1])
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Local checkpoint required; downloads disabled: {self.checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("This bounded replay requires the existing CUDA environment")
        self.encoder = build_encoder(torch.device("cuda")).requires_grad_(False)
        self.forward_batches = 0

    def coordinates(self, features):
        return ((features.astype(np.float64) - self.center) / self.scale) @ self.directions

    def forward(self, pixels):
        with torch.inference_mode():
            x = torch.as_tensor(pixels, device="cuda", dtype=torch.float32)
            x = (x / 255.0 - .5) / .25
            with torch.autocast("cuda", dtype=torch.bfloat16):
                spatial = self.encoder(x).flatten(2).transpose(1, 2)
            # Preserve the historical spatial-cache rounding before pooling.
            features = spatial.float().cpu().numpy().astype(np.float16).astype(np.float32).mean(1)
        self.forward_batches += 1
        return features, self.coordinates(features)

    def image(self, pixels, patches):
        original = pixels.astype(np.float32)
        fills = replacement_images(pixels)
        tasks = [(fi, pi) for fi in range(2) for pi in range(len(patches))]
        baseline = np.empty((2, len(patches), 2), dtype=np.float64)
        occluded = np.empty_like(baseline)
        baselines, null_spreads = [], []
        first_feature = None
        for lo in range(0, len(tasks), self.batch_size - 1):
            chunk = tasks[lo:lo + self.batch_size - 1]
            batch = np.repeat(original[None], self.batch_size, axis=0)
            for slot, (fi, pi) in enumerate(chunk, 1):
                y0, y1, x0, x1 = patches[pi]
                batch[slot, :, y0:y1, x0:x1] = fills[fi][:, y0:y1, x0:x1]
            features, values = self.forward(batch)
            if first_feature is None:
                first_feature = features[0].copy()
            baselines.append(values[0])
            if len(chunk) + 1 < self.batch_size:
                null_spreads.append(np.max(np.abs(values[len(chunk) + 1:] - values[0]), axis=0))
            for slot, (fi, pi) in enumerate(chunk, 1):
                baseline[fi, pi], occluded[fi, pi] = values[0], values[slot]
        return dict(response=baseline - occluded, baseline=baseline, occluded=occluded,
                    original_feature=first_feature,
                    baseline_batch_range=np.ptp(np.asarray(baselines), axis=0),
                    identical_input_spread=np.max(null_spreads, axis=0) if null_spreads else np.zeros(2))


def load_pixels(row):
    return np.stack([read_one(Path(row["image_relpath_vv"])),
                     read_one(Path(row["image_relpath_vh"]))])


def replay_record(probe, result, cached, row, cohort):
    delta = result["original_feature"] - cached
    actual = probe.coordinates(result["original_feature"])
    reference = probe.coordinates(cached)
    record = dict(cohort=cohort, manifest_row=int(row.manifest_row),
                  instance_id=row.instance_id, port=row.port, class_id=int(row.class_id),
                  feature_max_abs_error=float(np.max(np.abs(delta))),
                  feature_relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(cached)))
    for j, name in enumerate(DIRECTIONS):
        record.update({f"{name}_cached": float(reference[j]),
                       f"{name}_replay": float(actual[j]),
                       f"{name}_replay_minus_cached": float(actual[j] - reference[j]),
                       f"{name}_baseline_batch_range": float(result["baseline_batch_range"][j]),
                       f"{name}_identical_input_spread": float(result["identical_input_spread"][j])})
    return record


def summarize_replay(table):
    answer = {}
    for cohort, group in table.groupby("cohort", sort=True):
        values = dict(rows=len(group), feature_max_abs_error=float(group.feature_max_abs_error.max()),
                      feature_relative_l2_mean=float(group.feature_relative_l2.mean()),
                      feature_relative_l2_max=float(group.feature_relative_l2.max()))
        for name in DIRECTIONS:
            error = group[f"{name}_replay_minus_cached"].to_numpy()
            values[name] = dict(coordinate_max_abs_error=float(np.max(np.abs(error))),
                                coordinate_mean_abs_error=float(np.mean(np.abs(error))),
                                baseline_batch_range_max=float(group[f"{name}_baseline_batch_range"].max()),
                                identical_input_spread_max=float(group[f"{name}_identical_input_spread"].max()))
        answer[cohort] = values
    return answer


def summarize_no_op(table, batch_size):
    cohorts = []
    for cohort, group in table.groupby("cohort", sort=True):
        variants = 128 if cohort == "examples" else 10
        padding = (-variants) % (batch_size - 1)
        cohorts.append(dict(cohort=cohort, images=len(group), duplicate_pairs_per_image=padding,
                            duplicate_pairs=len(group)*padding,
                            max_abs_coordinate_difference={name: float(group[f"{name}_identical_input_spread"].max())
                                                           for name in DIRECTIONS}))
    return dict(method="Unmodified padding inputs versus slot0 original within the same fixed-size forward batch",
                pair_count_basis="Executed padding slots from 128 or 10 variants per image and batch_size-1 variant slots",
                total_duplicate_pairs=sum(item["duplicate_pairs"] for item in cohorts),
                cohorts=cohorts,
                interpretation="Observed numeric floor for these duplicate-input coordinate comparisons only; historical cache mismatch is not an occlusion-effect floor",
                extra_images=0, extra_forward_batches=0)


def map_sensitivity(rows, maps):
    records = []
    for i, row in rows.iterrows():
        for d, direction in enumerate(DIRECTIONS):
            a, b = maps[i, :, d].reshape(2, -1)
            nonzero = (a != 0) & (b != 0)
            records.append(dict(manifest_row=int(row.manifest_row), direction=direction,
                                signed_pearson=pearson(a, b), magnitude_pearson=pearson(abs(a), abs(b)),
                                sign_agreement=float(np.mean(np.sign(a) == np.sign(b))),
                                nonzero_both_count=int(nonzero.sum()),
                                nonzero_sign_agreement=float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else None,
                                globalmean_mean_abs=float(np.abs(a).mean()),
                                gaussian_mean_abs=float(np.abs(b).mean()),
                                gaussian_to_globalmean_mean_abs=float(np.abs(b).mean() / np.abs(a).mean()) if np.any(a) else None))
    return pd.DataFrame(records)


def save_montages(rows, pixels, maps, replay):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.unicode_minus": False})
    limits = np.max(np.abs(maps), axis=(0, 1, 3, 4))
    with PdfPages(OUT / "example_montage.pdf") as pdf:
        for page, start in enumerate((0, 4), 1):
            fig, axes = plt.subplots(4, 6, figsize=(16, 11))
            fig.subplots_adjust(left=.16, right=.98, bottom=.14, top=.88, wspace=.12, hspace=.27)
            handles = {}
            for rr, index in enumerate(range(start, start + 4)):
                row = rows.iloc[index]
                for pol in range(2):
                    axes[rr, pol].imshow(pixels[index, pol], cmap="gray", vmin=0, vmax=255,
                                         interpolation="nearest")
                for d in range(2):
                    for fi in range(2):
                        handles[d] = axes[rr, 2 + 2*d + fi].imshow(
                            maps[index, fi, d], cmap="RdBu_r", vmin=-limits[d], vmax=limits[d],
                            interpolation="nearest", extent=(0, 128, 128, 0))
                label = (f"row {int(row.manifest_row)} | {row.port}\n"
                         f"{row.class_name.replace('_', ' ')}\n"
                         f"previous {row.selection} selection\n"
                         f"S1={replay[index]['S1_replay']:+.3f}; D1={replay[index]['D1_replay']:+.3f}")
                axes[rr, 0].set_ylabel(label, rotation=0, ha="right", va="center", labelpad=14, fontsize=9)
                for ax in axes[rr]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                if rr == 0:
                    for ax, title in zip(axes[rr], ("VV | raw128", "VH | raw128", "S1 | global mean",
                                                   "S1 | blur sigma=4", "D1 | global mean", "D1 | blur sigma=4")):
                        ax.set_title(title, fontsize=11, pad=12)
            for d, left in enumerate((.455, .735)):
                cax = fig.add_axes([left, .085, .205, .013])
                fig.colorbar(handles[d], cax=cax, orientation="horizontal").set_label(
                    f"{DIRECTIONS[d]}: original - occluded", fontsize=10)
            fig.suptitle(f"Fixed-coordinate joint VV/VH occlusion | examples {start+1}-{start+4}",
                         fontsize=17, y=.977)
            fig.text(.57, .932, "Each cell replaces one 16 x 16 patch in both channels; independent response, not attention.",
                     ha="center", fontsize=10)
            fig.text(.57, .022, "Red: replacement lowered the coordinate. Blue: replacement raised it.\n"
                     "Shared scale per direction across both fills and all 8 examples; no semantic or polarization attribution.",
                     ha="center", fontsize=10)
            fig.savefig(OUT / f"example_montage_{page}.png", dpi=220, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)


def save_location_figure(cells, aggregate, normalized=False):
    center_key = "center_abs_response_per_l2" if normalized else "center_abs"
    corner_key = "corner_abs_response_per_l2_mean" if normalized else "corner_abs_mean"
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for d, direction in enumerate(DIRECTIONS):
        for fi, fill in enumerate(FILLS):
            ax = axes[d, fi]
            selected = cells[(cells.direction == direction) & (cells.fill == fill)]
            for label, color, marker in ((0, "#007C91", "o"), (4, "#B04458", "^")):
                part = selected[selected.class_id == label]
                ax.scatter(part[corner_key], part[center_key], color=color, marker=marker,
                           s=35, alpha=.8, label=f"class {label}")
            summary = next(s for s in aggregate if s["direction"] == direction and s["fill"] == fill)
            top = float(max(selected[corner_key].max(), selected[center_key].max()) * 1.07)
            ax.plot([0, top], [0, top], color="#777777", linestyle="--", linewidth=1)
            ax.scatter(summary["mean_corner_abs"], summary["mean_center_abs"], marker="X",
                       s=130, color="black", label="equal-cell mean", zorder=5)
            ax.set(xlim=(0, top), ylim=(0, top),
                   xlabel="Mean |response| / input L2: four corners" if normalized else "Mean absolute response: four separate corner patches",
                   ylabel="|response| / input L2: central patch" if normalized else "Absolute response: central patch")
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{direction} | {fill.replace('_', ' ')}\n"
                         f"center / periphery = {summary['center_to_corner_ratio']:.2f}", fontsize=12)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(("Center/periphery sensitivity | response normalized by input L2\n"
                  "Equal-cell descriptive ratios; not causal attribution or ship selectivity") if normalized else
                 ("Center/periphery sensitivity | equal 32 x 32 interventions\n"
                  "One point per source class-port cell; no ship/background masks"), fontsize=15)
    name = "center_periphery_input_l2_normalized" if normalized else "center_periphery"
    fig.savefig(OUT / f"{name}.png", dpi=220, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def augment_saved_results():
    """Reuse every saved coordinate response; reconstruct only pixel interventions."""
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    replay = pd.read_csv(OUT / "replay_discrepancy.csv")
    summary["no_op_control"] = summarize_no_op(replay, summary["batch_size"])
    with np.load(OUT / "example_maps.npz") as saved:
        archive = {key: saved[key] for key in saved.files}
    patches = [(y, y+16, x, x+16) for y in range(0, 128, 16) for x in range(0, 128, 16)]
    map_l2 = np.stack([input_perturbation_l2(pixels, patches) for pixels in archive["raw128_vv_vh"]])
    map_l2 = map_l2.reshape(8, 2, 8, 8)
    archive["input_perturbation_l2"] = map_l2
    archive["input_l2_axis_order"] = np.array(["example", "fill", "grid_y", "grid_x"])
    denominator = map_l2[:, :, None]
    archive["response_per_input_l2"] = np.divide(
        archive["response"], denominator, out=np.full_like(archive["response"], np.nan), where=denominator > 0)
    np.savez_compressed(OUT / "example_maps.npz", **archive)
    map_table = pd.read_csv(OUT / "example_map_responses.csv")
    row_order = {int(row): i for i, row in enumerate(archive["manifest_rows"])}
    map_table["input_perturbation_l2"] = [
        map_l2[row_order[int(row.manifest_row)], FILLS.index(row.fill), row.grid_y, row.grid_x]
        for row in map_table.itertuples()]
    map_table["response_per_input_l2"] = map_table.response / map_table.input_perturbation_l2.where(map_table.input_perturbation_l2 > 0)
    map_table["abs_response_per_input_l2"] = map_table.response_per_input_l2.abs()
    map_table.to_csv(OUT / "example_map_responses.csv", index=False)

    selected = pd.read_csv(OUT / "balanced_selection.csv", keep_default_na=False)
    region_l2 = {int(row.manifest_row): input_perturbation_l2(load_pixels(row), list(REGIONS.values()))
                 for _, row in selected.iterrows()}
    table = pd.read_csv(OUT / "center_periphery_per_row.csv")
    for pi, region in enumerate(REGIONS):
        table[f"{region}_input_l2"] = [region_l2[int(row.manifest_row)][FILLS.index(row.fill), pi]
                                        for row in table.itertuples()]
        denominator = table[f"{region}_input_l2"]
        table[f"{region}_response_per_l2"] = table[f"{region}_response"] / denominator.where(denominator > 0)
        table[f"{region}_abs_response_per_l2"] = table[f"{region}_response_per_l2"].abs()
    corner_names = list(REGIONS)[1:]
    table["corner_input_l2_mean"] = table[[f"{name}_input_l2" for name in corner_names]].mean(axis=1)
    # Require all four valid corner ratios instead of silently changing the corner set.
    table["corner_abs_response_per_l2_mean"] = table[[f"{name}_abs_response_per_l2" for name in corner_names]].mean(axis=1, skipna=False)
    table["corner_response_per_l2_mean"] = table[[f"{name}_response_per_l2" for name in corner_names]].mean(axis=1, skipna=False)
    table["all_region_l2_nonzero"] = (table[[f"{name}_input_l2" for name in REGIONS]] > 0).all(axis=1)
    table.to_csv(OUT / "center_periphery_per_row.csv", index=False)
    group_keys = ["direction", "fill", "class_id", "port"]
    derived_columns = [column for column in table if "input_l2" in column or "response_per_l2" in column]
    cells = pd.read_csv(OUT / "center_periphery_per_cell.csv")
    cells = cells.drop(columns=[column for column in derived_columns if column in cells], errors="ignore")
    cells = cells.drop(columns=["valid_normalized_rows"], errors="ignore")
    valid = table[table.all_region_l2_nonzero]
    normalized_cells = valid.groupby(group_keys, as_index=False)[derived_columns].mean()
    counts = valid.groupby(group_keys, as_index=False).agg(valid_normalized_rows=("manifest_row", "size"))
    cells = cells.merge(normalized_cells, on=group_keys, how="left").merge(counts, on=group_keys, how="left")
    cells.to_csv(OUT / "center_periphery_per_cell.csv", index=False)
    aggregate = []
    for (direction, fill), group in cells.groupby(["direction", "fill"], sort=True):
        usable = group.dropna(subset=["center_abs_response_per_l2", "corner_abs_response_per_l2_mean"])
        central = float(usable.center_abs_response_per_l2.mean())
        corners = float(usable.corner_abs_response_per_l2_mean.mean())
        aggregate.append(dict(direction=direction, fill=fill, cells=len(usable),
            rows=int(usable.valid_normalized_rows.sum()), mean_center_abs=central, mean_corner_abs=corners,
            mean_center_minus_corner_abs=central-corners,
            center_to_corner_ratio=central/corners if corners else None,
            fraction_cells_center_greater=float((usable.center_abs_response_per_l2 > usable.corner_abs_response_per_l2_mean).mean()),
            mean_center_input_l2=float(usable.center_input_l2.mean()),
            mean_corner_input_l2=float(usable.corner_input_l2_mean.mean()),
            center_to_corner_input_l2_ratio=float(usable.center_input_l2.mean()/usable.corner_input_l2_mean.mean()),
            region_equal_cell={region: dict(mean_input_l2=float(usable[f"{region}_input_l2"].mean()),
                mean_signed_response_per_l2=float(usable[f"{region}_response_per_l2"].mean()),
                mean_abs_response_per_l2=float(usable[f"{region}_abs_response_per_l2"].mean())) for region in REGIONS}))
    summary["input_perturbation_normalization"] = dict(
        formula="L2 norm across joint VV/VH of normalized replacement minus normalized original within the selected patch",
        normalized_input="(float32 pixel / 255 - .5) / .25; 128x128 resized input",
        response_per_l2="Signed original-minus-occluded response / L2; absolute response / L2 also exported",
        aggregation="Compute each region ratio per row first; average the four corner absolute ratios; average rows within class-port cell, then cells equally",
        denominators="Only nonzero L2; normalized regional aggregates require all five region denominators > 0",
        excluded_row_direction_fill_combinations=int((~table.all_region_l2_nonzero).sum()),
        zero_l2_grid_patches=int((map_l2 == 0).sum()), extra_gpu_forwards=0,
        note="Equal patch area does not mean equal perturbation magnitude. Raw center response is not evidence of dedicated ship selectivity. Response/L2 is descriptive, not causal attribution.")
    summary["center_periphery_equal_cell_response_per_input_l2"] = aggregate
    caveat = "Center brightness/input-change magnitude can confound equal-area raw response comparisons; response/L2 does not establish causal attribution or ship selectivity"
    if caveat not in summary["limitations"]:
        summary["limitations"].append(caveat)
    write_json(OUT / "summary.json", summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    save_location_figure(cells, aggregate, normalized=True)
    print(json.dumps(dict(stage="cpu_augmentation_complete", no_op=summary["no_op_control"],
                          normalized_center_periphery=[{key: value for key, value in item.items() if key != "region_equal_cell"}
                                                       for item in aggregate])), flush=True)


def brightness_only(batch_size):
    started = time.monotonic()
    selected = pd.read_csv(OUT / "balanced_selection.csv", keep_default_na=False)
    if len(selected) != 74 or selected.groupby(["class_id", "port"]).ngroups != 37:
        raise ValueError("Brightness follow-up requires the unchanged 74-row, 37-cell selection")
    with np.load(OUT / "coordinate_definition.npz") as saved:
        center, scale, directions = saved["feature_center"], saved["feature_scale"], saved["directions"]
    with np.load(PRIOR / "latent_vectors_AntwerpBruges.npz") as archive:
        np.testing.assert_array_equal(directions[:, 0], archive["illustrated_pair_direction"])
        np.testing.assert_array_equal(directions[:, 1], archive["qd"][:, 0])
        stable = archive["source_stable_coordinates"] @ (archive["qs"].T @ archive["illustrated_pair_direction"])
        drift = archive["source_drift_coordinates"][:, 0]
        source_sd = np.array([stable.std(ddof=0), drift.std(ddof=0)])
        reference_rows = len(stable)
    if not np.all(np.isfinite(source_sd) & (source_sd > 0)):
        raise ValueError("Source coordinate SD must be finite and nonzero")
    protocol = dict(date="2026-10-08", status="fixed_before_followup_execution",
        motivation="Follow-up motivated by the observed brightness correlation; NOT preregistered before the entire study",
        selected_rows=str(OUT / "balanced_selection.csv"), rows=74, cells=37, offsets=[8, -8],
        operation="Add fixed offset to both resized128 float32 pixel channels, then clip to [0,255]; retain the original preprocessing/encoder",
        response="Altered coordinate minus original coordinate in the SAME forward batch; sign convention differs from occlusion original-minus-occluded",
        clipping_fraction="Fraction of joint VV/VH pixels whose pre-clip value is <0 or >255; fractions per channel also saved",
        batch_size=batch_size, batch="slot0 original, slot1 +8, slot2 -8, remaining slots unmodified duplicates",
        source_sd_reference=dict(rows=reference_rows, ddof=0, population="All fixed Antwerp outer-source rows; not only the class0/4 sample",
                                 S1="source_stable_coordinates @ (qs.T @ illustrated_pair_direction)",
                                 D1="source_drift_coordinates[:,0]", values=dict(zip(DIRECTIONS, source_sd.tolist()))),
        aggregation="Average signed/absolute changes and sign fractions within class-port cells, then give all 37 cells equal weight",
        tuning=False, model_updates=False, additional_sampling=False,
        interpretation="Synthetic brightness response of this frozen pipeline; NOT real SAR incidence causality and NOT guaranteed label-preserving")
    write_json(OUT / "brightness_protocol.json", protocol)
    torch.set_num_threads(2)
    probe = CoordinateProbe(center, scale, directions, batch_size)
    previous = pd.read_csv(OUT / "replay_discrepancy.csv")
    previous = previous[previous.cohort.eq("balanced")].set_index("manifest_row")
    records = []
    for i, row in selected.iterrows():
        pixels = load_pixels(row).astype(np.float32)
        batch = np.repeat(pixels[None], batch_size, axis=0)
        clipping = []
        for slot, offset in enumerate((8, -8), 1):
            shifted = pixels + offset
            clipping.append(((shifted < 0) | (shifted > 255)).mean(axis=(1, 2)))
            batch[slot] = np.clip(shifted, 0, 255)
        _, coordinates = probe.forward(batch)
        no_op = np.max(np.abs(coordinates[3:] - coordinates[0]), axis=0)
        for slot, offset in enumerate((8, -8), 1):
            for d, direction in enumerate(DIRECTIONS):
                change = float(coordinates[slot, d] - coordinates[0, d])
                records.append(dict(manifest_row=int(row.manifest_row), instance_id=row.instance_id,
                    port=row.port, class_id=int(row.class_id), direction=direction, offset=offset,
                    original_coordinate=float(coordinates[0, d]), altered_coordinate=float(coordinates[slot, d]),
                    coordinate_change=change, absolute_coordinate_change=abs(change),
                    change_source_sd=change/source_sd[d], absolute_change_source_sd=abs(change)/source_sd[d],
                    change_sign=int(np.sign(change)), clipped_fraction=float(clipping[slot-1].mean()),
                    clipped_fraction_vv=float(clipping[slot-1][0]), clipped_fraction_vh=float(clipping[slot-1][1]),
                    actual_mean_pixel_change=float((batch[slot]-pixels).mean()),
                    no_op_max_abs=float(no_op[d]),
                    original_minus_previous_replay=float(coordinates[0, d]-previous.loc[int(row.manifest_row), direction+"_replay"])))
        if (i+1) % 20 == 0 or i+1 == len(selected):
            print(json.dumps(dict(stage="brightness", completed=i+1, total=len(selected))), flush=True)
    table = pd.DataFrame(records)
    table["positive_change"] = table.coordinate_change > 0
    table["negative_change"] = table.coordinate_change < 0
    table["zero_change"] = table.coordinate_change == 0
    table.to_csv(OUT / "brightness_per_row.csv", index=False)
    cells = table.groupby(["direction", "offset", "class_id", "port"], as_index=False).agg(
        rows=("manifest_row", "size"), mean_change=("coordinate_change", "mean"),
        mean_absolute_change=("absolute_coordinate_change", "mean"),
        mean_change_source_sd=("change_source_sd", "mean"), mean_absolute_change_source_sd=("absolute_change_source_sd", "mean"),
        positive_fraction=("positive_change", "mean"), negative_fraction=("negative_change", "mean"),
        zero_fraction=("zero_change", "mean"), mean_clipped_fraction=("clipped_fraction", "mean"),
        mean_clipped_fraction_vv=("clipped_fraction_vv", "mean"), mean_clipped_fraction_vh=("clipped_fraction_vh", "mean"),
        mean_actual_pixel_change=("actual_mean_pixel_change", "mean"))
    cells.to_csv(OUT / "brightness_per_cell.csv", index=False)
    aggregate = []
    for (direction, offset), group in cells.groupby(["direction", "offset"], sort=True):
        item = dict(direction=direction, offset=int(offset), cells=len(group), rows=int(group.rows.sum()),
                    positive_mean_change_cells=int((group.mean_change > 0).sum()),
                    negative_mean_change_cells=int((group.mean_change < 0).sum()),
                    zero_mean_change_cells=int((group.mean_change == 0).sum()))
        item.update({column: float(group[column].mean()) for column in group
                     if column.startswith("mean_") or column.endswith("_fraction")})
        aggregate.append(item)
    result = dict(status="complete", protocol=protocol, equal_cell_results=aggregate,
        no_op=dict(duplicate_pairs=len(selected)*(batch_size-3),
                   max_abs_by_direction={name: float(table.loc[table.direction.eq(name), "no_op_max_abs"].max()) for name in DIRECTIONS}),
        max_original_replay_change={name: float(table.loc[table.direction.eq(name), "original_minus_previous_replay"].abs().max()) for name in DIRECTIONS},
        forward_batches=probe.forward_batches, elapsed_seconds=time.monotonic()-started,
        historical_cache_discrepancy="Recorded separately in summary.json/replay_discrepancy.csv; not used as the brightness-effect baseline or numeric floor")
    write_json(OUT / "brightness_summary.json", result)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["brightness_followup"] = result
    write_json(summary_path, summary)
    print(json.dumps(dict(stage="brightness_complete", results=aggregate, no_op=result["no_op"],
                          original_replay_change=result["max_original_replay_change"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(16, 24, 32), default=24)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--augment-existing", action="store_true", help="CPU only: reuse saved responses and add perturbation L2 summaries")
    modes.add_argument("--brightness-only", action="store_true", help="Fixed +/-8 paired-channel brightness follow-up on the existing 74 rows")
    args = parser.parse_args()
    if args.brightness_only:
        brightness_only(args.batch_size)
        return
    if args.augment_existing:
        augment_saved_results()
        return
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw128").mkdir(exist_ok=True)
    torch.set_num_threads(2)
    with warnings.catch_warnings(record=True) as caught:
        with GEOMETRY.open("rb") as handle:
            geometry = pickle.load(handle)
    with np.load(PRIOR / "latent_vectors_AntwerpBruges.npz") as vectors:
        source_rows = vectors["source_rows"]
        s1 = vectors["illustrated_pair_direction"]
        np.testing.assert_array_equal(vectors["illustrated_pair"], [0, 4])
        np.testing.assert_array_equal(vectors["feature_center"], geometry["feature_center"])
        np.testing.assert_array_equal(vectors["feature_scale"], geometry["feature_scale"])
    directions = np.column_stack((s1, geometry["qd"][:, 0]))
    center, scale = geometry["feature_center"], geometry["feature_scale"]
    columns = ["instance_id", "port", "class_name", "class_id", "mmsi", "image_relpath_vv", "image_relpath_vh"]
    frame = pd.read_csv(DATA / "manifest.csv", usecols=columns, keep_default_na=False, dtype={"mmsi": str})
    frame.insert(0, "manifest_row", np.arange(len(frame)))
    examples = pd.read_csv(PRIOR / "example_selection.csv", keep_default_na=False)
    if len(examples) != 8:
        raise ValueError("Expected the eight existing examples")
    np.testing.assert_array_equal(frame.iloc[examples.manifest_row].instance_id, examples.instance_id)
    if not examples.manifest_row.isin(source_rows).all():
        raise ValueError("An illustrated example is not in the fixed source split")
    source = frame.iloc[source_rows]
    source = source[source.class_id.isin((0, 4))]
    if source.port.eq("Antwerp-Bruges").any():
        raise ValueError("Outer target appears in source sample")
    rng = np.random.default_rng(SEED)
    sampled = []
    for _, cell in source.groupby(["class_id", "port"], sort=True):
        chosen = np.sort(rng.choice(cell.index.to_numpy(), size=min(2, len(cell)), replace=False))
        part = cell.loc[chosen].copy()
        part["source_cell_size"] = len(cell)
        part["sampling_seed"] = SEED
        sampled.append(part)
    balanced = pd.concat(sampled, ignore_index=True)
    examples.to_csv(OUT / "example_selection.csv", index=False, encoding="utf-8-sig")
    balanced.to_csv(OUT / "balanced_selection.csv", index=False)
    np.savez_compressed(OUT / "coordinate_definition.npz", directions=directions,
                        direction_names=np.array(DIRECTIONS), feature_center=center, feature_scale=scale)
    probe = CoordinateProbe(center, scale, directions, args.batch_size)
    cached = np.load(DATA / "pooled_features.npy", mmap_mode="r")
    print(json.dumps(dict(stage="start", examples=len(examples), balanced_rows=len(balanced),
                          cells=len(sampled), batch_size=args.batch_size)), flush=True)
    patches = [(y, y+16, x, x+16) for y in range(0, 128, 16) for x in range(0, 128, 16)]
    raw, maps, map_originals, map_occluded, replay, map_long = [], [], [], [], [], []
    for i, row in examples.iterrows():
        pixels = load_pixels(row)
        raw.append(pixels)
        for pol, name in enumerate(("vv", "vh")):
            Image.fromarray(pixels[pol]).save(OUT / "raw128" / f"row_{int(row.manifest_row):05d}_{name}.png")
        result = probe.image(pixels, patches)
        record = replay_record(probe, result, cached[int(row.manifest_row)], row, "examples")
        replay.append(record)
        for key, collection in (("response", maps), ("baseline", map_originals), ("occluded", map_occluded)):
            collection.append(result[key].transpose(0, 2, 1).reshape(2, 2, 8, 8))
        for fi, fill in enumerate(FILLS):
            for pi, patch in enumerate(patches):
                for d, direction in enumerate(DIRECTIONS):
                    map_long.append(dict(manifest_row=int(row.manifest_row), fill=fill, direction=direction,
                                         grid_y=pi//8, grid_x=pi%8, y0=patch[0], y1=patch[1], x0=patch[2], x1=patch[3],
                                         matched_original=result["baseline"][fi, pi, d],
                                         occluded=result["occluded"][fi, pi, d], response=result["response"][fi, pi, d]))
        print(json.dumps(dict(stage="example", completed=i+1, total=8, row=int(row.manifest_row))), flush=True)
    raw, maps = np.stack(raw), np.stack(maps)
    np.savez_compressed(OUT / "example_maps.npz", manifest_rows=examples.manifest_row.to_numpy(),
                        raw128_vv_vh=raw, response=maps, matched_original=np.stack(map_originals),
                        occluded=np.stack(map_occluded), fills=np.array(FILLS), directions=np.array(DIRECTIONS),
                        map_axis_order=np.array(["example", "fill", "direction", "grid_y", "grid_x"]))
    pd.DataFrame(map_long).to_csv(OUT / "example_map_responses.csv", index=False)
    sensitivity = map_sensitivity(examples, maps)
    sensitivity.to_csv(OUT / "fill_pattern_sensitivity.csv", index=False)
    save_montages(examples, raw, maps, replay)
    locations = []
    for i, row in balanced.iterrows():
        result = probe.image(load_pixels(row), list(REGIONS.values()))
        replay.append(replay_record(probe, result, cached[int(row.manifest_row)], row, "balanced"))
        for fi, fill in enumerate(FILLS):
            for d, direction in enumerate(DIRECTIONS):
                response = result["response"][fi, :, d]
                record = dict(manifest_row=int(row.manifest_row), instance_id=row.instance_id, port=row.port,
                              class_id=int(row.class_id), fill=fill, direction=direction,
                              center_abs=float(abs(response[0])), corner_abs_mean=float(np.abs(response[1:]).mean()))
                record["center_minus_corner_abs"] = record["center_abs"] - record["corner_abs_mean"]
                for p, region in enumerate(REGIONS):
                    record[f"{region}_response"] = float(response[p])
                    record[f"{region}_matched_original"] = float(result["baseline"][fi, p, d])
                    record[f"{region}_occluded"] = float(result["occluded"][fi, p, d])
                locations.append(record)
        if (i+1) % 12 == 0 or i+1 == len(balanced):
            print(json.dumps(dict(stage="balanced", completed=i+1, total=len(balanced))), flush=True)
    locations = pd.DataFrame(locations)
    locations.to_csv(OUT / "center_periphery_per_row.csv", index=False)
    cells = locations.groupby(["direction", "fill", "class_id", "port"], as_index=False).agg(
        rows=("manifest_row", "size"), center_abs=("center_abs", "mean"),
        corner_abs_mean=("corner_abs_mean", "mean"), center_minus_corner_abs=("center_minus_corner_abs", "mean"))
    cells.to_csv(OUT / "center_periphery_per_cell.csv", index=False)
    aggregate = []
    for (direction, fill), group in cells.groupby(["direction", "fill"], sort=True):
        central, corners = float(group.center_abs.mean()), float(group.corner_abs_mean.mean())
        aggregate.append(dict(direction=direction, fill=fill, cells=len(group), rows=int(group.rows.sum()),
                              mean_center_abs=central, mean_corner_abs=corners,
                              mean_center_minus_corner_abs=central-corners,
                              center_to_corner_ratio=central/corners if corners else None,
                              fraction_cells_center_greater=float((group.center_abs > group.corner_abs_mean).mean())))
    replay_table = pd.DataFrame(replay)
    replay_table.to_csv(OUT / "replay_discrepancy.csv", index=False)
    save_location_figure(cells, aggregate)
    pattern = []
    for direction, group in sensitivity.groupby("direction", sort=True):
        pattern.append(dict(direction=direction, examples=len(group),
                            mean_signed_pearson=float(group.signed_pearson.mean()),
                            mean_magnitude_pearson=float(group.magnitude_pearson.mean()),
                            mean_sign_agreement=float(group.sign_agreement.mean()),
                            mean_gaussian_to_globalmean_magnitude=float(group.gaussian_to_globalmean_mean_abs.mean())))
    summary = dict(status="complete", date="2026-10-08", outer_source_geometry=str(GEOMETRY),
        vector_archive=str(PRIOR / "latent_vectors_AntwerpBruges.npz"),
        example_selection=str(PRIOR / "example_selection.csv"), manifest=str(DATA / "manifest.csv"),
        coordinate_formula="((pooled_float32 - feature_center) / feature_scale) @ direction; no display standardization",
        directions={"S1": "illustrated_pair_direction exactly; candidate class0vs4", "D1": "geometry qd[:,0] exactly"},
        response_formula="original rerun coordinate minus occluded coordinate in the SAME forward batch; never cached minus occluded",
        image_preprocessing="PIL L, bilinear resize128 independently, VV then VH, (pixels/255-.5)/.25",
        encoder="Frozen torchvision ImageNet ResNet50 V2; identical adapted two-channel kernels; 2x2 spatial pooling",
        precision="CUDA BF16 autocast; float16 spatial storage rounding then float32 token mean; float64 coordinate projection",
        checkpoint=str(probe.checkpoint), seed=SEED, batch_size=args.batch_size,
        batch_protocol="slot0 original; up to batch_size-1 occlusions; final batch padded with original duplicates",
        fills={"globalmean": "Each image/channel global mean over resized128 pixels, float32, no uint8 rerounding",
               "gaussian_sigma4": "Entire resized128 image blurred first: scipy gaussian_filter sigma=(0,4,4), reflect, truncate4; copy selected patch only"},
        map_shape=list(maps.shape), map_axes=["example", "fill", "direction", "grid_y", "grid_x"],
        grid_patch_size=16, center_periphery_patches_y0_y1_x0_x1=REGIONS,
        sample=dict(rows=len(balanced), cells=len(sampled), classes=[0,4], max_per_cell=2,
                    sampling="numpy default_rng(seed), sorted class_id/port groups, without replacement",
                    aggregation="Mean absolute coordinate change per region per row; mean four individual corners; average rows within cell then cells equally"),
        replay=summarize_replay(replay_table), no_op_control=summarize_no_op(replay_table, args.batch_size),
        fill_pattern_sensitivity=pattern,
        center_periphery_equal_cell=aggregate,
        prior_four_image_replay=dict(rows=[0,21195,25656,38090], batch4_bf16_relative_l2=[.007342251,.011440727,.007661806,.007742808],
                                    batch4_max_abs=[.0732421875,.0625,.02734375,.0546875],
                                    batch64_bf16_relative_l2=[.005714110,.008887738,.004406986,.007278112],
                                    fp32_relative_l2_range=[.045284986,.059055179],
                                    cause="Batch size explains part; remaining historical runtime/kernel discrepancy unconfirmed"),
        limitations=["Center/periphery only: no ship/background masks or semantic naming conclusions",
                     "Shared first-layer VV/VH weights mix channels; no absolute-gradient polarization interpretation",
                     "Fill changes the intervention; sign and magnitude patterns need not agree",
                     "Eight examples are previous illustrative selections, not a representative sample",
                     "Bounded source sample is descriptive; no target evaluation or model updates",
                     "Current replay is not bit-exact to historical cached features; all intervention effects use same-batch rerun baselines"],
        pickle_warnings=list(dict.fromkeys(str(item.message) for item in caught)),
        environment=dict(python=sys.executable, torch=torch.__version__, torchvision=torchvision.__version__,
                         numpy=np.__version__, matplotlib=matplotlib.__version__, gpu=torch.cuda.get_device_name(0)),
        forward_batches=probe.forward_batches, elapsed_seconds=time.monotonic()-started)
    write_json(OUT / "summary.json", summary)
    augment_saved_results()
    print(json.dumps(dict(stage="complete", output=str(OUT), elapsed_seconds=summary["elapsed_seconds"],
                          center_periphery=aggregate, fill_pattern=pattern, replay=summary["replay"])), flush=True)


if __name__ == "__main__":
    main()
