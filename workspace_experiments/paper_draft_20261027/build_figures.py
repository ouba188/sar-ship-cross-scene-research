"""Assemble measured SAR chips without altering the source pixels."""
from pathlib import Path
import csv
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})
mpl.rcParams.update({
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

SOURCE = ROOT.parent / "paper_evidence_20261027/expanded_controls_20261027/natural_pairs"
FIGURES = ROOT / "figures"
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial"]
NAMES = {0: "散货船", 2: "普通货船", 3: "成品油/化学品船",
         4: "集装箱船", 5: "原油轮", 6: "拖轮/拖带船"}
PORTS = {"Port Klang": "巴生港", "Richards Bay": "理查兹湾", "Singapore": "新加坡",
         "Los Angeles": "洛杉矶", "Santos": "桑托斯", "Jebel Ali": "杰贝阿里"}
MAIN = [23490, 24966, 6352]
SUPPLEMENT = [23490, 10057, 24966, 33849, 6352, 12577]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assemble(pairs, observations, name):
    # Identical native-pixel scale across every panel, including different crop sizes.
    width_mm = 183
    row_mm = 48
    height_mm = 13 + row_mm * len(pairs) + 8
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    positions = [35, 71, 111, 147]
    panel_mm = 33
    extent_px = 88
    fig.text(71 / width_mm, 1 - 4 / height_mm, "较早观测", ha="center", va="top", weight="bold")
    fig.text(147 / width_mm, 1 - 4 / height_mm, "较晚观测", ha="center", va="top", weight="bold")
    for row, pair in enumerate(pairs):
        top_mm = 13 + row * row_mm
        class_id = int(pair["class_id"])
        fig.text(2 / width_mm, 1 - (top_mm + 8) / height_mm,
                 f"({chr(97+row)}) {NAMES[class_id]}\nMMSI {pair['mmsi']}",
                 fontsize=6.6, ha="left", va="top", linespacing=1.55)
        for time_index, key in enumerate(("history_id", "query_id")):
            oid = int(pair[key]); obs = observations[oid]
            date = obs["vv_time"][:10]
            platform = obs["product"].split("_")[0]
            x_center = (71 if time_index == 0 else 147) / width_mm
            fig.text(x_center, 1 - top_mm / height_mm,
                     f"{PORTS[obs['port']]} | {date} | {platform}",
                     ha="center", va="top", fontsize=6.5)
            for p_index, pol in enumerate(("vv", "vh")):
                x_mm = positions[time_index * 2 + p_index]
                y_mm = top_mm + 5
                ax = fig.add_axes([x_mm / width_mm, 1 - (y_mm + panel_mm) / height_mm,
                                   panel_mm / width_mm, panel_mm / height_mm])
                with Image.open(obs[pol + "_image_path"]) as im:
                    pixels = np.asarray(im)
                assert pixels.ndim == 2 and pixels.dtype == np.uint8, (oid, pol, pixels.shape)
                h, w = pixels.shape
                assert (w, h) == (int(float(obs[pol + "_native_width"])), int(float(obs[pol + "_native_height"])))
                assert max(w, h) <= extent_px
                assert float(obs[pol + "_pixel_x_m"]) == float(obs[pol + "_pixel_y_m"]) == 10.0
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                          extent=(-w / 2, w / 2, h / 2, -h / 2))
                ax.set_xlim(-extent_px / 2, extent_px / 2)
                ax.set_ylim(extent_px / 2, -extent_px / 2)
                ax.set_facecolor("#e6e6e6")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.text((x_mm + panel_mm / 2) / width_mm,
                         1 - (y_mm + panel_mm + 1.2) / height_mm,
                         f"{pol.upper()}  |  #{oid}  |  {w} x {h} px", fontsize=6,
                         ha="center", va="top")
        if row < len(pairs) - 1:
            yy = 1 - (top_mm + row_mm - 2) / height_mm
            fig.add_artist(Line2D([.01, .99], [yy, yy], transform=fig.transFigure,
                                 color="#c7cccf", linewidth=.4))
    baseline_y = 5 / height_mm
    bar_mm = panel_mm * 20 / extent_px
    fig.add_artist(Line2D([.20, .20 + bar_mm / width_mm], [baseline_y, baseline_y],
                         transform=fig.transFigure, color="#222222", linewidth=2))
    fig.text(.20 + bar_mm / width_mm + .015, baseline_y,
             "200 m（20个原生像素）  |  统一空间采样比例；保留原8位显示值",
             va="center", fontsize=6.4)
    # Exact canvas width is retained for manuscript layout; only text/lines are vector.
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(FIGURES / f"{name}.png", dpi=300)
        fig.savefig(FIGURES / f"{name}.pdf", dpi=300)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    cases = {int(r["history_id"]): r for r in read_csv(SOURCE / "selected_cases.csv")}
    observations = {int(r["obs_id"]): r for r in read_csv(SOURCE / "selected_observations.csv")}
    rows = []
    for number, history in enumerate(SUPPLEMENT):
        pair = cases[history]
        left, right = [observations[int(pair[k])] for k in ("history_id", "query_id")]
        assert left["mmsi"] == right["mmsi"] == pair["mmsi"]
        assert left["class_id"] == right["class_id"] == pair["class_id"]
        assert left["overpass"] != right["overpass"]
        for obs in (left, right):
            for pol in ("vv", "vh"):
                rows.append({"main_figure": history in MAIN, "pair_history": history,
                             "pair_query": pair["query_id"], "selection_rule": pair["selection_rule"],
                             "pol": pol, "obs_id": obs["obs_id"], "mmsi": obs["mmsi"],
                             "class_name": obs["class_name"], "product": obs["product"],
                             "port": obs["port"], "source_path": obs[pol + "_image_path"],
                             "time_utc": obs[pol + "_time"],
                             "native_width": obs[pol + "_native_width"],
                             "native_height": obs[pol + "_native_height"],
                             "pixel_x_m": obs[pol + "_pixel_x_m"],
                             "pixel_y_m": obs[pol + "_pixel_y_m"],
                             "scale_source": obs[pol + "_scale_source"],
                             "incidence_deg": obs[pol + "_incidence_deg"],
                             "diagnostic_prediction": obs["base_pred"],
                             "diagnostic_correct": obs["correct"]})
    with (ROOT / "figure_sources.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assemble([cases[k] for k in MAIN], observations, "fig1_same_vessel_observation_shift")
    assemble([cases[k] for k in SUPPLEMENT], observations, "figS1_extended_observations")
    receipt = {"main_pairs": MAIN, "supplement_pairs": SUPPLEMENT, "unique_original_chips": len(rows),
               "main_chip_count": 12, "supplement_chip_count": 24, "native_extent_px": 88,
               "display_limits": [0, 255], "pixel_spacing_m": 10,
               "pixel_spacing_is_not_resolution": True,
               "processing": "No crop, rotation, denoising, per-chip contrast normalization, or synthetic imagery. Nearest-neighbor display only; gray outside native extent.",
               "selection": "Illustrative subset of six previously selected case strata, not a random sample or domain-shift rate estimate.",
               "checks": ["same MMSI and class within each pair", "different overpasses", "image dimensions", "uint8 grayscale", "known sampling", "all native pixels inside view"]}
    (ROOT / "figure_build_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
