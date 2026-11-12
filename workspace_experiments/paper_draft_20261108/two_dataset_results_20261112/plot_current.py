"""Render the current-model paper panels from saved aggregate statistics."""
import json
import numpy as np
import matplotlib.pyplot as plt

from build_figures import ROOT, SPECS, heatmap, save


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    results = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(
        1, 2, figsize=(8.6, 3.8), layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.8]},
    )
    for ax, (key, spec) in zip(axes, SPECS.items()):
        methods = results[key]["methods"]
        x = np.arange(len(spec["names"]))
        for offset, method, label, color in (
            (-0.17, "Plain", "Plain", "#91979e"),
            (0.17, "Full-Tuned", "Ours", "#c86b42"),
        ):
            values = methods[method]
            ax.bar(x + offset, values["recall"], .32, label=label, color=color,
                   yerr=values["recall_sd"], capsize=2,
                   error_kw={"elinewidth": .7})
        ax.set_xticks(x, spec["names"], rotation=35, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Recall (%)")
        ax.set_title(spec["title"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=7, loc="upper left", frameon=False)
    save(fig, "per_class_recall_current")

    for key, spec in SPECS.items():
        methods = results[key]["methods"]
        full = np.array(methods["Full-Tuned"]["cm"])
        delta = full - np.array(methods["Plain"]["cm"])
        fig, axes = plt.subplots(1, 2, figsize=(8.3, 4.2), layout="constrained")
        im = heatmap(axes[0], full, spec["names"], f"{spec['title']}: Ours")
        fig.colorbar(im, ax=axes[0], shrink=.70, label="Row-normalized (%)")
        im = heatmap(axes[1], delta, spec["names"], "Ours minus Plain", True, 22)
        fig.colorbar(im, ax=axes[1], shrink=.70, label="Difference (pp)")
        save(fig, f"{key}_full_delta_plain_current")


if __name__ == "__main__":
    main()
