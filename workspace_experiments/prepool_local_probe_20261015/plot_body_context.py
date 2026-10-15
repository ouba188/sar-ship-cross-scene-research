"""Plot measured readout effects; no physical attribution is inferred."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent / "body_context"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180})


def main():
    table = pd.read_csv(ROOT / "target_summary.csv")
    delta = pd.read_csv(ROOT / "cross_ba_deltas.csv")
    names = ["ship", "ring", "additive", "static", "conditional"]
    labels = ["Body", "Ring", "Body + ring", "Static", "Conditional"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for offset, origin, color in ((-.18, "same", "#527f9d"), (.18, "cross", "#b85a50")):
        group = table.loc[table.origin.eq(origin)].set_index("arm")
        values = group.loc[names, "ba"].to_numpy() - group.loc["original", "ba"]
        axes[0].bar(np.arange(5)+offset, values, width=.34, color=color,
                    label="Same acquisition" if origin == "same" else "Cross acquisition")
    axes[0].set_xticks(np.arange(5), labels, rotation=20, ha="right")
    axes[0].set_ylabel("BA change vs matched original (pp)")
    axes[0].set_title("Complete system: equal-fold mean")
    axes[0].legend(frameon=False, fontsize=9)
    scene_labels = ["Jebel Ali", "Ningbo", "Qingdao", "Shanghai", "Singapore 1", "Singapore 2"]
    axes[1].barh(scene_labels, delta.ring, color=["#3a8773" if x >= 0 else "#bb6256" for x in delta.ring])
    axes[1].set_xlabel("Ring-only BA change (pp)")
    axes[1].set_title("Cross acquisition: useful, but not universal")
    axes[1].invert_yaxis()
    axes[0].axhline(0, color="0.4", linewidth=.8)
    axes[1].axvline(0, color="0.4", linewidth=.8)
    fig.savefig(ROOT / "complete_model_effects.png")
    fig.savefig(ROOT / "complete_model_effects.pdf")
    plt.close(fig)

    effect = pd.read_csv(ROOT / "tail_effects.csv")
    diag = pd.read_csv(ROOT / "training_diagnostics.csv")
    merged = diag.merge(effect, on=["pool", "mode", "seed"], validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for mode, color in (("static", "#527f9d"), ("conditional", "#b85a50")):
        d = merged.loc[merged["mode"].eq(mode)]
        axes[0].scatter(100*d.adjustment_abs_mean, d.mean_probability_l1, label=mode,
                        color=color, s=42, alpha=.8)
    axes[0].set_xlabel("Mean absolute channel adjustment (%)")
    axes[0].set_ylabel("Mean probability-vector L1 change")
    axes[0].set_title("600-step tails: parameters move, probabilities move little")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(frameon=False)
    ckpt = pd.read_csv(ROOT / "checkpoints.csv")
    for i, mode in enumerate(("static", "conditional")):
        d = ckpt.loc[ckpt["mode"].eq(mode)]
        zero = int(d.step.eq(0).sum())
        axes[1].bar(i, zero, color="#9fa9ad", label="Zero / additive" if i == 0 else None)
        axes[1].bar(i, len(d)-zero, bottom=zero, color="#3a8773", label="Learned" if i == 0 else None)
    axes[1].set_xticks([0, 1], ["Static", "Conditional"])
    axes[1].set_ylim(0, 20)
    axes[1].set_ylabel("Selected checkpoints (5 pools x 3 seeds)")
    axes[1].set_title("Selection uses source-held BA, not target outcomes")
    axes[1].legend(frameon=False, fontsize=9)
    fig.savefig(ROOT / "gate_mechanism_diagnostics.png")
    fig.savefig(ROOT / "gate_mechanism_diagnostics.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
