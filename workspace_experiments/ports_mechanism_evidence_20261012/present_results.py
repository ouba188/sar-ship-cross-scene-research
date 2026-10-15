"""Evidence tables and a fixed, non-selected display-condition figure."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
ORDER = ["ports3", "oss1", "oss2", "oss2_grdh"]
LABELS = ["3 ports", "OSS1", "OSS2", "OSS2 GRDH"]


def main():
    table = pd.read_csv(OUT / "flow_summary.csv")
    primary = table.query("condition == 'display' and scope == 'original' and kind == 'mlp' and head == 'expert'")
    flow = primary[primary.contrast == "T-R"].copy()
    main_cols = ["dataset", "stratum", "query_mass_pct", "old_gain_pp", "new_gain_pp", "gain_change_pp",
                 "new_rescue_pp", "new_harm_pp", "new_risk_gain"]
    flow[main_cols].to_csv(OUT / "primary_evidence.csv", index=False)
    lost = primary[(primary.stratum == "lost") & primary.contrast.isin(["T-R", "P-R", "T-P", "PE-P"])]
    lost[["dataset", "contrast", "new_gain_pp", "new_gain_conditional_pp", "candidate_kept_conditional_macro_pct",
          "candidate_correct_conditional_macro_pct", "new_risk_gain"]].to_csv(OUT / "lost_replacement.csv", index=False)
    totals = primary[primary.stratum == "all"]
    totals[["dataset", "contrast", "old_gain_pp", "new_gain_pp", "gain_change_pp", "old_risk_gain", "new_risk_gain"]].to_csv(
        OUT / "replacement_totals.csv", index=False)
    rows = []
    for dataset in ORDER:
        part = flow[flow.dataset == dataset].set_index("stratum")
        rescue = part.loc["all", "new_rescue_pp"]
        rows.append(dict(dataset=dataset,
            rescue_mass_from_originally_difficult_pct=100*part.loc["stable_wrong", "new_gain_pp"]/rescue,
            rescue_mass_from_new_errors_pct=100*part.loc["lost", "new_gain_pp"]/rescue,
            net_gain_pp=part.loc["all", "new_gain_pp"],
            gain_change_pp=part.loc["all", "gain_change_pp"]))
    pd.DataFrame(rows).to_csv(OUT / "rescue_origin.csv", index=False)
    absolute = []
    for dataset in ORDER:
        part = flow[flow.dataset == dataset].set_index("stratum")
        r0 = part.loc["stable_correct", "query_mass_pct"]+part.loc["lost", "query_mass_pct"]
        r1 = part.loc["stable_correct", "query_mass_pct"]+part.loc["gained", "query_mass_pct"]
        absolute.append(dict(dataset=dataset, r_original_ba=r0, r_display_ba=r1,
            t_original_ba=r0+part.loc["all", "old_gain_pp"], t_display_ba=r1+part.loc["all", "new_gain_pp"],
            input_r_delta_ba=r1-r0, input_t_delta_ba=r1-r0+part.loc["all", "gain_change_pp"],
            original_T_rescue_also_fixed_by_R_display_pct=100*part.loc["gained", "old_rescue_pp"]/part.loc["all", "old_rescue_pp"]))
    pd.DataFrame(absolute).to_csv(OUT / "absolute_and_overlap.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(4)
    styles = [("lost", "Rescue: newly wrong", "#188B79"),
              ("stable_wrong", "Rescue: wrong in both", "#277DA8"),
              ("gained", "Harm: newly correct", "#CC743D"),
              ("stable_correct", "Harm: correct in both", "#B44059")]
    pos, neg = np.zeros(4), np.zeros(4)
    for state, label, color in styles:
        values = flow[flow.stratum == state].set_index("dataset").loc[ORDER, "new_gain_pp"].to_numpy()
        bottom = np.where(values >= 0, pos, neg)
        axes[0].bar(x, values, bottom=bottom, color=color, width=.6, label=label)
        pos += np.maximum(values, 0)
        neg += np.minimum(values, 0)
    net = flow[flow.stratum == "all"].set_index("dataset").loc[ORDER, "new_gain_pp"].to_numpy()
    axes[0].scatter(x, net, marker="D", s=36, color="#202020", zorder=5, label="Net T-R gain")
    for xx, value in zip(x, net):
        axes[0].annotate(f"{value:+.2f}", (xx, value), xytext=(10, 0), textcoords="offset points", fontsize=9)
    axes[0].axhline(0, color="#555555", lw=.8)
    axes[0].set(xticks=x, xticklabels=LABELS, ylabel="Contribution to balanced accuracy (pp)",
                title="A. Repairs and harms after display stretch")
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8, frameon=False)
    for offset, contrast, label, color in [(-.17, "P-R", "Reconstructed branch (P)", "#929BA1"),
                                           (.17, "T-R", "Actual second branch (T)", "#188B79")]:
        values = lost[lost.contrast == contrast].set_index("dataset").loc[ORDER, "new_gain_conditional_pp"].to_numpy()
        axes[1].bar(x+offset, values, width=.32, color=color, label=label)
        for xx, value in zip(x+offset, values):
            axes[1].text(xx, value+.35, f"{value:.1f}", ha="center", fontsize=9)
    axes[1].set(xticks=x, xticklabels=LABELS, ylim=(0, 23),
                ylabel="Conditional rescue rate, class-weighted (%)",
                title="B. On R's newly wrong queries only")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.15), fontsize=8, frameon=False)
    fig.suptitle("Second-branch repairs, harms, and reconstruction control", fontsize=14, y=.98)
    fig.text(.5, .02, "Fixed source models, B64; descriptive task/draw means. Panel B is not a full-query net gain.",
             ha="center", fontsize=9, color="#454545")
    fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.26, wspace=.32)
    fig.savefig(OUT / "mechanism_evidence.png", dpi=180, facecolor="white")
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(absolute).to_string(index=False))


if __name__ == "__main__":
    main()
