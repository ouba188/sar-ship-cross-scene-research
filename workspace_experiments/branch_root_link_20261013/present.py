"""Readable evidence tables and a compact figure, without new selection."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def main():
    effect = pd.read_csv(OUT / "effects_summary.csv")
    interaction = pd.read_csv(OUT / "interactions_summary.csv")
    fresh = effect.query("series == 'fresh' and head == 'expert' and comparison == 'T-R'")
    matched = fresh.query("scope == 'original' and condition in ['OO','OD','DD']")
    matched.to_csv(OUT / "MATCHED_BRANCH.csv", index=False)
    geometry = fresh.query("dataset == 'oss2' and scope in ['original','query_SLC','query_GRDH','paired_SLC'] and condition in ['OO','range_geometry','axis_sham']")
    geometry.to_csv(OUT / "GEOMETRY_BRANCH.csv", index=False)
    replacement = effect.query("series == 'archive_replacement' and head == 'expert' and scope in ['original','query_SLC','paired_SLC']")
    replacement.to_csv(OUT / "GEOMETRY_REPLACEMENT.csv", index=False)
    overlap = interaction.query("series == 'fresh' and head == 'expert' and comparison == 'T-R' and scope in ['original','query_SLC'] and transition in ['DD-OO','DD-OD','range_geometry-OO']").copy()
    for term in ("now_base_correct", "now_both_correct", "now_only_base_correct", "retained", "now_both_wrong"):
        overlap[term+"_percent_of_old_rescue"] = 100*overlap["old_rescue_"+term+"_ba"]/overlap.old_rescue_ba
    overlap.to_csv(OUT / "SAME_SAMPLE_OVERLAP.csv", index=False)
    print("GEOMETRY BRANCH\n", geometry[["scope","condition","rescue_ba","harm_ba","delta_ba","delta_oa","risk_gain_macro"]].to_string(index=False))
    print("REPLACEMENT MLP\n", replacement.query("kind == 'mlp'")[["scope","condition","comparison","delta_ba","delta_oa","risk_gain_macro"]].to_string(index=False))
    print("OVERLAP\n", overlap[["dataset","scope","transition","before_gain","after_gain","baseline_change","candidate_change","now_base_correct_percent_of_old_rescue","now_only_base_correct_percent_of_old_rescue","retained_percent_of_old_rescue","now_both_wrong_percent_of_old_rescue"]].to_string(index=False))

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.5))
    colors = ["#5e6b78", "#d07d40", "#258a77"]
    datasets = ["ports3", "oss1", "oss2", "oss2_grdh"]
    x = np.arange(4)
    for j, (condition, label) in enumerate((("OO","Original"),("OD","Target display only"),("DD","Matched source + target"))):
        values = matched[matched.condition == condition].set_index("dataset").loc[datasets,"delta_ba"]
        axs[0,0].bar(x+(j-1)*.25, values, width=.25, label=label, color=colors[j])
    axs[0,0].set(xticks=x, xticklabels=["Three ports","OSS1","OSS2","OSS2 GRDH"], ylabel="T - R, BA points", ylim=(0,4.3), title="A. Source/display matching: advantage remains")
    axs[0,0].legend(frameon=False, fontsize=8)
    scopes = ["original", "query_SLC", "query_GRDH", "paired_SLC"]
    for j, (condition, label) in enumerate((("OO","Original"),("range_geometry","Metadata axis"),("axis_sham","Wrong-axis sham"))):
        values = geometry[geometry.condition == condition].set_index("scope").loc[scopes,"delta_ba"]
        axs[0,1].bar(x+(j-1)*.25, values, width=.25, label=label, color=colors[j])
    axs[0,1].set(xticks=x, xticklabels=["Full OSS2","SLC query","GRDH query","Paired SLC"], ylabel="T - R, BA points", ylim=(0,3.5), title="B. Geometry: complementary benefit is conditional")
    axs[0,1].legend(frameon=False, fontsize=8)
    comparisons = ["T-R", "T-P", "E-R", "PE-P", "T-PE_scaled"]
    xx = np.arange(5)
    for j, (condition, label) in enumerate((("OO","Original"),("range_geometry","Metadata axis"),("axis_sham","Wrong-axis sham"))):
        values = replacement.query("kind == 'mlp' and scope == 'original'")
        values = values[values.condition == condition].set_index("comparison").loc[comparisons,"delta_ba"]
        axs[1,0].bar(xx+(j-1)*.25, values, width=.25, label=label, color=colors[j])
    axs[1,0].set(xticks=xx, xticklabels=comparisons, ylabel="Paired BA difference (points)", ylim=(0,3.25), title="C. Archived model: finite replacement controls")
    axs[1,0].legend(frameon=False, fontsize=8)
    selected = overlap.query("scope == 'original' and transition == 'DD-OO'").set_index("dataset").loc[datasets].copy()
    bottom = np.zeros(4)
    for column, label, color in (("now_base_correct_percent_of_old_rescue","R now also correct", "#258a77"),
        ("retained_percent_of_old_rescue","Still rescued by T", "#477daf"),
        ("now_both_wrong_percent_of_old_rescue","Both now wrong", "#bf6860")):
        values = selected[column].to_numpy()
        axs[1,1].bar(x, values, bottom=bottom, color=color, label=label, width=.62)
        bottom += values
    np.testing.assert_allclose(bottom,100.,atol=1e-10)
    axs[1,1].set(xticks=x, xticklabels=["Three ports","OSS1","OSS2","OSS2 GRDH"], ylabel="Share of original rescue mass (%)", ylim=(0,134), yticks=[0,20,40,60,80,100], title="D. The same original T-rescued samples after matching")
    axs[1,1].legend(frameon=False, fontsize=8)
    for ax in axs.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.axhline(0, color="#777777", lw=.7)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(11)
    fig.suptitle("Does the second branch still help after input corrections?", fontsize=15, y=.985)
    fig.text(.5,.025,"A/B/D: fresh source fits; C: separate archived source + MLP mapper. Same B64 supports within each comparison.\n"
        "Repeated folds/draws are not independent training seeds. Rescue shares use class-balanced mass, not raw sample counts.",
        ha="center", va="bottom", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0,.08,1,.96), h_pad=2.0, w_pad=2.0)
    fig.savefig(OUT / "branch_root_evidence.png", dpi=160)
    plt.close(fig)
    (OUT / "PRESENTATION_COMPLETE.json").write_text(json.dumps(dict(
        figure="branch_root_evidence.png", independent_seed_uncertainty_claimed=False),indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
