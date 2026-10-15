"""Compact evidence tables and a figure separating recovery from net benefit."""

from settings import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    table = pd.read_csv(OUT / "metrics.csv")
    effects = pd.read_csv(OUT / "effects.csv")
    summary = pd.read_csv(OUT / "summary.csv")
    es = pd.read_csv(OUT / "effect_summary.csv")
    cls = pd.read_csv(OUT / "classes.csv")
    key = ["fold","task","dataset","draw","scope","arm","head"]
    cls["class_count"] = cls.groupby(key+["cell"])["class_id"].transform("nunique")
    pieces = []
    for contrast,a,b in (("source_coordination","DD","OD"),("matched_net","DD","OO"),("target_only","OD","OO")):
        joined = cls[cls.cell == a].merge(cls[cls.cell == b],on=key+["class_id"],suffixes=("_a","_b"),validate="one_to_one")
        out = joined[key+["class_id"]].copy()
        out["contrast"] = contrast
        out["delta_recall"] = joined.recall_a-joined.recall_b
        out["ba_contribution"] = out.delta_recall/joined.class_count_a
        out["risk_gain_contribution"] = (joined.brier_b-joined.brier_a)/joined.class_count_a
        pieces.append(out)
    ce = pd.concat(pieces,ignore_index=True)
    summed = ce.groupby(key+["contrast"])["ba_contribution"].sum().reset_index()
    check = summed.merge(effects,on=key+["contrast"],validate="one_to_one")
    np.testing.assert_allclose(check.ba_contribution,check.delta_ba,atol=1e-10,rtol=0.)
    ce.to_csv(OUT / "class_effects.csv",index=False)
    ce.groupby(["dataset","scope","arm","head","contrast","class_id"])[["delta_recall","ba_contribution","risk_gain_contribution"]].mean().reset_index().to_csv(OUT / "class_effect_summary.csv",index=False)
    primary = summary[(summary.scope == "original") & (summary["head"] == "expert") & (summary.arm == "R")]
    primary.pivot(index="dataset",columns="cell",values="ba").to_csv(OUT / "PRIMARY_FOUR_CELLS.csv")
    spread = effects[effects.scope == "original"].groupby(["dataset","arm","head","contrast"]).delta_ba.agg(
        mean="mean",std="std",minimum="min",maximum="max",contexts="size",positive=lambda v:int((v > 0).sum())).reset_index()
    spread.to_csv(OUT / "draw_spread.csv",index=False)
    datasets = ["ports3","oss1","oss2","oss2_grdh"]
    names = ["Three ports","OSS1","OSS2","OSS2 GRDH"]
    colors = {"OO":"#6f7885","OD":"#dc785e","DO":"#8d78a5","DD":"#279a87"}
    fig,axs = plt.subplots(2,2,figsize=(13.8,9.2))
    x = np.arange(4)
    for j,cell in enumerate(("OO","OD","DO","DD")):
        v = primary[primary.cell == cell].set_index("dataset").loc[datasets,"ba"].to_numpy()
        axs[0,0].bar(x+(j-1.5)*.18,v,width=.18,color=colors[cell],label=cell)
    axs[0,0].set(xticks=x,xticklabels=names,ylabel="Balanced accuracy (%)",ylim=(0,48),title="A. Four cells: first-branch expert")
    axs[0,0].legend(ncol=4,frameon=False,fontsize=9)
    selected = es[(es.scope == "original") & (es["head"] == "expert") & (es.arm == "R")]
    for j,(contrast,label,color) in enumerate((("target_only","Target-only change",colors["OD"]),
        ("source_coordination","Source-coordination recovery",colors["DD"]),("matched_net","Matched net vs OO","#416da0"))):
        v = selected[selected.contrast == contrast].set_index("dataset").loc[datasets,"delta_ba"].to_numpy()
        axs[0,1].bar(x+(j-1)*.24,v,width=.24,color=color,label=label)
    axs[0,1].axhline(0,color="#777777",lw=.7)
    axs[0,1].set(xticks=x,xticklabels=names,ylabel="Paired BA change (pp)",title="B. Recovery is not overall improvement",ylim=(-7,10))
    axs[0,1].legend(frameon=False,fontsize=8)
    for j,(head,label,color) in enumerate((("prior","Source prior","#ce8a38"),("expert","After support adaptation","#327aa2"),("c4_late","Compact fusion","#279a87"))):
        v = es[(es.scope == "original") & (es["head"] == head) & (es.arm == "R") &
            (es.contrast == "source_coordination")].set_index("dataset").loc[datasets,"delta_ba"].to_numpy()
        axs[1,0].bar(x+(j-1)*.24,v,width=.24,color=color,label=label)
    axs[1,0].axhline(0,color="#777777",lw=.7)
    axs[1,0].set(xticks=x,xticklabels=names,ylabel="DD minus OD: BA change (pp)",title="C. Support adaptation changes the outcome",ylim=(-2,11))
    axs[1,0].legend(frameon=False,fontsize=8)
    pairs = pd.read_csv(OUT / "product_pair_summary.csv")
    pairs = pairs[(pairs.cell == "OO") & (pairs.scope == "paired_all") & (pairs["head"] == "expert")].set_index("arm").loc[list(ARMS)]
    xx = np.arange(3)
    axs[1,1].bar(xx-.17,pairs.grdh_ba,width=.34,color="#279a87",label="GRDH")
    axs[1,1].bar(xx+.17,pairs.slc_ba,width=.34,color="#a57557",label="SLC")
    axs[1,1].set(xticks=xx,xticklabels=["First branch (R)","Added branch (T)","Dual prior + branch"],
        ylabel="Balanced accuracy (%)",title="D. Same ship and SAR time, different products",ylim=(0,45))
    axs[1,1].legend(frameon=False,fontsize=9)
    for ax in axs.flat:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.16)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.title.set_fontsize(11)
    fig.suptitle("Cross-dataset diagnosis: source matching and paired product sensitivity",fontsize=15,y=.985)
    fig.text(.5,.025,"Cell labels: source / target input, O = original, D = display stretch. Frozen encoders; fixed B64 supports.\n"
        "233 label-consistent product pairs, ~224 query pairs per draw; natural pairs, not a randomized product intervention. No independent-seed confidence intervals.",
        ha="center",va="bottom",fontsize=9,color="#444444")
    fig.tight_layout(rect=(0,.08,1,.96),h_pad=2.0,w_pad=2.0)
    fig.savefig(OUT / "root_cause_evidence.png",dpi=160)
    plt.close(fig)
    write(OUT / "PRESENTATION_COMPLETE.json",dict(class_effect_rows=len(ce),class_sums_verified=True,
        figure="root_cause_evidence.png",descriptive_draw_spread_not_independent_uncertainty=True))


if __name__ == "__main__":
    main()
