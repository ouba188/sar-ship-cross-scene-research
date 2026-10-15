"""Create the public full-readout main table and paired module effects."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARMS = ("raw_base_plain", "c4_base", "expert_plain", "c4_g18_plain", "c4_dual_plain",
        "c4_class_fixed", "c4_class_learned", "c4_adaptive", "c4_budget5")
COMPARISONS = (("c4_dual_plain", "raw_base_plain", "paired_C4"),
               ("c4_dual_plain", "c4_base", "second_branch"),
               ("c4_dual_plain", "c4_g18_plain", "SAR_vs_ImageNet_second_branch"),
               ("c4_class_fixed", "c4_dual_plain", "fixed_class_relation"),
               ("c4_class_learned", "c4_class_fixed", "learned_class_weights"),
               ("c4_adaptive", "c4_class_learned", "adaptive_vs_all_fusion"),
               ("c4_budget5", "c4_class_learned", "budget5_vs_all_fusion"))
METRICS = ("chip_oa", "chip_ba", "ship_oa", "ship_ba")


def main():
    raw = pd.concat([pd.read_csv(file) for file in HERE.glob("oss*/*_metrics.csv")], ignore_index=True)
    assert raw.groupby("dataset").fold.nunique().to_dict() == {"oss1": 4, "oss2": 4}
    assert len(raw) == 8 * 2 * 5 * len(ARMS)
    assert raw.groupby(["dataset", "fold", "origin", "draw"]).arm.nunique().eq(len(ARMS)).all()
    raw.to_csv(HERE / "public_full_draw_metrics.csv", index=False)
    fold = raw.groupby(["dataset", "fold", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    fold.to_csv(HERE / "public_full_fold_metrics.csv", index=False)
    main = fold.groupby(["dataset", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    main.to_csv(HERE / "PUBLIC_MAIN_TABLE.csv", index=False)
    indexed = raw.set_index(["dataset", "fold", "origin", "draw", "arm"])
    effects = []
    for candidate, baseline, label in COMPARISONS:
        a = indexed.xs(candidate, level="arm")
        b = indexed.xs(baseline, level="arm")
        z = (a[list(METRICS)] - b[list(METRICS)]).reset_index()
        z["comparison"] = label
        effects.append(z)
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(HERE / "PUBLIC_PAIRED_EFFECTS.csv", index=False)
    group = effects.groupby(["dataset", "origin", "comparison"], as_index=False)[list(METRICS)].mean()
    positive = (effects.groupby(["dataset", "origin", "comparison", "fold"], as_index=False)
                .chip_ba.mean().assign(positive=lambda x: x.chip_ba > 0)
                .groupby(["dataset", "origin", "comparison"]).positive.sum().reset_index(name="positive_folds"))
    group = group.merge(positive, on=["dataset", "origin", "comparison"], validate="one_to_one")
    group.to_csv(HERE / "PUBLIC_MODULE_EFFECTS.csv", index=False)
    protocol = dict(datasets=["oss1", "oss2"], folds_per_dataset=4, support_budget_labels=16,
                    support_draws=5, source_only_ssl=True, source_only_relation=True,
                    target_labels_for_model_selection=False,
                    C4="Four right-angle ImageNet R50 pooled views; source-fitted PCA128 and balanced ridge prior",
                    expert="Original-view ImageNet R50 plus source-only SAR SSL R18, source-fitted PCA128 per branch and balanced ridge prior",
                    relation="28 source-class pair coordinates; source-train acquisition-held risk learning, rho=.25, source-only calibration",
                    handoff="Source-validation logistic utility; adaptive p>=.5 and target-batch top 5% disagreements as capped alternative",
                    source_head="Same train plus validation source rows in final heads; inner relation/gate fits use source folds only",
                    output="C4 base/expert probability average versus source-gated base/fusion decision",
                    boundaries="New matched integration variant, not exact historical 24-port legacy frequency/PCA model; no target-side model selection; one SSL seed per fold; five support draws are not training replicates",
                    aggregation="Mean five support draws within fold, then equal four folds; per-fold BA uses query-present classes")
    (HERE / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(main.pivot_table(index=["dataset", "origin"], columns="arm", values="chip_ba").round(2).to_string())
    print(group[["dataset", "origin", "comparison", "chip_oa", "chip_ba", "positive_folds"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
