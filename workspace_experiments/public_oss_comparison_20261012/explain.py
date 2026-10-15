"""Exact per-class contribution accounting from the frozen public comparison."""

from common import *
import pandas as pd


def main():
    classes = pd.read_csv(OUT / "classes.csv")
    keys = ["regime", "dataset", "scope", "kind", "draw", "head", "class_id"]
    baseline = classes[classes.arm == "R"][keys+["recall", "brier"]]
    paired = classes[classes.arm != "R"].merge(baseline, on=keys, suffixes=("", "_R"), validate="many_to_one")
    paired["delta_recall"] = paired.recall-paired.recall_R
    paired["risk_gain"] = paired.brier_R-paired.brier
    group = ["regime", "dataset", "scope", "kind", "draw", "head", "arm"]
    present = paired.groupby(group).class_id.transform("nunique")
    paired["ba_contribution"] = paired.delta_recall/present
    paired["macro_risk_contribution"] = paired.risk_gain/present
    paired.to_csv(OUT / "class_effects.csv", index=False)
    paired.groupby(["regime", "dataset", "scope", "kind", "head", "arm", "class_id"])[
        ["delta_recall", "risk_gain", "ba_contribution", "macro_risk_contribution", "support_n"]].mean().reset_index().to_csv(OUT / "class_effect_summary.csv", index=False)
    summed = paired.groupby(group)[["ba_contribution", "macro_risk_contribution"]].sum().reset_index()
    metrics = pd.read_csv(OUT / "metrics.csv")
    base = metrics[metrics.arm == "R"][group[:-1]+["ba", "brier_macro"]]
    m = metrics[metrics.arm != "R"].merge(base, on=group[:-1], suffixes=("", "_R"), validate="many_to_one")
    joined = m.merge(summed, on=group, validate="one_to_one")
    np.testing.assert_allclose(joined.ba-joined.ba_R, joined.ba_contribution, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(joined.brier_macro_R-joined.brier_macro, joined.macro_risk_contribution, atol=1e-10, rtol=1e-10)
    shared = [json.loads((OUT / "targets/shared_frozen" / d / "COMPLETE.json").read_text()) for d in ("oss1", "oss2", "oss2_grdh")]
    if len({r["shared_geometry_path"] for r in shared}) != 1 or len({r["mapper_path"] for r in shared}) != 1:
        raise ValueError("Shared-model outputs did not use the same fitted geometry and mappers")
    common = source_rows("shared_frozen", "oss1")
    for name in DATASETS:
        split = archive(original(name) / "source_split.npz")
        if np.intersect1d(common, split["excluded_source_rows"]).size:
            raise ValueError("Common source includes an excluded public-target identity row")
    write(OUT / "EXPLANATION_VERIFIED.json", dict(class_contributions_sum_exactly=True,
        common_source_n=len(common), common_source_disjoint_both_excluded_sets=True,
        identical_shared_geometry_and_mapper_files=True, no_additional_training_or_parameter_selection=True))
    print("CLASS ACCOUNTING VERIFIED", len(joined), "COMMON SOURCE", len(common), flush=True)


if __name__ == "__main__":
    main()
