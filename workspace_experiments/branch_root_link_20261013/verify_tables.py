"""Catch cache-version, class-denominator and transition-accounting mistakes."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
RC = OUT.parent / "ports_root_cause_20261013"


def main():
    scores = pd.read_csv(OUT / "scores.csv")
    original = pd.read_csv(RC / "metrics.csv").rename(columns={"cell":"condition"})
    geo = pd.read_csv(RC / "geometry_metrics.csv").rename(columns={"variant":"condition"})
    geo = geo[geo.condition != "original"].assign(task="oss2", dataset="oss2")
    expected = pd.concat([original, geo], ignore_index=True)
    expected = expected[expected["head"].isin(["expert", "c4_late"])]
    keys = ["fold", "task", "draw", "scope", "head", "arm", "condition"]
    joined = expected.merge(scores[scores.series == "fresh"], on=keys, suffixes=("_old","_new"), validate="one_to_one")
    assert len(joined) == len(expected)
    max_error = 0.
    for col in ("ba", "oa", "brier_macro", "brier_micro"):
        error = float(np.max(np.abs(joined[col+"_old"]-joined[col+"_new"])))
        max_error = max(max_error,error)
        np.testing.assert_allclose(joined[col+"_old"],joined[col+"_new"],atol=1e-10,rtol=0.)
    effects = pd.read_csv(OUT / "effects.csv")
    classes = pd.read_csv(OUT / "classes.csv")
    keys = ["series","kind","fold","task","dataset","draw","scope","head","condition","comparison"]
    sums = classes.groupby(keys)[["rescue_ba","harm_ba","delta_ba"]].sum().reset_index()
    check = effects.merge(sums,on=keys,suffixes=("_all","_class"),validate="one_to_one")
    assert len(check) == len(effects)
    for col in ("rescue_ba","harm_ba","delta_ba"):
        np.testing.assert_allclose(check[col+"_all"],check[col+"_class"],atol=1e-10,rtol=0.)
    inter = pd.read_csv(OUT / "interactions.csv")
    total = inter.old_rescue_now_base_correct_ba+inter.old_rescue_retained_ba+inter.old_rescue_now_both_wrong_ba
    np.testing.assert_allclose(total,inter.old_rescue_ba,atol=1e-10,rtol=0.)
    np.testing.assert_allclose(inter.old_rescue_now_base_correct_ba,
        inter.old_rescue_now_both_correct_ba+inter.old_rescue_now_only_base_correct_ba,atol=1e-10,rtol=0.)
    np.testing.assert_allclose(inter.gain_change,inter.candidate_change-inter.baseline_change,atol=1e-10,rtol=0.)
    result = dict(existing_metric_rows=len(joined),existing_metric_max_error=max_error,
                  class_contribution_rows=len(check),transition_partition_rows=len(inter),passed=True)
    (OUT / "TABLE_VERIFICATION.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
