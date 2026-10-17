"""Account for real B32 gains after the constant-context control."""

from pathlib import Path

import numpy as np
import pandas as pd

from report_geometry import HERE, KEY, METRICS, summarize_details


def main():
    original = pd.concat([pd.read_csv(f, dtype={"mmsi": str}) for f in (HERE / "evaluation").glob("*/*_predictions.csv")], ignore_index=True)
    extra = pd.read_csv(HERE / "dependency_diagnostics/predictions.csv", dtype={"mmsi": str})
    const = extra.loc[extra.arm.eq("constant")]
    paired = original.loc[original.arm.eq("B")].merge(const[KEY + ["pred", "correct", "brier"]], on=KEY,
        suffixes=("", "_constant"), validate="many_to_one")
    paired["gain"] = paired.correct.astype(int) - paired.correct_constant.astype(int)
    unit = ["seed", "budget", "origin", "port", "fold", "draw"]
    class_count = paired.groupby(unit).truth.transform("nunique")
    class_size = paired.groupby(unit + ["truth"]).truth.transform("size")
    tasks = paired[unit].drop_duplicates().groupby(unit[:4]).size().rename("task_count").reset_index()
    paired = paired.merge(tasks, on=unit[:4], validate="many_to_one", sort=False)
    ports = paired.groupby(unit[:3]).port.transform("nunique")
    paired["ba_contribution"] = 100 * paired.gain / (ports * paired.task_count * class_count.to_numpy() * class_size.to_numpy())
    paired["rescue"] = paired.gain.eq(1)
    paired["harm"] = paired.gain.eq(-1)
    paired.groupby(["seed", "budget", "origin", "port", "truth"])[["ba_contribution", "rescue", "harm"]].sum().to_csv(HERE / "B_vs_constant_class_contributions.csv")
    paired.loc[paired.gain.ne(0)].to_csv(HERE / "B_vs_constant_changed_cases.csv", index=False)
    pp = pd.read_csv(HERE / "target_per_port.csv")
    cp = pd.read_csv(HERE / "dependency_diagnostics/per_port.csv")
    ref = cp.loc[cp.arm.eq("constant")].drop(columns="seed")
    contrasts = []
    for arm in "ABC":
        table = pp.loc[pp.arm.eq(arm)].merge(ref, on=["port", "budget", "origin"], suffixes=("", "_constant"), validate="many_to_one")
        for col in METRICS:
            table["delta_" + col] = table[col] - table[col + "_constant"]
        contrasts.append(table[["port", "budget", "origin", "arm", "seed"] + ["delta_" + c for c in METRICS]])
    contrasts = pd.concat(contrasts, ignore_index=True)
    contrasts.to_csv(HERE / "constant_control_contrasts.csv", index=False)
    expected = contrasts.loc[contrasts.arm.eq("B")].groupby(["seed", "budget", "origin"]).delta_ba.mean()
    actual = paired.groupby(["seed", "budget", "origin"]).ba_contribution.sum()
    np.testing.assert_allclose(actual.sort_index(), expected.sort_index(), atol=1e-10)
    common_key = ["port", "fold", "origin", "draw", "sample_id"]
    present = original.loc[original.arm.eq("body")].groupby(common_key).budget.nunique()
    keep = present.loc[present.eq(3)].reset_index()[common_key]
    all_rows = pd.concat([original, const], ignore_index=True).merge(keep, on=common_key, validate="many_to_one")
    common = summarize_details(all_rows)
    common.to_csv(HERE / "common_budget_metrics.csv", index=False)
    means = common.groupby(["port", "budget", "origin", "arm", "seed"])[METRICS].mean().groupby(["budget", "origin", "arm", "seed"]).mean()
    means.to_csv(HERE / "common_budget_seed_means.csv")
    changed = original.loc[original.arm.eq("C")].merge(extra.loc[extra.arm.eq("C_axes45"), KEY + ["seed", "pred"]], on=KEY + ["seed"], suffixes=("", "_rotated"), validate="one_to_one")
    changed.loc[changed.pred.ne(changed.pred_rotated)].to_csv(HERE / "axis_changed_decisions.csv", index=False)
    print("CONSTANT CONTROLS", contrasts.loc[contrasts.origin.eq("cross")].groupby(["budget", "arm"])[["delta_oa", "delta_ba"]].mean().to_string(), flush=True)
    print("COMMON COHORT", means.reset_index().query("origin == 'cross'").groupby(["budget", "arm"])[["oa", "ba"]].mean().to_string(), flush=True)
    print("B32 CLASS", paired.loc[paired.budget.eq(32) & paired.origin.eq("cross")].groupby(["seed", "truth"]).ba_contribution.sum().groupby("truth").mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
