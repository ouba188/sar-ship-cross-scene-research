"""Descriptive paired subgroup analysis; no fitted deployment selector."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent / "body_context_expansion"
OUT = ROOT / "ring_conditions"
KEYS = ["port", "fold", "budget", "origin", "draw", "sample_id"]
EFFECTS = ["vs_ship", "vs_half", "vs_shuffle", "vs_original"]


def effect_summary(g):
    oa = g.groupby(["port", "fold"])[EFFECTS].mean().groupby("port").mean().mean() * 100
    ba = g.groupby(["port", "fold", "truth"])[EFFECTS].mean().groupby(["port", "fold"]).mean().groupby("port").mean().mean() * 100
    result = {f"delta_oa_{c}": oa[c] for c in EFFECTS}
    result.update({f"delta_ba_{c}": ba[c] for c in EFFECTS})
    result.update(observations=len(g), samples=g.sample_id.nunique(), ships=g.mmsi.nunique(),
        ports=g.port.nunique(), acquisitions=g.acquisition.nunique(), tasks=g.fold.nunique(),
        rescues=int(g.rescue.sum()), harms=int(g.harm.sum()),
        new_vs_original=int(g.new_rescue.sum()), restore_original=int(g.restore_original.sum()),
        rescue_not_half=int(g.rescue_not_half.sum()), rescue_none_shuffle=int(g.rescue_none_shuffle.sum()))
    return result


def bins(value):
    valid = value.dropna()
    edges = valid.quantile([1/3, 2/3]).to_numpy()
    out = pd.Series("missing", index=value.index)
    out.loc[value.notna()] = np.where(value.loc[value.notna()] <= edges[0], "low",
        np.where(value.loc[value.notna()] <= edges[1], "middle", "high"))
    return out, edges.tolist()


def main():
    pred = pd.read_csv(OUT / "predictions.csv", dtype={"mmsi": str})
    conditions = pd.read_csv(OUT / "conditions.csv", dtype={"mmsi": str})
    info = pred.loc[pred.arm.eq("ship"), KEYS + ["mmsi", "truth"]]
    wide = pred.pivot(index=KEYS, columns="arm", values="correct").astype(float).reset_index()
    data = info.merge(wide, on=KEYS, validate="one_to_one")
    data = data.merge(conditions.drop(columns=["port", "mmsi", "class_id"]), on="sample_id", validate="many_to_one")
    support = pd.read_csv(OUT / "support_conditions.csv").drop(columns=["mmsi", "truth"])
    data = data.merge(support, on=KEYS, validate="one_to_one")
    shufcols = [f"ring_shuffle_{i}" for i in range(3)]
    data["shuffle_mean"] = data[shufcols].mean(axis=1)
    for name, ref in (("ship", "ship"), ("half", "body_half"), ("shuffle", "shuffle_mean"), ("original", "original")):
        data["vs_" + name] = data.additive-data[ref]
    data["rescue"] = data.additive.eq(1) & data.ship.eq(0)
    data["harm"] = data.additive.eq(0) & data.ship.eq(1)
    data["new_rescue"] = data.rescue & data.original.eq(0)
    data["restore_original"] = data.rescue & data.original.eq(1)
    data["rescue_not_half"] = data.rescue & data.body_half.eq(0)
    data["rescue_none_shuffle"] = data.rescue & data[shufcols].eq(0).all(axis=1)
    data["rescue_all_controls_wrong"] = data.new_rescue & data.body_half.eq(0) & data[shufcols].eq(0).all(axis=1)
    thresholds = {}
    query_meta = conditions.loc[conditions.sample_id.isin(data.sample_id)].set_index("sample_id")
    fields = ["standardized_contrast_vv", "standardized_contrast_vh", "raw_ring_vv_std", "raw_ring_vh_std",
              "box_occupancy_vv", "relative_signed_deg"]
    group_columns = []
    for field in fields:
        mapped, edge = bins(query_meta[field])
        name = field + "_bin"
        data[name] = data.sample_id.map(mapped)
        thresholds[field] = edge
        group_columns.append(name)
    data["incidence_bin"] = pd.cut(data.incidence_deg, [-np.inf, 30, 40, np.inf], labels=["under30", "30to40", "over40"]).astype(str)
    data.loc[data.incidence_deg.isna(), "incidence_bin"] = "missing"
    group_columns.append("incidence_bin")
    for field in ("body_top2_gap", "body_support_gap", "ring_support_match"):
        name = field + "_bin"
        data[name] = "missing"
        thresholds[field] = {}
        for (budget, origin), index in data.groupby(["budget", "origin"]).groups.items():
            assignment, edges = bins(data.loc[index, field])
            data.loc[index, name] = assignment
            thresholds[field][f"{budget}_{origin}"] = edges
        group_columns.append(name)
    group_columns.append("support_class_agreement")
    data.to_csv(OUT / "paired_query_conditions.csv", index=False)
    (OUT / "condition_thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    records = []
    for column in ["port", "class_name", *group_columns]:
        for (budget, origin, value), g in data.groupby(["budget", "origin", column]):
            records.append(dict(budget=budget, origin=origin, condition=column, level=str(value), **effect_summary(g)))
    subgroups = pd.DataFrame(records)
    subgroups.to_csv(OUT / "subgroup_effects.csv", index=False)
    per_draw = []
    for (port, budget, origin, draw), g in data.groupby(["port", "budget", "origin", "draw"]):
        per_draw.append(dict(port=port, budget=budget, origin=origin, draw=draw, **{k:v for k,v in effect_summary(g).items() if k.startswith("delta_")}))
    pd.DataFrame(per_draw).to_csv(OUT / "port_draw_effects.csv", index=False)
    # Contrast bins inside the same acquisition/class, requiring both bins to have actual ships.
    contrasts = []
    strata = ["port", "fold", "budget", "origin", "truth"]
    for column in group_columns:
        if column in ("incidence_bin", "support_class_agreement"):
            continue
        cell = data.groupby(strata + [column])[EFFECTS].mean().unstack(column)
        size = data.groupby(strata + [column]).sample_id.nunique().unstack(column, fill_value=0)
        if not {"low", "high"} <= set(size):
            continue
        eligible = size.index[(size.low >= 3) & (size.high >= 3)]
        for idx in eligible:
            vals = {"high_minus_low_"+name: 100*(cell.loc[idx,(name,"high")]-cell.loc[idx,(name,"low")]) for name in EFFECTS}
            contrasts.append(dict(zip(strata,idx), condition=column, low_samples=int(size.loc[idx,"low"]), high_samples=int(size.loc[idx,"high"]), **vals))
    pd.DataFrame(contrasts).to_csv(OUT / "within_scene_class_condition_contrasts.csv", index=False)
    events = data.loc[data.rescue | data.harm]
    events.to_csv(OUT / "rescue_harm_conditions.csv", index=False)
    vessel = data.groupby(["budget", "origin", "sample_id", "mmsi", "class_name", "port", "fold"]).agg(
        trials=("draw", "size"), rescues=("rescue", "sum"), harms=("harm", "sum"),
        new_rescues=("new_rescue", "sum"), all_controls_wrong=("rescue_all_controls_wrong", "sum"),
        avg_vs_half=("vs_half", "mean"), avg_vs_shuffle=("vs_shuffle", "mean"),
        incidence_deg=("incidence_deg", "first"), contrast_vv=("standardized_contrast_vv", "first"),
        ring_std_vv=("raw_ring_vv_std", "first"), occupancy=("box_occupancy_vv", "first"))
    vessel.to_csv(OUT / "per_vessel_context.csv")
    fixed_query = []
    for (budget, origin, sample_id), g in data.loc[data.ship.eq(0)].groupby(["budget", "origin", "sample_id"]):
        hit, miss = g.loc[g.rescue], g.loc[~g.rescue]
        if len(hit) and len(miss):
            fixed_query.append(dict(budget=budget, origin=origin, sample_id=sample_id,
                rescue_draws=len(hit), not_rescued_draws=len(miss),
                ring_true_margin_difference=hit.ring_true_support_margin_diagnostic.mean()-miss.ring_true_support_margin_diagnostic.mean()))
    pd.DataFrame(fixed_query).to_csv(OUT / "within_query_support_changes.csv", index=False)
    status = []
    for (budget,origin), g in data.groupby(["budget","origin"]):
        status.append(dict(budget=budget,origin=origin,**effect_summary(g),
            all_controls_wrong=int(g.rescue_all_controls_wrong.sum()),
            rescue_ships=g.loc[g.rescue,"mmsi"].nunique(),harm_ships=g.loc[g.harm,"mmsi"].nunique()))
    pd.DataFrame(status).to_csv(OUT / "overall_effects.csv", index=False)
    print("OVERALL\n",pd.DataFrame(status).to_string(index=False))
    print("CROSS SUBGROUPS\n",subgroups.loc[subgroups.origin.eq("cross"),["budget","condition","level","ships","rescues","harms","delta_ba_vs_ship","delta_ba_vs_half","delta_ba_vs_shuffle"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
