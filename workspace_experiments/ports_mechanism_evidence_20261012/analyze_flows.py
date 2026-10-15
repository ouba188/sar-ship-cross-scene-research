"""Paired sample-level evidence for input-dependent second-branch utility."""

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / "ports_input_controls_20261012"
sys.path.insert(0, str(INPUT))
import context as ctx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ARMS = tuple(ctx.crossfit.public.ARMS)
KINDS = tuple(ctx.crossfit.public.KINDS)
HEADS = ("expert", "c4_late")
PAIRS = (("T", "R"), ("P", "R"), ("E", "R"), ("PE", "P"),
         ("T", "P"), ("T", "PE_scaled"), ("DD", "T"), ("DD", "R"))
STATES = ("lost", "gained", "stable_correct", "stable_wrong")


def masks(old, new):
    return dict(lost=old & ~new, gained=~old & new,
                stable_correct=old & new, stable_wrong=~old & ~new)


def weights(y):
    counts = np.bincount(y, minlength=8)
    return 1. / (np.count_nonzero(counts) * counts[y])


def weighted_sum(w, value, subset):
    return float(w[subset] @ np.asarray(value, dtype=float)[subset])


def describe_flow(w, subset, b0, b1, a0, a1, risk0, risk1):
    rescue0, harm0 = ~b0 & a0, b0 & ~a0
    rescue1, harm1 = ~b1 & a1, b1 & ~a1
    old_gain = a0.astype(float)-b0
    new_gain = a1.astype(float)-b1
    mass = float(w[subset].sum())
    wrong_mass = float(w[subset & ~b1].sum())
    result = dict(n=int(subset.sum()), query_mass_pct=100*mass,
        old_rescue_n=int(np.sum(subset & rescue0)), old_harm_n=int(np.sum(subset & harm0)),
        new_rescue_n=int(np.sum(subset & rescue1)), new_harm_n=int(np.sum(subset & harm1)),
        new_base_wrong_n=int(np.sum(subset & ~b1)),
        old_rescue_pp=100*weighted_sum(w, rescue0, subset),
        old_harm_pp=100*weighted_sum(w, harm0, subset),
        new_rescue_pp=100*weighted_sum(w, rescue1, subset),
        new_harm_pp=100*weighted_sum(w, harm1, subset),
        old_gain_pp=100*weighted_sum(w, old_gain, subset),
        new_gain_pp=100*weighted_sum(w, new_gain, subset),
        gain_change_pp=100*weighted_sum(w, new_gain-old_gain, subset),
        old_risk_gain=weighted_sum(w, risk0, subset),
        new_risk_gain=weighted_sum(w, risk1, subset),
        risk_gain_change=weighted_sum(w, risk1-risk0, subset),
        candidate_kept_correct_n=int(np.sum(subset & a0 & a1)),
        candidate_gained_n=int(np.sum(subset & ~a0 & a1)),
        candidate_lost_n=int(np.sum(subset & a0 & ~a1)),
        candidate_stayed_wrong_n=int(np.sum(subset & ~a0 & ~a1)),
        candidate_kept_correct_pp=100*weighted_sum(w, a0 & a1, subset),
        candidate_gained_pp=100*weighted_sum(w, ~a0 & a1, subset))
    result["new_gain_conditional_pp"] = result["new_gain_pp"]/mass if mass else np.nan
    result["new_rescue_conditional_macro_pct"] = result["new_rescue_pp"]/wrong_mass if wrong_mass else np.nan
    result["candidate_kept_conditional_macro_pct"] = result["candidate_kept_correct_pp"]/mass if mass else np.nan
    result["candidate_correct_conditional_macro_pct"] = (
        result["candidate_kept_correct_pp"]+result["candidate_gained_pp"])/mass if mass else np.nan
    np.testing.assert_allclose(result["new_gain_pp"], result["new_rescue_pp"]-result["new_harm_pp"], atol=1e-12)
    np.testing.assert_allclose(result["old_gain_pp"], result["old_rescue_pp"]-result["old_harm_pp"], atol=1e-12)
    return result


def target_frame(name, ids):
    base = "oss2" if name == "oss2_grdh" else name
    frame = ctx.frame(base).set_index("sample_id").loc[ids]
    np.testing.assert_array_equal(frame.index.to_numpy(str), ids)
    return frame


def probability_losses(prob, ki, draw, hi, y):
    p = np.asarray(prob[ki, draw, :, hi], dtype=float)
    return np.sum((p-np.eye(8)[y][None])**2, axis=-1)


def source_path(condition, fold, name):
    return INPUT / "predictions" / condition / f"fold_{fold}/targets/shared_frozen" / name


def aggregate(table, group, values):
    # Equal task weights remain intact after excluding contaminated support draws.
    return table.groupby(["fold", "task"]+group, dropna=False)[values].mean().groupby(
        group, dropna=False).mean().reset_index()


def main():
    if not (INPUT / "COMPLETE.json").exists() or not (OUT / "PROTOCOL_ZH.md").exists():
        raise RuntimeError("Completed inputs and this analysis protocol are required")
    flags = pd.read_csv(INPUT / "audit/primary_geometry_flags.csv")
    rows, class_rows, events = [], [], []
    prior = pd.read_csv(INPUT / "arm_effects.csv").set_index(
        ["fold", "task", "condition", "draw", "scope", "kind", "head", "contrast"])
    replays, metadata = [], []
    for fold in ctx.FOLDS:
        for name in ctx.tasks(fold):
            root = source_path("original", fold, name)
            old = ctx.archive(root / "predictions.npz")
            op = np.load(root / "probabilities.npy", mmap_mode="r")
            f = target_frame(name, old["ids"])
            yall = f.class_id.to_numpy(int)
            ds = "ports3" if name in ctx.PORTS else name
            flag_ds = "self24ports" if name in ctx.PORTS else name.split("_")[0]
            risky = np.isin(old["ids"], flags.loc[flags.dataset == flag_ds, "sample_id"])
            identity_column = "mmsi" if name in ctx.PORTS else "mmsi_or_ship_id"
            identities = f[identity_column].astype(str).str.strip().to_numpy()
            number = pd.to_numeric(f[identity_column], errors="coerce")
            valid = number.between(100000000, 999999999) & number.eq(np.floor(number))
            mmsi = number.where(valid, 0).fillna(0).to_numpy(np.int64)
            products = f["product" if name in ctx.PORTS else "scene_id"].astype(str).to_numpy()
            metadata.append(dict(fold=fold, task=name, n=len(f), identity_column=identity_column,
                valid_mmsi_rows=int(np.sum(mmsi > 0)), unique_valid_mmsi=len(np.unique(mmsi[mmsi > 0])),
                unknown_or_other_identity_rows=int(np.sum(mmsi == 0)), unique_products=len(np.unique(products))))
            for condition in ctx.CONDITIONS[1:]:
                dst = source_path(condition, fold, name)
                new = ctx.archive(dst / "predictions.npz")
                np.testing.assert_array_equal(new["arms"], np.array(ARMS))
                np.testing.assert_array_equal(new["kinds"], np.array(KINDS))
                for key in ("ids", "support", "eval_masks", "common_query"):
                    np.testing.assert_array_equal(new[key], old[key])
                npb = np.load(dst / "probabilities.npy", mmap_mode="r")
                for ki, kind in enumerate(KINDS):
                    for hi, head in enumerate(HEADS):
                        for draw, ev in enumerate(old["eval_masks"]):
                            p0 = old["predictions"][ki, draw, :, hi]
                            p1 = new["predictions"][ki, draw, :, hi]
                            c0, c1 = p0 == yall[None], p1 == yall[None]
                            l0 = probability_losses(op, ki, draw, hi, yall)
                            l1 = probability_losses(npb, ki, draw, hi, yall)
                            scopes = [("original", ev), ("common_query", old["common_query"])]
                            if not risky[old["support"][draw]].any():
                                scopes.append(("proxy_retained_query", ev & ~risky))
                            if condition == "display" and kind == "mlp" and head == "expert":
                                e = pd.DataFrame(dict(fold=fold, task=name, dataset=ds, draw=draw,
                                    sample_id=old["ids"][ev], class_id=yall[ev], mmsi=mmsi[ev],
                                    identity=identities[ev], product=products[ev],
                                    common_query=old["common_query"][ev], geometry_risky=risky[ev]))
                                for ai, arm in enumerate(ARMS):
                                    e[arm+"_original"] = p0[ai, ev]
                                    e[arm+"_display"] = p1[ai, ev]
                                events.append(e)
                            for scope, query in scopes:
                                y = yall[query]
                                w = weights(y)
                                states = masks(c0[0, query], c1[0, query])
                                np.testing.assert_array_equal(sum(s.astype(int) for s in states.values()), np.ones(len(y), int))
                                for a, b in PAIRS:
                                    ai, bi = ARMS.index(a), ARMS.index(b)
                                    meta = dict(fold=fold, task=name, dataset=ds, condition=condition,
                                        draw=draw, scope=scope, kind=kind, head=head, contrast=a+"-"+b)
                                    args = (c0[bi, query], c1[bi, query], c0[ai, query], c1[ai, query],
                                            l0[bi, query]-l0[ai, query], l1[bi, query]-l1[ai, query])
                                    parts = {state: describe_flow(w, mask, *args) for state, mask in states.items()}
                                    total = describe_flow(w, np.ones(len(y), bool), *args)
                                    for field in ("n", "query_mass_pct", "old_gain_pp", "new_gain_pp", "gain_change_pp",
                                                  "old_risk_gain", "new_risk_gain", "risk_gain_change"):
                                        np.testing.assert_allclose(sum(p[field] for p in parts.values()), total[field], atol=1e-10)
                                    for state, value in {"all": total, **parts}.items():
                                        rows.append(dict(**meta, stratum=state, **value))
                                    for cond, prefix in (("original", "old"), (condition, "new")):
                                        key = (fold, name, cond, draw, scope, kind, head, a+"-"+b)
                                        ref = prior.loc[key]
                                        error_ba = abs(total[prefix+"_gain_pp"]-ref.delta_ba)
                                        error_risk = abs(total[prefix+"_risk_gain"]-ref.risk_gain_macro)
                                        if error_ba > 1e-9 or error_risk > 1e-10:
                                            raise ValueError(f"Existing effect replay mismatch: {key}")
                                        replays.append((error_ba, error_risk))
                                    if kind == "mlp" and scope == "original" and (a, b) in (("T", "R"), ("P", "R"), ("T", "P"), ("PE", "P")):
                                        for c in np.unique(y):
                                            for state, mask in states.items():
                                                selected = mask & (y == c)
                                                class_rows.append(dict(**meta, stratum=state, class_id=int(c),
                                                    class_query_n=int(np.sum(y == c)), **describe_flow(w, selected, *args)))
                print("FLOWS", fold, name, condition, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "flows.csv", index=False)
    classes = pd.DataFrame(class_rows)
    classes.to_csv(OUT / "class_flows.csv", index=False)
    pd.concat(events, ignore_index=True).to_csv(OUT / "display_sample_events.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUT / "cohort_metadata.csv", index=False)
    group = ["dataset", "condition", "scope", "kind", "head", "contrast", "stratum"]
    value_cols = [c for c in table if c not in group+["fold", "task", "draw"]]
    aggregate(table, group, value_cols).to_csv(OUT / "flow_summary.csv", index=False)
    table.groupby(["fold", "task"]+group)[value_cols].mean().reset_index().to_csv(OUT / "task_summary.csv", index=False)
    cg = ["dataset", "condition", "scope", "kind", "head", "contrast", "stratum", "class_id"]
    aggregate(classes, cg, [c for c in value_cols if c != "class_query_n"]).to_csv(OUT / "class_summary.csv", index=False)
    total = table[table.stratum == "all"]
    total.groupby(group).agg(
        new_positive_draws=("new_gain_pp", lambda s: int((s > 1e-10).sum())),
        robustness_positive_draws=("gain_change_pp", lambda s: int((s > 1e-10).sum())),
        draw_contexts=("gain_change_pp", "size"),
        min_gain_change=("gain_change_pp", "min"), max_gain_change=("gain_change_pp", "max")
    ).reset_index().to_csv(OUT / "draw_spread.csv", index=False)
    ctx.write(OUT / "ANALYSIS_COMPLETE.json", dict(flow_rows=len(table), class_rows=len(classes),
        sample_event_rows=sum(len(e) for e in events), replay_comparisons=len(replays),
        replay_max_ba_error=float(np.max(replays, axis=0)[0]), replay_max_risk_error=float(np.max(replays, axis=0)[1]),
        four_strata_sum_to_original_query_effect=True, same_input_ids_and_supports=True,
        new_training_runs=0, independent_training_seeds_added=0,
        inferential_p_values_reported=False, analysis="Exploratory conditional error flows on existing development results"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
