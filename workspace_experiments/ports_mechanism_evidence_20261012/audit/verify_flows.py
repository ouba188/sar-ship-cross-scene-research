"""Independent CPU-only accounting of frozen MLP expert error flows."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parent.parent
INPUT = ROOT / "ports_input_controls_20261012"
sys.dont_write_bytecode = True
sys.path.insert(0, str(INPUT))
import context

CASES = ((2, "Los Angeles"), (2, "oss1"), (3, "oss1"), (3, "Port Klang"))
CONTRASTS = (("T", "R"), ("P", "R"), ("T", "P"), ("PE", "P"))
CONDITIONS = ("original", "display", "crop75", "crop75_display")


def compare_main(table, split, persistence, class_flows):
    main_path = AUDIT.parent / "flows.csv"
    if not main_path.exists():
        return dict(status="pending")
    main = pd.read_csv(main_path)
    main = main[(main["condition"] == "display") & (main["scope"] == "original") &
                (main["kind"] == "mlp") & (main["head"] == "expert")]
    keys = ["fold", "task", "draw", "scope", "kind", "head", "contrast", "stratum"]
    expected = pd.concat([table.assign(stratum="all"), split], ignore_index=True)
    errors, matched_rows = {}, 0
    for condition, prefix in (("original", "old"), ("display", "new")):
        ours = expected[expected["condition"] == condition].drop(columns="condition")
        pair = ours.merge(main, on=keys, how="left", validate="one_to_one", indicator=True,
                          suffixes=("_audit", "_main"))
        assert pair["_merge"].eq("both").all()
        matched_rows += len(pair)
        for ours_col, main_col in (("n_audit", "n_main"), ("rescue_n", prefix+"_rescue_n"),
                                   ("harm_n", prefix+"_harm_n"), ("rescue_pp", prefix+"_rescue_pp"),
                                   ("harm_pp", prefix+"_harm_pp"), ("gain_pp", prefix+"_gain_pp")):
            errors[condition+"/"+ours_col] = float((pair[ours_col]-pair[main_col]).abs().max())
            np.testing.assert_allclose(pair[ours_col], pair[main_col], atol=1e-10, rtol=0)
        selected = pair["stratum"] != "all"
        np.testing.assert_allclose(pair.loc[selected, "mass_pp"], pair.loc[selected, "query_mass_pct"],
                                   atol=1e-10, rtol=0)
        np.testing.assert_allclose(pair.loc[selected, "change_pp"], pair.loc[selected, "gain_change_pp"],
                                   atol=1e-10, rtol=0)
        selected = ~selected
        np.testing.assert_allclose(pair.loc[selected, "brier_macro_gain"],
                                   pair.loc[selected, prefix+"_risk_gain"], atol=1e-10, rtol=0)
    pkeys = ["fold", "task", "draw", "scope", "kind", "head"]
    lost = main[(main["stratum"] == "lost") & (main["contrast"] == "T-R")]
    pairs = persistence.merge(lost, on=pkeys, how="left", validate="one_to_one")
    for ours, theirs in (("lost_n", "n"), ("lost_mass_pp", "query_mass_pct"),
                         ("t_original_and_display_n", "candidate_kept_correct_n"),
                         ("t_original_only_n", "candidate_lost_n"),
                         ("t_display_only_n", "candidate_gained_n"),
                         ("t_neither_n", "candidate_stayed_wrong_n"),
                         ("t_original_and_display_mass_pp", "candidate_kept_correct_pp"),
                         ("t_display_only_mass_pp", "candidate_gained_pp")):
        np.testing.assert_allclose(pairs[ours], pairs[theirs], atol=1e-10, rtol=0)
    main_classes = pd.read_csv(AUDIT.parent / "class_flows.csv")
    main_classes = main_classes[(main_classes["condition"] == "display") &
                                (main_classes["scope"] == "original") & (main_classes["kind"] == "mlp") &
                                (main_classes["head"] == "expert")]
    ckeys = pkeys + ["contrast", "class_id"]
    values = [f"{prefix}_{metric}" for prefix in ("old", "new") for metric in
              ("rescue_n", "harm_n", "rescue_pp", "harm_pp", "gain_pp")]
    main_classes = main_classes.groupby(ckeys)[values].sum().reset_index()
    for condition, prefix in (("original", "old"), ("display", "new")):
        pairs = class_flows[class_flows["condition"] == condition].merge(
            main_classes, on=ckeys, how="left", validate="one_to_one", indicator=True)
        assert pairs["_merge"].eq("both").all()
        for field in ("rescue_n", "harm_n", "rescue_pp", "harm_pp", "gain_pp"):
            np.testing.assert_allclose(pairs[field], pairs[prefix+"_"+field], atol=1e-10, rtol=0)
    summary = pd.read_csv(AUDIT.parent / "task_summary.csv")
    summary = summary[(summary["condition"] == "display") & (summary["scope"] == "original") &
                      (summary["kind"] == "mlp") & (summary["head"] == "expert") &
                      (summary["stratum"] == "all")]
    skeys = ["fold", "task", "contrast"]
    for condition, prefix in (("original", "old"), ("display", "new")):
        means = table[table["condition"] == condition].groupby(skeys)[
            ["rescue_pp", "harm_pp", "gain_pp"]].mean().reset_index()
        pair = means.merge(summary, on=skeys, how="left", validate="one_to_one", indicator=True)
        assert pair["_merge"].eq("both").all()
        for field in ("rescue_pp", "harm_pp", "gain_pp"):
            np.testing.assert_allclose(pair[field], pair[prefix+"_"+field], atol=1e-10, rtol=0)
    return dict(status="VERIFIED", matched_flow_cells=matched_rows,
                persistence_rows=len(persistence), class_rows=len(class_flows),
                task_summary_condition_rows=32, max_abs_errors=errors)


def load(condition, fold, task):
    folder = INPUT / "predictions" / condition / f"fold_{fold}" / "targets/shared_frozen" / task
    with np.load(folder / "predictions.npz") as z:
        saved = {key: z[key] for key in z.files}
    return saved, np.load(folder / "probabilities.npy", mmap_mode="r")


def compare_metadata(metadata):
    main = pd.read_csv(AUDIT.parent / "cohort_metadata.csv")
    if "valid_mmsi_rows" not in main:
        return dict(status="pending_metadata_fix")
    pair = metadata.merge(main, on=["fold", "task"], validate="one_to_one", suffixes=("_audit", "_main"))
    assert len(pair) == len(metadata)
    for ours, theirs in (("rows", "n"), ("missing_or_nonpositive_identity_rows", "unknown_or_other_identity_rows"),
                         ("distinct_positive_identity_values", "unique_valid_mmsi"),
                         ("distinct_acquisition_values", "unique_products")):
        np.testing.assert_array_equal(pair[ours], pair[theirs])
    events = pd.read_csv(AUDIT.parent / "display_sample_events.csv", keep_default_na=False,
                         dtype={"sample_id": str, "identity": str})
    checked = 0
    for fold, task in CASES:
        selected = events[(events["fold"] == fold) & (events["task"] == task)]
        f = context.frame(task).set_index("sample_id").loc[selected["sample_id"]]
        column = "mmsi" if "mmsi" in f else "mmsi_or_ship_id"
        np.testing.assert_array_equal(selected["identity"], f[column].astype(str).str.strip())
        expected = pd.to_numeric(f[column], errors="coerce").fillna(0).to_numpy()
        np.testing.assert_array_equal(selected["mmsi"], expected)
        checked += len(selected)
    return dict(status="VERIFIED", cohort_rows=len(pair), event_identity_rows=checked,
                completion_marker=str(AUDIT.parent / "ANALYSIS_COMPLETE.json"))


def flow(base, candidate, weights, selection):
    rescued = selection & ~base & candidate
    harmed = selection & base & ~candidate
    rescue = float(weights[rescued].sum() * 100)
    harm = float(weights[harmed].sum() * 100)
    return dict(n=int(selection.sum()), rescue_n=int(rescued.sum()), harm_n=int(harmed.sum()),
                rescue_pp=rescue, harm_pp=harm, gain_pp=rescue-harm)


def main():
    totals, strata, persistence, class_flows, metadata, alignments = [], [], [], [], [], []
    for fold, task in CASES:
        old, old_prob = load("original", fold, task)
        new, new_prob = load("display", fold, task)
        for condition in CONDITIONS[1:]:
            other = new if condition == "display" else load(condition, fold, task)[0]
            for key in ("ids", "support", "eval_masks", "common_query", "arms", "kinds"):
                np.testing.assert_array_equal(old[key], other[key], err_msg=f"{fold}/{task}/{condition}/{key}")
            alignments.append(dict(fold=fold, task=task, condition=condition, matched=True))
        f = context.frame(task).set_index("sample_id")
        assert f.index.is_unique
        ids = old["ids"].astype(str)
        f = f.loc[ids]
        np.testing.assert_array_equal(f.index.to_numpy(str), ids)
        yall = f["class_id"].to_numpy(int)
        identity_col = "mmsi" if "mmsi" in f.columns else "mmsi_or_ship_id"
        identity = pd.to_numeric(f[identity_col], errors="coerce").fillna(0).to_numpy()
        known = identity > 0
        metadata.append(dict(fold=fold, task=task, rows=len(f), identity_column=identity_col,
                             missing_or_nonpositive_identity_rows=int((~known).sum()),
                             distinct_positive_identity_values=int(np.unique(identity[known]).size),
                             acquisition_column="product" if "product" in f.columns else "scene_id",
                             distinct_acquisition_values=int(f["product" if "product" in f.columns else "scene_id"].nunique())))
        kinds, arms = old["kinds"].tolist(), old["arms"].tolist()
        ki = kinds.index("mlp")
        np.testing.assert_array_equal(old["common_query"], old["eval_masks"].all(axis=0))
        for draw, query in enumerate(old["eval_masks"]):
            assert not query[old["support"][draw]].any()
            y = yall[query]
            present, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
            weights = 1.0 / (len(present) * counts[inverse])
            all_rows = np.ones(len(y), dtype=bool)
            correct = {}
            risks = {}
            for condition, saved, probabilities in (("original", old, old_prob), ("display", new, new_prob)):
                correct[condition], risks[condition] = {}, {}
                for arm in ("R", "P", "T", "PE"):
                    pred = saved["predictions"][ki, draw, arms.index(arm), 0, query]
                    prob = np.asarray(probabilities[ki, draw, arms.index(arm), 0, query], dtype=float)
                    np.testing.assert_array_equal(pred, prob.argmax(axis=1))
                    correct[condition][arm] = pred == y
                    risks[condition][arm] = ((prob - np.eye(prob.shape[1])[y])**2).sum(axis=1)
            r0, r1 = correct["original"]["R"], correct["display"]["R"]
            groups = dict(lost=r0 & ~r1, gained=~r0 & r1, stable_correct=r0 & r1,
                          stable_wrong=~r0 & ~r1)
            np.testing.assert_array_equal(np.stack(list(groups.values())).sum(axis=0), np.ones(len(y)))
            meta = dict(fold=fold, task=task, draw=draw, scope="original", kind="mlp", head="expert")
            for cand, base in CONTRASTS:
                parts = []
                for condition in ("original", "display"):
                    b, a = correct[condition][base], correct[condition][cand]
                    result = flow(b, a, weights, all_rows)
                    direct_ba = 100 * np.mean([a[y == c].mean()-b[y == c].mean() for c in present])
                    np.testing.assert_allclose(result["gain_pp"], direct_ba, atol=1e-12, rtol=0)
                    losses = risks[condition][base]-risks[condition][cand]
                    totals.append(dict(**meta, condition=condition, contrast=f"{cand}-{base}",
                                       k=len(present), **result, brier_macro_gain=float(weights @ losses),
                                       brier_micro_gain=float(losses.mean())))
                    for name, selection in groups.items():
                        part = flow(b, a, weights, selection)
                        record = dict(**meta, condition=condition, contrast=f"{cand}-{base}",
                                      stratum=name, mass_pp=float(weights[selection].sum()*100), **part)
                        strata.append(record)
                        parts.append(record)
                    for c in present:
                        class_flows.append(dict(**meta, condition=condition, contrast=f"{cand}-{base}",
                                                class_id=int(c), **flow(b, a, weights, y == c)))
                    for metric in ("rescue_pp", "harm_pp", "gain_pp"):
                        np.testing.assert_allclose(sum(p[metric] for p in parts if p["condition"] == condition),
                                                   result[metric], atol=1e-12, rtol=0)
                for name in groups:
                    p0, p1 = [p for p in parts if p["stratum"] == name]
                    p0["change_pp"] = p1["gain_pp"]-p0["gain_pp"]
                    p1["change_pp"] = p0["change_pp"]
            lost = groups["lost"]
            t0, t1 = correct["original"]["T"], correct["display"]["T"]
            cells = dict(t_original_and_display=lost & t0 & t1, t_original_only=lost & t0 & ~t1,
                         t_display_only=lost & ~t0 & t1, t_neither=lost & ~t0 & ~t1)
            persistence.append(dict(**meta, lost_n=int(lost.sum()), lost_mass_pp=float(weights[lost].sum()*100),
                                    **{key+"_n": int(value.sum()) for key, value in cells.items()},
                                    **{key+"_mass_pp": float(weights[value].sum()*100) for key, value in cells.items()},
                                    display_t_correct_given_lost=float(t1[lost].mean()) if lost.any() else None,
                                    original_t_correct_given_lost=float(t0[lost].mean()) if lost.any() else None))
    table, split = pd.DataFrame(totals), pd.DataFrame(strata)
    prior = pd.read_csv(INPUT / "arm_effects.csv")
    keys = ["fold", "task", "draw", "scope", "kind", "head", "condition", "contrast"]
    compared = table.merge(prior, on=keys, how="left", validate="one_to_one", indicator=True)
    assert compared["_merge"].eq("both").all()
    errors = {}
    for ours, theirs in (("gain_pp", "delta_ba"), ("brier_macro_gain", "risk_gain_macro"),
                         ("brier_micro_gain", "risk_gain_micro")):
        errors[ours] = float((compared[ours]-compared[theirs]).abs().max())
        np.testing.assert_allclose(compared[ours], compared[theirs], atol=1e-10, rtol=0)
    for filename, value in (("independent_flows.csv", table), ("independent_strata.csv", split),
                            ("independent_class_flows.csv", pd.DataFrame(class_flows)),
                            ("independent_persistence.csv", pd.DataFrame(persistence)),
                            ("identity_metadata.csv", pd.DataFrame(metadata))):
        value.to_csv(AUDIT / filename, index=False)
    table.groupby(["fold", "task", "condition", "contrast"])[
        ["rescue_pp", "harm_pp", "gain_pp"]].mean().reset_index().to_csv(AUDIT / "flow_means.csv", index=False)
    split.groupby(["fold", "task", "condition", "contrast", "stratum"])[
        ["rescue_pp", "harm_pp", "gain_pp", "change_pp"]].mean().reset_index().to_csv(
            AUDIT / "stratum_means.csv", index=False)
    main_result = compare_main(table, split, pd.DataFrame(persistence), pd.DataFrame(class_flows))
    metadata_result = compare_metadata(pd.DataFrame(metadata)) if main_result["status"] == "VERIFIED" else None
    receipt = dict(status="INDEPENDENT_RAW_AND_PRIOR_VERIFIED", no_model_runs=True,
                   original_query_only=True, totals=len(table), strata=len(split),
                   matched_condition_pairs=alignments, prior_max_abs_error=errors,
                   main_output_comparison=main_result, metadata_fix_comparison=metadata_result,
                   units="BA percentage points; global class denominators")
    (AUDIT / "VERIFICATION.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(table.groupby(["fold", "task", "condition", "contrast"])["gain_pp"].mean().to_string())


if __name__ == "__main__":
    main()
