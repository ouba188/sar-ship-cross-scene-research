"""Independent selected numeric audit. Read saved predictions; no model execution."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
OUT = Path(__file__).resolve().parents[1]
RC = OUT.parent / "ports_root_cause_20261013"
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship2_transfer_20261007/target_manifest.csv")
TABLES = {name: pd.read_csv(OUT / (name + ".csv")) for name in
          ("scores", "effects", "interactions", "flows")}
CHECKS = {name: [] for name in TABLES}


def compare(table, key, values):
    frame = TABLES[table]
    mask = np.ones(len(frame), bool)
    for name, value in key.items():
        mask &= frame[name].eq(value).to_numpy()
    found = frame.loc[mask]
    assert len(found) == 1, (table, key, len(found))
    row = found.iloc[0]
    for name, value in values.items():
        error = abs(float(row[name]) - float(value))
        assert error < 1e-10, (table, key, name, value, row[name])
        CHECKS[table].append(error)


def archive(path):
    with np.load(path / "predictions.npz", allow_pickle=False) as stored:
        data = {key: stored[key] for key in stored.files}
    return data, np.load(path / "probabilities.npy", mmap_mode="r")


def score(y, pred, prob):
    correct = pred == y
    labels = np.unique(y)
    loss = np.square(prob - np.eye(8)[y]).sum(axis=1)
    nll = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-15))
    return dict(n=len(y), oa=100 * correct.mean(),
                ba=100 * np.mean([correct[y == c].mean() for c in labels]),
                brier_micro=loss.mean(), brier_macro=np.mean([loss[y == c].mean() for c in labels]),
                nll_micro=nll.mean(), nll_macro=np.mean([nll[y == c].mean() for c in labels]))


def mass(y, selected):
    return 100 * np.mean([selected[y == c].mean() for c in np.unique(y)])


def effect(y, a, b):
    ca, cb = a[0] == y, b[0] == y
    rescue, harm = ca & ~cb, ~ca & cb
    sa, sb = score(y, *a), score(y, *b)
    return dict(rescue_ba=mass(y, rescue), harm_ba=mass(y, harm),
                delta_ba=sa["ba"] - sb["ba"], delta_oa=sa["oa"] - sb["oa"],
                risk_gain_macro=sb["brier_macro"] - sa["brier_macro"],
                risk_gain_micro=sb["brier_micro"] - sa["brier_micro"],
                rescue_n=int(rescue.sum()), harm_n=int(harm.sum()))


def main():
    assert (OUT / "ANALYSIS_COMPLETE.json").exists()
    cases = {}
    for name, source, target in (("OO", "original", "original"),
                                 ("DD", "display", "display"),
                                 ("range_geometry", "original", "range_geometry")):
        data, prob = archive(RC / "predictions" / source / target / "fold_2/oss2")
        hi = list(data["heads"]).index("expert")
        cases[("fresh", name)] = (data, data["predictions"][:, :, hi], prob[:, :, hi])
    for name in ("original", "range_geometry", "axis_sham"):
        data, prob = archive(OUT / "replacement_predictions" / name / "fold_2/targets/shared_frozen/oss2")
        ki = list(data["kinds"]).index("mlp")
        cases[("archive_replacement", "OO" if name == "original" else name)] = (
            data, data["predictions"][ki, :, :, 0], prob[ki, :, :, 0])
    reference = cases[("fresh", "OO")][0]
    for data, _, _ in cases.values():
        for key in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(data[key], reference[key])
    frame = pd.read_csv(MANIFEST, keep_default_na=False).set_index("sample_id").loc[reference["ids"]]
    yall = frame.class_id.to_numpy(int)
    pairs = pd.read_csv(RC / "dataset_audit/oss2_grdh_slc_pairs.csv")
    assert pairs.same_sar_datetime.all()
    pairs = pairs[pairs.same_label]
    assert len(pairs) == 233
    index = pd.Index(reference["ids"])
    g, s = index.get_indexer(pairs.sample_id_grdh), index.get_indexer(pairs.sample_id_slc)
    assert min(g.min(), s.min()) >= 0
    np.testing.assert_array_equal(yall[g], yall[s])
    values, records, denominators = {}, [], []
    for draw in range(5):
        query = reference["eval_masks"][draw]
        assert not query[reference["support"][draw]].any()
        paired = np.zeros(len(query), bool)
        paired[s[query[g] & query[s]]] = True
        scopes = dict(original=query, query_SLC=query & frame.scene_id.str.contains("_SLC_", regex=False).to_numpy(),
                      paired_SLC=paired)
        for scope, mask in scopes.items():
            y = yall[mask]
            denominators.append(dict(draw=draw, scope=scope, n=len(y), classes=len(np.unique(y))))
            for (series, condition), (data, pred, prob) in cases.items():
                key = dict(series=series, kind="none" if series == "fresh" else "mlp", fold=2,
                           task="oss2", dataset="oss2", draw=draw, scope=scope, head="expert")
                arm_values = {}
                for ai, arm in enumerate(data["arms"]):
                    item = pred[draw, ai, mask], np.asarray(prob[draw, ai, mask], float)
                    arm_values[str(arm)] = item
                    compare("scores", dict(**key, condition=condition, arm=arm), score(y, *item))
                values[(series, condition, draw, scope)] = arm_values
                contrasts = ("T-R",) if series == "fresh" else ("T-P", "PE-P", "E-R", "T-PE_scaled")
                for contrast in contrasts:
                    a, b = contrast.split("-")
                    result = effect(y, arm_values[a], arm_values[b])
                    compare("effects", dict(**key, condition=condition, comparison=contrast), result)
                    records.append(dict(series=series, condition=condition, scope=scope,
                                        draw=draw, comparison=contrast, **result))
            before = values[("fresh", "OO", draw, scope)]
            after = values[("fresh", "DD", draw, scope)]
            e0, e1 = effect(y, before["T"], before["R"]), effect(y, after["T"], after["R"])
            r0, t0 = before["R"][0] == y, before["T"][0] == y
            r1, t1 = after["R"][0] == y, after["T"][0] == y
            oldres = ~r0 & t0
            key = dict(series="fresh", kind="none", fold=2, task="oss2", dataset="oss2", draw=draw,
                       scope=scope, head="expert", transition="DD-OO", comparison="T-R")
            interaction = dict(before_gain=e0["delta_ba"], after_gain=e1["delta_ba"],
                gain_change=e1["delta_ba"] - e0["delta_ba"],
                baseline_change=score(y, *after["R"])["ba"] - score(y, *before["R"])["ba"],
                candidate_change=score(y, *after["T"])["ba"] - score(y, *before["T"])["ba"],
                risk_gain_change=e1["risk_gain_macro"] - e0["risk_gain_macro"],
                old_rescue_ba=mass(y, oldres), old_rescue_now_base_correct_ba=mass(y, oldres & r1),
                old_rescue_now_both_correct_ba=mass(y, oldres & r1 & t1),
                old_rescue_now_only_base_correct_ba=mass(y, oldres & r1 & ~t1),
                old_rescue_retained_ba=mass(y, oldres & ~r1 & t1),
                old_rescue_now_both_wrong_ba=mass(y, oldres & ~r1 & ~t1))
            compare("interactions", key, interaction)
            for state, subset in dict(lost=r0 & ~r1, gained=~r0 & r1,
                                      stable_correct=r0 & r1, stable_wrong=~r0 & ~r1).items():
                br, bh = mass(y, subset & oldres), mass(y, subset & r0 & ~t0)
                ar, ah = mass(y, subset & ~r1 & t1), mass(y, subset & r1 & ~t1)
                compare("flows", dict(**key, state=state), dict(n=int(subset.sum()), mass_ba=mass(y, subset),
                        before_rescue=br, before_harm=bh, after_rescue=ar, after_harm=ah,
                        before_gain=br-bh, after_gain=ar-ah, gain_change=(ar-ah)-(br-bh)))
            if scope == "original":
                print("FRESH_INTERACTION", json.dumps(dict(draw=draw, **interaction)))
    summary = pd.DataFrame(records).groupby(["series", "scope", "condition", "comparison"])[
        ["delta_ba", "risk_gain_macro", "rescue_ba", "harm_ba"]].mean()
    print("SELECTED_MEANS_CSV")
    print(summary.to_csv(float_format="%.9f"))
    print("DENOMINATORS", json.dumps(denominators))
    print("PASS", json.dumps({name: dict(numeric_checks=len(errors), max_error=max(errors))
                              for name, errors in CHECKS.items()}))


if __name__ == "__main__":
    main()
