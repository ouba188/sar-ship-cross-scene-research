"""Paired input effects, preserved cue decomposition and manipulation checks."""

from context import *
from report import metrics, CONTRASTS, HEADS
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


def labels(name, ids):
    base = "oss2" if name == "oss2_grdh" else name
    f = frame(base).set_index("sample_id").loc[ids]
    np.testing.assert_array_equal(f.index.to_numpy(str), ids)
    return f.class_id.to_numpy(int)


def main():
    if not (OUT / "PREDICTIONS_FROZEN.json").exists():
        raise RuntimeError("Freeze all input-condition predictions first")
    records, features, cls, replay = [], [], [], []
    geometry_flags = pd.read_csv(OUT / "audit/primary_geometry_flags.csv")
    for fold in FOLDS:
        for name in tasks(fold):
            reference = None
            for condition in CONDITIONS:
                dst = OUT / "predictions" / condition / f"fold_{fold}/targets/shared_frozen" / name
                saved = archive(dst / "predictions.npz")
                prob = np.load(dst / "probabilities.npy", mmap_mode="r")
                if reference is None:
                    reference = saved
                else:
                    for key in ("ids", "support", "eval_masks", "common_query"):
                        np.testing.assert_array_equal(saved[key], reference[key])
                yall = labels(name, saved["ids"])
                geometry_dataset = "self24ports" if name in PORTS else name.split("_")[0]
                flagged_ids = geometry_flags.loc[geometry_flags.dataset == geometry_dataset, "sample_id"]
                risky = np.isin(saved["ids"], flagged_ids)
                meta = dict(fold=fold, task=name, dataset="ports3" if name in PORTS else name, condition=condition)
                complete = json.loads((dst / "COMPLETE.json").read_text(encoding="utf-8"))
                features.extend(dict(**meta, **d) for d in complete["feature_diagnostics"])
                if condition == "original":
                    oldroot = PREVIOUS / f"fold_{fold}/targets/shared_frozen" / name
                    old = archive(oldroot / "predictions.npz")
                    op = np.load(oldroot / "probabilities.npy", mmap_mode="r")
                    for key in ("ids", "support", "eval_masks"):
                        np.testing.assert_array_equal(old[key], saved[key])
                    for draw, mask in enumerate(saved["eval_masks"]):
                        for ai, arm in enumerate(crossfit.public.ARMS):
                            for hi, head in enumerate(HEADS):
                                a = metrics(yall[mask], saved["predictions"][1, draw, ai, hi, mask], np.asarray(prob[1, draw, ai, hi, mask], float))
                                b = metrics(yall[mask], old["predictions"][1, draw, ai, hi, mask], np.asarray(op[1, draw, ai, hi, mask], float))
                                replay.append(dict(**meta, draw=draw, arm=arm, head=head,
                                    delta_ba=a["ba"]-b["ba"], delta_brier_macro=a["brier_macro"]-b["brier_macro"],
                                    changed_predictions=int(np.sum(saved["predictions"][1, draw, ai, hi, mask] != old["predictions"][1, draw, ai, hi, mask])),
                                    max_probability_error=float(np.max(np.abs(prob[1, draw, ai, hi, mask]-op[1, draw, ai, hi, mask])))))
                for draw, ev in enumerate(saved["eval_masks"]):
                    sc = np.bincount(yall[saved["support"][draw]], minlength=8)
                    support_risk_n = int(risky[saved["support"][draw]].sum())
                    for scope, mask in (("original", ev), ("common_query", saved["common_query"]),
                                        ("proxy_retained_query", ev & ~risky)):
                        y = yall[mask]
                        for ki, kind in enumerate(crossfit.public.KINDS):
                            for hi, head in enumerate(HEADS):
                                for ai, arm in enumerate(crossfit.public.ARMS):
                                    p = np.asarray(prob[ki, draw, ai, hi, mask], float)
                                    pred = saved["predictions"][ki, draw, ai, hi, mask]
                                    row = dict(**meta, draw=draw, scope=scope, kind=kind, head=head, arm=arm,
                                        support_risk_n=support_risk_n, **metrics(y, pred, p))
                                    np.testing.assert_allclose(p.sum(1), 1., atol=1e-7, rtol=0)
                                    records.append(row)
                                    losses = np.sum((p-np.eye(8)[y])**2, axis=1)
                                    for c in np.unique(y):
                                        ix = y == c
                                        cls.append(dict(**meta, draw=draw, scope=scope, kind=kind, head=head, arm=arm,
                                            support_risk_n=support_risk_n,
                                            class_id=int(c), n=int(ix.sum()), support_n=int(sc[c]),
                                            recall=100*np.mean(pred[ix] == c), brier=float(losses[ix].mean())))
                print("READOUT", fold, name, condition, flush=True)
    table = pd.DataFrame(records)
    classes = pd.DataFrame(cls)
    keys = ["fold", "task", "dataset", "condition", "draw", "scope", "kind", "head"]
    cmean = classes.groupby(keys+["arm"])[["recall", "brier"]].mean().reset_index()
    check = table.merge(cmean, on=keys+["arm"], validate="one_to_one")
    np.testing.assert_allclose(check.ba, check.recall, atol=1e-10, rtol=0)
    np.testing.assert_allclose(check.brier_macro, check.brier, atol=1e-10, rtol=0)
    table.to_csv(OUT / "metrics.csv", index=False)
    classes.to_csv(OUT / "classes.csv", index=False)
    pd.DataFrame(features).to_csv(OUT / "feature_diagnostics.csv", index=False)
    pd.DataFrame(replay).to_csv(OUT / "baseline_replay.csv", index=False)
    table.groupby(["condition", "dataset", "scope", "kind", "head", "arm"])[
        ["ba", "oa", "brier_macro", "brier_micro", "nll_macro"]].mean().reset_index().to_csv(OUT / "summary.csv", index=False)
    arm_effects = []
    for a, b in CONTRASTS:
        match = table[table.arm == a].merge(table[table.arm == b], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        out = match[keys].copy()
        out["contrast"] = a+"-"+b
        out["delta_ba"] = match.ba_a-match.ba_b
        out["risk_gain_macro"] = match.brier_macro_b-match.brier_macro_a
        out["risk_gain_micro"] = match.brier_micro_b-match.brier_micro_a
        arm_effects.append(out)
    effects = pd.concat(arm_effects, ignore_index=True)
    effects.to_csv(OUT / "arm_effects.csv", index=False)
    effects.groupby(["condition", "dataset", "scope", "kind", "head", "contrast"])[
        ["delta_ba", "risk_gain_macro", "risk_gain_micro"]].mean().reset_index().to_csv(OUT / "arm_effect_summary.csv", index=False)
    input_keys = [k for k in keys if k != "condition"] + ["arm"]
    matched = table[table.condition != "original"].merge(table[table.condition == "original"],
        on=input_keys, suffixes=("_new", "_old"), validate="many_to_one")
    changed = matched[input_keys].copy()
    changed["condition"] = matched.condition_new
    changed["support_risk_n"] = matched.support_risk_n_old
    changed["delta_ba"] = matched.ba_new-matched.ba_old
    changed["delta_oa"] = matched.oa_new-matched.oa_old
    changed["risk_gain_macro"] = matched.brier_macro_old-matched.brier_macro_new
    changed["risk_gain_micro"] = matched.brier_micro_old-matched.brier_micro_new
    changed.to_csv(OUT / "input_effects.csv", index=False)
    changed.groupby(["condition", "dataset", "scope", "kind", "head", "arm"])[
        ["delta_ba", "delta_oa", "risk_gain_macro", "risk_gain_micro"]].mean().reset_index().to_csv(OUT / "input_effect_summary.csv", index=False)
    clean = changed[(changed.scope == "proxy_retained_query") & (changed.support_risk_n == 0)]
    group = ["condition", "dataset", "kind", "head", "arm"]
    measures = ["delta_ba", "delta_oa", "risk_gain_macro", "risk_gain_micro"]
    # Keep equal target-task weights despite excluding one contaminated support draw.
    clean.groupby(["fold", "task"]+group)[measures].mean().groupby(group).mean().reset_index().to_csv(
        OUT / "proxy_retained_no_support_risk_summary.csv", index=False)
    manipulation_checks()
    write(OUT / "READOUT_COMPLETE.json", dict(metric_rows=len(table), input_effect_rows=len(changed),
        all_support_query_ids_match=True, class_contributions_match=True,
        inference_frozen_first=True, original_replay_metrics_saved=True,
        aggregation="Equal three ports or equal two public source folds, then equal five support draws; not independent test replicates"))


def manipulation_checks():
    summaries, ordering = [], []
    fields = ("contrast_proxy", "visible_footprint_proxy_fraction", "center_excess_proxy", "elongation_proxy")
    for base in DATASETS:
        f = frame(base)
        for condition in CONDITIONS:
            d = pd.read_csv(OUT / "features" / base / condition / "descriptors.csv")
            np.testing.assert_array_equal(d.sample_id.to_numpy(str), f.sample_id.to_numpy(str))
            d["class_id"] = f.class_id.to_numpy(int)
            datasets = [(base, d)]
            if base == "oss2":
                datasets.append(("oss2_grdh", d[f.scene_id.str.contains("_GRDH_", regex=False).to_numpy()]))
            for name, part in datasets:
                for c, group in [(int(c), g) for c, g in part.groupby("class_id")] + [(-1, part)]:
                    for field in fields:
                        summaries.append(dict(task=name, dataset="ports3" if name in PORTS else name,
                            condition=condition, class_id=c, descriptor=field, n=len(group), valid=int(group[field].notna().sum()),
                            mean=float(group[field].mean()), median=float(group[field].median())))
                pair = part[part.class_id.isin([1, 4])]
                for field in fields:
                    ok = pair[field].notna()
                    if pair.loc[ok, "class_id"].nunique() == 2:
                        ordering.append(dict(task=name, dataset="ports3" if name in PORTS else name,
                            condition=condition, descriptor=field, n_fishing=int(np.sum(pair.loc[ok, "class_id"] == 1)),
                            n_container=int(np.sum(pair.loc[ok, "class_id"] == 4)),
                            auc_container_higher=float(roc_auc_score(pair.loc[ok, "class_id"] == 4, pair.loc[ok, field]))))
    pd.DataFrame(summaries).to_csv(OUT / "image_descriptor_summary.csv", index=False)
    pd.DataFrame(ordering).to_csv(OUT / "class_cue_ordering.csv", index=False)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
