"""Frozen prediction readout and label/coverage matched domain comparisons."""

from crossfit import *
from report import metrics, CONTRASTS, HEADS
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from dual_reconstruction_core import reconstruct


def target_frame(name, ids):
    if name in PUBLIC_TASKS:
        frame = pd.read_csv(public.original(name.split("_")[0]) / "target_manifest.csv", keep_default_na=False)
        key = "sample_id"
    else:
        frame = pd.read_csv(SOURCE / "manifest.csv", usecols=["instance_id", "class_id"])
        key = "instance_id"
    frame = frame.set_index(key).loc[ids].reset_index()
    np.testing.assert_array_equal(frame[key].to_numpy(str), ids)
    return frame


def main():
    if not (OUT / "PREDICTIONS_FROZEN.json").exists():
        raise RuntimeError("Freeze all predictions before outcome analysis")
    records, effects, classes, features = [], [], [], []
    error = 0.
    for fold in range(5):
        for name in fold_ports(fold) + list(PUBLIC_TASKS):
            dst = OUT / f"fold_{fold}/targets/shared_frozen" / name
            saved = archive(dst / "predictions.npz")
            p = np.load(dst / "probabilities.npy", mmap_mode="r")
            yall = target_frame(name, saved["ids"]).class_id.to_numpy(int)
            meta = dict(fold=fold, dataset=name if name in PUBLIC_TASKS else "self24ports", task=name)
            receipt = json.loads((dst / "COMPLETE.json").read_text(encoding="utf-8"))
            features.extend(dict(**meta, **x) for x in receipt["feature_diagnostics"])
            for draw, ev in enumerate(saved["eval_masks"]):
                sc = np.bincount(yall[saved["support"][draw]], minlength=8)
                for scope, mask in (("original", ev), ("common_query", saved["common_query"])):
                    y = yall[mask]
                    for ki, kind in enumerate(public.KINDS):
                        for hi, head in enumerate(HEADS):
                            byarm = {}
                            for ai, arm in enumerate(public.ARMS):
                                prob = np.asarray(p[ki, draw, ai, hi, mask], float)
                                pred = saved["predictions"][ki, draw, ai, hi, mask]
                                np.testing.assert_allclose(prob.sum(1), 1, atol=1e-7, rtol=0)
                                row = dict(**meta, scope=scope, kind=kind, draw=draw, head=head, arm=arm,
                                    **metrics(y, pred, prob))
                                records.append(row)
                                byarm[arm] = row
                                loss = np.sum((prob-np.eye(8)[y])**2, axis=1)
                                for c in np.unique(y):
                                    cm = y == c
                                    classes.append(dict(**meta, scope=scope, kind=kind, draw=draw, head=head, arm=arm,
                                        class_id=int(c), n=int(cm.sum()), support_n=int(sc[c]),
                                        recall=100*np.mean(pred[cm] == c), brier=float(loss[cm].mean())))
                                if draw == 0 and scope == "original":
                                    _, inv, ns = np.unique(y, return_inverse=True, return_counts=True)
                                    checks = [100*balanced_accuracy_score(y, pred),
                                        brier_score_loss(y, prob, labels=np.arange(8), scale_by_half=False, sample_weight=1/ns[inv])]
                                    want = [row["ba"], row["brier_macro"]]
                                    error = max(error, float(np.max(np.abs(np.array(checks)-want))))
                                    np.testing.assert_allclose(checks, want, atol=1e-10, rtol=1e-10)
                            for a, b in CONTRASTS:
                                effects.append(dict(**meta, scope=scope, kind=kind, draw=draw, head=head, contrast=a+"-"+b,
                                    delta_ba=byarm[a]["ba"]-byarm[b]["ba"], delta_oa=byarm[a]["oa"]-byarm[b]["oa"],
                                    risk_gain_macro=byarm[b]["brier_macro"]-byarm[a]["brier_macro"],
                                    risk_gain_micro=byarm[b]["brier_micro"]-byarm[a]["brier_micro"]))
            print("READOUT", fold, name, flush=True)
    table, effect, cls = pd.DataFrame(records), pd.DataFrame(effects), pd.DataFrame(classes)
    table.to_csv(OUT / "metrics.csv", index=False)
    effect.to_csv(OUT / "effects.csv", index=False)
    cls.to_csv(OUT / "classes.csv", index=False)
    pd.DataFrame(features).to_csv(OUT / "feature_diagnostics.csv", index=False)
    keys = ["dataset", "scope", "kind", "head", "arm"]
    table.groupby(keys)[["ba", "oa", "brier_macro", "brier_micro"]].mean().reset_index().to_csv(OUT / "summary.csv", index=False)
    keys = ["dataset", "scope", "kind", "head", "contrast"]
    effect.groupby(keys)[["delta_ba", "delta_oa", "risk_gain_macro", "risk_gain_micro"]].mean().reset_index().to_csv(OUT / "effect_summary.csv", index=False)
    task_effect = effect.groupby(["fold", "task"]+keys)[["delta_ba", "delta_oa", "risk_gain_macro", "risk_gain_micro"]].mean().reset_index()
    task_effect.to_csv(OUT / "task_effects.csv", index=False)

    ckeys = ["fold", "dataset", "task", "scope", "kind", "draw", "head", "class_id"]
    ce = []
    for a, b in CONTRASTS:
        joined = cls[cls.arm == a].merge(cls[cls.arm == b], on=ckeys, suffixes=("_a", "_b"), validate="one_to_one")
        out = joined[ckeys].copy()
        out["contrast"] = a+"-"+b
        out["n"], out["support_n"] = joined.n_a, joined.support_n_a
        out["delta_recall"] = joined.recall_a-joined.recall_b
        out["risk_gain"] = joined.brier_b-joined.brier_a
        ce.append(out)
    ce = pd.concat(ce, ignore_index=True)
    ce.to_csv(OUT / "class_effects.csv", index=False)
    accounted = ce.groupby(ckeys[:-1]+["contrast"])[["delta_recall", "risk_gain"]].mean().reset_index()
    check = effect.merge(accounted, on=ckeys[:-1]+["contrast"], validate="one_to_one")
    np.testing.assert_allclose(check.delta_ba, check.delta_recall, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(check.risk_gain_macro, check.risk_gain, atol=1e-10, rtol=1e-10)

    paired = []
    matchkeys = ["fold", "scope", "kind", "draw", "head", "class_id", "contrast"]
    port = ce[ce.dataset == "self24ports"]
    for target in PUBLIC_TASKS:
        match = port.merge(ce[ce.dataset == target], on=matchkeys, suffixes=("_port", "_public"), validate="many_to_one")
        for coverage, subset in (("query_class_matched", match),
                ("support_present_both", match[(match.support_n_port > 0)&(match.support_n_public > 0)])):
            group = subset.groupby(["fold", "task_port", "scope", "kind", "draw", "head", "contrast"])
            item = group.agg(classes=("class_id", "size"), port_risk_gain=("risk_gain_port", "mean"),
                public_risk_gain=("risk_gain_public", "mean"), port_delta_ba=("delta_recall_port", "mean"),
                public_delta_ba=("delta_recall_public", "mean")).reset_index()
            item["public_target"], item["coverage"] = target, coverage
            paired.append(item)
    pair = pd.concat(paired, ignore_index=True)
    pair.to_csv(OUT / "paired_class_matched.csv", index=False)
    # Each public score is paired to a port under the identical fold model and class set.
    pair.groupby(["public_target", "coverage", "scope", "kind", "head", "contrast"])[
        ["classes", "port_risk_gain", "public_risk_gain", "port_delta_ba", "public_delta_ba"]].mean().reset_index().to_csv(
            OUT / "paired_class_matched_summary.csv", index=False)
    write(OUT / "READOUT_COMPLETE.json", dict(targets=38, ports=23, folds=5, draws=5,
        metric_rows=len(table), sklearn_max_error=error, class_sum_identity=True,
        public_aggregate="Equal five source-fold weights", port_aggregate="Equal23 port weights, then equal5draws",
        class_match="Equal23 port-paired weights; each public target weighted over the same query-present classes as its paired port",
        support_match="Restrict to classes present in BOTH supports; not a matched support-count experiment"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
