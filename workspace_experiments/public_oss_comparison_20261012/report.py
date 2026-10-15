"""Post-freeze public-target metrics, paired effects and support composition."""

from common import *
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss
from backbone_ablation_report import ba
import warnings

TASKS = ("oss1", "oss2", "oss2_grdh")
HEADS = ("expert", "c4_late")
CONTRASTS = (("P", "R"), ("E", "R"), ("PE", "P"), ("T", "P"), ("T", "PE"),
             ("T", "R"), ("T", "PE_scaled"), ("PE_scaled", "PE"), ("DD", "T"), ("DD", "R"))


def metrics(y, pred, p):
    brier = np.sum((p-np.eye(8)[y])**2, axis=1)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
    return dict(n=len(y), oa=100*np.mean(y == pred), ba=ba(y, pred),
                brier_micro=float(brier.mean()), brier_macro=float(np.mean([brier[y == c].mean() for c in np.unique(y)])),
                nll_micro=float(nll.mean()), nll_macro=float(np.mean([nll[y == c].mean() for c in np.unique(y)])))


def main():
    if not (OUT / "PREDICTIONS_FROZEN.json").exists() or not (OUT / "GRDH_PREDICTIONS_FROZEN.json").exists():
        raise RuntimeError("Finish all predefined target predictions first")
    records, effects, classes, features, supports, counts = [], [], [], [], [], []
    max_metric_error = max_probability_sum_error = 0.
    for regime in REGIMES:
        for task in TASKS:
            name = task.split("_")[0]
            dst = OUT / "targets" / regime / task
            complete = json.loads((dst / "COMPLETE.json").read_text())
            saved = archive(dst / "predictions.npz")
            probability = np.load(dst / "probabilities.npy", mmap_mode="r")
            frame = pd.read_csv(original(name) / "target_manifest.csv", keep_default_na=False)
            frame = frame.set_index("sample_id").loc[saved["ids"]].reset_index()
            yall = frame.class_id.to_numpy(int)
            common = dict(regime=regime, dataset=task)
            features.extend(dict(**common, **d) for d in complete["feature_diagnostics"])
            if regime == REGIMES[0]:
                for c in range(8):
                    counts.append(dict(dataset=task, class_id=c, class_name=frame.loc[frame.class_id == c, "class_name"].iloc[0],
                                       n=int(np.sum(yall == c))))
            for draw, ev in enumerate(saved["eval_masks"]):
                sc = np.bincount(yall[saved["support"][draw]], minlength=8)
                if regime == REGIMES[0]:
                    sf = frame.iloc[saved["support"][draw]]
                    qf = frame.iloc[np.flatnonzero(ev)]
                    known = sf.mmsi_or_ship_id.astype(str)
                    known = known[~known.isin(["", "0", "nan", "None"])]
                    supports.append(dict(dataset=task, draw=draw, query_n=int(ev.sum()),
                        common_query_n=int(saved["common_query"].sum()), classes=int(np.sum(sc>0)),
                        distinct_known_support_vessels=int(known.nunique()),
                        query_same_known_vessel=int(qf.mmsi_or_ship_id.astype(str).isin(known).sum()),
                        query_same_product=int(qf.scene_id.isin(sf.scene_id).sum()),
                        **{f"class_{c}": int(sc[c]) for c in range(8)}))
                for scope, mask in (("original", ev), ("common_query", saved["common_query"])):
                    y = yall[mask]
                    for ki, kind in enumerate(KINDS):
                        for hi, head in enumerate(HEADS):
                            byarm = {}
                            for ai, arm in enumerate(ARMS):
                                p = np.asarray(probability[ki, draw, ai, hi, mask], float)
                                pred = saved["predictions"][ki, draw, ai, hi, mask]
                                deviation = float(np.max(np.abs(p.sum(1)-1)))
                                max_probability_sum_error = max(max_probability_sum_error, deviation)
                                np.testing.assert_allclose(p.sum(1), 1., atol=1e-7, rtol=0.)
                                row = dict(**common, scope=scope, kind=kind, draw=draw, head=head, arm=arm,
                                           **metrics(y, pred, p))
                                records.append(row)
                                byarm[arm] = (row, pred)
                                loss = np.sum((p-np.eye(8)[y])**2, axis=1)
                                for c in np.unique(y):
                                    cm = y == c
                                    classes.append(dict(**common, scope=scope, kind=kind, draw=draw, head=head, arm=arm,
                                        class_id=int(c), n=int(cm.sum()), support_n=int(sc[c]),
                                        recall=100*np.mean(pred[cm] == c), brier=float(loss[cm].mean())))
                                if draw == 0 and scope == "original":
                                    _, inv, ns = np.unique(y, return_inverse=True, return_counts=True)
                                    w = 1./ns[inv]
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore", message="The y_prob values do not sum to one.*")
                                        independent = dict(ba=100*balanced_accuracy_score(y, pred),
                                            brier_micro=brier_score_loss(y, p, labels=np.arange(8), scale_by_half=False),
                                            brier_macro=brier_score_loss(y, p, labels=np.arange(8), scale_by_half=False, sample_weight=w),
                                            nll_micro=log_loss(y, p, labels=np.arange(8)),
                                            nll_macro=log_loss(y, p, labels=np.arange(8), sample_weight=w))
                                    for key, value in independent.items():
                                        max_metric_error = max(max_metric_error, abs(value-row[key]))
                                        np.testing.assert_allclose(value, row[key], atol=1e-10, rtol=1e-10)
                            for candidate, control in CONTRASTS:
                                a, pa = byarm[candidate]
                                b, pb = byarm[control]
                                effects.append(dict(**common, scope=scope, kind=kind, draw=draw, head=head,
                                    contrast=candidate+"-"+control, delta_ba=a["ba"]-b["ba"], delta_oa=a["oa"]-b["oa"],
                                    risk_gain_macro=b["brier_macro"]-a["brier_macro"],
                                    risk_gain_micro=b["brier_micro"]-a["brier_micro"],
                                    rescues=int(np.sum((pa == y)&(pb != y))), harms=int(np.sum((pa != y)&(pb == y)))))
    table = pd.DataFrame(records)
    effect = pd.DataFrame(effects)
    table.to_csv(OUT / "metrics.csv", index=False)
    effect.to_csv(OUT / "effects.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "classes.csv", index=False)
    pd.DataFrame(features).to_csv(OUT / "feature_diagnostics.csv", index=False)
    pd.DataFrame(supports).to_csv(OUT / "support_profile.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "class_counts.csv", index=False)
    keys = ["regime", "dataset", "scope", "kind", "head", "arm"]
    summary = table.groupby(keys)[["oa", "ba", "brier_micro", "brier_macro", "nll_micro", "nll_macro"]].mean().reset_index()
    summary.to_csv(OUT / "summary.csv", index=False)
    spread = table.groupby(keys).agg(ba_mean=("ba", "mean"), ba_std=("ba", "std"),
        ba_min=("ba", "min"), ba_max=("ba", "max"), brier_macro_std=("brier_macro", "std")).reset_index()
    spread.to_csv(OUT / "support_spread.csv", index=False)
    effects_summary = effect.groupby(["regime", "dataset", "scope", "kind", "head", "contrast"]).agg(
        delta_ba=("delta_ba", "mean"), delta_oa=("delta_oa", "mean"),
        risk_gain_macro=("risk_gain_macro", "mean"), risk_gain_micro=("risk_gain_micro", "mean"),
        ba_wins=("delta_ba", lambda s: int((s>0).sum())), risk_wins=("risk_gain_macro", lambda s: int((s>0).sum())),
        ba_min=("delta_ba", "min"), ba_max=("delta_ba", "max"),
        rescues=("rescues", "sum"), harms=("harms", "sum")).reset_index()
    effects_summary.to_csv(OUT / "effect_summary.csv", index=False)
    old = pd.read_csv(OUT.parent / "dual_cue_decomposition_20261012/metrics.csv")
    new = table[(table.regime == "task_trained") & (table.dataset == "oss2") & (table.scope == "original") & (table["head"] == "expert") & table.arm.isin(["R", "P", "E", "PE", "T"])]
    old = old[(old.dataset == "oss2") & (old["head"] == "expert")]
    paired = new.merge(old, on=["dataset", "kind", "draw", "head", "arm"], suffixes=("_new", "_old"), validate="one_to_one")
    for key in ("oa", "ba", "brier_micro", "brier_macro"):
        np.testing.assert_allclose(paired[key+"_new"], paired[key+"_old"], atol=1e-8, rtol=1e-8)
    write(OUT / "COMPLETE.json", dict(regimes=2, targets_per_regime=3, draws_per_target=5,
        support_budget=64, metric_rows=len(table), all_predictions_frozen_first=True,
        sklearn_metric_max_error=max_metric_error, probability_sum_max_error=max_probability_sum_error,
        old_oss2_all5expert_arms_replay=True, backbone_retraining=False,
        new_mapper_fits="Two linear plus two8epoch MLP fits; OSS2 task mapper reused", no_target_selection=True))
    print(effects_summary[(effects_summary.scope == "original") & (effects_summary["head"] == "expert")
        & effects_summary.contrast.isin(["P-R", "E-R", "PE-P", "T-PE", "T-R", "DD-R"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
