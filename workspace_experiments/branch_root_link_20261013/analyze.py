"""Matched branch utility, repair origins and finite-replacement controls."""

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
RC = OUT.parent / "ports_root_cause_20261013"
sys.path.insert(0, str(RC))
import settings as cfg
import analyze_factorial as previous
import numpy as np
import pandas as pd
from report import metrics
from threadpoolctl import threadpool_limits

HEADS = ("expert", "c4_late")
FRESH_PAIRS = (("T", "R"), ("DD", "T"), ("DD", "R"))
REPLACEMENT_PAIRS = FRESH_PAIRS + (("P", "R"), ("T", "P"), ("E", "R"),
                                  ("PE", "P"), ("T", "PE_scaled"))
TRANSITIONS = (("OD", "OO"), ("DO", "OO"), ("DD", "OD"), ("DD", "OO"),
               ("range_geometry", "OO"), ("axis_sham", "OO"),
               ("range_geometry", "axis_sham"))


def weights(y):
    count = np.bincount(y, minlength=8)
    return 100. / (np.count_nonzero(count) * count[y])


def effect(y, old, new, pold, pnew, w):
    cb, ca = old == y, new == y
    rescue, harm = ~cb & ca, cb & ~ca
    loss_b = np.sum((pold - np.eye(8)[y])**2, axis=1)
    loss_a = np.sum((pnew - np.eye(8)[y])**2, axis=1)
    return dict(rescue_ba=float(w[rescue].sum()), harm_ba=float(w[harm].sum()),
        delta_ba=float(w @ (ca.astype(float)-cb)), delta_oa=100*float(np.mean(ca.astype(float)-cb)),
        risk_gain_macro=float(w @ (loss_b-loss_a)/100), risk_gain_micro=float(np.mean(loss_b-loss_a)),
        rescue_n=int(rescue.sum()), harm_n=int(harm.sum()))


def cache_fresh(fold, task, source, condition):
    saved, prob = previous.load(fold, task, source, condition)
    return saved, saved["predictions"][:, :, 1:], prob[:, :, 1:], tuple(saved["arms"])


def cache_replacement(fold, condition, kind):
    path = OUT / "replacement_predictions" / condition / f"fold_{fold}/targets/shared_frozen/oss2"
    saved = cfg.archive(path / "predictions.npz")
    ki = list(saved["kinds"]).index(kind)
    prob = np.load(path / "probabilities.npy", mmap_mode="r")
    return saved, saved["predictions"][ki], prob[ki], tuple(saved["arms"])


def masks_for(task, frame, saved, draw):
    scopes = previous.scopes(task, frame, saved, draw)
    if task == "oss2":
        pairs = pd.read_csv(RC / "dataset_audit/oss2_grdh_slc_pairs.csv", keep_default_na=False)
        pairs = pairs[pairs.same_label]
        ix = pd.Index(saved["ids"])
        g, s = ix.get_indexer(pairs.sample_id_grdh), ix.get_indexer(pairs.sample_id_slc)
        assert (g >= 0).all() and (s >= 0).all()
        ev = saved["eval_masks"][draw]
        eligible = ev[g] & ev[s]
        for label, keep in (("paired", eligible), ("paired_clean", eligible &
            (pairs.noise_xml_grdh.astype(str) == "0").to_numpy() &
            (pairs.noise_xml_slc.astype(str) == "0").to_numpy())):
            for product, indices in (("GRDH", g), ("SLC", s)):
                mask = np.zeros(len(ev), bool)
                mask[indices[keep]] = True
                scopes[label+"_"+product] = mask
    return {k: v for k, v in scopes.items() if v.any()}


def summarize(df, groups, name):
    numbers = [k for k in df.select_dtypes(include=np.number).columns if k not in ("fold", "draw")+tuple(groups)]
    task = df.groupby(["fold", "task"]+groups, dropna=False)[numbers].mean().reset_index()
    task.to_csv(OUT / (name+"_tasks.csv"), index=False)
    summary = task.groupby(groups, dropna=False)[numbers].mean().reset_index()
    summary.to_csv(OUT / (name+"_summary.csv"), index=False)
    return summary


def analyze_context(fold, task, series, kind, caches, tables):
    saved = caches["OO"][0]
    arms = caches["OO"][3]
    comparisons = FRESH_PAIRS if series == "fresh" else REPLACEMENT_PAIRS
    for other, pred, prob, names in caches.values():
        assert names == arms
        for key in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(other[key], saved[key])
    yall = cfg.labels(task, saved["ids"])
    frame = previous.metadata(task, saved["ids"])
    for draw in range(5):
        for scope, mask in masks_for(task, frame, saved, draw).items():
            y, w = yall[mask], weights(yall[mask])
            meta = dict(series=series, kind=kind, fold=fold, task=task,
                        dataset="ports3" if task in cfg.PORTS else task, draw=draw, scope=scope)
            for hi, head in enumerate(HEADS):
                m = dict(**meta, head=head)
                current = {}
                for condition, (_, pred, prob, _) in caches.items():
                    values = {}
                    for ai, arm in enumerate(arms):
                        p = np.asarray(prob[draw, ai, hi, mask], float)
                        pp = pred[draw, ai, hi, mask]
                        value = metrics(y, pp, p)
                        tables["scores"].append(dict(**m, condition=condition, arm=arm, **value))
                        values[arm] = (pp, p, value)
                    current[condition] = values
                    for a, b in comparisons:
                        pa, proba, va = values[a]
                        pb, probb, vb = values[b]
                        stats = effect(y, pb, pa, probb, proba, w)
                        np.testing.assert_allclose(stats["delta_ba"], va["ba"]-vb["ba"], atol=1e-10)
                        tables["effects"].append(dict(**m, condition=condition, comparison=a+"-"+b, **stats))
                        for c in np.unique(y):
                            cm = y == c
                            rescue, harm = (pb != y) & (pa == y), (pb == y) & (pa != y)
                            tables["classes"].append(dict(**m, condition=condition, comparison=a+"-"+b,
                                class_id=int(c), n=int(cm.sum()), rescue_ba=float(w[cm & rescue].sum()),
                                harm_ba=float(w[cm & harm].sum()),
                                delta_ba=float(w[cm] @ ((pa[cm] == c).astype(float)-(pb[cm] == c)))))
                for after, before in TRANSITIONS:
                    if after not in caches or before not in caches:
                        continue
                    r0 = current[before]["R"][0] == y
                    r1 = current[after]["R"][0] == y
                    states = dict(lost=r0 & ~r1, gained=~r0 & r1,
                                  stable_correct=r0 & r1, stable_wrong=~r0 & ~r1)
                    transition = after+"-"+before
                    for a, b in comparisons:
                        pa0, ppa0, va0 = current[before][a]
                        pb0, ppb0, vb0 = current[before][b]
                        pa1, ppa1, va1 = current[after][a]
                        pb1, ppb1, vb1 = current[after][b]
                        a0, b0, a1, b1 = pa0 == y, pb0 == y, pa1 == y, pb1 == y
                        oldres, newres = ~b0 & a0, ~b1 & a1
                        oldharm, newharm = b0 & ~a0, b1 & ~a1
                        e0 = effect(y, pb0, pa0, ppb0, ppa0, w)
                        e1 = effect(y, pb1, pa1, ppb1, ppa1, w)
                        delta = e1["delta_ba"]-e0["delta_ba"]
                        tables["interactions"].append(dict(**m, transition=transition, comparison=a+"-"+b,
                            before_gain=e0["delta_ba"], after_gain=e1["delta_ba"], gain_change=delta,
                            baseline_change=vb1["ba"]-vb0["ba"], candidate_change=va1["ba"]-va0["ba"],
                            risk_gain_change=e1["risk_gain_macro"]-e0["risk_gain_macro"],
                            old_rescue_ba=float(w[oldres].sum()),
                            old_rescue_now_base_correct_ba=float(w[oldres & b1].sum()),
                            old_rescue_now_both_correct_ba=float(w[oldres & b1 & a1].sum()),
                            old_rescue_now_only_base_correct_ba=float(w[oldres & b1 & ~a1].sum()),
                            old_rescue_retained_ba=float(w[oldres & ~b1 & a1].sum()),
                            old_rescue_now_both_wrong_ba=float(w[oldres & ~b1 & ~a1].sum())))
                        total = 0.
                        for state, subset in states.items():
                            oldnet = float(w[subset] @ (a0[subset].astype(float)-b0[subset]))
                            newnet = float(w[subset] @ (a1[subset].astype(float)-b1[subset]))
                            total += newnet-oldnet
                            tables["flows"].append(dict(**m, transition=transition, comparison=a+"-"+b,
                                state=state, n=int(subset.sum()), mass_ba=float(w[subset].sum()),
                                before_rescue=float(w[subset & oldres].sum()), before_harm=float(w[subset & oldharm].sum()),
                                after_rescue=float(w[subset & newres].sum()), after_harm=float(w[subset & newharm].sum()),
                                before_gain=oldnet, after_gain=newnet, gain_change=newnet-oldnet))
                        np.testing.assert_allclose(total, delta, atol=1e-10)
                        if (a, b) == ("T", "R") and head == "expert" and scope in ("original", "query_SLC") and (
                            (series == "fresh" and transition in ("DD-OO", "DD-OD", "range_geometry-OO")) or
                            (series == "archive_replacement" and kind == "mlp" and transition == "range_geometry-OO")):
                            rows = pd.DataFrame(dict(sample_id=saved["ids"][mask], class_id=y,
                                weight_ba=w, r_before=r0, r_after=r1, t_before=a0, t_after=a1))
                            for key, value in dict(**m, transition=transition).items():
                                rows[key] = value
                            tables["events"].append(rows)
    print("ANALYZED", series, kind, fold, task, flush=True)


def main():
    assert (OUT / "REPLACEMENT_PREDICTIONS_FROZEN.json").exists()
    tables = {k: [] for k in ("scores", "effects", "classes", "interactions", "flows", "events")}
    for fold in cfg.FOLDS:
        for task in cfg.pilot.tasks(fold):
            caches = {cell: cache_fresh(fold, task, *pair) for cell, pair in previous.CELLS.items()}
            if task == "oss2":
                for variant in ("range_geometry", "axis_sham"):
                    caches[variant] = cache_fresh(fold, task, "original", variant)
            analyze_context(fold, task, "fresh", "none", caches, tables)
        for kind in ("linear", "mlp"):
            caches = {("OO" if c == "original" else c): cache_replacement(fold, c, kind)
                      for c in ("original", "range_geometry", "axis_sham")}
            analyze_context(fold, "oss2", "archive_replacement", kind, caches, tables)
    common = ["series", "kind", "dataset", "scope", "head"]
    extra = dict(scores=["condition", "arm"], effects=["condition", "comparison"],
        classes=["condition", "comparison", "class_id"], interactions=["transition", "comparison"],
        flows=["transition", "comparison", "state"])
    counts = {}
    for name in extra:
        df = pd.DataFrame(tables[name])
        df.to_csv(OUT / (name+".csv"), index=False)
        summarize(df, common+extra[name], name)
        counts[name] = len(df)
        if name == "effects":
            spread = df.groupby(common+extra[name])["delta_ba"].agg(
                contexts="size", positive=lambda a: int((a>1e-10).sum()), negative=lambda a: int((a < -1e-10).sum()),
                minimum="min", maximum="max", descriptive_std="std").reset_index()
            spread.to_csv(OUT / "support_spread.csv", index=False)
    events = pd.concat(tables["events"], ignore_index=True)
    events.to_csv(OUT / "sample_events.csv", index=False)
    counts["event_rows_with_repeated_draws"] = len(events)
    cfg.write(OUT / "ANALYSIS_COMPLETE.json", dict(counts=counts, source_series_separate=True,
        exact_support_query_pairing=True, flow_sum_checks=True, training_seeds=0))
    print(pd.read_csv(OUT / "effects_summary.csv").query(
        "series == 'fresh' and scope == 'original' and head == 'expert' and comparison == 'T-R'").to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
