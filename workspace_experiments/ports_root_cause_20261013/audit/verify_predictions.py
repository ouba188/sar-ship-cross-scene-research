"""Independent saved-prediction audit. Does not import experiment modules or fit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
WORK = ROOT.parent
INPUT = WORK / "ports_input_controls_20261012"
SOURCE = WORK / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916"
SENSITIVITY = Path("E:/Hermes/paper_tracking/sensitivity")
ALL_TASKS = {2: ("Los Angeles", "oss1", "oss2", "oss2_grdh"),
             3: ("Port Klang", "Santos", "oss1", "oss2", "oss2_grdh")}
TASKS = {2: ("Los Angeles", "oss1", "oss2"), 3: ("Port Klang", "oss1", "oss2")}
ARMS = ("R", "T", "DD")
HEADS = ("prior", "expert", "c4_late")
CONDITIONS = ("original", "display")
METRICS = ("oa", "ba", "brier_micro", "brier_macro", "nll_micro", "nll_macro")
KEYS = ["fold", "task", "dataset", "draw", "scope", "arm", "head"]


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def folder(source, target, fold, task):
    return ROOT / "predictions" / source / target / f"fold_{fold}" / task


def read_context(source, target, fold, task):
    path = folder(source, target, fold, task)
    if not (path / "COMPLETE.json").is_file():
        raise RuntimeError(f"Unfinished prediction context: {path}")
    saved = archive(path / "predictions.npz")
    np.testing.assert_array_equal(saved["arms"], ARMS)
    np.testing.assert_array_equal(saved["heads"], HEADS)
    assert len(np.unique(saved["ids"])) == len(saved["ids"])
    for draw, support in enumerate(saved["support"]):
        assert len(np.unique(support)) == 64
        assert not saved["eval_masks"][draw, support].any()
    probability = np.load(path / "probabilities.npy", mmap_mode="r")
    assert probability.shape == (5, 3, 3, len(saved["ids"]), 8)
    np.testing.assert_array_equal(probability[:, 0, 0], probability[:, 1, 0])
    for draw in range(1, 5):
        np.testing.assert_array_equal(probability[draw, :, 0], probability[0, :, 0])
    return saved, probability


def replay():
    rows = []
    for fold, tasks in ALL_TASKS.items():
        for task in tasks:
            for target in CONDITIONS:
                new, prob = read_context("archive", target, fold, task)
                oldpath = INPUT / "predictions" / target / f"fold_{fold}/targets/shared_frozen" / task
                old = archive(oldpath / "predictions.npz")
                oldprob = np.load(oldpath / "probabilities.npy", mmap_mode="r")
                for key in ("ids", "support", "eval_masks"):
                    np.testing.assert_array_equal(new[key], old[key])
                for ai, arm in enumerate(ARMS):
                    oldai = list(old["arms"]).index(arm)
                    for hi in (1, 2):
                        np.testing.assert_array_equal(new["predictions"][:, ai, hi],
                                                      old["predictions"][1, :, oldai, hi-1])
                        error = float(np.max(np.abs(prob[:, ai, hi] - oldprob[1, :, oldai, hi-1])))
                        assert error <= 2e-7, (fold, task, target, arm, hi, error)
                        rows.append(dict(fold=fold, task=task, target_condition=target,
                                         arm=arm, head=HEADS[hi], max_probability_error=error,
                                         predictions_equal=True))
    pd.DataFrame(rows).to_csv(OUT / "independent_replay.csv", index=False)
    result = dict(status="passed", contexts=18, comparisons=len(rows),
                  max_probability_error=max(row["max_probability_error"] for row in rows),
                  prediction_argmax_equal=True, prior_expert_R_T_equal=True)
    (OUT / "REPLAY_AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def metric_values(y, prediction, probability):
    counts = np.bincount(y, minlength=8)
    present = counts > 0
    confusion = np.bincount(8*y + prediction, minlength=64).reshape(8, 8)
    recalls = np.diag(confusion)[present] / counts[present]
    correct_probability = probability[np.arange(len(y)), y]
    loss = (probability*probability).sum(axis=1) - 2*correct_probability + 1
    nll = -np.log(np.clip(correct_probability, 1e-15, 1))
    w = 1 / (present.sum()*counts[y])
    return dict(n=len(y), oa=100*np.trace(confusion)/len(y), ba=100*recalls.mean(),
                brier_micro=loss.mean(), brier_macro=w@loss,
                nll_micro=nll.mean(), nll_macro=w@nll), loss, w


def evaluate_all():
    if not (ROOT / "PREDICTIONS_FROZEN.json").is_file():
        raise RuntimeError("Wait for all four-cell predictions to freeze before reading query labels")
    manifests = {"ports": pd.read_csv(SOURCE / "manifest.csv", usecols=["instance_id", "class_id"])
                 .rename(columns={"instance_id": "sample_id"}).set_index("sample_id")}
    for task in ("oss1", "oss2"):
        manifests[task] = pd.read_csv(SENSITIVITY / f"opensarship{task[-1]}_transfer_20261007/target_manifest.csv",
                                      usecols=["sample_id", "class_id"]).set_index("sample_id")
    records, classes, event_rows = [], [], []
    max_sum_error = 0.
    rounded_tie_disagreements = 0
    for fold, tasks in TASKS.items():
        for task in tasks:
            values = {}
            baseline = archive(folder("archive", "original", fold, task) / "predictions.npz")
            manifest = manifests[task.split("_")[0] if task.startswith("oss") else "ports"]
            yall = manifest.loc[baseline["ids"], "class_id"].to_numpy(int)
            dataset = task if task.startswith("oss") else "ports3"
            for source in ("archive", *CONDITIONS):
                for target in CONDITIONS:
                    saved, prob = read_context(source, target, fold, task)
                    for key in ("ids", "support", "eval_masks"):
                        np.testing.assert_array_equal(saved[key], baseline[key])
                    values[source, target] = saved["predictions"]
                    for draw, mask in enumerate(saved["eval_masks"]):
                        for scope, query in (("original", mask), ("common_query", saved["eval_masks"].all(axis=0))):
                            y = yall[query]
                            for ai, arm in enumerate(ARMS):
                                for hi, head in enumerate(HEADS):
                                    p = np.asarray(prob[draw, ai, hi, query], float)
                                    pred = saved["predictions"][draw, ai, hi, query]
                                    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                                    max_sum_error = max(max_sum_error, float(np.abs(p.sum(axis=1)-1).max()))
                                    assert max_sum_error <= 1e-7
                                    mismatch = pred != p.argmax(axis=1)
                                    assert np.all(p[np.arange(len(y)), pred][mismatch] == p.max(axis=1)[mismatch])
                                    rounded_tie_disagreements += int(mismatch.sum())
                                    stats, loss, _ = metric_values(y, pred, p)
                                    meta = dict(fold=fold, task=task, dataset=dataset, source_condition=source,
                                                target_condition=target, draw=draw, scope=scope, arm=arm, head=head)
                                    records.append(dict(**meta, **stats))
                                    for cls in np.unique(y):
                                        ix = y == cls
                                        classes.append(dict(**meta, class_id=int(cls), n=int(ix.sum()),
                                                            recall=100*np.mean(pred[ix] == cls), brier=loss[ix].mean()))
            # Direct event-level reconstruction of the two main metric differences.
            for draw, mask in enumerate(baseline["eval_masks"]):
                for scope, query in (("original", mask), ("common_query", baseline["eval_masks"].all(axis=0))):
                    y = yall[query]
                    counts = np.bincount(y, minlength=8)
                    w = 1/(np.count_nonzero(counts)*counts[y])
                    for ai, arm in enumerate(ARMS):
                        for hi, head in enumerate(HEADS):
                            end = values["display", "display"][draw, ai, hi, query] == y
                            for contrast, cell in (("DD-OD", ("original", "display")),
                                                   ("DD-OO", ("original", "original"))):
                                start = values[cell][draw, ai, hi, query] == y
                                rescue, harm = (~start & end), (start & ~end)
                                event_rows.append(dict(fold=fold, task=task, dataset=dataset, draw=draw, scope=scope,
                                                       arm=arm, head=head, contrast=contrast,
                                                       rescue_ba_pp=100*(w@rescue), harm_ba_pp=100*(w@harm),
                                                       direct_delta_ba=100*(w@(end.astype(int)-start))))
    table = pd.DataFrame(records)
    table.to_csv(OUT / "independent_metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(OUT / "independent_classes.csv", index=False)
    group = ["source_condition", "target_condition", "dataset", "scope", "arm", "head"]
    table.groupby(["fold", "task"] + group)[list(METRICS)].mean().groupby(group).mean().reset_index().to_csv(
        OUT / "independent_summary.csv", index=False)
    cells = {source[0].upper() + target[0].upper():
             table[(table.source_condition == source) & (table.target_condition == target)]
             .set_index(KEYS)[list(METRICS)] for source in CONDITIONS for target in CONDITIONS}
    deltas = {"DD-OD": cells["DD"]-cells["OD"], "DD-OO": cells["DD"]-cells["OO"],
              "interaction": cells["DD"]-cells["DO"]-cells["OD"]+cells["OO"],
              "OD-OO": cells["OD"]-cells["OO"], "DO-OO": cells["DO"]-cells["OO"],
              "DD-DO": cells["DD"]-cells["DO"]}
    contrasts = pd.concat([value.reset_index().assign(contrast=key) for key, value in deltas.items()], ignore_index=True)
    contrasts = contrasts.rename(columns={key: "delta_"+key for key in METRICS})
    contrasts.to_csv(OUT / "independent_contrasts.csv", index=False)
    measures = ["delta_"+key for key in METRICS]
    group = ["dataset", "scope", "arm", "head", "contrast"]
    contrasts.groupby(["fold", "task"]+group)[measures].mean().groupby(group).mean().reset_index().to_csv(
        OUT / "independent_contrast_summary.csv", index=False)
    events = pd.DataFrame(event_rows)
    check = events.merge(contrasts, on=KEYS+["contrast"], validate="one_to_one")
    np.testing.assert_allclose(check.direct_delta_ba, check.delta_ba, atol=1e-10, rtol=0)
    np.testing.assert_allclose(check.rescue_ba_pp-check.harm_ba_pp, check.delta_ba, atol=1e-10, rtol=0)
    events.to_csv(OUT / "independent_rescue_harm.csv", index=False)
    fresh = table[table.source_condition == "original"].set_index(KEYS+["target_condition"])[list(METRICS)]
    old = table[table.source_condition == "archive"].set_index(KEYS+["target_condition"])[list(METRICS)]
    (fresh-old).reset_index().to_csv(OUT / "independent_original_fit_drift.csv", index=False)
    return dict(status="passed", prediction_contexts=36, factorial_contexts=24, selected_tasks=TASKS, metric_rows=len(table),
                contrast_rows=len(contrasts), rescue_harm_rows=len(events), max_probability_sum_error=max_sum_error,
                rounded_probability_tie_disagreements=rounded_tie_disagreements,
                brier_difference_sign="positive delta is worse", query_labels_used_only_after_prediction_freeze=True,
                independent_of_main_analysis=True, main_csv_comparison="not yet performed")


def compare_main():
    if not (ROOT / "READOUT_COMPLETE.json").is_file():
        raise RuntimeError("Wait for main readout to complete")
    reports = []

    def compare(left, right, keys, measures, name):
        right = right[right.scope.isin(["original", "common_query"])].copy()
        right = right[[task in TASKS.get(fold, ()) for fold, task in zip(right.fold, right.task)]]
        merged = left.merge(right, on=keys, how="outer", suffixes=("_audit", "_main"),
                            validate="one_to_one", indicator=True)
        assert merged._merge.eq("both").all(), (name, merged._merge.value_counts().to_dict())
        errors = {}
        for metric in measures:
            a, b = merged[metric+"_audit"].to_numpy(), merged[metric+"_main"].to_numpy()
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=0, err_msg=name+":"+metric)
            errors[metric] = float(np.max(np.abs(a-b)))
        reports.append(dict(table=name, rows=len(merged), max_errors=errors))

    metrics = pd.read_csv(OUT / "independent_metrics.csv")
    fresh = metrics[metrics.source_condition != "archive"].copy()
    fresh["cell"] = fresh.source_condition.str[0].str.upper()+fresh.target_condition.str[0].str.upper()
    compare(fresh, pd.read_csv(ROOT / "metrics.csv"), KEYS+["cell"], ["n", *METRICS], "metrics")
    cls = pd.read_csv(OUT / "independent_classes.csv")
    cls = cls[cls.source_condition != "archive"].copy()
    cls["cell"] = cls.source_condition.str[0].str.upper()+cls.target_condition.str[0].str.upper()
    compare(cls, pd.read_csv(ROOT / "classes.csv"), KEYS+["cell", "class_id"], ["n", "recall", "brier"], "classes")
    rename = {"DD-OD": "source_coordination", "DD-OO": "matched_net", "OD-OO": "target_only",
              "DO-OO": "source_only", "DD-DO": "target_after_source", "interaction": "interaction"}
    effects = pd.read_csv(OUT / "independent_contrasts.csv")
    effects["contrast"] = effects.contrast.map(rename)
    effects["risk_gain_macro"] = -effects.delta_brier_macro
    effects["risk_gain_micro"] = -effects.delta_brier_micro
    primary = pd.concat([pd.read_csv(ROOT / "effects.csv"), pd.read_csv(ROOT / "interaction.csv")])
    measures = ["delta_ba", "delta_oa", "risk_gain_macro", "risk_gain_micro"]
    compare(effects, primary, KEYS+["contrast"], measures, "effects_and_interaction")
    grouping = ["fold", "task", "dataset", "scope", "arm", "head", "cell"]
    summary = fresh.groupby(grouping)[list(METRICS)].mean().reset_index()
    compare(summary, pd.read_csv(ROOT / "task_cells.csv"), grouping,
            ["oa", "ba", "brier_macro", "brier_micro", "nll_macro"], "task_cells")
    grouping[-1] = "contrast"
    es = effects.groupby(grouping)[measures].mean().reset_index()
    compare(es, pd.read_csv(ROOT / "task_effects.csv"), grouping, measures, "task_effects")
    flows = pd.read_csv(OUT / "independent_rescue_harm.csv").rename(
        columns={"rescue_ba_pp": "rescue_ba", "harm_ba_pp": "harm_ba", "direct_delta_ba": "delta_ba"})
    flows["contrast"] = flows.contrast.map(rename)
    compare(flows, primary[primary.contrast.isin(["source_coordination", "matched_net"])],
            KEYS+["contrast"], ["rescue_ba", "harm_ba", "delta_ba"], "rescue_harm")
    drift = pd.read_csv(OUT / "independent_original_fit_drift.csv")
    drift = drift[drift.scope == "original"].rename(
        columns={"ba": "delta_ba", "oa": "delta_oa", "target_condition": "condition"})
    compare(drift, pd.read_csv(ROOT / "source_cache_drift.csv"), KEYS+["condition"],
            ["delta_ba", "delta_oa"], "source_cache_drift")
    result = dict(status="passed", scope="Selected24 factorial contexts, all arms/heads/draws; original and common_query scopes",
                  selected_tasks=TASKS,
                  tables=reports, additional_identity_product_sensitivities="not independently recalculated")
    (OUT / "MAIN_COMPARISON.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    numeric = json.loads((OUT / "NUMERICAL_AUDIT.json").read_text(encoding="utf-8"))
    numeric["main_csv_comparison"] = "passed; see MAIN_COMPARISON.json"
    (OUT / "NUMERICAL_AUDIT.json").write_text(json.dumps(numeric, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["replay", "full", "compare"])
    args = parser.parse_args()
    if args.stage == "replay":
        result = replay()
    elif args.stage == "compare":
        result = compare_main()
    else:
        result = evaluate_all()
        (OUT / "NUMERICAL_AUDIT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
