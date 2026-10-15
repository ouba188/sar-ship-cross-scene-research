"""Describe frozen class/draw gains; no model import, fitting, or selection.

Run with the ClearSAR interpreter. All outputs stay beside this script.
IDs in the prediction archives are positional target-manifest indices.
"""

import os

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
DEFAULT_BUDGET = OUT.parents[1] / "frozen_candidate_coverage_20261012" / "budget"
DATASETS = ("f0921", "oss2")
BUDGETS = (16, 64, 128)
LAYERS = ("fusion", "expert")
PAIRS = (("initial", "raw"), ("simple", "raw"), ("initial", "simple"))
KEYS = ["dataset", "task", "budget", "draw"]
GROUP = ["dataset", "budget", "layer", "arm", "reference"]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)


def correlation(x, y):
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    return float(np.corrcoef(x, y)[0, 1]), float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def main(budget_root):
    # Reuse the existing evaluations; NPZ is needed only for paired identities.
    tables = {}
    for name in ("classes", "supports", "paired", "metrics"):
        table = pd.read_csv(budget_root / (name + ".csv"))
        table = table[table.dataset.isin(DATASETS)]
        if "arm" in table:
            table = table[table.arm.isin(("raw", "initial", "simple"))]
        tables[name] = table
    classes = tables["classes"].set_index(KEYS + ["layer", "arm", "class_id"])
    supports = tables["supports"].set_index(KEYS)
    metrics = tables["metrics"].set_index(KEYS + ["layer", "arm"])
    paired = tables["paired"].set_index(KEYS + ["layer", "arm", "reference"])
    for table in (classes, supports, metrics, paired):
        assert table.index.is_unique

    class_rows, draw_rows, support_rows, receipts = [], [], [], []
    checks = dict(prediction_units=0, class_recall_replays=0, metric_replays=0,
                  paired_replays=0, manifest_support_matches=0)
    for dataset in DATASETS:
        folder = budget_root / "strong_external" / dataset / dataset
        receipt_path = folder / "PREDICTIONS_FROZEN.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        manifest = pd.read_csv(receipt["target_manifest"], usecols=["class_id", "class_name"])
        class_names = manifest.drop_duplicates(["class_id", "class_name"])
        assert class_names.class_id.is_unique
        names = class_names.set_index("class_id").class_name.to_dict()
        reference_query = None
        receipts.append(dict(dataset=dataset, receipt=str(receipt_path),
                             target_manifest=receipt["target_manifest"],
                             baseline=receipt["baseline"], query_n=receipt["query_n"]))
        for budget in BUDGETS:
            for draw in range(5):
                key = (dataset, dataset, budget, draw)
                ident = dict(zip(KEYS, key))
                with np.load(folder / f"{draw}_{budget}.npz", allow_pickle=False) as archive:
                    query_ids, support_ids = archive["query_ids"], archive["support_ids"]
                    if reference_query is None:
                        reference_query = query_ids.copy()
                        assert len(np.unique(query_ids)) == len(query_ids)
                    else:
                        np.testing.assert_array_equal(query_ids, reference_query)
                    assert len(query_ids) == receipt["query_n"]
                    assert len(np.unique(support_ids)) == budget
                    assert np.intersect1d(query_ids, support_ids).size == 0
                    y = manifest.iloc[query_ids].class_id.to_numpy(int)
                    sy = manifest.iloc[support_ids].class_id.to_numpy(int)
                    np.testing.assert_array_equal(sy, archive["support_labels"])
                    checks["manifest_support_matches"] += 1
                    cids, ns = np.unique(y, return_counts=True)
                    support_counts = {int(c): int((sy == c).sum()) for c in cids}
                    existing_support = supports.loc[key]
                    saved_counts = {int(c): n for c, n in json.loads(existing_support.class_counts).items()}
                    assert {c: n for c, n in support_counts.items() if n} == saved_counts
                    assert existing_support.n_query == len(y)
                    covered = sum(n > 0 for n in support_counts.values())
                    srow = dict(**ident, n_query=len(y), covered_classes=covered,
                                missing_classes=";".join(names[c] for c in cids if not support_counts[c]),
                                ship_field=existing_support.ship_field,
                                known_support_ships=int(existing_support.known_support_ships),
                                query_support_ship_overlap=float(existing_support.query_support_ship_overlap))
                    srow.update({"support_n_" + names[c]: support_counts[c] for c in cids})
                    support_rows.append(srow)
                    for layer in LAYERS:
                        predictions = {a: archive[f"{layer}_{a}"].argmax(1) for a in ("raw", "initial", "simple")}
                        correct = {a: pred == y for a, pred in predictions.items()}
                        for arm, values in correct.items():
                            old_metric = metrics.loc[key + (layer, arm)]
                            recalls = []
                            for c, n in zip(cids, ns):
                                old = classes.loc[key + (layer, arm, c)]
                                assert int(old.n) == n and int(old.support_n) == support_counts[c]
                                recall = 100 * values[y == c].mean()
                                close(recall, old.recall)
                                recalls.append(recall)
                                checks["class_recall_replays"] += 1
                            close([100 * values.mean(), np.mean(recalls)], [old_metric.oa, old_metric.ba])
                            checks["metric_replays"] += 1
                        for arm, reference in PAIRS:
                            a, r = correct[arm], correct[reference]
                            rescued, harmed = ~r & a, r & ~a
                            local = []
                            for c, n in zip(cids, ns):
                                mask = y == c
                                rescue_n, harm_n = int(rescued[mask].sum()), int(harmed[mask].sum())
                                net = rescue_n - harm_n
                                ref_n, arm_n = int(r[mask].sum()), int(a[mask].sum())
                                assert net == arm_n - ref_n
                                old_a = classes.loc[key + (layer, arm, c)]
                                old_r = classes.loc[key + (layer, reference, c)]
                                row = dict(**ident, layer=layer, arm=arm, reference=reference,
                                    class_id=int(c), class_name=names[c], n_query=int(n),
                                    support_n=support_counts[c], covered_classes=covered,
                                    reference_correct_n=ref_n, arm_correct_n=arm_n,
                                    rescue_n=rescue_n, harm_n=harm_n, net_n=net,
                                    reference_recall_pp=100 * ref_n / n, arm_recall_pp=100 * arm_n / n,
                                    delta_recall_pp=100 * net / n,
                                    rescue_ba_pp=100 * rescue_n / n / len(cids),
                                    harm_ba_pp=100 * harm_n / n / len(cids),
                                    delta_ba_pp=100 * net / n / len(cids),
                                    delta_oa_pp=100 * net / len(y),
                                    class_brier_gain=float(old_r.brier - old_a.brier))
                                local.append(row)
                                class_rows.append(row)
                            old_a = metrics.loc[key + (layer, arm)]
                            old_r = metrics.loc[key + (layer, reference)]
                            drow = dict(**ident, layer=layer, arm=arm, reference=reference,
                                n_query=len(y), covered_classes=covered,
                                rescue_n=int(rescued.sum()), harm_n=int(harmed.sum()),
                                net_n=int(rescued.sum() - harmed.sum()),
                                reference_oa_pp=float(old_r.oa), arm_oa_pp=float(old_a.oa),
                                reference_ba_pp=float(old_r.ba), arm_ba_pp=float(old_a.ba),
                                delta_oa_pp=100 * (a.mean() - r.mean()),
                                delta_ba_pp=sum(t["delta_ba_pp"] for t in local),
                                rescue_ba_pp=sum(t["rescue_ba_pp"] for t in local),
                                harm_ba_pp=sum(t["harm_ba_pp"] for t in local),
                                class_brier_gain=float(old_r.class_brier - old_a.class_brier),
                                micro_brier_gain=float(old_r.brier - old_a.brier))
                            close(drow["delta_ba_pp"], old_a.ba - old_r.ba)
                            close(sum(t["delta_oa_pp"] for t in local), drow["delta_oa_pp"])
                            if reference == "raw":
                                old = paired.loc[key + (layer, arm, reference)]
                                close([drow[k] for k in ("delta_ba_pp", "delta_oa_pp", "rescue_ba_pp", "harm_ba_pp")],
                                      [old[k] for k in ("delta_ba", "delta_oa", "rescue_ba", "harm_ba")])
                                checks["paired_replays"] += 1
                            draw_rows.append(drow)
                checks["prediction_units"] += 1
            print(f"{dataset} B{budget}: paired class/draw attribution complete", flush=True)

    cf, df, sf = pd.DataFrame(class_rows), pd.DataFrame(draw_rows), pd.DataFrame(support_rows)
    draw_groups = df.groupby(GROUP)["delta_ba_pp"]
    draw_count = draw_groups.transform("size")
    df["contribution_to_mean_delta_ba_pp"] = df.delta_ba_pp / draw_count
    df["leave_one_draw_out_mean_delta_ba_pp"] = (draw_groups.transform("sum") - df.delta_ba_pp) / (draw_count - 1)
    summary = df.groupby(GROUP).agg(
        draws=("draw", "size"), n_query=("n_query", "first"),
        rescue_events=("rescue_n", "sum"), harm_events=("harm_n", "sum"), net_events=("net_n", "sum"),
        reference_oa_pp=("reference_oa_pp", "mean"), arm_oa_pp=("arm_oa_pp", "mean"),
        reference_ba_pp=("reference_ba_pp", "mean"), arm_ba_pp=("arm_ba_pp", "mean"),
        delta_oa_pp=("delta_oa_pp", "mean"), delta_ba_pp=("delta_ba_pp", "mean"),
        rescue_ba_pp=("rescue_ba_pp", "mean"), harm_ba_pp=("harm_ba_pp", "mean"),
        sd_delta_ba_pp=("delta_ba_pp", "std"), min_delta_ba_pp=("delta_ba_pp", "min"),
        max_delta_ba_pp=("delta_ba_pp", "max"), positive_draws=("delta_ba_pp", lambda x: int((x > 1e-9).sum())),
        negative_draws=("delta_ba_pp", lambda x: int((x < -1e-9).sum())),
        tie_draws=("delta_ba_pp", lambda x: int((abs(x) <= 1e-9).sum())),
        class_brier_gain=("class_brier_gain", "mean"), micro_brier_gain=("micro_brier_gain", "mean")).reset_index()
    cs = cf.groupby(GROUP + ["class_id", "class_name"]).agg(
        draws=("draw", "size"), n_query=("n_query", "first"),
        mean_support_n=("support_n", "mean"), min_support_n=("support_n", "min"),
        max_support_n=("support_n", "max"), supported_draws=("support_n", lambda x: int((x > 0).sum())),
        rescue_events=("rescue_n", "sum"), harm_events=("harm_n", "sum"), net_events=("net_n", "sum"),
        reference_recall_pp=("reference_recall_pp", "mean"), arm_recall_pp=("arm_recall_pp", "mean"),
        delta_recall_pp=("delta_recall_pp", "mean"), delta_ba_pp=("delta_ba_pp", "mean"),
        delta_oa_pp=("delta_oa_pp", "mean"), rescue_ba_pp=("rescue_ba_pp", "mean"),
        harm_ba_pp=("harm_ba_pp", "mean"), positive_draws=("delta_ba_pp", lambda x: int((x > 1e-9).sum())),
        negative_draws=("delta_ba_pp", lambda x: int((x < -1e-9).sum())),
        class_brier_gain=("class_brier_gain", "mean")).reset_index()

    associations = []
    for key, group in cf.groupby(GROUP + ["class_id", "class_name"]):
        x, y = group.support_n.to_numpy(), group.delta_recall_pp.to_numpy()
        pearson, spearman = correlation(x, y)
        associations.append(dict(zip(GROUP + ["class_id", "class_name"], key)) | dict(
            predictor="own_class_support_n", outcome="delta_recall_pp", n_draws=len(group),
            predictor_unique=len(np.unique(x)), pearson_r=pearson, spearman_r=spearman,
            unsupported_draws=int((x == 0).sum()), supported_draws=int((x > 0).sum()),
            unsupported_mean_delta=float(y[x == 0].mean()) if (x == 0).any() else np.nan,
            supported_mean_delta=float(y[x > 0].mean()) if (x > 0).any() else np.nan))
    for key, group in df.groupby(GROUP):
        x, y = group.covered_classes.to_numpy(), group.delta_ba_pp.to_numpy()
        pearson, spearman = correlation(x, y)
        associations.append(dict(zip(GROUP, key)) | dict(
            class_id=-1, class_name="ALL", predictor="covered_classes", outcome="delta_ba_pp",
            n_draws=len(group), predictor_unique=len(np.unique(x)), pearson_r=pearson, spearman_r=spearman))

    products = dict(class_draw_attribution=cf, draw_attribution=df, class_summary=cs,
                    summary=summary, support_draws=sf, support_associations=pd.DataFrame(associations))
    for name, frame in products.items():
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    report = dict(status="complete", budget_root=str(budget_root),
                  runtime=sys.executable, cpu_threads=2, gpu=False, training=False,
                  id_semantics="Positional target_manifest iloc, not target_manifest_row column",
                  checks=checks, sources=receipts,
                  output_rows={name: len(frame) for name, frame in products.items()},
                  limitations=["Five reused support draws, not independent training seeds",
                               "Events can count the same query on multiple draws",
                               "Correlations are descriptive only; no CI, p-value or causal inference",
                               "Common query cohort and historical row-only exclusion preserved",
                               "Probability risks reused from existing matched CSV evaluations"])
    (OUT / "DIAGNOSTIC_RECEIPT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(summary[summary.layer.eq("fusion")][
        ["dataset", "budget", "arm", "reference", "delta_oa_pp", "delta_ba_pp", "positive_draws", "negative_draws"]].to_string(index=False))
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-root", type=Path, default=DEFAULT_BUDGET)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.budget_root)
