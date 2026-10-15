"""Aggregate paired frozen-feature innovation controls without model selection."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
METRICS = ("chip_oa", "chip_ba", "ship_oa", "ship_ba")


def load(root):
    folders = [p for dataset in ("oss1", "oss2") for p in sorted((root / dataset).iterdir())
               if p.is_dir() and (p / "metrics.csv").exists()]
    metrics = pd.concat([pd.read_csv(p / "metrics.csv") for p in folders], ignore_index=True)
    cases = pd.concat([pd.read_csv(p / "cases.csv", dtype={"mmsi": str}) for p in folders], ignore_index=True)
    diagnostics = pd.DataFrame([dict(dataset=p.parent.name, fold=p.name,
                                     **json.loads((p / "mapper_diagnostics.json").read_text()))
                                for p in folders])
    return metrics, cases, diagnostics


def summarize(root, metrics, cases, diagnostics):
    arms = [arm for arm in ("full", "reconstructed", "innovation", "single_backbone")
            if arm in set(metrics.arm)]
    fold = metrics.groupby(["dataset", "fold", "origin", "arm"], as_index=False)[list(METRICS)].mean()
    summary = fold.groupby(["dataset", "origin", "arm"], as_index=False)[list(METRICS)].mean()
    reference = fold[fold.arm == "base"].set_index(["dataset", "fold", "origin"])
    paired = fold.join(reference[list(METRICS)], on=["dataset", "fold", "origin"], rsuffix="_base")
    for field in METRICS:
        paired[f"d_{field}_vs_base"] = paired[field] - paired[f"{field}_base"]
    paired = paired[paired.arm != "base"].copy()
    contrasts = paired.groupby(["dataset", "origin", "arm"], as_index=False)[
        [f"d_{field}_vs_base" for field in METRICS]].mean()
    full = fold[fold.arm == "full"].set_index(["dataset", "fold", "origin"])
    for field in METRICS:
        paired[f"d_{field}_vs_full"] = paired[field] - full.loc[
            pd.MultiIndex.from_frame(paired[["dataset", "fold", "origin"]]), field].to_numpy()

    keys = ["dataset", "fold", "origin", "draw", "sample_id"]
    predictions = cases.pivot(index=keys, columns="arm", values="prediction").reset_index()
    truth = cases[cases.arm == "base"][keys + ["truth", "mmsi", "true_class_support_count"]]
    predictions = predictions.merge(truth, on=keys, validate="one_to_one")
    for arm in arms:
        predictions[f"{arm}_rescue"] = (predictions.base != predictions.truth) & (
            predictions[arm] == predictions.truth)
        predictions[f"{arm}_harm"] = (predictions.base == predictions.truth) & (
            predictions[arm] != predictions.truth)
    events = predictions.groupby(["dataset", "origin"], as_index=False)[
        [f"{a}_{event}" for a in arms
         for event in ("rescue", "harm")]].sum()
    events["query_exposures"] = predictions.groupby(["dataset", "origin"]).size().to_numpy()
    events["unique_query_ships"] = predictions.groupby(["dataset", "origin"]).mmsi.nunique().to_numpy()

    count = predictions.groupby(["dataset", "fold"], as_index=False).agg(
        query_rows=("sample_id", "nunique"), query_ships=("mmsi", "nunique"),
        query_classes=("truth", "nunique"))
    support = predictions.groupby(["dataset", "fold", "origin"], as_index=False).agg(
        mean_true_class_support=("true_class_support_count", "mean"),
        missing_true_support_fraction=("true_class_support_count", lambda x: float((x == 0).mean())))
    margins = cases.pivot(index=keys, columns="arm", values="true_rival_margin").reset_index()
    margin_summary = []
    for (dataset, origin), group in margins.groupby(["dataset", "origin"]):
        for arm in arms:
            delta = group[arm] - group.base
            margin_summary.append(dict(dataset=dataset, origin=origin, arm=arm,
                                       mean_margin_gain=float(delta.mean()),
                                       gain_fraction=float((delta > 0).mean())))
    class_results = cases.assign(correct=cases.prediction.eq(cases.truth)).groupby(
        ["dataset", "fold", "origin", "truth", "arm"], as_index=False).agg(
            query_exposures=("correct", "size"),
            query_ships=("mmsi", "nunique"), recall=("correct", "mean"))
    class_reference = class_results[class_results.arm == "base"].set_index(
        ["dataset", "fold", "origin", "truth"])
    class_results["d_recall_vs_base"] = class_results.recall - class_reference.loc[
        pd.MultiIndex.from_frame(class_results[["dataset", "fold", "origin", "truth"]]),
        "recall"].to_numpy()

    for name, frame in (("FOLD_RESULTS", fold), ("MAIN_RESULTS", summary),
                        ("PAIRED_VS_BASE", contrasts), ("FOLD_CONTRASTS", paired),
                        ("RESCUE_HARM", events), ("QUERY_COMPOSITION", count),
                        ("SUPPORT_COVERAGE", support), ("MARGIN_EFFECTS", pd.DataFrame(margin_summary)),
                        ("CLASS_EFFECTS", class_results),
                        ("MAPPER_DIAGNOSTICS", diagnostics)):
        frame.to_csv(root / f"{name}.csv", index=False)
    print("MAIN RESULTS\n", summary.round(3).to_string(index=False))
    print("\nPAIRED VS BASE\n", contrasts.round(3).to_string(index=False))
    print("\nRESCUES / HARMS\n", events.to_string(index=False))
    print("\nMAPPER\n", diagnostics.groupby("dataset")[["source_oof_r2", "target_diagnostic_r2"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapper", choices=("linear", "mlp", "raw_mlp_scalefix", "raw_mlp_scalefix_single"),
                        default="linear")
    args = parser.parse_args()
    root = HERE if args.mapper == "linear" else HERE / args.mapper
    summarize(root, *load(root))
