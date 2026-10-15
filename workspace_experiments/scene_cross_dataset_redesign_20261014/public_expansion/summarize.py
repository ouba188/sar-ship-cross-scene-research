"""Summarize matched public SAR pilot and expansion folds without model selection."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
PILOTS = {
    "oss1": ROOT / "strict_training" / "oss1",
    "oss2": ROOT / "strict_training" / "oss2_v2",
}
MODES = ("G50", "G50_G18", "G50_D18", "fused_G50_G18", "fused_G50_D18")


def main():
    records = []
    for dataset in PILOTS:
        folders = [("pilot", PILOTS[dataset])]
        folders += [(folder.name, folder / "fit") for folder in sorted((HERE / dataset).iterdir())
                    if folder.is_dir() and (folder / "fit" / "pilot_metrics.csv").exists()]
        for fold, folder in folders:
            frame = pd.read_csv(folder / "pilot_metrics.csv")
            predictions = pd.read_csv(folder / "pilot_predictions.csv", dtype={"mmsi": str})
            curve = json.loads((folder / "training_curve.json").read_text(encoding="utf-8"))
            train = json.loads((folder / "training_complete.json").read_text(encoding="utf-8"))
            episode = None if fold == "pilot" else json.loads(
                (folder.parent / "episode_summary.json").read_text(encoding="utf-8"))
            subset = frame[frame.support_origin.isin(("same", "cross"))]
            for origin in ("same", "cross"):
                block = subset[subset.support_origin.eq(origin)]
                assert set(block.draw) == set(range(5))
                means = block.groupby("mode")[["chip_oa", "chip_ba", "ship_oa", "ship_ba"]].mean()
                assert set(MODES) <= set(means.index)
                reference = predictions[(predictions.support_origin == origin)
                                        & (predictions.draw == 0) & (predictions["mode"] == "G50")]
                class_ships = reference.groupby("truth").mmsi.nunique()
                stable_classes = set(class_ships[class_ships >= 3].index)
                stable = predictions[(predictions.support_origin == origin)
                                     & predictions.truth.isin(stable_classes)
                                     & predictions["mode"].isin(MODES)].copy()
                stable["correct"] = stable.truth.eq(stable.prediction)
                stable_ba = stable.groupby(["mode", "draw", "truth"]).correct.mean().groupby(
                    ["mode", "draw"]).mean().groupby("mode").mean() * 100
                record = {
                    "dataset": dataset, "fold": fold, "support_origin": origin,
                    "source_train_rows": train["source_train_rows"],
                    "source_validation_rows": train["source_validation_rows"],
                    "source_val_initial_ba": 100 * curve[0]["source_val_ba"],
                    "source_val_trained_ba": 100 * max(item["source_val_ba"] for item in curve[1:]),
                    "source_prefers_ssl": max(item["source_val_ba"] for item in curve[1:]) > curve[0]["source_val_ba"],
                    "query_rows": int(block.chips.iloc[0]),
                    "query_ships": int(block.ships.iloc[0]),
                    "query_classes": None if episode is None else episode["query_classes"],
                    "query_classes_ge3_ships": len(stable_classes),
                }
                for mode in MODES:
                    for metric in ("chip_oa", "chip_ba", "ship_oa", "ship_ba"):
                        record[f"{mode}_{metric}"] = means.loc[mode, metric]
                    record[f"{mode}_stable_ba"] = stable_ba.loc[mode]
                for mode in MODES[1:]:
                    record[f"{mode}_delta_chip_ba"] = means.loc[mode, "chip_ba"] - means.loc["G50", "chip_ba"]
                    record[f"{mode}_delta_chip_oa"] = means.loc[mode, "chip_oa"] - means.loc["G50", "chip_oa"]
                    record[f"{mode}_delta_ship_ba"] = means.loc[mode, "ship_ba"] - means.loc["G50", "ship_ba"]
                    record[f"{mode}_delta_ship_oa"] = means.loc[mode, "ship_oa"] - means.loc["G50", "ship_oa"]
                    record[f"{mode}_delta_stable_ba"] = stable_ba.loc[mode] - stable_ba.loc["G50"]
                records.append(record)
    result = pd.DataFrame(records)
    result.to_csv(HERE / "fold_summary.csv", index=False)
    summary = result.groupby(["dataset", "support_origin"]).agg(
        folds=("fold", "nunique"), train_min=("source_train_rows", "min"),
        train_max=("source_train_rows", "max"),
        g50_ba=("G50_chip_ba", "mean"),
        g50_oa=("G50_chip_oa", "mean"),
        joint_ssl_delta_ba=("G50_D18_delta_chip_ba", "mean"),
        joint_ssl_delta_stable_ba=("G50_D18_delta_stable_ba", "mean"),
        joint_ssl_delta_oa=("G50_D18_delta_chip_oa", "mean"),
        joint_ssl_delta_ship_ba=("G50_D18_delta_ship_ba", "mean"),
        joint_ssl_delta_ship_oa=("G50_D18_delta_ship_oa", "mean"),
        joint_ssl_positive=("G50_D18_delta_chip_ba", lambda x: int((x > 0).sum())),
        joint_imagenet_delta_ba=("G50_G18_delta_chip_ba", "mean"),
        joint_imagenet_delta_stable_ba=("G50_G18_delta_stable_ba", "mean"),
        joint_imagenet_delta_oa=("G50_G18_delta_chip_oa", "mean"),
        fused_ssl_delta_ba=("fused_G50_D18_delta_chip_ba", "mean"),
        fused_ssl_delta_oa=("fused_G50_D18_delta_chip_oa", "mean"),
        fused_imagenet_delta_ba=("fused_G50_G18_delta_chip_ba", "mean"),
    ).reset_index()
    summary.to_csv(HERE / "dataset_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    print(result[["dataset", "fold", "support_origin", "source_train_rows",
                  "source_prefers_ssl", "G50_D18_delta_chip_ba",
                  "G50_G18_delta_chip_ba", "fused_G50_D18_delta_chip_ba"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
