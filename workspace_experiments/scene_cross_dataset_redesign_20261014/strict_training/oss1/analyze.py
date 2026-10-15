"""Summarize paired OSS1 branch effects without treating draws as independent ships."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main():
    metrics = pd.read_csv(HERE / "pilot_metrics.csv")
    selected = metrics[metrics.support_origin.ne("none")]
    pivot = selected.pivot(index=["draw", "support_origin"], columns="mode",
                           values=["chip_oa", "chip_ba"])
    rows = []
    for (draw, origin), row in pivot.iterrows():
        for mode, control in (("G50_D18", "G50_G18"),
                              ("fused_G50_D18", "fused_G50_G18"),
                              ("G50_D18", "G50")):
            rows.append({"draw": draw, "support_origin": origin, "mode": mode,
                         "control": control,
                         "delta_oa": row["chip_oa", mode] - row["chip_oa", control],
                         "delta_ba": row["chip_ba", mode] - row["chip_ba", control]})
    pd.DataFrame(rows).to_csv(HERE / "paired_deltas.csv", index=False)

    predictions = pd.read_csv(HERE / "pilot_predictions.csv")
    predictions["correct"] = predictions.truth.eq(predictions.prediction)
    predictions.groupby(["draw", "support_origin", "mode", "truth"],
                        as_index=False).agg(n=("correct", "size"),
                                            correct=("correct", "sum")).to_csv(
                                                HERE / "classwise.csv", index=False)
    base = predictions[predictions["mode"].eq("G50")][
        ["draw", "support_origin", "sample_id", "correct"]].rename(
            columns={"correct": "base_correct"})
    paired = predictions.merge(base, on=["draw", "support_origin", "sample_id"])
    paired["rescued"] = (~paired.base_correct) & paired.correct
    paired["harmed"] = paired.base_correct & (~paired.correct)
    paired.groupby(["draw", "support_origin", "mode", "truth"],
                   as_index=False).agg(n=("correct", "size"),
                                       rescued=("rescued", "sum"),
                                       harmed=("harmed", "sum")).to_csv(
                                           HERE / "rescue_harm.csv", index=False)


if __name__ == "__main__":
    main()
