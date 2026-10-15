"""Summarize paired class-pair-selection effects without target-side tuning."""

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ARMS = ("plain", "all28", "confusing14", "easy14", "random14")
COMPARISONS = (("confusing14", "all28"), ("confusing14", "easy14"),
               ("confusing14", "random14"), ("confusing14", "plain"),
               ("easy14", "all28"))
METRICS = ("chip_oa", "chip_ba", "ship_oa", "ship_ba")


def main():
    files = sorted(HERE.glob("*/*/metrics.csv"))
    assert len(files) == 16
    raw = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    raw["port"] = raw.port.fillna("")
    assert len(raw) == 16 * 2 * 5 * len(ARMS)
    assert raw.groupby(["dataset", "fold", "origin", "draw"]).arm.nunique().eq(len(ARMS)).all()
    raw.to_csv(HERE / "ALL_DRAW_METRICS.csv", index=False)
    indexed = raw.set_index(["dataset", "port", "fold", "origin", "draw", "arm"])
    effects = []
    for candidate, baseline in COMPARISONS:
        delta = (indexed.xs(candidate, level="arm")[list(METRICS)]
                 - indexed.xs(baseline, level="arm")[list(METRICS)])
        effects.append(delta.reset_index().assign(comparison=f"{candidate}_vs_{baseline}"))
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(HERE / "PAIRED_DRAW_EFFECTS.csv", index=False)

    fold = effects.groupby(["dataset", "port", "fold", "origin", "comparison"], as_index=False)[list(METRICS)].mean()
    fold.to_csv(HERE / "FOLD_EFFECTS.csv", index=False)
    pub = fold[fold.dataset.ne("ports")].groupby(["dataset", "origin", "comparison"], as_index=False)[list(METRICS)].mean()
    port = fold[fold.dataset.eq("ports")].groupby(["port", "origin", "comparison"], as_index=False)[list(METRICS)].mean()
    port.to_csv(HERE / "PORT_EFFECTS.csv", index=False)
    total = port.groupby(["origin", "comparison"], as_index=False)[list(METRICS)].mean().assign(dataset="ports")
    total = pd.concat([pub, total], ignore_index=True)
    total.to_csv(HERE / "MAIN_EFFECTS.csv", index=False)

    cases = pd.concat((pd.read_csv(file, dtype={"mmsi": str}) for file in HERE.glob("*/*/cases.csv")), ignore_index=True)
    cases["port"] = cases.port.fillna("")
    assert len(cases.groupby(["dataset", "fold", "origin", "draw"])) == 16 * 2 * 5
    for arm in ARMS:
        cases[f"correct_{arm}"] = cases[f"pred_{arm}"].eq(cases.truth)
    cases.to_csv(HERE / "ALL_CASES.csv", index=False)
    changes = cases.groupby(["dataset", "port", "fold", "origin", "draw", "truth"], as_index=False).agg(
        n=("truth", "size"),
        top_correct=("correct_confusing14", "sum"),
        all_correct=("correct_all28", "sum"),
        easy_correct=("correct_easy14", "sum"))
    changes["top_minus_all_recall"] = 100 * (changes.top_correct - changes.all_correct) / changes.n
    changes.to_csv(HERE / "CLASS_CHANGES.csv", index=False)
    event = cases.assign(rescue=(cases.correct_confusing14 & ~cases.correct_all28),
                         harm=(~cases.correct_confusing14 & cases.correct_all28))
    event.groupby(["dataset", "origin", "truth"], as_index=False)[["rescue", "harm"]].sum().to_csv(
        HERE / "RESCUE_HARM_BY_CLASS.csv", index=False)

    selection = []
    for file in HERE.glob("*/*/source_selection.json"):
        item = json.loads(file.read_text(encoding="utf-8"))
        selection.append(dict(dataset=item["dataset"], port=item["port"], fold=item["fold"],
                              min_source_val_class=min(item["source_val_class_counts"]),
                              source_val=item["source_val"],
                              top_pairs=";".join(f"{item['pairs'][i][0]}-{item['pairs'][i][1]}"
                                                 for i in item["confusing14"]),
                              top_scale=item["mask_scales"]["confusing14"],
                              easy_scale=item["mask_scales"]["easy14"]))
    pd.DataFrame(selection).to_csv(HERE / "SOURCE_SELECTION.csv", index=False)
    print(total[["dataset", "origin", "comparison", "chip_oa", "chip_ba"]].round(3).to_string(index=False))
    print("positive folds top-vs-all", fold[fold.comparison.eq("confusing14_vs_all28")]
          .groupby(["dataset", "origin"]).chip_ba.apply(lambda x: int((x > 1e-8).sum())).to_dict())
    print("source validation minimum class count", min(item["min_source_val_class"] for item in selection))


if __name__ == "__main__":
    main()
