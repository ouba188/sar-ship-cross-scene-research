"""Check whether source-hard class pairs match actual target errors (diagnostic only)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent


def main():
    rows = []
    for file in sorted(HERE.glob("*/*/source_selection.json")):
        selected = json.loads(file.read_text(encoding="utf-8"))
        cases = pd.read_csv(file.with_name("cases.csv"))
        pair_to_index = {tuple(pair): i for i, pair in enumerate(selected["pairs"])}
        top = set(selected["confusing14"])
        easy = set(selected["easy14"])
        source_scores = np.asarray(selected["confusion_scores"])
        for origin, group in cases.groupby("origin"):
            present = set(group.truth)
            for baseline in ("plain", "all28"):
                errors = np.zeros(28, dtype=int)
                for truth, prediction in zip(group.truth, group[f"pred_{baseline}"]):
                    if truth != prediction:
                        errors[pair_to_index[tuple(sorted((int(truth), int(prediction))))]] += 1
                error_total = int(errors.sum())
                corr = spearmanr(source_scores, errors).statistic if error_total else np.nan
                rows.append(dict(dataset=selected["dataset"], port=selected["port"], fold=selected["fold"],
                                 origin=origin, baseline=baseline, query_present_classes=len(present),
                                 repeated_error_events=error_total,
                                 top_error_fraction=(float(errors[list(top)].sum() / error_total)
                                                     if error_total else np.nan),
                                 easy_error_fraction=(float(errors[list(easy)].sum() / error_total)
                                                      if error_total else np.nan),
                                 source_target_pair_rank_corr=corr,
                                 top_pairs_both_query_classes=sum(set(selected["pairs"][i]) <= present for i in top)))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "TARGET_CONFUSION_ALIGNMENT.csv", index=False)
    print(out[out.baseline.eq("plain")].groupby(["dataset", "origin"])[
        ["top_error_fraction", "source_target_pair_rank_corr", "top_pairs_both_query_classes"]].mean().round(3).to_string())
    print("Three-class held ports:")
    print(out[(out.dataset == "ports") & (out.query_present_classes == 3)
              & (out.origin == "cross") & (out.baseline == "plain")][
                  ["port", "top_error_fraction", "source_target_pair_rank_corr", "top_pairs_both_query_classes"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
