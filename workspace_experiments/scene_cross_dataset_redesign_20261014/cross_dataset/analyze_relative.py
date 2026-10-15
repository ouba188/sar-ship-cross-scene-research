"""Paired branch gains on the same queries and frozen support draws."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = "G50"
ALTERNATIVES = ("fused_G50_G18", "fused_G50_D18")
SEED = 20261014
BOOTSTRAPS = 1000


def main():
    frame = pd.read_csv(ROOT / "external_predictions.csv")
    frame = frame[frame.model.isin((BASE, *ALTERNATIVES))].copy()
    frame["correct"] = frame.truth.eq(frame.prediction)
    paired = frame.pivot(index=["dataset", "budget", "draw", "sample_id", "truth"],
                         columns="model", values="correct").reset_index()
    draws, classes, intervals = [], [], []
    rng = np.random.default_rng(SEED)
    for (dataset, budget), group in paired.groupby(["dataset", "budget"], sort=False):
        n_draws = group.draw.nunique()
        for alternative in ALTERNATIVES:
            group = group.copy()
            group["delta"] = group[alternative].astype(int) - group[BASE].astype(int)
            group["rescue"] = (~group[BASE] & group[alternative]).astype(int)
            group["harm"] = (group[BASE] & ~group[alternative]).astype(int)
            for draw, block in group.groupby("draw"):
                draws.append({"dataset": dataset, "budget": budget, "draw": draw,
                              "alternative": alternative, "query_n": len(block),
                              "oa_delta_pp": 100 * block.delta.mean(),
                              "ba_delta_pp": 100 * block.groupby("truth").delta.mean().mean(),
                              "rescue": int(block.rescue.sum()), "harm": int(block.harm.sum())})
            by_query = group.groupby(["sample_id", "truth"]).delta.mean().reset_index()
            truth = by_query.truth.to_numpy(int)
            gain = by_query.delta.to_numpy(float)
            sample_indices = [np.flatnonzero(truth == c) for c in np.unique(truth)]
            oa_samples = np.empty(BOOTSTRAPS)
            ba_samples = np.empty(BOOTSTRAPS)
            for i in range(BOOTSTRAPS):
                picks = [rng.choice(indices, len(indices), replace=True)
                         for indices in sample_indices]
                oa_samples[i] = 100 * np.concatenate([gain[p] for p in picks]).mean()
                ba_samples[i] = 100 * np.mean([gain[p].mean() for p in picks])
            intervals.append({"dataset": dataset, "budget": budget,
                              "alternative": alternative, "query_n": len(by_query),
                              "support_draws": n_draws, "oa_delta_pp": 100 * gain.mean(),
                              "ba_delta_pp": 100 * np.mean([gain[p].mean() for p in sample_indices]),
                              "oa_ci_low": np.quantile(oa_samples, .025),
                              "oa_ci_high": np.quantile(oa_samples, .975),
                              "ba_ci_low": np.quantile(ba_samples, .025),
                              "ba_ci_high": np.quantile(ba_samples, .975)})
            for cls, block in group.groupby("truth"):
                classes.append({"dataset": dataset, "budget": budget,
                                "alternative": alternative, "class_id": cls,
                                "query_n": block.sample_id.nunique(),
                                "draws": n_draws,
                                "recall_delta_pp": 100 * block.delta.mean(),
                                "rescue": int(block.rescue.sum()),
                                "harm": int(block.harm.sum())})
    pd.DataFrame(draws).to_csv(ROOT / "relative_draws.csv", index=False)
    pd.DataFrame(classes).to_csv(ROOT / "relative_classes.csv", index=False)
    pd.DataFrame(intervals).to_csv(ROOT / "relative_bootstrap.csv", index=False)
    print(pd.DataFrame(intervals).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
