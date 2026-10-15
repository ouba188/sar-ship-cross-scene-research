"""Summarize fixed-weight and paired-support sensitivity without target tuning."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
frame = pd.concat([pd.read_csv(path) for path in sorted(HERE.glob("results_*.csv"))],
                  ignore_index=True)
names = ["gap_pca256", "early_stem_gap_pca64", "early_stem_peak_pca64",
         "fusion_early_stem_gap", "fusion_early_stem_peak_w0.25",
         "fusion_early_stem_peak", "fusion_early_stem_peak_w0.75"]
print("EQUAL-FOLD MEANS")
print(frame.loc[frame.representation.isin(names)].groupby(
    ["origin", "representation"], sort=False)[
        ["source_port_map", "target_map", "target_macro_map",
         "target_b16_map", "target_b16_macro_map"]].mean().round(4).to_string())

for origin in ("same", "cross"):
    subset = frame.loc[frame.origin.eq(origin)].set_index(["fold", "representation"])
    base = subset.xs("gap_pca256", level="representation")
    for name in names[2:]:
        candidate = subset.xs(name, level="representation")
        deltas = candidate[["target_b16_map", "target_b16_macro_map"]] - base[
            ["target_b16_map", "target_b16_macro_map"]]
        paired = np.array([np.asarray(json.loads(candidate.loc[fold, "target_b16_draws"])) -
                           np.asarray(json.loads(base.loc[fold, "target_b16_draws"]))
                           for fold in base.index])
        print("DELTA", origin, name,
              "mean", np.round(deltas.mean().to_numpy(), 4),
              "positive_folds", (deltas > 0).sum().to_numpy().tolist(),
              "positive_draws", (paired > 0).sum(axis=(0, 1)).tolist(),
              "draw_ranges", np.round(paired.min(axis=(0, 1)), 4),
              np.round(paired.max(axis=(0, 1)), 4))
    name = "fusion_early_stem_peak"
    candidate = subset.xs(name, level="representation")
    print("FOLD_DELTA", origin, name)
    print((candidate[["target_b16_map", "target_b16_macro_map"]] -
           base[["target_b16_map", "target_b16_macro_map"]]).round(4).to_string())
    print("DRAW_DELTA", origin, name)
    for fold in base.index:
        paired = (np.asarray(json.loads(candidate.loc[fold, "target_b16_draws"])) -
                  np.asarray(json.loads(base.loc[fold, "target_b16_draws"])))
        print(fold, np.round(paired, 4).tolist())

cross = frame.loc[frame.origin.eq("cross")].set_index(["fold", "representation"])
for fold in cross.index.get_level_values(0).unique():
    base = cross.loc[(fold, "gap_pca256")]
    candidate = cross.loc[(fold, "fusion_early_stem_peak")]
    by_base = json.loads(base.target_by_class)
    by_candidate = json.loads(candidate.target_by_class)
    print("CLASS_UNION_DELTA", fold,
          {cls: round(by_candidate[cls] - score, 4)
           for cls, score in by_base.items()})
