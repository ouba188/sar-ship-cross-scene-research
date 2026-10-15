"""Read existing experiment tables and contracts; never run models or write data."""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "ports_input_controls_20261012"
CONTRASTS = ("P-R", "E-R", "PE-P", "T-P", "T-R", "T-PE_scaled")


def main():
    metrics = pd.read_csv(OLD / "metrics.csv")
    selected = metrics[
        metrics.condition.isin(["original", "display"])
        & metrics.dataset.isin(["ports3", "oss1", "oss2"])
        & (metrics.scope == "original") & (metrics["head"] == "expert")
    ]
    keys = ["fold", "task", "dataset", "condition", "draw", "kind"]
    effects = pd.read_csv(OLD / "arm_effects.csv")
    effects = effects[(effects.scope == "original") & (effects["head"] == "expert")]
    records = []
    max_error = 0.0
    for contrast in CONTRASTS:
        a, b = contrast.split("-")
        pair = selected[selected.arm == a].merge(
            selected[selected.arm == b], on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        pair["delta_ba"] = pair.ba_a - pair.ba_b
        pair["risk_gain_macro"] = pair.brier_macro_b - pair.brier_macro_a
        pair["positive_ba"] = pair.delta_ba > 0
        check = pair.merge(effects[effects.contrast == contrast], on=keys,
                           suffixes=("", "_stored"), validate="one_to_one")
        assert len(check) == len(pair)
        for col in ("delta_ba", "risk_gain_macro"):
            error = float(np.max(np.abs(check[col] - check[col + "_stored"])))
            max_error = max(max_error, error)
            assert error < 1e-10, (contrast, col, error)
        for group, rows in pair.groupby(["dataset", "condition", "kind"]):
            task = rows.groupby(["fold", "task"])[["delta_ba", "risk_gain_macro"]].mean()
            records.append(dict(zip(["dataset", "condition", "kind"], group),
                                contrast=contrast, delta_ba=task.delta_ba.mean(),
                                risk_gain_macro=task.risk_gain_macro.mean(),
                                positive_ba=int(rows.positive_ba.sum()), contexts_draws=len(rows)))
    summary = pd.DataFrame(records).sort_values(["dataset", "kind", "contrast", "condition"])
    mechanism = pd.read_csv(ROOT / "ports_mechanism_evidence_20261012/replacement_totals.csv")
    for row in summary[summary.kind == "mlp"].itertuples():
        old = mechanism[(mechanism.dataset == row.dataset) & (mechanism.contrast == row.contrast)].iloc[0]
        prefix = "old" if row.condition == "original" else "new"
        assert abs(row.delta_ba - old[prefix + "_gain_pp"]) < 1e-10
        assert abs(row.risk_gain_macro - old[prefix + "_risk_gain"]) < 1e-10
    print("SUMMARY_CSV")
    print(summary.to_csv(index=False, float_format="%.9f"))
    print("VERIFICATION", json.dumps(dict(rows=len(summary), max_old_effect_error=max_error)))
    for fold in (2, 3):
        fits = ROOT / "ports_public_data_study_20261012" / f"fold_{fold}/fits"
        rows = np.load(fits / "source_rows.npy")
        with np.load(fits / "mapper.npz") as mapper:
            shapes = {k: list(mapper[k].shape) for k in mapper.files}
        assert shapes["linear_weight"] == [4096, 512]
        assert shapes["mlp_weight1"] == [4096, 512]
        assert shapes["mlp_weight2"] == [512, 512]
        with (fits / "geometry.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        np.testing.assert_array_equal(rows, bundle["source_rows"])
        for condition in ("original", "display"):
            path = ROOT / "ports_root_cause_20261013/fits" / condition / f"fold_{fold}/geometry.pkl"
            with path.open("rb") as handle:
                fresh = pickle.load(handle)
            np.testing.assert_array_equal(rows, fresh["source_rows"])
            assert fresh["model"]["pca_ssl"].n_features_in_ == 512
            assert fresh["model"]["pca"].n_features_in_ == 2048
        print("CONTRACT", json.dumps(dict(fold=fold, source_n=len(rows), shapes=shapes,
                                         source_rows_equal_archive_original_display=True)))


if __name__ == "__main__":
    main()
