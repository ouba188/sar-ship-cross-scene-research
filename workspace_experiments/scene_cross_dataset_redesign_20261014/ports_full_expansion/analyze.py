"""Aggregate four held-port scenes without treating scenes as independent ports."""

import argparse
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "public_full_20261014"))
from analyze import ARMS, COMPARISONS, METRICS  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    root = parser.parse_args().root
    files = sorted(root.glob("*__*"))
    files = [path / "full_metrics.csv" for path in files if (path / "full_metrics.csv").exists()]
    assert len(files) >= 4
    raw = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    assert len(raw) == len(files) * 2 * 5 * len(ARMS)
    assert raw.groupby(["port", "fold", "origin", "draw"]).arm.nunique().eq(len(ARMS)).all()
    raw.to_csv(root / "PORTS_FULL_DRAW_METRICS.csv", index=False)
    fold = raw.groupby(["port", "fold", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    fold.to_csv(root / "PORTS_FULL_FOLD_METRICS.csv", index=False)
    port = fold.groupby(["port", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    port.to_csv(root / "PORTS_MAIN_TABLE.csv", index=False)
    indexed = raw.set_index(["port", "fold", "origin", "draw", "arm"])
    effects = []
    for candidate, baseline, label in COMPARISONS:
        delta = indexed.xs(candidate, level="arm")[list(METRICS)] - indexed.xs(baseline, level="arm")[list(METRICS)]
        effects.append(delta.reset_index().assign(comparison=label))
    effects = pd.concat(effects, ignore_index=True)
    effects.to_csv(root / "PORTS_PAIRED_EFFECTS.csv", index=False)
    result = effects.groupby(["port", "origin", "comparison"], as_index=False)[list(METRICS)].mean()
    result.to_csv(root / "PORTS_MODULE_EFFECTS.csv", index=False)
    print(port.pivot_table(index=["port", "origin"], columns="arm", values="chip_ba").round(2).to_string())
    print(result[["port", "origin", "comparison", "chip_oa", "chip_ba"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
