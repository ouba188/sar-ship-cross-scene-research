"""Join strict held-port scenes from the two frozen 24-port cohorts."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "public_full_20261014"))
from analyze import ARMS, COMPARISONS, METRICS  # noqa: E402

SOURCES = (ROOT / "ports_full_expansion", ROOT / "ports_broad_expansion_20261014")
OUT = ROOT / "ports_six_results_20261015"


def main():
    files = [file for folder in SOURCES for file in folder.glob("*__*/full_metrics.csv")]
    assert len(files) == 8
    raw = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    assert raw.groupby("port").fold.nunique().to_dict() == {
        "Singapore": 2, "Port Klang": 2, "Ningbo-Zhoushan": 1,
        "Shanghai": 1, "Jebel Ali": 1, "Qingdao": 1}
    assert len(raw) == 8 * 2 * 5 * len(ARMS)
    assert raw.groupby(["port", "fold", "origin", "draw"]).arm.nunique().eq(len(ARMS)).all()
    OUT.mkdir(exist_ok=True)
    raw.to_csv(OUT / "DRAW_METRICS.csv", index=False)
    fold = raw.groupby(["port", "fold", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    fold.to_csv(OUT / "FOLD_MAIN_TABLE.csv", index=False)
    port = fold.groupby(["port", "origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    port.to_csv(OUT / "PORT_MAIN_TABLE.csv", index=False)
    overall = port.groupby(["origin", "arm"], as_index=False)[list(METRICS) + ["selected_fraction"]].mean()
    overall.to_csv(OUT / "SIX_PORT_MAIN_TABLE.csv", index=False)
    indexed = raw.set_index(["port", "fold", "origin", "draw", "arm"])
    paired = []
    for candidate, baseline, label in COMPARISONS:
        delta = indexed.xs(candidate, level="arm")[list(METRICS)] - indexed.xs(baseline, level="arm")[list(METRICS)]
        paired.append(delta.reset_index().assign(comparison=label))
    paired = pd.concat(paired, ignore_index=True)
    paired.to_csv(OUT / "PAIRED_DRAW_EFFECTS.csv", index=False)
    port_delta = (paired.groupby(["port", "origin", "comparison", "fold"], as_index=False)[list(METRICS)].mean()
                  .groupby(["port", "origin", "comparison"], as_index=False)[list(METRICS)].mean())
    port_delta.to_csv(OUT / "PORT_MODULE_EFFECTS.csv", index=False)
    total_delta = port_delta.groupby(["origin", "comparison"], as_index=False)[list(METRICS)].mean()
    total_delta.to_csv(OUT / "SIX_PORT_MODULE_EFFECTS.csv", index=False)
    print(overall[["origin", "arm", "chip_oa", "chip_ba", "selected_fraction"]].round(3).to_string(index=False))
    print(total_delta[["origin", "comparison", "chip_oa", "chip_ba"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
