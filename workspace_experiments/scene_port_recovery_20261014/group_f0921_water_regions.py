"""Coarse water-region strata for the current F_0921 eight-class queue.

These geographic strata are descriptive, not official sea/port boundaries or
independent acquisition groups. The order below resolves coastal transitions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "f0921_scene_port_recovered.csv"
WPI = ROOT / "f0921_wpi_port_candidates.csv"


def region_labels(lon, lat):
    labels = np.full(len(lon), "Other_or_transition", dtype=object)

    def put(name, condition):
        labels[(labels == "Other_or_transition") & condition] = name

    put("Hawaii", (-163 <= lon) & (lon <= -153) & (18 <= lat) & (lat <= 24))
    put("Great_Lakes", (-94 <= lon) & (lon <= -76) & (40 <= lat) & (lat <= 50.5))
    put("Pacific_Northwest", (-135 <= lon) & (lon <= -117) & (42 <= lat) & (lat <= 52))
    put("California_Baja_Pacific", (-135 <= lon) & (lon <= -108) & (25 <= lat) & (lat < 42))
    put("Gulf_of_Mexico", (-102 <= lon) & (lon < -81.5) & (18 <= lat) & (lat < 31.5))
    put("Chesapeake_Mid_Atlantic", (-79 <= lon) & (lon <= -72) & (36 <= lat) & (lat < 40.5))
    put("Caribbean", (-85 <= lon) & (lon <= -59) & (15 <= lat) & (lat < 24))
    put("Florida_Bahamas", (-85 <= lon) & (lon <= -72) & (24 <= lat) & (lat < 31))
    put("Southeast_Atlantic", (-83 <= lon) & (lon <= -69) & (31 <= lat) & (lat < 36))
    put("Northeast_Atlantic", (-79 <= lon) & (lon <= -59) & (lat <= 51)
        & ((lat >= 40.5) | ((lat >= 38) & (lon >= -72))))
    return labels


def main():
    cols = ["sample_id", "scene", "acquisition", "mmsi", "class_id", "lon", "lat"]
    data = pd.read_csv(INPUT, usecols=cols, dtype={"sample_id": str, "mmsi": str})
    data["water_region"] = region_labels(data.lon.to_numpy(), data.lat.to_numpy())
    names = pd.read_csv(
        WPI, usecols=["sample_id", "wpi_named_vicinity_25km"],
        dtype={"sample_id": str},
    )
    data = data.merge(names, on="sample_id", validate="one_to_one")
    data.to_csv(ROOT / "f0921_water_regions.csv", index=False)

    summary = data.groupby("water_region").agg(
        samples=("sample_id", "size"),
        ships=("mmsi", "nunique"),
        acquisitions=("acquisition", "nunique"),
        classes=("class_id", "nunique"),
        named_port_vicinity=("wpi_named_vicinity_25km", lambda x: x.notna().sum()),
    )
    summary["sample_share_pct"] = (100 * summary.samples / len(data)).round(2)
    summary.sort_values("samples", ascending=False).to_csv(ROOT / "f0921_water_region_summary.csv")
    by_class = pd.crosstab(data.water_region, data.class_id)
    by_class.to_csv(ROOT / "f0921_water_region_classes.csv")
    print(summary.sort_values("samples", ascending=False).to_string())
    print("rows:", len(data), "regions:", len(summary), "other:", int((data.water_region == "Other_or_transition").sum()))


if __name__ == "__main__":
    main()
