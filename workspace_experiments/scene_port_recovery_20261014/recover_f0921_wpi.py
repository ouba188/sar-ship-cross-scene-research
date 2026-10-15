"""Attach World Port Index point-vicinity candidates to F_0921 chips."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "f0921_scene_port_recovered.csv"
WPI = ROOT / "wpi_na_noaa.geojson"
EARTH_RADIUS_KM = 6371.0088


def unit_sphere(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack((np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon), np.sin(lat)))


def main():
    chips = pd.read_csv(INPUT, dtype={"sample_id": str, "mmsi": str})
    features = json.loads(WPI.read_text(encoding="utf-8"))["features"]
    ports = pd.DataFrame([
        {
            "wpi_id": f["properties"]["OBJECTID"],
            "port_name": f["properties"]["PORT_NAME"],
            "country": f["properties"]["COUNTRY"],
            "harbor_size": f["properties"]["HARBORSIZE"],
            "lon": f["geometry"]["coordinates"][0],
            "lat": f["geometry"]["coordinates"][1],
        }
        for f in features if f["geometry"]["type"] == "Point"
    ])
    xyz = unit_sphere(chips.lat.to_numpy(float), chips.lon.to_numpy(float))
    tree = cKDTree(unit_sphere(ports.lat.to_numpy(float), ports.lon.to_numpy(float)))
    chord, idx = tree.query(xyz, k=2)
    distance_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.minimum(chord / 2, 1))

    for suffix, col in (("first", 0), ("second", 1)):
        found = ports.iloc[idx[:, col]].reset_index(drop=True)
        for key in ("wpi_id", "port_name", "country", "harbor_size"):
            chips[f"wpi_{suffix}_{key}"] = found[key].to_numpy()
        chips[f"wpi_{suffix}_km"] = np.round(distance_km[:, col], 3)

    gap = distance_km[:, 1] - distance_km[:, 0]
    chips["wpi_second_gap_km"] = np.round(gap, 3)
    port_key = chips.wpi_first_country + "/" + chips.wpi_first_port_name
    for radius in (10, 25):
        near = distance_km[:, 0] <= radius
        distinct = gap >= 5
        chips[f"wpi_named_vicinity_{radius}km"] = port_key.where(near & distinct, "")
        chips[f"wpi_ambiguous_{radius}km"] = near & ~distinct

    keep = [
        "sample_id", "scene", "acquisition", "mmsi", "class_id", "lon", "lat",
        "historical_scene_port", "port_vicinity_50km",
        "wpi_first_wpi_id", "wpi_first_port_name", "wpi_first_country",
        "wpi_first_harbor_size", "wpi_first_km", "wpi_second_port_name",
        "wpi_second_km", "wpi_second_gap_km", "wpi_named_vicinity_10km",
        "wpi_ambiguous_10km", "wpi_named_vicinity_25km", "wpi_ambiguous_25km",
    ]
    chips[keep].to_csv(ROOT / "f0921_wpi_port_candidates.csv", index=False)

    named = chips.loc[chips.wpi_named_vicinity_25km.ne("")].groupby(
        ["wpi_first_wpi_id", "wpi_first_country", "wpi_first_port_name"],
        dropna=False,
    ).agg(samples=("sample_id", "size"), ships=("mmsi", "nunique"),
          acquisitions=("acquisition", "nunique"), classes=("class_id", "nunique"))
    named.sort_values("samples", ascending=False).to_csv(ROOT / "wpi_named_port_summary.csv")

    summary = {
        "rows": int(len(chips)),
        "wpi_point_count": int(len(ports)),
        "within_10km": int((distance_km[:, 0] <= 10).sum()),
        "unique_10km": int(chips.wpi_named_vicinity_10km.ne("").sum()),
        "ambiguous_10km": int(chips.wpi_ambiguous_10km.sum()),
        "within_25km": int((distance_km[:, 0] <= 25).sum()),
        "unique_25km": int(chips.wpi_named_vicinity_25km.ne("").sum()),
        "ambiguous_25km": int(chips.wpi_ambiguous_25km.sum()),
        "distinct_named_points_25km": int(len(named)),
        "source": "NOAA ArcGIS mirror of NGA World Port Index, layer 30",
        "source_scope_note": "NOAA layer describes a subset and cites 2019 WPI; unmatched points do not imply no nearby port",
        "source_query_bbox_lonlat": [-170, 10, -60, 60],
        "distance_note": "WPI is a point catalog, not a port boundary; named fields are vicinity candidates",
    }
    (ROOT / "wpi_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
