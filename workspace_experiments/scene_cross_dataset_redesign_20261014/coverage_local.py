"""Audit exact same/cross-acquisition support capacity for F_0921 and 24 ports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
F_FILE = ROOT.parent / "scene_port_recovery_20261014/f0921_water_regions.csv"
PORT_FILE = ROOT.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"


def load_data(dataset):
    if dataset == "F_0921":
        data = pd.read_csv(F_FILE, usecols=["sample_id", "mmsi", "acquisition",
                                             "water_region", "class_id"],
                           dtype={"sample_id": str, "mmsi": str, "acquisition": str})
        data = data.rename(columns={"water_region": "region"})
        data = data[data.region.ne("Other_or_transition") & data.acquisition.notna()]
    else:
        data = pd.read_csv(PORT_FILE, usecols=["instance_id", "mmsi", "product",
                                                 "port", "class_id"],
                           dtype={"mmsi": str})
        data = data.rename(columns={"instance_id": "sample_id", "port": "region"})
        data["acquisition"] = data["product"].str.split("_").map(
            lambda x: "|".join((x[0], x[6], x[7])))
    data = data[data.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")].copy()
    conflicts = data.groupby("mmsi").class_id.nunique()
    data = data[~data.mmsi.isin(conflicts[conflicts > 1].index)]
    return data


def audit_region(dataset, region, frame):
    unique = frame.sort_values("sample_id").drop_duplicates(["acquisition", "mmsi"])
    records = []
    for ordinal, (acquisition, same_all) in enumerate(unique.groupby("acquisition", sort=True)):
        n = len(same_all)
        base = {"dataset": dataset, "region": region, "acquisition": acquisition,
                "cell_rows": int(frame.acquisition.eq(acquisition).sum()),
                "cell_ships": n, "query_n": 0, "query_classes": 0,
                "paired_capacity": 0, "paired_classes": 0,
                "b16_feasible": False, "b64_feasible": False}
        if n < 24 or same_all.class_id.nunique() < 2:
            records.append(base)
            continue
        rng = np.random.default_rng(20261014 + ordinal)
        q_size = max(8, int(np.ceil(.25 * n)))
        representatives = [rng.choice(part.index.to_numpy())
                           for _, part in same_all.groupby("class_id")]
        remaining = same_all.index.difference(representatives).to_numpy()
        extra = rng.choice(remaining, size=q_size - len(representatives), replace=False)
        query = same_all.loc[np.r_[representatives, extra]]
        query_ships = set(query.mmsi)
        same = same_all[~same_all.mmsi.isin(query_ships)]
        other = unique[unique.acquisition.ne(acquisition) & ~unique.mmsi.isin(query_ships)]
        other = other.drop_duplicates("mmsi")
        same_counts = same.class_id.value_counts()
        other_counts = other.class_id.value_counts()
        capacity = np.minimum(same_counts.reindex(range(8), fill_value=0).to_numpy(),
                              other_counts.reindex(range(8), fill_value=0).to_numpy())
        base.update(query_n=len(query), query_classes=query.class_id.nunique(),
                    paired_capacity=int(capacity.sum()), paired_classes=int(sum(capacity > 0)),
                    b16_feasible=bool(capacity.sum() >= 16 and sum(capacity > 0) >= 2),
                    b64_feasible=bool(capacity.sum() >= 64 and sum(capacity > 0) >= 2))
        records.append(base)
    return records


def main():
    rows = []
    populations = {}
    for dataset in ("F_0921", "24ports"):
        data = load_data(dataset)
        populations[dataset] = {"eligible_rows": len(data),
                                "regions": data.region.nunique(),
                                "acquisitions": data.acquisition.nunique()}
        for region, frame in data.groupby("region", sort=True):
            rows.extend(audit_region(dataset, region, frame))
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / "coverage_local.csv", index=False)
    summary = result.groupby(["dataset", "region"]).agg(
        cells=("acquisition", "size"), cell_rows=("cell_rows", "sum"),
        b16_cells=("b16_feasible", "sum"), b64_cells=("b64_feasible", "sum"),
        b16_query_rows=("query_n", lambda x: int(x[result.loc[x.index, "b16_feasible"]].sum())),
        b64_query_rows=("query_n", lambda x: int(x[result.loc[x.index, "b64_feasible"]].sum())))
    summary.to_csv(ROOT / "coverage_local_summary.csv")
    (ROOT / "coverage_local_population.json").write_text(
        json.dumps(populations, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
