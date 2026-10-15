"""Check acquisition, vessel, and paired-support isolation in frozen local pilots."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
F_DATA = ROOT.parent / "scene_port_recovery_20261014" / "f0921_water_regions.csv"
PORT_DATA = ROOT.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv"


def check(task, folder, data, region_field):
    with np.load(folder / "split.npz") as split:
        train = split["ssl_train"]
        validation = split["source_validation"]
        target = split["target_acquisition"]
    with np.load(folder / "pilot_episode.npz") as episode:
        query = episode["query"]
        same = episode["support_same"]
        other = episode["support_other"]
        support_classes = episode["support_classes"]

    def values(rows, field):
        return set(data.iloc[rows][field].dropna())

    assert not (values(train, "acquisition") & values(validation, "acquisition"))
    assert not (values(np.r_[train, validation], "acquisition") & values(target, "acquisition"))
    assert not (values(train, "mmsi") & values(validation, "mmsi"))
    assert not (values(np.r_[train, validation], "mmsi") & values(target, "mmsi"))
    assert not (values(query, "mmsi") & values(np.r_[same, other], "mmsi"))
    assert set(np.r_[query, same, other]).issubset(set(target))
    query_acq = values(query, "acquisition")
    assert len(query_acq) == 1
    assert values(same, "acquisition") == query_acq
    assert not (values(other, "acquisition") & query_acq)
    assert values(np.r_[query, same, other], region_field) == {task}
    assert np.array_equal(data.iloc[same].class_id.to_numpy(), support_classes)
    assert np.array_equal(data.iloc[other].class_id.to_numpy(), support_classes)
    print(f"{task}: train={len(train)} validation={len(validation)} "
          f"query={len(query)} paired_support={len(same)} OK")


def main():
    f = pd.read_csv(F_DATA, usecols=["mmsi", "acquisition", "water_region", "class_id"],
                    dtype={"mmsi": str, "acquisition": str})
    for task in ("Gulf_of_Mexico", "California_Baja_Pacific"):
        check(task, ROOT / "f0921_split" / task, f, "water_region")

    ports = pd.read_csv(PORT_DATA, usecols=["mmsi", "product", "port", "class_id"],
                        dtype={"mmsi": str, "product": str})
    ports["acquisition"] = ports["product"].str.split("_").map(
        lambda x: "_".join((x[0], x[6], x[7])))
    for task in ("Rotterdam", "Singapore"):
        check(task, ROOT / "ports_split" / task, ports, "port")

    public = ROOT / "public_split"
    balanced = ROOT / "strict_training" / "oss2_v2"
    train = pd.read_csv(balanced / "source_train.csv", usecols=["acquisition", "mmsi"])
    validation = pd.read_csv(balanced / "source_validation.csv",
                             usecols=["acquisition", "mmsi", "class_id"])
    query = pd.read_csv(public / "oss2_query.csv", usecols=["acquisition", "mmsi"])
    candidates = pd.read_csv(public / "oss2_support_candidates.csv",
                             usecols=["acquisition", "mmsi"])
    target_acq = set(query.acquisition) | set(candidates.acquisition)
    target_mmsi = set(query.mmsi) | set(candidates.mmsi)
    assert not (set(train.acquisition) & set(validation.acquisition))
    assert not (set(train.mmsi) & set(validation.mmsi))
    assert not (set(train.acquisition) | set(validation.acquisition)) & target_acq
    assert not (set(train.mmsi) | set(validation.mmsi)) & target_mmsi
    assert set(validation.class_id) == set(range(8))
    print(f"OSS2 balanced: train={len(train)} validation={len(validation)} "
          f"all_eight_source_validation_classes OK")


if __name__ == "__main__":
    main()
