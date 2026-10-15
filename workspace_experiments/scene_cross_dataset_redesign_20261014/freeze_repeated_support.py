"""Freeze five same/cross-acquisition support draws for existing F/port pilots."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "f0921_split"))
from freeze_pilot import one_per_vessel  # noqa: E402


def data_and_region(group):
    if group == "f0921_split":
        data = pd.read_csv(ROOT.parent / "scene_port_recovery_20261014/f0921_water_regions.csv",
                           usecols=["sample_id", "mmsi", "acquisition", "water_region", "class_id"],
                           dtype={"sample_id": str, "mmsi": str})
        return data, "water_region"
    data = pd.read_csv(ROOT.parent /
                       "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
                       usecols=["instance_id", "mmsi", "product", "port", "class_id"],
                       dtype={"mmsi": str})
    data["sample_id"] = data.instance_id
    data["acquisition"] = data["product"].str.split("_").map(
        lambda x: "_".join((x[0], x[6], x[7])))
    return data, "port"


def freeze(group, task, data, region_field):
    folder = ROOT / group / task
    with np.load(folder / "split.npz") as split:
        target = data.iloc[split["target_acquisition"]]
    with np.load(folder / "pilot_episode.npz") as episode:
        query_rows = episode["query"]
        tokens = episode["support_classes"]
        same_first = episode["support_same"]
        other_first = episode["support_other"]
    query_acq = data.acquisition.iloc[query_rows].unique()
    assert len(query_acq) == 1
    query_ships = set(data.mmsi.iloc[query_rows])
    pools = {}
    for origin, select in (("same", target.acquisition.eq(query_acq[0])),
                           ("other", target.acquisition.ne(query_acq[0]))):
        pool = one_per_vessel(target[select & target[region_field].eq(task)])
        pools[origin] = pool[~pool.mmsi.isin(query_ships)]

    draws = {"same": [same_first], "other": [other_first]}
    for draw in range(1, 5):
        rng = np.random.default_rng(20261014 + draw + (0 if group == "f0921_split" else 100)
                                    + (0 if task in ("Gulf_of_Mexico", "Rotterdam") else 10))
        for origin in ("same", "other"):
            pool = pools[origin]
            chosen = {}
            for cls in np.unique(tokens):
                candidates = pool.loc[pool.class_id.eq(cls)].index.to_numpy()
                chosen[cls] = list(rng.choice(candidates, size=int(sum(tokens == cls)),
                                              replace=False))
            draws[origin].append(np.array([chosen[cls].pop() for cls in tokens], dtype=np.int64))

    same = np.stack(draws["same"])
    other = np.stack(draws["other"])
    for support in (same, other):
        assert np.array_equal(data.class_id.to_numpy(int)[support], np.broadcast_to(tokens, support.shape))
        for row in support:
            assert len(row) == len(set(data.mmsi.iloc[row]))
            assert not set(data.mmsi.iloc[row]) & query_ships
    np.savez_compressed(folder / "support_draws.npz", same=same, other=other,
                        support_classes=tokens, query=query_rows)
    (folder / "support_draws_protocol.json").write_text(json.dumps({
        "query_unchanged": True, "draw0_is_original": True,
        "draws": 5, "budget_max": len(tokens),
        "class_counts": np.bincount(tokens, minlength=8).tolist(),
        "same_and_other_class_order_matched": True,
    }, indent=2), encoding="utf-8")
    print(f"{task}: 5 paired draws, B{len(tokens)}, query {len(query_rows)}", flush=True)


def main():
    for group, tasks in (("f0921_split", ("Gulf_of_Mexico", "California_Baja_Pacific")),
                         ("ports_split", ("Rotterdam", "Singapore"))):
        data, region_field = data_and_region(group)
        for task in tasks:
            freeze(group, task, data, region_field)


if __name__ == "__main__":
    main()
