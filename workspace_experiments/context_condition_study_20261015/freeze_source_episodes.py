"""Freeze source-validation pseudo-scenes before testing context-conditioned readout."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "scene_cross_dataset_redesign_20261014"
REGION = HERE.parent / "region_cue_replication_20261013"
FOLD = ROOT / "ports_full_expansion/Singapore__S1A_060968_079865"
sys.path.insert(0, str(ROOT / "ports_broad_expansion_20261014"))
from freeze import matched_quota  # noqa: E402
from audit_ports_context import chip_overlap  # noqa: E402

SEED = 20261015


def choose_support(pool, quota, rng):
    selected = []
    for class_id, n in enumerate(quota):
        if n:
            rows = pool.loc[pool.class_id == class_id, "sample_id"].to_numpy(str)
            selected.extend(rng.choice(rows, int(n), replace=False))
    assert len(selected) == 16 and len(set(selected)) == 16
    return selected


def main():
    frame = pd.read_csv(FOLD / "source_validation.csv", dtype={"mmsi": str})
    geometry = pd.read_csv(HERE / "geometry_by_sample.csv", usecols=["sample_id", "far_ok"])
    frame = frame.merge(geometry, on="sample_id", validate="one_to_one")
    frame = frame.loc[frame.far_ok].sort_values("sample_id")
    coords = pd.read_csv(REGION / "source_region_metadata.csv", usecols=[
        "sample_id", "resolved_crop_left_vv", "resolved_crop_top_vv",
        "resolved_crop_left_vh", "resolved_crop_top_vh"])
    shape = pd.read_csv(REGION / "ports/frame.csv", usecols=[
        "sample_id", "scene_id", "native_h_vv", "native_w_vv", "native_h_vh", "native_w_vh"])
    coords = coords.merge(shape, on="sample_id", validate="one_to_one").set_index("sample_id")
    episodes = []
    for idx, (acquisition, all_rows) in enumerate(frame.groupby("acquisition", sort=True)):
        all_rows = all_rows.drop_duplicates("mmsi")
        if len(all_rows) < 32 or all_rows.class_id.nunique() < 4:
            continue
        rng = np.random.default_rng(SEED + idx)
        query_ids = []
        for _, group in all_rows.groupby("class_id", sort=True):
            ids = group.sample_id.to_numpy(str)
            query_ids.extend(rng.choice(ids, (len(ids) + 1) // 2, replace=False))
        query = all_rows.loc[all_rows.sample_id.isin(query_ids)]
        same = all_rows.loc[~all_rows.sample_id.isin(query_ids)]
        qpos = coords.loc[query.sample_id]
        same = same.loc[[not any(chip_overlap(coords.loc[row.sample_id], q)
                                 for _, q in qpos.iterrows()) for row in same.itertuples()]]
        if len(query) < 16:
            continue
        choices = []
        for cross_acquisition, other in frame.loc[
            frame.port.eq(all_rows.port.iloc[0]) & frame.acquisition.ne(acquisition)
        ].groupby("acquisition", sort=True):
            cross = other.drop_duplicates("mmsi")
            cross = cross.loc[~cross.mmsi.isin(query.mmsi)]
            quota, capacity = matched_quota(same, cross, query)
            if quota is None:
                continue
            choices.append((int((quota > 0).sum()), int(sum(capacity)), len(cross),
                            cross_acquisition, cross, quota))
        if not choices:
            continue
        _, _, _, cross_acquisition, cross, quota = max(choices, key=lambda item: item[:4])
        draws = []
        for draw in range(5):
            draws.append({"same": choose_support(same, quota, np.random.default_rng(SEED + idx * 100 + draw)),
                          "cross": choose_support(cross, quota, np.random.default_rng(SEED + idx * 100 + draw + 10000))})
        assert set(query.mmsi).isdisjoint(same.mmsi)
        assert set(query.mmsi).isdisjoint(cross.mmsi)
        episodes.append(dict(port=str(all_rows.port.iloc[0]), query_acquisition=acquisition,
                             cross_acquisition=cross_acquisition, query=query.sample_id.tolist(),
                             quota=quota.tolist(), draws=draws))
    assert episodes
    (HERE / "source_episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    rows = [{"port": e["port"], "query_acquisition": e["query_acquisition"],
             "cross_acquisition": e["cross_acquisition"], "query_ships": len(e["query"]),
             "query_classes": frame.set_index("sample_id").loc[e["query"], "class_id"].nunique(),
             "support_classes": int(sum(np.array(e["quota"]) > 0))}
            for e in episodes]
    pd.DataFrame(rows).to_csv(HERE / "source_episodes.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
