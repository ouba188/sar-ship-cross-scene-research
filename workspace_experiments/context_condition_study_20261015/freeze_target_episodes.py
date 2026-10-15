"""Freeze native-far-eligible held-port B16 episodes without spatial crop overlap."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_ports_context import chip_overlap

HERE = Path(__file__).resolve().parent
SCENE = HERE.parent / "scene_cross_dataset_redesign_20261014"
REGION = HERE.parent / "region_cue_replication_20261013"
SEED = 20261017


def main():
    quotas = pd.read_csv(HERE / "matched_support_quota.csv")
    geometry = pd.read_csv(HERE / "geometry_by_sample.csv", usecols=["sample_id", "far_ok"])
    eligible = set(geometry.loc[geometry.far_ok, "sample_id"].astype(str))
    positions = pd.read_csv(REGION / "source_region_metadata.csv", usecols=[
        "sample_id", "resolved_crop_left_vv", "resolved_crop_top_vv",
        "resolved_crop_left_vh", "resolved_crop_top_vh"])
    shape = pd.read_csv(REGION / "ports/frame.csv", usecols=[
        "sample_id", "scene_id", "native_h_vv", "native_w_vv", "native_h_vh", "native_w_vh"])
    coords = positions.merge(shape, on="sample_id", validate="one_to_one").set_index("sample_id")
    manifest_ids = pd.read_csv(
        HERE.parent / "visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv",
        usecols=["instance_id"]).instance_id.to_numpy(str)
    episodes = []
    for fold, frozen in quotas.groupby("fold", sort=True):
        if not frozen.far_no_overlap_quota_feasible.all():
            continue
        folder = next((root / fold for root in
                       (SCENE / "ports_full_expansion", SCENE / "ports_broad_expansion_20261014")
                       if (root / fold / "query.csv").exists()))
        feature_rows = np.load(folder.parent / fold.split("__")[0] / "rows.npy")
        available = set(manifest_ids[feature_rows])
        query = pd.read_csv(folder / "query.csv", dtype={"mmsi": str})
        query = query.loc[query.sample_id.isin(eligible)]
        assert set(query.sample_id).issubset(available), fold
        candidates = pd.read_csv(folder / "support_candidates.csv", dtype={"mmsi": str})
        candidates = candidates.loc[candidates.sample_id.isin(eligible & available)]
        qpos = coords.loc[query.sample_id]
        same = candidates.support_origin.eq("same")
        overlap = [origin and any(chip_overlap(coords.loc[row.sample_id], q)
                                      for _, q in qpos.iterrows())
                   for origin, row in zip(same, candidates.itertuples())]
        candidates = candidates.loc[~np.asarray(overlap)]
        candidates = candidates.loc[~candidates.mmsi.isin(query.mmsi)]
        candidates = candidates.sort_values("sample_id").drop_duplicates(["support_origin", "mmsi"])
        quota = {int(row.class_id): int(row.quota) for row in frozen.itertuples()}
        assert sum(quota.values()) == 16 and len(query) > 0
        draws = []
        for draw in range(5):
            pair = {}
            for origin in ("same", "cross"):
                rng = np.random.default_rng(SEED + len(episodes) * 100 + draw * 2 + (origin == "cross"))
                selected = []
                for class_id, count in quota.items():
                    pool = candidates.loc[(candidates.support_origin == origin)
                                          & (candidates.class_id == class_id), "sample_id"].to_numpy(str)
                    assert len(pool) >= count, (fold, origin, class_id, len(pool), count)
                    selected.extend(rng.choice(pool, count, replace=False).tolist())
                assert len(selected) == len(set(selected)) == 16
                assert set(selected).isdisjoint(query.sample_id)
                assert all(not chip_overlap(coords.loc[sid], q)
                           for sid in selected for _, q in qpos.iterrows()), (fold, origin, draw)
                pair[origin] = selected
            draws.append(pair)
        episodes.append(dict(fold=fold, folder=str(folder), port=str(query.port.iloc[0]),
                             query=query.sample_id.tolist(), quota=quota, draws=draws))
    assert len(episodes) == 6
    (HERE / "target_episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    pd.DataFrame([dict(fold=e["fold"], port=e["port"], query=len(e["query"]),
                       classes=len(e["quota"]), same_candidates=int((pd.read_csv(
                           Path(e["folder"]) / "support_candidates.csv").support_origin == "same").sum()))
                  for e in episodes]).to_csv(HERE / "target_episodes.csv", index=False)
    print(pd.read_csv(HERE / "target_episodes.csv").to_string(index=False))


if __name__ == "__main__":
    main()
