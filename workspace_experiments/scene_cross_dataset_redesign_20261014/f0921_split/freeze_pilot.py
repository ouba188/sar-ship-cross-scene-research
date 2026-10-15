"""Freeze source-only SSL splits and small matched-support F_0921 pilots."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parents[1] / "scene_port_recovery_20261014" / "f0921_water_regions.csv"
PIXELS = Path("E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/inputs128.npy")
TARGETS = ("Gulf_of_Mexico", "California_Baja_Pacific")
SEED = 20261014


def one_per_vessel(frame):
    return frame.sort_values("sample_id").drop_duplicates("mmsi")


def pilot_episode(target, region, seed, balanced_quota=False):
    groups = target[target.water_region.eq(region)].groupby("acquisition")
    sizes = groups.mmsi.nunique()
    eligible = sizes[sizes >= 32]
    if eligible.empty:
        raise ValueError(f"No B16 pilot acquisition in {region}")
    median = eligible.median()
    order = sorted(eligible.index, key=lambda a: (abs(sizes[a] - median), a))

    for acquisition in order:
        same = one_per_vessel(groups.get_group(acquisition))
        if same.class_id.nunique() < 2:
            continue
        other = one_per_vessel(target[
            target.water_region.eq(region) & target.acquisition.ne(acquisition)
        ])
        rng = np.random.default_rng(seed)
        q_size = max(16, min(len(same) - 16, int(np.ceil(len(same) * .25))))
        class_representatives = [
            rng.choice(part.index.to_numpy()) for _, part in same.groupby("class_id")
        ]
        if len(class_representatives) > q_size:
            continue
        remainder = same.index.difference(class_representatives).to_numpy()
        extra = rng.choice(remainder, size=q_size - len(class_representatives), replace=False)
        query = same.loc[np.r_[class_representatives, extra]].sort_index()
        query_mmsi = set(query.mmsi)
        same_pool = same[~same.mmsi.isin(query_mmsi)]
        other_pool = other[~other.mmsi.isin(query_mmsi)]
        classes = sorted(set(same_pool.class_id) & set(other_pool.class_id))
        capacities = {
            cls: min(int((same_pool.class_id == cls).sum()),
                     int((other_pool.class_id == cls).sum()))
            for cls in classes
        }
        if sum(capacities.values()) < 16 or len(classes) < 2:
            continue
        budget = 64 if sum(capacities.values()) >= 64 else 16
        quota = {cls: 1 for cls in classes}
        for _ in range(budget - len(classes)):
            available = [cls for cls in classes if quota[cls] < capacities[cls]]
            if balanced_quota:
                cls = min(available, key=lambda c: (quota[c], -capacities[c], c))
            else:
                cls = max(available, key=lambda c: (capacities[c] - quota[c], -c))
            if capacities[cls] <= quota[cls]:
                raise AssertionError("Insufficient support capacity")
            quota[cls] += 1
        tokens = np.concatenate([np.repeat(cls, quota[cls]) for cls in classes])
        rng.shuffle(tokens)
        sampled = {}
        for role, pool in (("same", same_pool), ("other", other_pool)):
            queues = {}
            for cls in classes:
                rows = pool.loc[pool.class_id.eq(cls)].index.to_numpy()
                queues[cls] = list(rng.choice(rows, size=quota[cls], replace=False))
            sampled[role] = np.array([queues[cls].pop() for cls in tokens], dtype=np.int64)
        return {
            "acquisition": acquisition,
            "query": query.index.to_numpy(np.int64),
            "support_same": sampled["same"],
            "support_other": sampled["other"],
            "budget_max": budget,
            "support_classes": tokens,
        }
    raise ValueError(f"No matched B16 episode in {region}")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT, dtype={"sample_id": str, "mmsi": str})
    pixels = np.load(PIXELS, mmap_mode="r")
    ids = pd.read_csv(
        "E:/Hermes/paper_tracking/sensitivity/f0921_detector_recrop_20261007/target_manifest.csv",
        usecols=["sample_id"], dtype={"sample_id": str},
    )
    np.testing.assert_array_equal(data.sample_id, ids.sample_id)
    if len(data) != len(pixels):
        raise ValueError("Pixel cache and metadata lengths differ")

    for i, region in enumerate(TARGETS):
        target_acq = set(data.loc[data.water_region.eq(region), "acquisition"].dropna())
        target = data[data.acquisition.isin(target_acq)]
        target_mmsi = set(target.mmsi)
        source = data[data.acquisition.notna() & ~data.acquisition.isin(target_acq)]
        source = source[~source.mmsi.isin(target_mmsi)]
        rng = np.random.default_rng(SEED + i)
        acqs = np.sort(source.acquisition.unique())
        val_acq = set(rng.choice(acqs, size=max(1, len(acqs) // 10), replace=False))
        validation = source[source.acquisition.isin(val_acq)]
        train = source[~source.acquisition.isin(val_acq)]
        train = train[~train.mmsi.isin(set(validation.mmsi))]
        episode = pilot_episode(target, region, SEED + i)

        dest = ROOT / region
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "split.npz",
            ssl_train=train.index.to_numpy(np.int64),
            source_validation=validation.index.to_numpy(np.int64),
            target_acquisition=target.index.to_numpy(np.int64),
            target_region=target[target.water_region.eq(region)].index.to_numpy(np.int64),
        )
        np.savez_compressed(dest / "pilot_episode.npz",
            query=episode["query"],
            support_same=episode["support_same"],
            support_other=episode["support_other"],
            support_classes=episode["support_classes"],
        )
        summary = {
            "target_region": region,
            "seed": SEED + i,
            "source_train_rows": len(train),
            "source_val_rows": len(validation),
            "target_acquisition_rows": len(target),
            "target_region_rows": int(target.water_region.eq(region).sum()),
            "target_acquisitions": len(target_acq),
            "pilot_acquisition": episode["acquisition"],
            "pilot_query_rows": len(episode["query"]),
            "pilot_max_support_budget": episode["budget_max"],
            "pilot_query_classes": sorted(data.loc[episode["query"], "class_id"].unique().tolist()),
            "pilot_support_class_counts": np.bincount(episode["support_classes"], minlength=8).tolist(),
            "source_train_classes": np.bincount(train.class_id, minlength=8).tolist(),
            "source_val_classes": np.bincount(validation.class_id, minlength=8).tolist(),
            "split_unit": "satellite|orbit|datatake",
            "selection": "median-size acquisition with matched support capacity, metadata/labels only; no predictions",
            "support_note": "class-matched retrospective diagnostic; pool labeling cost exceeds B",
            "ssl_target_pixels": 0,
        }
        (dest / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
