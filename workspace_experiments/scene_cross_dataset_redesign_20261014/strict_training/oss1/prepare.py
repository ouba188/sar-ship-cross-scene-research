"""Freeze an eight-class source validation fold and cache native OSS1 pixel pairs."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPLIT = ROOT / "public_split"
MANIFEST = Path("E:/Hermes/paper_tracking/sensitivity/opensarship1_transfer_20261007/target_manifest.csv")
SEED = 20261014


def main():
    old_train = pd.read_csv(SPLIT / "oss1_source_train.csv", dtype={"mmsi": str})
    old_val = pd.read_csv(SPLIT / "oss1_source_validation.csv", dtype={"mmsi": str})
    pool = pd.concat((old_train, old_val), ignore_index=True)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(folds.split(pool, pool.class_id, pool.acquisition))
    validation = pool.iloc[val_idx].copy()
    train = pool.iloc[train_idx].copy()
    train = train[~train.mmsi.isin(set(validation.mmsi))].copy()
    query = pd.read_csv(SPLIT / "oss1_query.csv", dtype={"mmsi": str})
    candidates = pd.read_csv(SPLIT / "oss1_support_candidates.csv", dtype={"mmsi": str})
    held_acqs = set(query.acquisition) | set(candidates.acquisition)
    held_ships = set(query.mmsi) | set(candidates.mmsi)
    assert not held_acqs & (set(train.acquisition) | set(validation.acquisition))
    assert not held_ships & (set(train.mmsi) | set(validation.mmsi))
    assert not set(train.acquisition) & set(validation.acquisition)
    assert not set(train.mmsi) & set(validation.mmsi)
    assert set(train.class_id) == set(validation.class_id) == set(range(8))
    assert set(train.sample_id) | set(validation.sample_id) <= set(pool.sample_id)
    train.to_csv(HERE / "source_train.csv", index=False)
    validation.to_csv(HERE / "source_validation.csv", index=False)
    train.drop(columns=["class_id", "class_name"]).to_csv(HERE / "ssl.csv", index=False)

    manifest = pd.read_csv(MANIFEST, usecols=["sample_id", "mmsi_or_ship_id", "scene_id",
                                            "chip_path_vv", "chip_path_vh"],
                           dtype={"mmsi_or_ship_id": str})
    needed = pd.Index(pd.concat((train.sample_id, validation.sample_id, query.sample_id,
                                 candidates.sample_id)).unique())
    assert manifest.sample_id.is_unique
    selected = manifest.set_index("sample_id").loc[needed].reset_index()
    selected = selected.rename(columns={"index": "sample_id"})
    selected.to_csv(HERE / "pixel_index.csv", index=False,
                    columns=["sample_id", "mmsi_or_ship_id", "scene_id"])
    pixels = np.lib.format.open_memmap(HERE / "pixels128.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(selected), 2, 128, 128))
    for i, row in enumerate(selected.itertuples(index=False)):
        for channel, path in enumerate((row.chip_path_vv, row.chip_path_vh)):
            with Image.open(path) as image:
                pixels[i, channel] = np.asarray(image.convert("L").resize(
                    (128, 128), Image.Resampling.BILINEAR))
        if (i + 1) % 250 == 0:
            print("cached", i + 1, "/", len(selected), flush=True)
    pixels.flush()
    print("source train/validation", len(train), len(validation), flush=True)
    print("source validation classes", validation.class_id.value_counts().sort_index().to_dict(), flush=True)
    print("pixel pairs", len(selected), flush=True)


if __name__ == "__main__":
    main()
