"""Freeze two source-port-only R18 SSL pilots on the 24-port cohort."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "f0921_split"))
from freeze_pilot import pilot_episode  # noqa: E402


MANIFEST = Path("E:/临时会话/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv")
PIXELS = Path("D:/SAR_completion_cache_20261010/source_inputs128.npy")
PORTS = ("Rotterdam", "Singapore")
SEED = 20261014


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(MANIFEST,
        usecols=["instance_id", "port", "product", "mmsi", "class_id"],
        dtype={"mmsi": str, "product": str})
    ids = np.load("D:/SAR_completion_cache_20261010/source_ids.npy").astype(str)
    pixels = np.load(PIXELS, mmap_mode="r")
    np.testing.assert_array_equal(data.instance_id.to_numpy(str), ids)
    if len(data) != len(pixels):
        raise ValueError("24-port pixel cache length mismatch")
    tokens = data["product"].str.split("_")
    if not tokens.str.len().eq(9).all():
        raise ValueError("Unexpected Sentinel-1 product key")
    data["acquisition"] = tokens.map(lambda x: "_".join((x[0], x[6], x[7])))
    data = data[data.mmsi.fillna("").str.fullmatch("[1-9][0-9]{8}")].copy()
    data["water_region"] = data.port
    data["sample_id"] = data.instance_id

    for i, port in enumerate(PORTS):
        target_acq = set(data.loc[data.port.eq(port), "acquisition"])
        target = data[data.acquisition.isin(target_acq)]
        source = data[~data.acquisition.isin(target_acq)]
        source = source[~source.mmsi.isin(set(target.mmsi))]
        folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + i)
        tr, va = next(folds.split(source, source.class_id, source.acquisition))
        validation = source.iloc[va]
        train = source.iloc[tr]
        train = train[~train.mmsi.isin(set(validation.mmsi))]
        episode = pilot_episode(target, port, SEED + i, balanced_quota=True)

        folder = ROOT / port.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "split.npz",
            ssl_train=train.index.to_numpy(np.int64),
            source_validation=validation.index.to_numpy(np.int64),
            target_acquisition=target.index.to_numpy(np.int64),
            target_port=target[target.port.eq(port)].index.to_numpy(np.int64),
        )
        np.savez_compressed(folder / "pilot_episode.npz",
            query=episode["query"],
            support_same=episode["support_same"],
            support_other=episode["support_other"],
            support_classes=episode["support_classes"],
        )
        summary = {
            "target_port": port, "seed": SEED + i,
            "source_train_rows": len(train), "source_val_rows": len(validation),
            "source_val_class_counts": np.bincount(validation.class_id, minlength=8).tolist(),
            "target_port_rows": int(target.port.eq(port).sum()),
            "heldout_acquisitions": len(target_acq),
            "pilot_acquisition": episode["acquisition"],
            "pilot_query_rows": len(episode["query"]),
            "pilot_budget_max": episode["budget_max"],
            "support_class_counts": np.bincount(episode["support_classes"], minlength=8).tolist(),
            "selection": "port chosen a priori; acquisition chosen by metadata capacity, not model outcomes",
            "ssl_target_pixels": 0,
        }
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
