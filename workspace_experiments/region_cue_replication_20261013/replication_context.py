"""Frozen task/model identities for cross-dataset regional cue replication."""

from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
OLD = OUT.parent / "region_cue_probe_20261013"
sys.path.insert(0, str(OLD))
import region_context as rc
from transforms_regions import transform, read_pair

CONDITIONS = rc.CONDITIONS
PATHS = rc.PATHS
CROSS = rc.run.crossfit
SOURCE = CROSS.SOURCE


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def all_tasks():
    result = [(name, fold) for fold in range(5) for name in CROSS.fold_ports(fold)]
    return result + [(name, fold) for name in ("oss1", "oss2") for fold in range(5)]


def feature_root(dataset):
    if dataset == "ports":
        return Path("C:/CodexArtifacts/region_cue_replication_20261013/ports/features")
    return OLD / "features" if dataset == "oss2" else OUT / dataset / "features"


def frame_for(dataset):
    if dataset == "oss2":
        return pd.read_csv(OLD / "cohort.csv", keep_default_na=False)
    return pd.read_csv(OUT / dataset / "frame.csv", keep_default_na=False)


def task_data(name):
    if name in ("oss1", "oss2"):
        # The old crossfit predictions used original release caches, not later input-control reextractions.
        data = CROSS.target_arrays(name)
        frame = frame_for(name).set_index("sample_id").loc[data["ids"]].reset_index()
        positions = np.arange(len(frame))
        dataset = name
    else:
        saved = rc.run.archive(CROSS.TASKROOT / name / "target.npz")
        positions = saved["rows"]
        frame = frame_for("ports").iloc[positions].reset_index(drop=True)
        data = {k: saved[k] for k in ("ids", "support", "support_labels", "eval_masks")}
        data["raw"] = np.load(SOURCE / "pooled_features.npy", mmap_mode="r")[positions]
        data["extra"] = np.load(CROSS.ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy",
                                 mmap_mode="r")[positions]
        dataset = "ports"
    np.testing.assert_array_equal(frame.sample_id.to_numpy(str), data["ids"])
    np.testing.assert_array_equal(frame.class_id.to_numpy(int)[data["support"]], data["support_labels"])
    controls_path = OLD / "region_metadata.csv" if dataset == "oss2" else OUT / dataset / "region_metadata.csv"
    controls = pd.read_csv(controls_path).iloc[positions].reset_index(drop=True)
    np.testing.assert_array_equal(controls.sample_id.to_numpy(str), data["ids"])
    return data, frame, controls, dataset, positions


def load_features(dataset, positions, condition):
    folder = feature_root(dataset) / ("replay_original" if condition == "original" else condition)
    if not (folder / "COMPLETE.json").exists():
        raise RuntimeError("Unfinished features: " + str(folder))
    return {k: np.load(folder / (k+".npy"), mmap_mode="r")[positions] for k in ("raw", "extra")}


def bundle_for(fold):
    with (CROSS.OUT / f"fold_{fold}/fits/geometry.pkl").open("rb") as handle:
        bundle = pickle.load(handle)
    np.testing.assert_array_equal(bundle["source_rows"], CROSS.permitted_rows(fold))
    return bundle
