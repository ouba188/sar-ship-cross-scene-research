"""Frozen OSS2 cue intervention cohort and existing expert operator."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import sys
import json
import pickle
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "ports_root_cause_20261013"
sys.path.insert(0, str(PREVIOUS))
import run_factorial as run
from analyze_factorial import metadata, load

CONDITIONS = ("original", "target_flat", "outer_flat", "sham_flat",
              "target_shuffle", "outer_shuffle", "target_display", "outer_display",
              "wide_target_flat", "wide_outer_flat", "replay_original")
PATHS = ("R_query", "T_Aonly", "T_Bonly", "T_query", "R_joint", "T_joint")


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def cohort():
    data = run.target("oss2", "original")
    frame = run.pilot.frame("oss2").set_index("sample_id").loc[data["ids"]].reset_index()
    detail = pd.read_csv(PREVIOUS / "dataset_audit/public_metadata.csv", keep_default_na=False)
    detail = detail[detail.dataset.eq("oss2")].set_index("sample_id").loc[data["ids"]]
    for key in ("center_x", "center_y", "head_x", "head_y", "tail_x", "tail_y",
                "noise_xml", "datatake_key", "identity"):
        frame[key] = detail[key].to_numpy()
    return data, frame


def encode(data, bundle):
    model, stats, heads = (bundle[k] for k in ("model", "stats", "heads"))
    raw, extra = data["raw"], np.asarray(data["extra"])
    rr, rd = run.evaluate.r50_views(raw, model, stats)
    x = run.evaluate._expert_inputs(raw, extra, model)
    return dict(r=np.c_[rr, rd]*np.sqrt(.5), t=run.evaluate.ssl_map(extra, model, stats),
                pr=heads["r50"].decision_function(x[:, :128]).astype(np.float32).astype(float))


def features(condition, original):
    if condition == "original":
        return original
    folder = OUT / "features" / condition
    if not (folder / "COMPLETE.json").exists():
        raise RuntimeError("Unfinished region features: " + condition)
    np.testing.assert_array_equal(np.load(folder / "ids.npy"), original["ids"])
    return {k: np.load(folder / (k + ".npy"), mmap_mode="r") for k in ("raw", "extra")}


def macro(a, y):
    return float(np.mean([np.mean(a[y == c]) for c in np.unique(y)]))


def weights(y):
    classes, counts = np.unique(y, return_counts=True)
    w = np.empty(len(y), float)
    for c, n in zip(classes, counts):
        w[y == c] = 100. / (len(classes)*n)
    return w


def class_margin(z, y):
    value = z[np.arange(len(y)), y]
    return value - (z.sum(1) - value) / (z.shape[1] - 1)
