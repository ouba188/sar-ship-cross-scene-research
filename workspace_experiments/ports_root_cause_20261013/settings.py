"""Fixed source/target factorial using the existing held-port pilot."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / "ports_input_controls_20261012"
sys.path.insert(0, str(INPUT))
import context as pilot
import predict_inputs

SOURCE, ROOT = pilot.SOURCE, pilot.ROOT
crossfit, write, archive = pilot.crossfit, pilot.write, pilot.archive
FOLDS, PORTS = pilot.FOLDS, pilot.PORTS
CONDITIONS = ("original", "display")
ARMS = ("R", "T", "DD")
HEADS = ("prior", "expert", "c4_late")


def source_frame():
    return pd.read_csv(SOURCE / "manifest.csv", keep_default_na=False, low_memory=False).rename(
        columns={"instance_id": "sample_id", "image_relpath_vv": "path_vv", "image_relpath_vh": "path_vh"})


def permitted():
    return np.union1d(*(crossfit.permitted_rows(f) for f in FOLDS))


def source_arrays(fold, condition):
    rows = crossfit.permitted_rows(fold)
    folder = OUT / "features" / condition
    if not (folder / "COMPLETE.json").exists():
        raise RuntimeError("Unfinished source features")
    np.testing.assert_array_equal(np.intersect1d(rows, np.load(folder / "rows.npy")), rows)
    np.testing.assert_array_equal(np.load(folder / "ids.npy")[rows], source_frame().sample_id.to_numpy(str)[rows])
    values = {k: np.load(folder / (k+".npy"), mmap_mode="r") for k in ("raw", "c4", "extra")}
    return rows, values["raw"], values["c4"], values["extra"][rows]


def target(name, condition):
    crossfit.configure_evaluation(2)
    return predict_inputs.data_for(name, condition)


def labels(name, ids):
    base = "oss2" if name == "oss2_grdh" else name
    return pilot.frame(base).set_index("sample_id").loc[ids].class_id.to_numpy(int)
