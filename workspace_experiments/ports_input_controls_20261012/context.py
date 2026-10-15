"""Fixed-input intervention pilot using existing port/public source models."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / "ports_public_data_study_20261012"
sys.path.insert(0, str(PREVIOUS))
import crossfit
ROOT, SOURCE = crossfit.ROOT, crossfit.SOURCE
archive, write = crossfit.archive, crossfit.write
PORTS = ("Los Angeles", "Port Klang", "Santos")
DATASETS = (*PORTS, "oss1", "oss2")
CONDITIONS = ("original", "display", "crop75", "crop75_display")
FOLDS = (2, 3)


def tasks(fold):
    return [p for p in PORTS if p in crossfit.fold_ports(fold)] + list(crossfit.PUBLIC_TASKS)


def frame(name):
    if name in PORTS:
        f = pd.read_csv(SOURCE / "manifest.csv", low_memory=False, keep_default_na=False)
        rows = archive(crossfit.TASKROOT / name / "target.npz")["rows"]
        f = f.iloc[rows].copy()
        return f.rename(columns={"instance_id": "sample_id", "image_relpath_vv": "path_vv", "image_relpath_vh": "path_vh"})
    return pd.read_csv(crossfit.public.original(name) / "target_manifest.csv", keep_default_na=False).rename(
        columns={"chip_path_vv": "path_vv", "chip_path_vh": "path_vh"})


def archived(name):
    evaluate = crossfit.configure_evaluation(2)
    return crossfit.target_arrays(name)


def freeze():
    value = dict(date="2026-10-12", datasets=DATASETS, conditions=CONDITIONS, folds=FOLDS,
        targets={str(f): tasks(f) for f in FOLDS}, selection="Existing diagnostic trio LA/Klang/Santos, not all23ports; all public rows retained",
        inputs="Identity, pooled-two-polarization q01/q99 shared affine display stretch, central75percent native crop, crop then display",
        crop="floor(.75*width/height), centered with integer left/top; same fractional rule for each polarization",
        display="Quantiles of pooled VV/VH bilinear128 L pixels; apply shared affine to native VV/VH and VH RGB; round,clip,uint8; flat pair unchanged",
        frozen="Existing prior5fold source models2/3: encoders, source heads, mapper, PCA/scales, ridge8, temperature1; no source transforms/refits",
        adaptation="Apply input intervention to BOTH support and query, re-solve the same five64 support systems",
        readouts="Original seven cue arms, both linear/MLP, expert and compactC4; no learned-frequency/port-specific modules",
        baseline="Reextract original native images with exact legacy encoder preprocessing; compare archived feature and prediction replay",
        outcomes="Freeze predictions first; BA/OA, macro/microBrier, featureR2/shift, contrast/footprint and class cue ordering; paired input effects",
        limitations=["Target-input intervention under a fixed source model, NOT matched transformed-source retraining",
            "Crop75 is framing perturbation, NOT physical occupancy harmonization; may truncate targets",
            "Display mapping is not radiometric calibration; clipping and polarization contrast may change",
            "Frozen pretrained exposure remains; three ports only; B64/one mapper seed; not independent confirmation",
            "No target outcome tuning or automated selection among four inputs"])
    value = json.loads(json.dumps(value))
    p = OUT / "PROTOCOL.json"
    if p.exists():
        if json.loads(p.read_text(encoding="utf-8")) != value:
            raise ValueError("Changed input-control protocol")
    else:
        write(p, value)
