"""Port-group-held-out, matched-source frozen cue comparison.

Reuse the already verified public operator and fixed mapper recipe verbatim.
Only input cohorts and output destinations change; historical outputs stay intact.
"""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

from pathlib import Path
import sys
import json
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PUBLIC = OUT.parent / "public_oss_comparison_20261012"
sys.path.insert(0, str(PUBLIC))
import common as public

ROOT, SOURCE = public.ROOT, public.SOURCE
TASKROOT = ROOT / "learned_frequency_residual_20261009/tasks/self24ports"
PORTS = [p.name for p in sorted(TASKROOT.iterdir()) if (p / "target.npz").exists()]
PUBLIC_TASKS = ("oss1", "oss2", "oss2_grdh")
archive, write = public.archive, public.write


def fold_ports(fold):
    return PORTS[fold::5]


def permitted_rows(fold):
    rows = public.source_rows("shared_frozen", "oss1")
    held = []
    for port in fold_ports(fold):
        saved = archive(TASKROOT / port / "target.npz")
        rows = np.intersect1d(rows, saved["source_rows"])
        held.extend(saved["rows"])
    if np.intersect1d(rows, held).size:
        raise ValueError("Held port rows entered source fit")
    frame = pd.read_csv(SOURCE / "manifest.csv", usecols=["mmsi", "port", "class_id"])
    mmsi = pd.to_numeric(frame.mmsi, errors="coerce").fillna(0).astype(np.int64)
    known = set(mmsi.iloc[held]) - {0}
    if known.intersection(set(mmsi.iloc[rows])):
        raise ValueError("Known held-port vessel entered source fit")
    if set(frame.port.iloc[rows]).intersection(fold_ports(fold)):
        raise ValueError("Held port entered source fit")
    if len(np.unique(frame.class_id.iloc[rows])) != 8:
        raise ValueError("Source lost a class")
    return rows


def source_arrays(fold):
    rows = permitted_rows(fold)
    raw = np.load(SOURCE / "pooled_features.npy", mmap_mode="r")
    c4 = np.load(ROOT / "c4_base_trial_20261009/pooled_c4.npy", mmap_mode="r")
    extra = np.load(ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy", mmap_mode="r")[rows]
    return rows, raw, c4, extra


def target_arrays(name):
    import evaluate
    if name in PUBLIC_TASKS:
        return ORIGINAL_TARGET_ARRAYS("shared_frozen", name.split("_")[0], name.endswith("_grdh"))
    saved = archive(TASKROOT / name / "target.npz")
    rows = saved["rows"]
    frame = pd.read_csv(SOURCE / "manifest.csv", usecols=["instance_id", "class_id"])
    np.testing.assert_array_equal(frame.instance_id.to_numpy(str)[rows], saved["ids"])
    np.testing.assert_array_equal(frame.class_id.to_numpy()[rows[saved["support"]]], saved["support_labels"])
    return dict(raw=np.load(SOURCE / "pooled_features.npy", mmap_mode="r")[rows],
        c4=np.load(ROOT / "c4_base_trial_20261009/pooled_c4.npy", mmap_mode="r")[rows],
        extra=np.load(ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy", mmap_mode="r")[rows],
        ids=saved["ids"], support=saved["support"], support_labels=saved["support_labels"],
        eval_masks=saved["eval_masks"])


def freeze():
    protocol = dict(date="2026-10-12", ports=PORTS, folds=[dict(fold=i, ports=fold_ports(i),
        source_n=len(permitted_rows(i))) for i in range(5)], public_targets=PUBLIC_TASKS,
        partition="Alphabetically sorted 23 eligible ports, round-robin five folds; no outcome-based selection",
        source="Intersection of both public-purged source sets and every held-port permitted source set",
        fitting="One source geometry, source heads and linear/MLP mapper pair per fold; identical across that fold's port/public targets",
        features="Unchanged pre-task e206 R18 + ImageNet R50 original/C4 features. No new encoder training",
        operator="Exact public_oss_comparison_20261012.evaluate operator; no port drift/stable/incidence/frequency modules",
        mapper="Unchanged ridge.01 and fixed 8epoch MLP, seed20261010; final checkpoint, no sweep",
        support="Original five64 supports per port/OSS1/OSS2; prior GRDH five64 supports retained",
        inference="R/P/E/PE/T/PE_scaled/DD; fixed R50 prior except DD; expert and compact C4 late readouts",
        primary="Mean within-port macro Brier/BA, equal port weight; public fold average and same-fold class-matched comparison",
        diagnostics="Class-matched risks; support-present/absent strata; source-standardized input distance; source-defined shift bins",
        limitations=["Five source models, not one model across all 23 ports", "Grouped-port holdout is harder/different from historic single-port LOPO",
            "Frozen encoder historical/pretraining exposure unchanged; not a from-scratch inductive benchmark",
            "Support-query can share vessel/product; five draws and public folds are not independent test datasets",
            "Residual is mapper-relative, not proof of new information; correlations do not identify physical causes",
            "64-label operating point only; compact readout is not latest full-system model"])
    path = OUT / "PROTOCOL.json"
    value = json.loads(json.dumps(protocol))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Protocol changed")
    else:
        write(path, protocol)


def configure_evaluation(fold):
    import evaluate
    global ORIGINAL_TARGET_ARRAYS
    if "ORIGINAL_TARGET_ARRAYS" not in globals():
        ORIGINAL_TARGET_ARRAYS = evaluate.target_arrays
    dst = OUT / f"fold_{fold}"
    evaluate.OUT = dst
    evaluate.fit_dir = lambda regime, name: dst / "fits"
    evaluate.mapper_path = lambda regime, name: dst / "fits/mapper.npz"
    evaluate.source_arrays = lambda regime, name: source_arrays(fold)
    evaluate.target_arrays = lambda regime, name, grdh_only=False: target_arrays(name)
    return evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["fit", "predict"])
    args = parser.parse_args()
    freeze()
    with threadpool_limits(limits=2):
        if args.stage == "fit":
            import fit_mappers
            fit_mappers.torch.set_num_threads(2)
            fit_mappers.torch.backends.cuda.matmul.allow_tf32 = False
            fit_mappers.torch.backends.cudnn.allow_tf32 = False
            for fold in range(5):
                fit_mappers.fit_dir = lambda regime, name: OUT / f"fold_{fold}/fits"
                fit_mappers.source_arrays = lambda regime, name: source_arrays(fold)
                fit_mappers.fit("shared_frozen", str(fold))
        else:
            for fold in range(5):
                evaluate = configure_evaluation(fold)
                bundle = evaluate.prepare("shared_frozen", "shared")
                for name in fold_ports(fold) + list(PUBLIC_TASKS):
                    evaluate.predict("shared_frozen", name, bundle)
            write(OUT / "PREDICTIONS_FROZEN.json", dict(folds=5, port_targets=23, public_targets=15,
                all_requested_predictions_saved=True, query_classification_metrics_evaluated=False))


if __name__ == "__main__":
    main()
