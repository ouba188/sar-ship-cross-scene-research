"""Frozen inputs for the public OpenSARShip cue comparison."""

import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "2"

from pathlib import Path
import json
import sys
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = Path("E:/Hermes/paper_tracking/sensitivity")
sys.path.insert(0, str(ROOT / "oof_reliability_check"))
from compact_fusion_source import SOURCE, CACHE
from dual_reconstruction_core import reconstruct

DATASETS = ("oss1", "oss2")
REGIMES = ("task_trained", "shared_frozen")
KINDS = ("linear", "mlp")
ARMS = ("R", "P", "E", "PE", "T", "PE_scaled", "DD")
BATCH = 512


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def original(name):
    return ROOT / f"opensarship{name[-1]}_transfer_20261007"


def joint(name):
    return ROOT / "module_completion_20261008" / name


def source_rows(regime, name):
    if regime == "shared_frozen":
        return np.intersect1d(*(archive(original(d) / "source_split.npz")["source_rows"] for d in DATASETS))
    return archive(original(name) / "source_split.npz")["source_rows"]


def fit_dir(regime, name):
    return OUT / "fits" / ("shared_frozen" if regime == "shared_frozen" else name)


def source_arrays(regime, name):
    rows = source_rows(regime, name)
    raw = np.load(SOURCE / "pooled_features.npy", mmap_mode="r")
    c4 = np.load(ROOT / "c4_base_trial_20261009/pooled_c4.npy", mmap_mode="r")
    if regime == "shared_frozen":
        extra = np.asarray(np.load(ROOT / "neural_complement_pilot_20261007/frozen_replay/source_extra.npy", mmap_mode="r")[rows])
    else:
        cached = archive(CACHE / f"{name}_source_extra.npz")
        np.testing.assert_array_equal(cached["rows"], rows)
        extra = cached["features"]
    return rows, raw, c4, extra


def mapper_path(regime, name):
    if regime == "task_trained" and name == "oss2":
        return ROOT / "dual_reconstruction_20261010/targets/oss2/oss2/mapper.npz"
    return fit_dir(regime, name) / "mapper.npz"


def freeze():
    p = dict(date="2026-10-12", datasets=DATASETS, regimes=REGIMES,
        task_trained="Original dataset-specific purged source, existing task-trained R18 tail and geometry; OSS2 mapper reused, OSS1 same fixed recipe",
        shared_frozen="Identical ImageNet R50 and pre-task e206 R18 weights; common source-row intersection, one shared PCA/scales, heads and mapper pair",
        source="Existing self-collected source corpus remains training data; public datasets are targets, not a public-source-only benchmark",
        mapper="R50original2048+C4pooled2048 -> R18512; source standardization floor1e-4; linear ridge.01; MLP4096-512-512 GELU AdamW lr.001 wd.0001,8epochs,batch512,seed20261010",
        arms=ARMS, decomposition="P=deployed reconstructed SSL coordinates; E=T-P; preserve original sqrt192/sqrt2560 scales",
        prior="R50 prior identical for R/P/E/PE/T/PE_scaled; DD additionally replaces prior with dual source head",
        scaled="PE_scaled analytically matches added SSL support trace to T separately each draw, no search/query use",
        base="New matched compact C4-PCA128 source ridge1 plus target ridge8, identical between arms; NO learned-frequency/stable/incidence modules",
        readouts=["expert", "c4_late"], support="Historical5 draws of64 rows; same supports per method; retain original and fixed-common-query scopes",
        evaluation="All four regime/dataset predictions frozen before new query metrics; macro/micro Brier,NLL,OA,BA,class recalls and support coverage",
        limitations=["Architecture/recipe matches in task_trained, not identical weights or sources",
                     "shared_frozen controls weights and source but omits task-tail adaptation; not the full recent system",
                     "Historical encoder/development exposure remains; target support/query is row-disjoint, not necessarily vessel-disjoint",
                     "F0921 public status unestablished here; historical reference only, no new F or self-port target evaluation",
                     "Residual usefulness is mapper-relative, not information-theoretic novelty; one new mapper training seed",
                     "64 is one budget; five overlapping supports are not independent neural seeds"])
    path = OUT / "PROTOCOL.json"
    encoded = json.loads(json.dumps(p))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != encoded:
            raise ValueError("Changed frozen protocol")
    else:
        write(path, p)
