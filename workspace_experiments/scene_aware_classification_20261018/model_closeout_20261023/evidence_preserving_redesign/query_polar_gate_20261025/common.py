"""Shared frozen-score and descriptor paths for query-level polar gating."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'hybrid_polar_fusion_20261025'
BASE = HERE.parents[1]
LOCAL = BASE / 'local_evidence_transport'
SEEDS = (17, 29, 43)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


previous = load('gate_frozen_fusion', PREVIOUS / 'train.py')
Data, metrics, write_json, exact_j = previous.Data, previous.metrics, previous.write_json, previous.exact_j


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def scores(data, seed):
    out = {}
    for arm in ('base', 'prior_vv', 'center_vh', 'old_full', 'prior_late', 'hybrid_source', 'hybrid_equal'):
        with np.load(PREVIOUS / 'results' / f'{arm}_{seed}.npz') as z:
            np.testing.assert_array_equal(z['up_index'], data.up[:data.n])
            out[arm] = z['score'].copy()
    return out
