"""Fold-local support correction followed by residual-utility fitting."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import importlib.util
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'frozen_candidate_coverage_20261012/budget'))
import run_coverage as c

spec = importlib.util.spec_from_file_location('utility_core', ROOT / 'classwise_utility_adaptation_20261012/core.py')
utility = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utility)

ALPHA = .25
TAU = .03


def corrected_support(raw, model):
    delta = c.predict(model, raw[:, c.COLS], np.zeros(len(raw), int), 'image_history')
    return c.adapt_support(raw, delta, 1.)


def make_parts(y, groups):
    if len(np.unique(groups)) < 4:
        raise ValueError('The fixed grouped protocol requires four support identities')
    parts, notes = [], []
    for repeat, seed in enumerate(utility.SEEDS):
        with warnings.catch_warnings(record=True) as captured:
            split = list(StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(
                np.zeros((len(y), 1)), y, groups))
        notes.extend(str(w.message) for w in captured)
        for fold, (train, val) in enumerate(split):
            parts.append(dict(repeat=repeat, fold=fold, train=train.tolist(), validation=val.tolist()))
    return parts, notes


def paired_oof(raw, prior, kb, base_prior, labels, groups, model, parts):
    n, width = prior.shape
    values = np.full((3, 4, n, width), np.nan)
    seen = np.zeros((3, n), int)
    times = dict(oof_base_seconds=0., oof_raw_seconds=0., oof_initial_seconds=0.)
    start = time.perf_counter()
    raw_kernel = raw @ raw.T
    times['oof_raw_seconds'] += time.perf_counter()-start
    for part in parts:
        r = part['repeat']
        train, val = np.asarray(part['train']), np.asarray(part['validation'])
        assert not set(groups[train]) & set(groups[val])
        assert not np.intersect1d(train, val).size
        assert len(train)+len(val) == n
        ridge = len(train)/8.
        start = time.perf_counter()
        ab = c.coefficients(kb[np.ix_(train, train)], base_prior[train], labels[train], ridge)
        values[r, 3, val] = softmax(base_prior[val] + kb[np.ix_(val, train)] @ ab, axis=1)
        times['oof_base_seconds'] += time.perf_counter()-start
        start = time.perf_counter()
        ar = c.coefficients(raw_kernel[np.ix_(train, train)], prior[train], labels[train], ridge)
        values[r, 1, val] = raw_kernel[np.ix_(val, train)] @ ar
        times['oof_raw_seconds'] += time.perf_counter()-start
        start = time.perf_counter()
        # Each inner fit rebuilds support centering/energy; validation stays raw.
        changed = corrected_support(raw[train], model)
        ai = c.coefficients(changed @ changed.T, prior[train], labels[train], ridge)
        values[r, 2, val] = (raw[val] @ changed.T) @ ai
        times['oof_initial_seconds'] += time.perf_counter()-start
        values[r, 0, val] = prior[val]
        seen[r, val] += 1
    np.testing.assert_array_equal(seen, np.ones((3, n), int))
    assert np.isfinite(values).all()
    components = [values[:, j].reshape(-1, width) for j in range(4)]
    return components, np.tile(labels, 3), times


def fit_pair(raw, prior, kb, base_prior, labels, groups, model, parts):
    times = {}
    start = time.perf_counter()
    ab = c.coefficients(kb, base_prior, labels, len(labels)/8.)
    times['base_fit_seconds'] = time.perf_counter()-start
    start = time.perf_counter()
    wr = raw.T @ c.coefficients(raw @ raw.T, prior, labels, len(labels)/8.)
    times['raw_fit_seconds'] = time.perf_counter()-start
    start = time.perf_counter()
    changed = corrected_support(raw, model)
    wi = changed.T @ c.coefficients(changed @ changed.T, prior, labels, len(labels)/8.)
    times['initial_fit_seconds'] = time.perf_counter()-start
    start = time.perf_counter()
    wg = wi.copy()
    wg[:, np.bincount(labels, minlength=8) == 0] = 0.
    times['guard_extra_seconds'] = time.perf_counter()-start
    oof, oy, oof_times = paired_oof(raw, prior, kb, base_prior, labels, groups, model, parts)
    times.update(oof_times)
    gates, optim = {}, {}
    for arm, index in [('light', 1), ('combined', 2)]:
        start = time.perf_counter()
        gates[arm], optim[arm] = utility.fit_gate(oof[0], oof[index], oof[3], oy, TAU, 'joint')
        times[arm+'_gate_seconds'] = time.perf_counter()-start
    return dict(base_coef=ab, raw_weight=wr, initial_weight=wi, guard_weight=wg,
                gate_light=gates['light'], gate_combined=gates['combined'],
                oof_prior=oof[0], oof_raw=oof[1], oof_initial=oof[2],
                oof_base=oof[3], oof_labels=oy), times, optim


def scores(base, prior, raw_residual, initial_residual, guard_residual, gate_light, gate_combined):
    logits = dict(raw=prior+raw_residual, initial=prior+initial_residual,
        light=prior+(1+ALPHA*(gate_light-1))*raw_residual,
        combined=prior+(1+ALPHA*(gate_combined-1))*initial_residual,
        initial_guard=prior+guard_residual)
    pb = softmax(base, axis=1)
    return {k: .5*(pb+softmax(v, axis=1)) for k, v in logits.items()}
