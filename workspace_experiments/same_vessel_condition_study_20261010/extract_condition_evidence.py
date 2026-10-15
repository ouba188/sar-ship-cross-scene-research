"""Compact source-only descriptors of frozen CNN spatial activation maps."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dctn

BASE = Path(__file__).resolve().parent
OUT = BASE / 'condition_evidence_20261011'
CACHE = Path('E:/Hermes/paper_tracking/sensitivity/neural_complement_pilot_20261007')
ROLES = ('factor_fit', 'test_source', 'val_new_ship')
NAMES = ('low_log_energy', 'middle_log_energy', 'high_log_energy',
         'low_energy_share', 'middle_energy_share', 'center_border_log_ratio')
EPS = 1e-12


def spectral_functions():
    # Reuse the two existing pure functions without importing its Torch training stack.
    path = BASE / 'spectral_residual.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('masks', 'transform')]
    scope = {'np': np, 'dctn': dctn}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['masks'], scope['transform']


def describe_maps(h, masks, transform):
    h = np.asarray(h, dtype=np.float32)
    coefficients = transform(h).astype(np.float64)
    bands = masks(h.shape[-1]).astype(np.float64)
    energy = np.stack([np.mean((coefficients * band) ** 2, axis=(1, 2, 3))
                       for band in bands], axis=1)
    total = energy.sum(1, keepdims=True)
    shares = np.divide(energy, total, out=np.zeros_like(energy), where=total > 0)
    power = np.mean(h.astype(np.float64) ** 2, axis=1)
    center = np.zeros(h.shape[-2:], dtype=bool)
    center[2:6, 2:6] = True
    ratio = np.log((power[:, center].mean(1) + EPS)
                   / (power[:, ~center].mean(1) + EPS))
    values = np.column_stack([np.log(energy + EPS), shares[:, :2], ratio])
    return values.astype(np.float32), energy


def checks(masks, transform):
    np.testing.assert_allclose(masks().sum(0), 1, atol=1e-7)
    h = np.zeros((2, 3, 8, 8), dtype=np.float32)
    h[1] = 2
    values, energy = describe_maps(h, masks, transform)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[0, 3:], 0)
    np.testing.assert_allclose(energy[1], [4, 0, 0], atol=1e-6)
    np.testing.assert_allclose(values[1, 3:], [1, 0, 0], atol=1e-6)
    rng = np.random.default_rng(20261011)
    h = rng.normal(size=(3, 4, 8, 8)).astype(np.float32)
    values, energy = describe_maps(h, masks, transform)
    scaled, doubled_energy = describe_maps(2 * h, masks, transform)
    np.testing.assert_allclose(doubled_energy, 4 * energy, rtol=1e-6)
    np.testing.assert_allclose(scaled[:, 3:], values[:, 3:], atol=1e-6)
    np.testing.assert_allclose(scaled[:, :3] - values[:, :3], np.log(4), atol=1e-6)
    return ['mask_partition', 'zero_energy_finite', 'constant_map_dc_only',
            'uniform_center_border_zero', 'amplitude_and_share_scaling']


def main():
    start = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    output = OUT / 'evidence_features.npz'
    if output.exists():
        raise ValueError('Preserve the completed extraction')
    masks, transform = spectral_functions()
    passed = checks(masks, transform)
    meta = pd.read_csv(BASE / 'observations.csv')
    np.testing.assert_array_equal(meta.obs_id, np.arange(len(meta)))
    split_path = BASE / 'grouped_condition_anchor_20261011/source_split.npz'
    with np.load(split_path) as saved:
        splits = {role: saved[role].copy() for role in ROLES}
    requested = np.concatenate(list(splits.values()))
    obs_ids = np.unique(requested)
    assert len(obs_ids) == len(requested)
    selected = meta.iloc[obs_ids]
    assert not selected.port.isin(('Santos', 'Port Klang', 'Los Angeles')).any()
    fit_ships = set(meta.iloc[splits['factor_fit']].mmsi)
    assert not fit_ships & set(meta.iloc[np.r_[splits['test_source'], splits['val_new_ship']]].mmsi)
    roles = np.empty(len(obs_ids), dtype='<U12')
    for role, ids in splits.items():
        roles[np.searchsorted(obs_ids, ids)] = role
    rows = selected.manifest_row.to_numpy(dtype=np.int64)
    maps = np.load(CACHE / 'source_layer3.npy', mmap_mode='r')
    manifest_path = Path('E:/\u4e34\u65f6\u4f1a\u8bdd/visual_reliable_baseline/artifacts/eight_class_adaptive_20260916/manifest.csv')
    keys = ['instance_id', 'product', 'port', 'mmsi', 'class_id']
    manifest = pd.read_csv(manifest_path, usecols=keys, dtype=str, keep_default_na=False)
    observation_keys = pd.read_csv(BASE / 'observations.csv', usecols=keys, dtype=str, keep_default_na=False)
    assert len(manifest) == len(maps) and maps.shape[1:] == (256, 8, 8)
    np.testing.assert_array_equal(manifest.iloc[rows][keys].to_numpy(),
                                  observation_keys.iloc[obs_ids][keys].to_numpy())
    values = np.empty((len(obs_ids), len(NAMES)), dtype=np.float32)
    zero_energy = 0
    for lo in range(0, len(rows), 64):
        h = np.array(maps[rows[lo:lo + 64]], dtype=np.float32)
        assert np.isfinite(h).all()
        part, energy = describe_maps(h, masks, transform)
        values[lo:lo + len(h)] = part
        zero_energy += int((energy.sum(1) == 0).sum())
    assert np.isfinite(values).all()
    np.savez_compressed(output, obs_ids=obs_ids, manifest_rows=rows, features=values,
                        feature_names=np.array(NAMES), roles=roles)
    with np.load(output) as saved:
        np.testing.assert_array_equal(saved['obs_ids'], obs_ids)
        np.testing.assert_array_equal(saved['manifest_rows'], rows)
        np.testing.assert_array_equal(saved['features'], values)
    coverage = {}
    for role, ids in splits.items():
        part = meta.iloc[ids]
        coverage[role] = dict(observations=len(part), ships=int(part.mmsi.nunique()),
            ports=int(part.port.nunique()), incidence_known=int(part.incidence_deg.notna().sum()))
    receipt = dict(status='complete', seconds=time.perf_counter() - start,
        output=str(output), shape=list(values.shape), feature_names=list(NAMES),
        source_map=str(CACHE / 'source_layer3.npy'), source_map_shape=list(maps.shape),
        source_map_dtype=str(maps.dtype), split=str(split_path), coverage=coverage,
        observation_alignment='Global obs_id equals observations.csv row; manifest_row checked against original manifest on instance_id, product, port, mmsi, class_id.',
        extraction='Memory-mapped source cache, batches64, NumPy/SciPy only; no network fitting, no GPU, no target query evaluation.',
        frequency='CNN layer3 spatial frequencies, NOT radar carrier/polarization frequency bands. Reuse masks/transform pure function AST from spectral_residual.py; orthonormal DCT-II, low Gaussian width.2, upper.7, middle=upper-low, high=1-upper.',
        energy='Mean squared masked DCT coefficients over channels and8x8 grid; natural log(E+1e-12). Shares=Eband/sum(Ebands). Soft masks overlap, so this denominator is branch-energy sum, NOT total original-map energy. Omit redundant high share.',
        center_border='Natural log ratio of mean squared activation at central4x4 versus remaining48cells, with1e-12 floor. Proxy only: not segmented ship/background, deep receptive fields overlap.',
        upstream='e206 epoch40 SAR SSL R18 frozen prefix through layer3, original float16 cache. Cache maker uses channel index1 repeated3channels and /255 from historical128pixel chips; no new image encoding. Historical protocol acknowledges inherited F0921 unlabeled pretraining. Backbone not refit for current ship split; no full-pipeline unseen-ship or unseen-port guarantee.',
        provenance_files=[str(CACHE / 'protocol.json'),
            'E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/train_neural_complement.py',
            str(BASE / 'spectral_residual.py')],
        zero_energy_observations=zero_energy, checks=passed + ['source_only_cohort',
            'exact_manifest_alignment', 'fit_vs_held_ship_disjoint', 'export_replay'],
        feature_min=values.min(0).tolist(), feature_max=values.max(0).tolist())
    (OUT / 'EXTRACTION.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='complete', shape=list(values.shape), coverage=coverage,
                         zero_energy_observations=zero_energy, seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
