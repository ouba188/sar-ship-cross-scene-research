"""Source-held-port readouts and latent contributions for calibration/gates."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import argparse
import gc
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from conditional_invariant_directions import nuisance_penalized_basis
from latent_usage_attribution import span_basis, raw_readout_gradients, score_contributions
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from residual_expert import fit_correction
from sar_joint_completion import SOURCE, write_json
from stable_port_directions import fit_direction_bank


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'module_completion_20261008/ports_projection'
OUT = ROOT/'evidence_aware_handoff_20261008'


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def pair_geometry(key, raw, labels, domains):
    path = OUT/'source_pair_cache'/f'{key}.npz'
    geometry = archive(PARENT/'episode_geometry_cache'/f'{key}.npz')
    if path.exists():
        result = archive(path)
        np.testing.assert_array_equal(result['train_rows'], geometry['train_rows'])
        return geometry, result
    started = time.monotonic()
    train = geometry['train_rows']
    x = np.array(raw[train], dtype=np.float32)
    pc = geometry['base_components']
    xs = ((x@pc.T-geometry['base_mean']@pc.T-geometry['base_center'])/geometry['base_scale']).astype(np.float32)
    base = RidgeClassifier(alpha=1., class_weight='balanced').fit(xs, labels[train])
    np.testing.assert_array_equal(base.classes_, np.arange(8))
    bc, bs = xs.mean(0), xs.std(0)+1e-6
    del xs
    center = x.mean(0, dtype=np.float64)
    variance = np.zeros(x.shape[1])
    for lo in range(0, len(x), 1024):
        difference = x[lo:lo+1024].astype(float)-center
        variance += np.square(difference).sum(0)
    scale = np.sqrt(variance/len(x))+1e-6
    x -= center
    x /= scale
    bank = fit_direction_bank(x, labels[train], domains[train])
    candidate = nuisance_penalized_basis(x, labels[train], domains[train], bank['pairs'], bank['pooled_basis'])
    q, singular = span_basis(candidate['basis'])
    result = dict(train_rows=train, raw_center=center, raw_scale=scale, q=q, singular_values=singular,
        base_coef=base.coef_, base_intercept=base.intercept_, base_residual_center=bc, base_residual_scale=bs)
    np.savez_compressed(path, **result)
    del x, bank
    gc.collect()
    print(f'PAIR {key}: {len(train)} source rows, {q.shape[1]} directions, {time.monotonic()-started:.1f}s', flush=True)
    return geometry, result


def make_episode(geometry, fitted, episode, raw, extra):
    rows, support, xs = episode['rows'], episode['support'], episode['xs']
    labels = episode['labels']
    phi = np.column_stack(((xs-fitted['base_residual_center'])/fitted['base_residual_scale']/np.sqrt(xs.shape[1]),
                           np.ones(len(rows))))
    bprior = xs@fitted['base_coef'].T+fitted['base_intercept']
    bcoef = fit_correction(phi[support], labels[support], bprior[support])
    base = bprior+phi@bcoef
    pca_ssl = SimpleNamespace(mean_=episode['ssl_mean'], components_=episode['ssl_components'])
    z = ((extra-episode['ssl_mean'])@episode['ssl_components'].T).astype(np.float32)
    xe = (np.column_stack((xs, z))-episode['expert_center'])/episode['expert_scale']
    prior = (xe@episode['expert_coef'].T+episode['expert_intercept']).astype(np.float32).astype(float)
    retained = (xe/np.sqrt(xe.shape[1])).astype(float)
    split = raw.shape[1]
    extra_discarded = ((discarded_coordinates(extra, pca_ssl)-episode['discarded_center'][split:])/
                       episode['discarded_scale'][split:])
    complement = np.column_stack((episode['base_discarded'], extra_discarded))/np.sqrt(split+extra.shape[1])
    kernel = .5*(retained@retained[support].T)+.5*(complement@complement[support].T)+1.
    beta = fit_kernel_correction(kernel[support], labels[support], prior[support])
    expert = prior+kernel@beta
    model = dict(pca=SimpleNamespace(components_=geometry['base_components']), scale=geometry['base_scale'],
        base=SimpleNamespace(coef_=fitted['base_coef']), expert=SimpleNamespace(coef_=episode['expert_coef']),
        expert_scale=episode['expert_scale'])
    gradients = raw_readout_gradients(model, bcoef, fitted['base_residual_scale'], retained[support],
                                     complement[support], beta, episode['discarded_scale'])
    _, contribution = score_contributions(np.asarray(raw[rows], dtype=float), fitted['raw_center'],
                                         fitted['raw_scale'], fitted['q'], gradients)
    query = np.zeros(len(rows), dtype=bool)
    query[episode['query']] = True
    if query[support].any():
        raise ValueError('Support enters source evaluation')
    return dict(rows=rows, support=support, query=query, labels=labels, base_scores=base, base_prior=bprior,
        expert_scores=expert, stable_source=contribution['base_source'],
        stable_base=contribution['base_source']+contribution['base_adaptation'],
        stable_expert=contribution['expert_source']+contribution['expert_retained']+contribution['expert_complement'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', nargs='*')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'source_pair_cache').mkdir(exist_ok=True)
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    labels, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    all_ports = list(np.unique(domains))
    raw = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=2):
        for item in inventory:
            port = item['port']
            if args.ports and port not in args.ports:
                continue
            folder = OUT/'source_episodes'/port
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'COMPLETE.json').exists():
                continue
            feature_folder = OUT/'source_features'/port
            if not (feature_folder/'COMPLETE.json').exists():
                print('WAITING FOR SOURCE FEATURES '+port, flush=True)
                break
            source_rows = np.load(feature_folder/'rows.npy')
            source_features = np.load(feature_folder/'features.npy', mmap_mode='r')
            descriptor = json.loads((PARENT/port/'EPISODES_COMPLETE.json').read_text(encoding='utf-8'))
            receipts = []
            for item_episode in descriptor['episodes']:
                ep = archive(Path(item_episode['path']))
                target_port = item_episode['port']
                key = '_'.join(f'{i:02d}' for i in sorted((all_ports.index(port), all_ports.index(target_port))))
                dest = folder/(Path(item_episode['path']).stem+'.npz')
                if not dest.exists():
                    geometry, fitted = pair_geometry(key, raw, labels, domains)
                    np.testing.assert_array_equal(ep['train_rows'], geometry['train_rows'])
                    if np.any(domains[ep['train_rows']] == port) or np.any(domains[ep['rows']] == port):
                        raise ValueError('Outer target enters source training')
                    if np.any(domains[ep['train_rows']] == target_port):
                        raise ValueError('Inner held port enters head/direction fitting')
                    local = np.searchsorted(source_rows, ep['rows'])
                    np.testing.assert_array_equal(source_rows[local], ep['rows'])
                    arrays = make_episode(geometry, fitted, ep, raw, source_features[local])
                    np.savez_compressed(dest, **arrays)
                receipts.append(dict(port=target_port, path=str(dest), pair_key=key,
                    rows=len(ep['rows']), query=len(ep['query']), support=64))
            write_json(folder/'COMPLETE.json', dict(outer_port=port, episodes=receipts,
                source_only=True, directions_and_heads_inner_port_held=True,
                encoder_inner_port_held=False, source_support_seed=0))
            print(f'SOURCE EPISODES COMPLETE {port}: {len(receipts)} ports', flush=True)


if __name__ == '__main__':
    main()
