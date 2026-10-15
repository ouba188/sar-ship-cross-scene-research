"""Nested source-port selection for the conditional directional residual."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from conditional_direction_residual import fit_predict, normalize_direction, rank_conditions
from probe_nested_direction_geometry import archive, inputs, EPISODES, ROOT, C4, TRIAL


OUT = ROOT/'conditional_direction_residual_20261009'
ETAS = (0., .25, 1., 4.)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def source_port(port, frame, context, zero, averaged, descriptors):
    folder = OUT/'source'/port
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        return
    start = time.perf_counter()
    ports = sorted(frame.port.unique())
    items = json.loads((EPISODES/port/'EPISODES_COMPLETE.json').read_text())['episodes']
    labels = frame.class_id.to_numpy()
    records = []
    for i, item in enumerate(items):
        ep = archive(item['path'])
        train, rows, s, q = ep['train_rows'], ep['rows'], ep['support'], ep['query']
        key = '_'.join(f'{j:02d}' for j in sorted([ports.index(port), ports.index(item['port'])]))
        geo = archive(EPISODES/'episode_geometry_cache'/f'{key}.npz')
        pair = archive(ROOT/'evidence_aware_handoff_20261008/source_pair_cache'/f'{key}.npz')
        head = archive(TRIAL/'nested_geometry_gpu/pair_fit'/f'{key}.npz')
        for cache in (geo, pair, head):
            np.testing.assert_array_equal(train, cache['train_rows'])
        if frame.port.iloc[train].isin([port, item['port']]).any() or np.intersect1d(s, q).size:
            raise ValueError('Held source port or support enters training/evaluation')
        pc = geo['base_components']
        raw = np.asarray(averaged[rows], dtype=float)
        xs = ((raw-geo['base_mean'])@pc.T-geo['base_center'])/geo['base_scale']
        z = (raw-pair['raw_center'])/pair['raw_scale']
        stable = (z@pair['q']-head['stable_center'])/head['stable_scale']
        zd = z@head['qd']
        u = context[rows, :2]
        bank = np.c_[xs, stable, zd, u]
        prior = bank@head['original_coef']+head['original_intercept']
        dxs = (head['qd'].T*pair['raw_scale'][None, :])@pc.T/geo['base_scale']
        core = xs-.5*zd@dxs
        phi = np.c_[(core-pair['base_residual_center'])/pair['base_residual_scale']/np.sqrt(128),
                    stable/np.sqrt(stable.shape[1]), u/np.sqrt(2), np.ones(len(rows))]
        delta = ((np.asarray(zero[rows], dtype=float)-raw)@pc.T/geo['base_scale'])/pair['base_residual_scale']
        direction = normalize_direction(delta)
        condition = rank_conditions(descriptors.iloc[rows])
        yq = labels[rows[q]]
        baseline = None
        for eta in ETAS:
            for conditional in (False, True):
                score = fit_predict(phi, prior, direction, condition, s, labels[rows[s]], eta, conditional)
                pred = score[q].argmax(1)
                correct = pred == yq
                if baseline is None:
                    baseline = correct
                records.append(dict(outer_port=port, held_port=item['port'],
                    mode='conditional' if conditional else 'unconditional', eta=eta,
                    oa=float(correct.mean()), ba=float(np.mean([correct[yq == c].mean() for c in np.unique(yq)])),
                    rescues=int((correct & ~baseline).sum()), harms=int((~correct & baseline).sum()),
                    n=len(q), support_n=len(s)))
        if (i+1)%6 == 0 or i+1 == len(items):
            print(f'SOURCE {port} {i+1}/{len(items)} {time.perf_counter()-start:.1f}s', flush=True)
    pd.DataFrame(records).to_csv(folder/'metrics.csv', index=False)
    write_json(folder/'COMPLETE.json', dict(episodes=len(items), seconds=time.perf_counter()-start))


def select(ports):
    data = pd.concat([pd.read_csv(OUT/'source'/p/'metrics.csv') for p in ports], ignore_index=True)
    data.to_csv(OUT/'source_metrics.csv', index=False)
    selection = {'ports': {}, 'external': {}}
    for p in ports:
        selection['ports'][p] = {}
        for mode in ('unconditional', 'conditional'):
            means = data[(data.outer_port == p) & (data['mode'] == mode)].groupby('eta')[['oa', 'ba']].mean()
            means = means.reset_index().sort_values(['ba', 'eta'], ascending=[False, True])
            selection['ports'][p][mode] = float(means.iloc[0].eta)
    for mode in ('unconditional', 'conditional'):
        means = data[data['mode'] == mode].groupby(['outer_port', 'eta'])[['oa', 'ba']].mean().groupby('eta').mean()
        means = means.reset_index().sort_values(['ba', 'eta'], ascending=[False, True])
        selection['external'][mode] = float(means.iloc[0].eta)
    selection['criterion'] = 'Mean inner-port BA, ties prefer smaller eta; external mean over outer-source tasks'
    selection['target_outcomes_used'] = False
    write_json(OUT/'source_selection.json', selection)
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ports', nargs='+')
    parser.add_argument('--select-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ports = sorted(p.name for p in (C4/'ports').iterdir() if (p/'policies.npz').exists())
    if not args.select_only:
        write_json(OUT/'PROTOCOL.json', dict(
            purpose='Test directional information and condition-dependent use, not physical causality',
            etas=ETAS, ridge=8., condition_bandwidth=.5, selected_by='Inner held-port BA',
            condition_columns=['elongation_proxy', 'center_excess_proxy'],
            condition_transform='Unlabelled ranks within each target/inner episode cohort; missing median .5',
            direction='L2 normalized source-PCA-scaled original-zero minus C4 feature, first128 coordinates',
            kernel='Kbase + eta * cosine_direction * exp(-rank_distance_squared/(2*.5**2))',
            ablations=['C4 baseline', 'unconditional source-selected', 'conditional source-selected',
                       'unconditional eta1 diagnostic', 'conditional eta1 diagnostic',
                       'uniform condition attenuation eta1 diagnostic'],
            zero_option=True, budget_labels=64, draws=5, expert_handoff=.05,
            frozen='Source head, geometry, expert, gate; same cached features and support draws',
            scope=['self24ports', 'f0921', 'oss2'], independent_confirmation=False,
            mechanism_readout='Rescues/harms, retention of existing C4 rescues, class/port heterogeneity',
            excluded='No query-label fitting, no global permutation, no FUSAR/OSS1/generic run'))
        frame, context = inputs()
        descriptors = pd.read_csv(ROOT/'c4_condition_diagnosis_20261009/descriptors/self24ports.csv')
        np.testing.assert_array_equal(descriptors.row_id, np.arange(len(frame)))
        np.testing.assert_array_equal(descriptors.instance_id, frame.instance_id)
        zero = np.load(C4/'pooled_zero.npy', mmap_mode='r')
        averaged = np.load(C4/'pooled_c4.npy', mmap_mode='r')
        for p in args.ports or ports:
            source_port(p, frame, context, zero, averaged, descriptors)
    if all((OUT/'source'/p/'COMPLETE.json').exists() for p in ports):
        select(ports)


if __name__ == '__main__':
    main()
