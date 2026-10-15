"""Bounded six-port base-only pilot; reuse target scores and preserve expert."""

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from base_only_preserved_expert import (
    ARMS, FACTORS, PORTS, base_transforms, base_views, fit_episode_gate,
    gate_features, predict_bases, route_episode, verify_target_alignment)
from evidence_aware_handoff import predict_gate, route_scores
from protected_drift_activation import fit_source_heads
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'base_only_preserved_expert_pilot_20261009'
CACHE = ROOT / 'protected_drift_activation_20261008/ports'
PARENT = ROOT / 'module_completion_20261008/ports_projection'
EVIDENCE = ROOT / 'evidence_aware_handoff_20261008/source_episodes'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def archive(path, keys=None):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in (saved.files if keys is None else keys)}


def record(port, arm, labels, base, expert, routed, **extra):
    final = routed['predictions']
    b, e = base.argmax(1), expert.argmax(1)
    fm, bm, em = [metrics(labels, p, 8) for p in (final, b, e)]
    return dict(port=port, arm=arm, oa=fm['acc'], ba=fm['ba'],
        base_oa=bm['acc'], base_ba=bm['ba'], expert_oa=em['acc'], expert_ba=em['ba'],
        rescues=int(((b != labels) & (final == labels)).sum()),
        harms=int(((b == labels) & (final != labels)).sum()),
        admissions=int(routed['selected'].sum()), cap=int(routed['cap']), query_rows=len(labels), **extra)


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-09', ports=list(PORTS), arms=list(ARMS), draws=5,
        labels_per_draw=64, handoff_budget=.05, cpu_threads=2,
        expert='Exact historical scores from original source episodes and target joint_policies; no refit',
        base='Same source balanced ridge1 and support residual ridge8 as protected_drift_activation_20261008',
        matched_baseline='Refitted base without stable/drift modifications + historical expert + compatible refit gate',
        stable_only='Source-standardized candidate Qs appended to base source head and base64 residual features',
        joint='Stable activation plus .5 attenuation of16 source drift directions orthogonal to Qs, base only',
        target_reuse='Original saved base scores, independently replayed against saved source heads and64 labels',
        source_reuse='Rebuild only base readouts from original banks and purged source episode rows',
        gate='Original4confidence binary LR C1, eligible informative rescue/harm rows; equal episode/class weights',
        source_diagnostic='Three blocked folds by original episode index modulo3; no recipe selection',
        boundary='Qs/Qd/PCA/scales/encoder use all outer-source; source head fits exclude inner held rows. NOT fully inner OOF',
        target_labels='Only original64 support labels before freezing ALL six target policies',
        readout='Five-draw mean per port then equal-port mean; present-class BA and OA',
        no_sweeps=True, main_model_changed=False, independent_confirmation=False,
        target_query_selection=False, expansion_to_all23=False)
    path = OUT / 'PROTOCOL.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Changed protocol requires a different directory')
    write_json(path, protocol)


def train_source(port, geometry, source_labels):
    folder = OUT / 'ports' / port
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'SOURCE_FITTED.json').exists():
        with (folder / 'gates.pkl').open('rb') as stream:
            return pickle.load(stream)
    started = time.monotonic()
    source = np.load(CACHE / port / 'source_rows.npy')
    bank = np.load(CACHE / port / 'source_bank.npy', mmap_mode='r')
    descriptions = read_json(PARENT / port / 'EPISODES_COMPLETE.json')['episodes']
    old_episodes = {e['port']: e for e in read_json(EVIDENCE / port / 'COMPLETE.json')['episodes']}
    previous_metrics = pd.read_csv(CACHE / port / 'source_episode_metrics.csv').set_index(['held_source', 'arm'])
    transforms = base_transforms(geometry)
    groups = {arm: [] for arm in FACTORS}
    historical, verification = [], []
    for index, item in enumerate(descriptions):
        ep = archive(item['path'], ['rows', 'train_rows', 'support', 'query'])
        old = archive(old_episodes[item['port']]['path'],
                      ['rows', 'support', 'query', 'base_scores', 'expert_scores', 'labels'])
        rows, train, support, query = [ep[k] for k in ('rows', 'train_rows', 'support', 'query')]
        held_local, train_local = np.searchsorted(source, rows), np.searchsorted(source, train)
        np.testing.assert_array_equal(source[held_local], rows)
        np.testing.assert_array_equal(source[train_local], train)
        np.testing.assert_array_equal(old['rows'], rows)
        np.testing.assert_array_equal(old['support'], support)
        np.testing.assert_array_equal(np.flatnonzero(old['query']), query)
        labels = source_labels[held_local]
        np.testing.assert_array_equal(labels, old['labels'])
        if np.intersect1d(train, rows).size or len(support) != 64 or old['query'][support].any():
            raise ValueError('Source train/held or support/query contamination')
        heads = fit_source_heads(bank[train_local], source_labels[train_local], transforms)
        held_bank = np.asarray(bank[held_local])
        bases = predict_bases(held_bank, base_views(held_bank, geometry), heads, support, labels[support])
        expert = old['expert_scores'][query]
        _, inverse, counts = np.unique(labels[query], return_inverse=True, return_counts=True)
        weights = 1. / (len(descriptions) * len(counts) * counts[inverse])
        for arm, value in bases.items():
            base = value[query]
            replay = metrics(labels[query], base.argmax(1), 8)
            expected = previous_metrics.loc[(item['port'], arm)]
            np.testing.assert_allclose([replay['acc'], replay['ba']],
                                       [expected.base_acc, expected.base_ba], atol=1e-12, rtol=0)
            eligible = ((base.argmax(1) != expert.argmax(1)) &
                        np.isin(expert.argmax(1), labels[support]))
            groups[arm].append(dict(base=base, expert=expert, labels=labels[query], weights=weights,
                features=gate_features(base, expert), eligible=eligible, rows=rows[query],
                support_labels=labels[support], held_source=item['port']))
        hb = metrics(labels[query], old['base_scores'][query].argmax(1), 8)
        he = metrics(labels[query], expert.argmax(1), 8)
        historical.append(dict(port=port, held_source=item['port'], base_oa=hb['acc'], base_ba=hb['ba'],
                               expert_oa=he['acc'], expert_ba=he['ba']))
        verification.append(dict(held_source=item['port'], query_rows=len(query), source_row_alignment=True,
                                 support_query_alignment=True, base_metrics_replayed=True,
                                 expert_source='historical cache, no transformation'))
        del heads, held_bank, bases, ep, old
        if (index + 1) % 6 == 0 or index + 1 == len(descriptions):
            print(f'SOURCE {port}: {index + 1}/{len(descriptions)} ({time.monotonic()-started:.1f}s)', flush=True)
    gates, records, fit_records = {}, [], []
    for arm, episodes in groups.items():
        gates[arm] = fit_episode_gate(episodes)
        for episode in episodes:
            routed = route_episode(gates[arm], episode)
            fit_records.append(record(port, arm, episode['labels'], episode['base'], episode['expert'],
                                      routed, held_source=episode['held_source']))
        # Gate-only blocked crossfit: upstream geometry/head predictions stay fixed.
        for fold in range(3):
            train = [e for i, e in enumerate(episodes) if i % 3 != fold]
            gate = fit_episode_gate(train)
            for i, episode in enumerate(episodes):
                if i % 3 != fold:
                    continue
                routed = route_episode(gate, episode)
                records.append(record(port, arm, episode['labels'], episode['base'], episode['expert'],
                    routed, held_source=episode['held_source'], gate_fold=fold))
        payload = {key: np.concatenate([e[key] for e in episodes])
                   for key in ('base', 'expert', 'labels', 'features', 'eligible', 'weights', 'rows')}
        payload['episode_offsets'] = np.r_[0, np.cumsum([len(e['rows']) for e in episodes])]
        payload['support_labels'] = np.stack([e['support_labels'] for e in episodes])
        payload['held_sources'] = np.asarray([e['held_source'] for e in episodes])
        np.savez_compressed(folder / (arm + '_source_scores.npz'), **payload)
    with (folder / 'gates.pkl').open('wb') as stream:
        pickle.dump(gates, stream)
    pd.DataFrame(records).to_csv(folder / 'source_crossfit_metrics.csv', index=False)
    pd.DataFrame(fit_records).to_csv(folder / 'source_fit_metrics.csv', index=False)
    pd.DataFrame(historical).to_csv(folder / 'historical_source_classifiers.csv', index=False)
    write_json(folder / 'SOURCE_FITTED.json', dict(port=port, episodes=len(descriptions),
        checks=verification, target_labels_used=0, gate_features=4,
        gate_counts={arm: g['counts'] for arm, g in gates.items()}, elapsed_seconds=time.monotonic()-started))
    return gates


def freeze_target(port, geometry, gates, support_truth):
    folder = OUT / 'ports' / port
    if (folder / 'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT / port / 'matched_joint/joint_policies.npz')
    cached = archive(CACHE / port / 'policies.npz',
        ['query_rows', 'source_rows', 'support_rows', 'eval_mask', 'arm_names', 'base_scores'])
    rows = np.load(CACHE / port / 'target_rows.npy')
    verify_target_alignment(old, cached, rows)
    bank = np.load(CACHE / port / 'target_bank.npy', mmap_mode='r')
    with (CACHE / port / 'source_models.pkl').open('rb') as stream:
        heads = pickle.load(stream)['heads']
    views = base_views(bank, geometry)
    outputs = {k: [] for k in ('base_scores', 'gate_scores', 'predictions', 'selected')}
    replay_error = 0.
    for draw, evaluation in enumerate(old['eval_mask']):
        support = np.searchsorted(rows, old['support_rows'][draw])
        replay = predict_bases(bank, views, heads, support, support_truth[draw])
        result = {key: [old[key][draw]] for key in outputs}
        for arm in FACTORS:
            index = list(cached['arm_names']).index(arm)
            base, expert = cached['base_scores'][draw, index], old['expert_scores'][draw]
            np.testing.assert_allclose(replay[arm], base, atol=2e-10, rtol=2e-10)
            replay_error = max(replay_error, float(np.max(np.abs(replay[arm]-base))))
            score = predict_gate(gates[arm], gate_features(base, expert))
            routed = route_scores(base, expert, score, support_truth[draw], evaluation, rows)
            if routed['cap'] != old['cap'][draw] or routed['selected'][~evaluation].any():
                raise ValueError('Handoff quota/support exclusion changed')
            for key, value in dict(base_scores=base, gate_scores=score,
                                  predictions=routed['predictions'], selected=routed['selected']).items():
                result[key].append(value)
        for key in outputs:
            outputs[key].append(np.stack(result[key]))
    np.savez_compressed(folder / 'policies.npz', arms=np.array(ARMS), query_rows=rows,
        source_rows=old['source_rows'], support_rows=old['support_rows'], eval_mask=old['eval_mask'],
        cap=old['cap'], expert_scores=old['expert_scores'], **{k: np.stack(v) for k, v in outputs.items()})
    saved = archive(folder / 'policies.npz', ['expert_scores', 'base_scores', 'gate_scores', 'predictions', 'selected'])
    np.testing.assert_array_equal(saved['expert_scores'], old['expert_scores'])
    for key in outputs:
        np.testing.assert_array_equal(saved[key][:, 0], old[key])
    write_json(folder / 'TARGET_FROZEN.json', dict(port=port, draws=5, labels_per_draw=64,
        query_labels_used=0, expert_exactly_unchanged=True, original_reference_exactly_unchanged=True,
        base_replay_max_abs_error=replay_error, all_arms=list(ARMS)))
    print(f'TARGET FROZEN {port}; base replay max error {replay_error:.3g}', flush=True)


def evaluate(frame):
    if read_json(OUT / 'ALL_FROZEN.json')['ports'] != list(PORTS):
        raise ValueError('All six policies must freeze before evaluation')
    records = []
    for port in PORTS:
        saved = archive(OUT / 'ports' / port / 'policies.npz')
        y = frame.class_id.to_numpy()[saved['query_rows']]
        for draw, evaluation in enumerate(saved['eval_mask']):
            for index, arm in enumerate(ARMS):
                routed = dict(predictions=saved['predictions'][draw, index, evaluation],
                    selected=saved['selected'][draw, index, evaluation], cap=saved['cap'][draw])
                records.append(record(port, arm, y[evaluation], saved['base_scores'][draw, index, evaluation],
                    saved['expert_scores'][draw, evaluation], routed, draw=draw))
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'metrics.csv', index=False)
    numeric = [k for k in df.select_dtypes(include='number').columns if k != 'draw']
    ports = df.groupby(['port', 'arm'], as_index=False)[numeric].mean()
    means = ports.groupby('arm', as_index=False)[numeric].mean()
    ports.to_csv(OUT / 'port_means.csv', index=False)
    means.to_csv(OUT / 'equal_port_means.csv', index=False)
    df.groupby(['draw', 'arm'], as_index=False)[numeric].mean().to_csv(OUT / 'draw_means.csv', index=False)
    for name in ('source_crossfit_metrics', 'source_fit_metrics', 'historical_source_classifiers'):
        pd.concat([pd.read_csv(OUT / 'ports' / p / (name + '.csv')) for p in PORTS]).to_csv(
            OUT / (name + '.csv'), index=False)
    write_json(OUT / 'COMPLETE.json', dict(status='complete', ports=list(PORTS), draws=5, rows=len(df),
        means=means.to_dict('records'), all_target_policies_frozen_before_readout=True,
        expert_exactly_unchanged=True, main_model_changed=False, independent_confirmation=False))
    print(means.to_string(index=False), flush=True)


def main():
    if (OUT / 'COMPLETE.json').exists():
        print('Completed output preserved: ' + str(OUT))
        return
    initialize()
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    with threadpool_limits(limits=2):
        for port in PORTS:
            if (OUT / 'ports' / port / 'TARGET_FROZEN.json').exists():
                continue
            with (CACHE / port / 'geometry.pkl').open('rb') as stream:
                geometry = pickle.load(stream)
            source = np.load(CACHE / port / 'source_rows.npy')
            if np.any(frame.port.to_numpy()[source] == port):
                raise ValueError('Outer target enters source fitting')
            gates = train_source(port, geometry, frame.class_id.to_numpy()[source])
            support_rows = archive(PARENT / port / 'matched_joint/joint_policies.npz', ['support_rows'])['support_rows']
            freeze_target(port, geometry, gates, frame.class_id.to_numpy()[support_rows])
            del geometry, gates
            gc.collect()
        if not all((OUT / 'ports' / p / 'TARGET_FROZEN.json').exists() for p in PORTS):
            raise ValueError('Missing frozen target')
        write_json(OUT / 'ALL_FROZEN.json', dict(ports=list(PORTS), arms=list(ARMS), query_metrics_read=False))
        evaluate(frame)


if __name__ == '__main__':
    main()
