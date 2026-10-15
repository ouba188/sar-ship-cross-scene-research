"""Fixed six-port, five-draw CPU spatial pooling pilot; freeze before readout."""

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '2'

import gc
import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from receiver_consistent_handoff import confidence_parts
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json
from stable_guided_spatial_pooling import (
    TokenCache, coordinated_expert, fit_residual_head, source_rms, spatial_residual)


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
OUT = ROOT / 'stable_spatial_pooling_pilot_20261008'
PARENT = ROOT / 'module_completion_20261008/ports_projection'
EVIDENCE = ROOT / 'evidence_aware_handoff_20261008'
DIRECTIONS = ROOT / 'conditional_invariance_probe_20261008/ports'
OLD_TOKENS = Path('E:/SAR_AIS_T_shipchip_dataset/v5_context_dynamic_64_128_merged_20260915/'
                  'shared_assets_local_knowledge_v6/resnet50_spatial.float16.npy')
PORTS = ('Antwerp-Bruges', 'Jebel Ali', 'Port Klang', 'Santos', 'Singapore', 'Tanger Med')
ARMS = ('original', 'matched_refit', 'ordinary_pool', 'stable_pool')
POOL_ARMS = ARMS[2:]


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def gate_features(base, expert):
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    return np.column_stack((em, bm, eh, bh))


def initialize(frame, raw):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(date='2026-10-08', ports=list(PORTS), arms=list(ARMS),
        target_supports='Original existing64, all five original draws', handoff_budget=.05,
        backbone='Frozen historical R50 four-token caches; current M3 R18 cached source/target features',
        old_cache_rows=25656, new_cache_rows=12435, old_row_mapping='manifest.old_cache_index',
        new_row_mapping='manifest rows 25656:38091; verified against new_features/manifest.csv instance_id',
        attention='Absolute within-chip centered token evidence along source-base predicted top-two pair direction',
        ordinary_direction='pooled_basis', stable_direction='conditional_penalty_basis',
        evidence_scale='Per-direction RMS over four centered tokens of all outer-source rows',
        temperature=1., residual='sum_t (alpha_t - 1/4) * token_t',
        projection='Original outer-source 128-dimensional R50 PCA components, without adding PCA mean',
        residual_scale='Per-coordinate RMS on outer source under outer source-base selection; no centering',
        auxiliary_head='Class-balanced ridge1, no intercept, targets 2*onehot-1-original expert prior',
        auxiliary_strength=1., spatial_kernel_strength=.25, kernel_rank=128,
        expert_kernel='Original .5 retained192 + .5 complement2560 + one bias, PLUS .25 spatial/128',
        expert_adaptation='One coordinated ridge8 support fit using augmented source prior and augmented kernel',
        base='Exact historical Base64 scores unchanged in every arm',
        matched_refit='Exact original base/expert scores; four-confidence-feature LR C1 gate refit only',
        gates='Each arm uses source-held episodes, equal-port/equal-class query weights, LR C1',
        source_heads='Each auxiliary head and base pair-selector exclude held source labels using original purged train rows',
        boundary='Directions, outer PCA, evidence/residual RMS and frozen encoder use all outer source: NOT fully inner-OOF',
        target_labels='Only original64 labels during adaptation; query labels accessed for evaluation after ALL_FROZEN',
        selection='Fixed rank/strength/temperature; no target tuning, no random/permutation runs',
        reporting='Five-draw means within each port, then equal-port means; overlapping draws are not independent',
        main_model_changed=False, cpu_threads=2, physical_interpretation=False)
    path = OUT / 'PROTOCOL.json'
    if path.exists() and read_json(path) != protocol:
        raise ValueError('Protocol changed; preserve existing run')
    write_json(path, protocol)
    old = np.load(OLD_TOKENS, mmap_mode='r')
    new = np.load(SOURCE / 'new_features/resnet50_spatial.float16.npy', mmap_mode='r')
    new_ids = pd.read_csv(SOURCE / 'new_features/manifest.csv', usecols=['instance_id'])
    np.testing.assert_array_equal(frame.instance_id.iloc[25656:], new_ids.instance_id)
    cache = TokenCache(old, new, frame.old_cache_index.to_numpy())
    rows = np.array([0, 19, 12828, 25655, 25656, 27000, 38090])
    error = float(np.max(np.abs(cache.read(rows).mean(1) - raw[rows])))
    np.testing.assert_array_equal(cache.read(rows).mean(1), raw[rows])
    write_json(OUT / 'TOKEN_ALIGNMENT.json', dict(rows=rows.tolist(), max_mean_error=error,
        old_shape=list(old.shape), new_shape=list(new.shape), new_manifest_aligned=True))
    print(f'TOKEN ALIGNMENT: {len(rows)} rows, exact FP32 means, new manifest aligned', flush=True)
    return cache


def prepare_geometry(port, cache, raw):
    folder = OUT / 'ports' / port
    folder.mkdir(parents=True, exist_ok=True)
    with (PARENT / port / 'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    candidate = archive(DIRECTIONS / port / 'frozen.npz')
    old = archive(PARENT / port / 'matched_joint/joint_policies.npz')
    source, target = old['source_rows'], old['query_rows']
    np.testing.assert_array_equal(source, candidate['source_rows'])
    np.testing.assert_array_equal(target, candidate['query_rows'])
    if np.intersect1d(source, target).size:
        raise ValueError('Outer source/target overlap')
    rows = np.sort(np.r_[source, target])
    if (folder / 'GEOMETRY_READY.json').exists():
        g = archive(folder / 'geometry.npz')
        np.testing.assert_array_equal(g['rows'], rows)
        return model, old, g
    started = time.monotonic()
    projected = np.lib.format.open_memmap(folder / 'projected_tokens.npy', mode='w+',
        dtype=np.float32, shape=(len(rows), 4, 128))
    directions = [candidate['pooled_basis'], candidate['conditional_penalty_basis']]
    direction_matrix = np.column_stack(directions) / candidate['feature_scale'][:, None]
    pairs = len(candidate['pairs'])
    evidence = {arm: np.lib.format.open_memmap(folder / (arm + '_evidence.npy'), mode='w+',
        dtype=np.float32, shape=(len(rows), 4, pairs)) for arm in POOL_ARMS}
    for lo in range(0, len(rows), 256):
        tokens = cache.read(rows[lo:lo + 256])
        centered = tokens - tokens.mean(1, keepdims=True)
        projected[lo:lo + len(tokens)] = centered @ model['pca'].components_.T
        values = centered @ direction_matrix
        for index, arm in enumerate(POOL_ARMS):
            evidence[arm][lo:lo + len(tokens)] = values[:, :, index * pairs:(index + 1) * pairs]
    projected.flush()
    for value in evidence.values():
        value.flush()
    local = np.searchsorted(rows, source)
    with np.load(ROOT / 'pred_cache2' / f'{port}.npz') as saved:
        xs = saved['Xs'][source]
    selector = model['base'].decision_function(xs)
    g = dict(rows=rows, source_rows=source, target_rows=target, pairs=candidate['pairs'])
    diagnostics = {}
    for arm in POOL_ARMS:
        squares = np.zeros(pairs)
        for lo in range(0, len(local), 512):
            values = np.asarray(evidence[arm][local[lo:lo + 512]], dtype=float)
            squares += np.square(values).sum((0, 1))
        g[arm + '_evidence_scale'] = np.maximum(np.sqrt(squares / (4 * len(local))), 1e-6)
        residual_squares, max_attention, deviation = np.zeros(128), 0., 0.
        for lo in range(0, len(local), 512):
            chosen = local[lo:lo + 512]
            residual, alpha = spatial_residual(projected[chosen], evidence[arm][chosen],
                selector[lo:lo + 512], g['pairs'], g[arm + '_evidence_scale'])
            residual_squares += np.square(residual).sum(0)
            max_attention += alpha.max(1).sum()
            deviation += np.abs(alpha - .25).sum()
        g[arm + '_residual_scale'] = np.maximum(np.sqrt(residual_squares / len(local)), 1e-6)
        diagnostics[arm] = dict(mean_max_attention=float(max_attention / len(local)),
            mean_uniform_deviation=float(deviation / len(local)),
            raw_residual_rms=float(np.sqrt(residual_squares.mean() / len(local))))
    np.savez_compressed(folder / 'geometry.npz', **g)
    write_json(folder / 'GEOMETRY_READY.json', dict(port=port, source_count=len(source),
        target_count=len(target), diagnostics=diagnostics, fully_inner_oof=False,
        elapsed_seconds=time.monotonic() - started))
    print(f'GEOMETRY {port}: {time.monotonic() - started:.1f}s', flush=True)
    return model, old, g


def feature_bank(folder, g, rows, selector):
    local = np.searchsorted(g['rows'], rows)
    np.testing.assert_array_equal(g['rows'][local], rows)
    projected = np.load(folder / 'projected_tokens.npy', mmap_mode='r')
    answer = {}
    for arm in POOL_ARMS:
        evidence = np.load(folder / (arm + '_evidence.npy'), mmap_mode='r')
        answer[arm] = np.empty((len(rows), 128), dtype=np.float32)
        for lo in range(0, len(rows), 512):
            chosen = local[lo:lo + 512]
            residual, _ = spatial_residual(projected[chosen], evidence[chosen], selector[lo:lo + 512],
                                           g['pairs'], g[arm + '_evidence_scale'])
            answer[arm][lo:lo + len(chosen)] = residual / g[arm + '_residual_scale']
    return answer


def inner_train_priors(raw, train, extra, geometry, ep, fitted):
    base, expert = [], []
    pc = geometry['base_components']
    offset = geometry['base_mean'] @ pc.T + geometry['base_center']
    for lo in range(0, len(train), 512):
        xs = ((raw[train[lo:lo + 512]] @ pc.T - offset) / geometry['base_scale']).astype(np.float32)
        z = ((extra[lo:lo + 512] - ep['ssl_mean']) @ ep['ssl_components'].T).astype(np.float32)
        xe = (np.column_stack((xs, z)) - ep['expert_center']) / ep['expert_scale']
        base.append(xs @ fitted['base_coef'].T + fitted['base_intercept'])
        expert.append((xe @ ep['expert_coef'].T + ep['expert_intercept']).astype(np.float32).astype(float))
    return np.concatenate(base), np.concatenate(expert)


def inner_held_views(ep, extra):
    ssl = SimpleNamespace(mean_=ep['ssl_mean'], components_=ep['ssl_components'])
    z = ((extra - ssl.mean_) @ ssl.components_.T).astype(np.float32)
    xe = (np.column_stack((ep['xs'], z)) - ep['expert_center']) / ep['expert_scale']
    prior = (xe @ ep['expert_coef'].T + ep['expert_intercept']).astype(np.float32).astype(float)
    retained = (xe / np.sqrt(192)).astype(float)
    discarded_extra = (discarded_coordinates(extra, ssl) - ep['discarded_center'][2048:]) / ep['discarded_scale'][2048:]
    complement = np.column_stack((ep['base_discarded'], discarded_extra)) / np.sqrt(2560)
    support = ep['support']
    kernel = .5 * (retained @ retained[support].T) + .5 * (complement @ complement[support].T) + 1
    return prior, kernel


def collect_record(port, held, arm, y, base, expert, pred, selected):
    bm, em, fm = [metrics(y, v, 8) for v in (base.argmax(1), expert.argmax(1), pred)]
    good_base, good_final = base.argmax(1) == y, pred == y
    return dict(port=port, held_source=held, arm=arm, n=len(y), oa=fm['acc'], ba=fm['ba'],
        base_oa=bm['acc'], base_ba=bm['ba'], expert_oa=em['acc'], expert_ba=em['ba'],
        rescues=int((~good_base & good_final).sum()), harms=int((good_base & ~good_final).sum()),
        admissions=int(selected.sum()), expert_changed=int((base.argmax(1) != expert.argmax(1)).sum()))


def train_source(port, model, old, g, raw, source_labels):
    folder = OUT / 'ports' / port
    if (folder / 'SOURCE_FITTED.json').exists():
        with (folder / 'source_models.pkl').open('rb') as stream:
            return pickle.load(stream)
    started = time.monotonic()
    source = g['source_rows']
    extra_rows = np.load(EVIDENCE / 'source_features' / port / 'rows.npy')
    np.testing.assert_array_equal(source, extra_rows)
    extra = np.load(EVIDENCE / 'source_features' / port / 'features.npy', mmap_mode='r')
    descriptions = read_json(PARENT / port / 'EPISODES_COMPLETE.json')['episodes']
    cached = {item['port']: item for item in read_json(EVIDENCE / 'source_episodes' / port / 'COMPLETE.json')['episodes']}
    groups = {arm: [] for arm in ARMS[1:]}
    max_replay_error = 0.
    diagnostics = []
    for i, item in enumerate(descriptions):
        ep = archive(item['path'])
        receipt = cached[item['port']]
        saved = archive(receipt['path'])
        train, held, support, query = ep['train_rows'], ep['rows'], ep['support'], ep['query']
        train_local, held_local = np.searchsorted(source, train), np.searchsorted(source, held)
        np.testing.assert_array_equal(source[train_local], train)
        np.testing.assert_array_equal(source[held_local], held)
        np.testing.assert_array_equal(saved['rows'], held)
        np.testing.assert_array_equal(saved['support'], support)
        if np.intersect1d(train, held).size or saved['query'][support].any():
            raise ValueError('Source held rows or support labels enter head fitting/query')
        geometry = archive(PARENT / 'episode_geometry_cache' / (receipt['pair_key'] + '.npz'))
        fitted = archive(EVIDENCE / 'source_pair_cache' / (receipt['pair_key'] + '.npz'))
        np.testing.assert_array_equal(geometry['train_rows'], train)
        np.testing.assert_array_equal(fitted['train_rows'], train)
        btrain, etrain = inner_train_priors(raw, train, extra[train_local], geometry, ep, fitted)
        train_features = feature_bank(folder, g, train, btrain)
        held_selector = ep['xs'] @ fitted['base_coef'].T + fitted['base_intercept']
        held_features = feature_bank(folder, g, held, held_selector)
        prior, kernel = inner_held_views(ep, extra[held_local])
        y = source_labels[held_local]
        replay = prior + kernel @ fit_kernel_correction(kernel[support], y[support], prior[support])
        error = float(np.max(np.abs(replay - saved['expert_scores'])))
        max_replay_error = max(max_replay_error, error)
        np.testing.assert_allclose(replay, saved['expert_scores'], atol=2e-6, rtol=2e-6)
        experts = {'matched_refit': saved['expert_scores']}
        for arm in POOL_ARMS:
            head = fit_residual_head(train_features[arm], source_labels[train_local], etrain)
            experts[arm], _ = coordinated_expert(prior, kernel, held_features[arm], head, support, y[support])
            diagnostics.append(dict(port=port, held_source=item['port'], arm=arm,
                source_prior_added_rms=float(np.sqrt(np.mean((held_features[arm] @ head) ** 2))),
                changed_expert_predictions=int(np.sum(experts[arm][query].argmax(1) != replay[query].argmax(1))),
                expert_score_delta_rms=float(np.sqrt(np.mean((experts[arm][query] - replay[query]) ** 2)))))
        base = saved['base_scores']
        _, inverse, counts = np.unique(y[query], return_inverse=True, return_counts=True)
        weight = 1. / (len(descriptions) * len(counts) * counts[inverse])
        for arm, expert in experts.items():
            eligible = (base.argmax(1) != expert.argmax(1)) & np.isin(expert.argmax(1), y[support])
            groups[arm].append(dict(base=base[query], expert=expert[query], labels=y[query],
                features=gate_features(base, expert)[query], eligible=eligible[query], weights=weight,
                rows=held[query], support_labels=y[support], held_source=item['port']))
        if (i + 1) % 4 == 0 or i + 1 == len(descriptions):
            print(f'SOURCE {port}: {i + 1}/{len(descriptions)}, {time.monotonic() - started:.1f}s', flush=True)
        del train_features, held_features, ep, saved, geometry, fitted
    gates, records = {}, []
    for arm, episodes in groups.items():
        joined = {key: np.concatenate([e[key] for e in episodes])
                  for key in ('base', 'expert', 'labels', 'features', 'eligible', 'weights')}
        gates[arm] = fit_gate(joined['features'], joined['base'], joined['expert'],
                             joined['labels'], joined['eligible'], joined['weights'])
        for episode in episodes:
            score = predict_gate(gates[arm], episode['features'])
            routed = route_scores(episode['base'], episode['expert'], score,
                episode['support_labels'], np.ones(len(score), dtype=bool), episode['rows'])
            records.append(collect_record(port, episode['held_source'], arm, episode['labels'],
                episode['base'], episode['expert'], routed['predictions'], routed['selected']))
    # The deployment auxiliary head uses all purged outer-source labels.
    with np.load(ROOT / 'pred_cache2' / f'{port}.npz') as saved:
        xs = saved['Xs'][source]
    selector = model['base'].decision_function(xs)
    z = model['pca_ssl'].transform(extra).astype(np.float32)
    xe = (np.column_stack((xs, z)) - model['expert_center']) / model['expert_scale']
    prior = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    features = feature_bank(folder, g, source, selector)
    heads = {arm: fit_residual_head(features[arm], source_labels, prior) for arm in POOL_ARMS}
    result = dict(gates=gates, heads=heads)
    with (folder / 'source_models.pkl').open('wb') as stream:
        pickle.dump(result, stream)
    pd.DataFrame(records).to_csv(folder / 'source_episode_metrics.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(folder / 'source_module_diagnostics.csv', index=False)
    write_json(folder / 'SOURCE_FITTED.json', dict(port=port, episodes=len(descriptions),
        max_original_expert_replay_error=max_replay_error, target_labels_used=0,
        source_gate_metrics='Gate fitting-sample diagnostic on source-held expert predictions; not gate OOF',
        elapsed_seconds=time.monotonic() - started))
    return result


def freeze_target(port, model, old, g, fitted, raw, support_truth):
    folder = OUT / 'ports' / port
    if (folder / 'TARGET_FROZEN.json').exists():
        return
    rows = old['query_rows']
    extra = np.load(PARENT / port / 'matched_joint/joint_extra.npy', mmap_mode='r')
    with np.load(ROOT / 'pred_cache2' / f'{port}.npz') as saved:
        xs = saved['Xs'][rows]
    selector = model['base'].decision_function(xs)
    features = feature_bank(folder, g, rows, selector)
    z = model['pca_ssl'].transform(extra).astype(np.float32)
    xe = (np.column_stack((xs, z)) - model['expert_center']) / model['expert_scale']
    prior = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    retained = (xe / np.sqrt(192)).astype(float)
    stats = archive(PARENT / port / 'deployment_complement.npz')
    discarded = np.column_stack((discarded_coordinates(raw[rows], model['pca']),
                                  discarded_coordinates(extra, model['pca_ssl'])))
    complement = (discarded - stats['discarded_center']) / stats['discarded_scale'] / np.sqrt(2560)
    outputs = {key: [] for key in ('expert_scores', 'gate_scores', 'predictions', 'selected')}
    coefficients, diagnostics, max_error = {}, [], 0.
    for draw, ev in enumerate(old['eval_mask']):
        support = np.searchsorted(rows, old['support_rows'][draw])
        np.testing.assert_array_equal(rows[support], old['support_rows'][draw])
        if len(support) != 64 or len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Existing64 support/query protocol changed')
        y = support_truth[draw]
        kernel = .5 * (retained @ retained[support].T) + .5 * (complement @ complement[support].T) + 1
        replay = prior + kernel @ fit_kernel_correction(kernel[support], y, prior[support])
        max_error = max(max_error, float(np.max(np.abs(replay - old['expert_scores'][draw]))))
        np.testing.assert_allclose(replay, old['expert_scores'][draw], atol=2e-6, rtol=2e-6)
        base = old['base_scores'][draw]
        experts = {'matched_refit': old['expert_scores'][draw]}
        for arm in POOL_ARMS:
            experts[arm], coef = coordinated_expert(prior, kernel, features[arm], fitted['heads'][arm], support, y)
            coefficients.setdefault(arm, []).append(coef)
            diagnostics.append(dict(port=port, draw=draw, arm=arm,
                expert_changed_vs_original=int(np.sum(experts[arm][ev].argmax(1) != replay[ev].argmax(1))),
                expert_score_delta_rms=float(np.sqrt(np.mean((experts[arm][ev] - replay[ev]) ** 2)))))
        results = {key: [old[key][draw]] for key in outputs}
        for arm in ARMS[1:]:
            score = predict_gate(fitted['gates'][arm], gate_features(base, experts[arm]))
            routed = route_scores(base, experts[arm], score, y, ev, rows)
            if routed['cap'] != old['cap'][draw] or routed['selected'][~ev].any():
                raise ValueError('Handoff budget or support exclusion changed')
            for key, value in dict(expert_scores=experts[arm], gate_scores=score,
                                  predictions=routed['predictions'], selected=routed['selected']).items():
                results[key].append(value)
        for key in outputs:
            outputs[key].append(np.stack(results[key]))
    np.savez_compressed(folder / 'target_policies.npz', arms=np.array(ARMS),
        query_rows=rows, source_rows=old['source_rows'], support_rows=old['support_rows'],
        eval_mask=old['eval_mask'], cap=old['cap'], base_scores=old['base_scores'],
        **{key: np.stack(value) for key, value in outputs.items()},
        **{arm + '_coefficients': np.stack(value) for arm, value in coefficients.items()})
    pd.DataFrame(diagnostics).to_csv(folder / 'target_module_diagnostics_no_labels.csv', index=False)
    write_json(folder / 'TARGET_FROZEN.json', dict(port=port, draws=5, arms=list(ARMS),
        max_original_expert_replay_error=max_error, base_unchanged=True, query_labels_used=0))
    print(f'TARGET FROZEN {port}: all five draws, unchanged base', flush=True)


def evaluate(frame):
    if read_json(OUT / 'ALL_FROZEN.json')['ports'] != list(PORTS):
        raise ValueError('All six target policies must freeze before query readout')
    records = []
    for port in PORTS:
        saved = archive(OUT / 'ports' / port / 'target_policies.npz')
        labels = frame.class_id.to_numpy()[saved['query_rows']]
        for draw, ev in enumerate(saved['eval_mask']):
            for index, arm in enumerate(ARMS):
                record = collect_record(port, '', arm, labels[ev], saved['base_scores'][draw, ev],
                    saved['expert_scores'][draw, index, ev], saved['predictions'][draw, index, ev],
                    saved['selected'][draw, index, ev])
                record['draw'] = draw
                record['cap'] = int(saved['cap'][draw])
                records.append(record)
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'target_draw_metrics.csv', index=False)
    numeric = ['oa', 'ba', 'base_oa', 'base_ba', 'expert_oa', 'expert_ba', 'rescues', 'harms', 'admissions', 'expert_changed', 'cap']
    ports = df.groupby(['port', 'arm'], as_index=False)[numeric].mean()
    overall = ports.groupby('arm', as_index=False)[numeric].mean()
    ports.to_csv(OUT / 'target_port_means.csv', index=False)
    overall.to_csv(OUT / 'target_equal_port_means.csv', index=False)
    source = pd.concat([pd.read_csv(OUT / 'ports' / p / 'source_episode_metrics.csv') for p in PORTS])
    source.to_csv(OUT / 'source_episode_metrics.csv', index=False)
    source.groupby('arm')[numeric[:-1]].mean().to_csv(OUT / 'source_equal_episode_means.csv')
    means = overall.set_index('arm')
    lines = ['# Stable-guided spatial pooling pilot', '',
        'Completed six fixed ports, five existing64 draws each, CPU ClearSAR with two threads. '
        'All six policies froze before target query evaluation. Historical main model and base scores are unchanged.', '',
        '| Arm | OA % | BA % | Expert BA % | Rescues | Harms | Admissions |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        r = means.loc[arm]
        lines.append(f'| {arm} | {100*r.oa:.4f} | {100*r.ba:.4f} | {100*r.expert_ba:.4f} | '
                     f'{r.rescues:.2f} | {r.harms:.2f} | {r.admissions:.2f} |')
    lines.extend(['', 'Metrics are equal-port means after averaging five draws. Counts are mean counts per port/draw.', '',
        '## Matched comparisons', ''])
    for arm, control in [('matched_refit', 'original'), ('ordinary_pool', 'matched_refit'),
                         ('stable_pool', 'matched_refit'), ('stable_pool', 'ordinary_pool')]:
        delta = means.loc[arm] - means.loc[control]
        lines.append(f'- {arm} minus {control}: OA {100*delta.oa:+.4f} pp; BA {100*delta.ba:+.4f} pp; '
                     f'expert BA {100*delta.expert_ba:+.4f} pp.')
    lines.extend(['', '| Port | Ordinary vs matched BA pp | Stable vs matched BA pp | Stable vs ordinary BA pp |',
                  '|---|---:|---:|---:|'])
    pivot = ports.pivot(index='port', columns='arm', values='ba')
    for port in PORTS:
        r = pivot.loc[port]
        lines.append(f'| {port} | {100*(r.ordinary_pool-r.matched_refit):+.4f} | '
                     f'{100*(r.stable_pool-r.matched_refit):+.4f} | {100*(r.stable_pool-r.ordinary_pool):+.4f} |')
    diag = pd.concat([pd.read_csv(OUT / 'ports' / p / 'source_module_diagnostics.csv') for p in PORTS])
    diag.groupby('arm').mean(numeric_only=True).to_csv(OUT / 'source_module_diagnostics_mean.csv')
    lines.extend(['', '## Pipeline and scope', '',
        'Source-base predicted top-two class pairs select ordinary pooled or conditional-penalty directions. '
        'Absolute within-chip centered token evidence, divided by per-direction source RMS, enters temperature-one softmax. '
        'The mean-zero attention residual is projected into the original 128 PCA coordinates and scaled by source RMS. '
        'A balanced ridge1 residual head fits desired scores minus the historical source expert prior. '
        'The target64 ridge8 correction is refit jointly with that augmented prior and the original expert kernel '
        'plus .25 times the normalized spatial kernel. The base is byte-for-byte unchanged. '
        'Each new arm refits the same four-confidence-feature LR C1 gate; the matched control only refits that gate. '
        'The historical original keeps its saved gate. This pilot uses existing binary fit_gate, which drops zero-utility '
        'rows and learns rescue versus harm on informative eligible disagreements. The concurrent drift-response pilot '
        'retains zero-utility rows in a net-utility classifier, so its refit control is not directly the same baseline.', '',
        'Residual heads and pair-selector heads exclude held source labels using existing purged train rows. '
        'Directions, PCA, RMS moments and encoder use all outer source, so this is NOT fully inner-OOF geometry. '
        'Source expert metrics use held episodes; routed source metrics reuse gate fitting episodes and are descriptive, '
        'not an independent gate evaluation. Only existing64 target labels enter fitting; all query labels are read after freezing.', '',
        'Historical token means exactly reproduce the original pooled features on seven fixed alignment rows; '
        'new-cache manifest identities also match. Original expert-kernel replay is checked for all source episodes and target draws. '
        'The focused tests cover row mapping/mean reconstruction, uniform no-op, label-free predicted-pair inference, '
        'source-only scaling, explicit primal/dual consistency and zero-branch replay.', '',
        '## Minimal diagnosis', ''])
    source_means = source.groupby('arm')[['ba', 'expert_ba']].mean()
    for arm in POOL_ARMS:
        d = diag[diag.arm == arm]
        delta = source_means.loc[arm] - source_means.loc['matched_refit']
        lines.append(f'- {arm}: source-held expert BA delta {100*delta.expert_ba:+.4f} pp; '
                     f'mean held prior addition RMS {d.source_prior_added_rms.mean():.4f}; '
                     f'mean changed held expert predictions {d.changed_expert_predictions.mean():.1f}.')
    lines.extend(['', 'Nonzero module diagnostics establish that the branch is active; they do not establish transfer value. '
        'For this completed pilot, both pooling arms regress on source-held expert BA as well as target expert BA. '
        'Thus the negative result is already present before target routing; target-only shift or a no-op does not explain it. '
        'The remaining cause within attention representation, residual fitting and regularization is unresolved. '
        'This fixed recipe does not support merging the new expert into the drift-response gate. '
        'Negative results and every port are retained. No target parameter search or physical interpretation was performed. '
        'These are development comparisons, with overlapping draws and no independent confirmation. '
        'Ordinary versus stable isolates this guidance choice within the matched pooling implementation; '
        'it does not identify stability causally.', ''])
    (OUT / 'FINDINGS.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(OUT / 'COMPLETE.json', dict(status='complete', ports=list(PORTS), draws=5,
        metrics=overall.to_dict('records'), all_target_policies_frozen_before_query_readout=True,
        main_model_changed=False))
    print(overall.to_string(index=False), flush=True)


def main():
    if (OUT / 'COMPLETE.json').exists():
        print('Completed pilot preserved: ' + str(OUT / 'FINDINGS.md'))
        return
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['instance_id', 'old_cache_index', 'port', 'class_id'])
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    with threadpool_limits(limits=2):
        cache = initialize(frame, raw)
        for port in PORTS:
            started = time.monotonic()
            model, old, g = prepare_geometry(port, cache, raw)
            if np.any(frame.port.to_numpy()[g['source_rows']] == port):
                raise ValueError('Outer target enters source fitting')
            source_labels = frame.class_id.to_numpy()[g['source_rows']]
            fitted = train_source(port, model, old, g, raw, source_labels)
            support_truth = frame.class_id.to_numpy()[old['support_rows']]
            freeze_target(port, model, old, g, fitted, raw, support_truth)
            write_json(OUT / 'STATUS.json', dict(last_frozen=port,
                frozen_ports=[p for p in PORTS if (OUT / 'ports' / p / 'TARGET_FROZEN.json').exists()],
                last_port_seconds=time.monotonic() - started))
            del model, old, g, fitted
            gc.collect()
        write_json(OUT / 'ALL_FROZEN.json', dict(ports=list(PORTS), arms=list(ARMS), draws=5,
            query_labels_read_for_metrics=False))
        evaluate(frame)


if __name__ == '__main__':
    main()
