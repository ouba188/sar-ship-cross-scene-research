"""Locate existing use of the candidate span without training a new model."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from budgeted_residual_handoff import ranked_mask
from latent_usage_attribution import (COMPONENTS, span_basis, raw_readout_gradients,
                                     centered_energy_fraction, score_contributions)
from projection_residual_expert import discarded_coordinates
from receiver_consistent_handoff import gate_scores
from run_ports_projection_completion import matched_views
from run_generic_transfer import metrics
from sar_joint_completion import SOURCE, write_json


ROOT = Path('E:/Hermes/paper_tracking/sensitivity')
PARENT = ROOT/'module_completion_20261008/ports_projection'
CANDIDATE = ROOT/'conditional_invariance_probe_20261008'
OUT = ROOT/'latent_usage_attribution_20261008'
ARMS = dict(reference=(), remove_base_source=('base_source',),
    remove_base_adaptation=('base_adaptation',), remove_expert_source=('expert_source',),
    remove_expert_retained=('expert_retained',), remove_expert_complement=('expert_complement',),
    remove_base=('base_source', 'base_adaptation'),
    remove_expert=('expert_source', 'expert_retained', 'expert_complement'),
    remove_both=COMPONENTS)


def archive(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def route(base, expert, support_labels, ev, ids, gate, cap):
    b, e = base.argmax(1), expert.argmax(1)
    score = gate_scores(base, expert, gate)
    observed = np.bincount(support_labels, minlength=8) > 0
    eligible = ev & (b != e) & observed[e] & (score > 0)
    selected = ranked_mask(score, eligible, ids, cap)
    return dict(predictions=np.where(selected, e, b), base_predictions=b,
                expert_predictions=e, selected=selected, gate_scores=score)


def verify_readouts(model, xs, retained, complement, prior, old, adapted):
    """Catch support-geometry mismatches that rerouting cached scores misses."""
    phi = np.column_stack(((xs-adapted['source_center'])/adapted['source_scale']/np.sqrt(xs.shape[1]),
                           np.ones(len(xs))))
    bprior = model['base'].decision_function(xs)
    max_b, max_e = 0., 0.
    for draw in range(5):
        support = np.searchsorted(old['query_rows'], old['support_rows'][draw])
        kernel = .5*(retained@retained[support].T)+.5*(complement@complement[support].T)+1.
        replay = prior+kernel@old['coefficients'][draw]
        max_e = max(max_e, float(np.max(np.abs(replay-old['expert_scores'][draw]))))
        b = bprior+phi@adapted['visual_correction'][draw]
        max_b = max(max_b, float(np.max(np.abs(b-old['base_scores'][draw]))))
    if max_e > 1e-8 or max_b > 1e-4:
        raise ValueError(f'Saved readout reconstruction differs: expert={max_e}, base={max_b}')
    return dict(max_expert_score_replay_error=max_e, max_base_score_replay_error=max_b)


def freeze(port, base, frame):
    parent = PARENT/port
    old = archive(parent/'matched_joint/joint_policies.npz')
    adapted = archive(ROOT/'oof_base_adapt_results_20261007'/port/'policies.npz')
    bank = archive(CANDIDATE/'ports'/port/'frozen.npz')
    with (parent/'source_model.pkl').open('rb') as f:
        model = pickle.load(f)
    stats = archive(parent/'deployment_complement.npz')
    ids, source = old['query_rows'], old['source_rows']
    np.testing.assert_array_equal(ids, bank['query_rows'])
    np.testing.assert_array_equal(source, bank['source_rows'])
    np.testing.assert_array_equal(old['base_scores'], adapted['scores'][:, 1])
    raw = np.asarray(base[ids], dtype=float)
    q, singular = span_basis(bank['conditional_penalty_basis'])
    rc, rs = bank['feature_center'], bank['feature_scale']
    with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
        xs = saved['Xs'][ids]
    extra = np.load(parent/'matched_joint/joint_extra.npy', mmap_mode='r')
    bd = (discarded_coordinates(raw, model['pca'])-stats['discarded_center'][:raw.shape[1]])/stats['discarded_scale'][:raw.shape[1]]
    prior, retained, complement = matched_views(xs, bd, extra, model, stats, True)
    readout_replay = verify_readouts(model, xs, retained, complement, prior, old, adapted)
    replay_xs = (model['pca'].transform(raw)-model['center'])/model['scale']
    relative_xs_error = float(np.linalg.norm(replay_xs-xs)/np.linalg.norm(xs))
    if relative_xs_error > 1e-3:
        raise ValueError('Recovered raw-to-cached-PCA map exceeds parent replay tolerance')
    raw_covectors = bank['conditional_penalty_basis']/rs[:, None]
    retained_energy = np.square(model['pca'].components_@raw_covectors).sum(0)/np.square(raw_covectors).sum(0)
    geometry = dict(port=port, rank=q.shape[1], candidate_directions=len(singular), **readout_replay,
        pca_relative_replay_error=relative_xs_error,
        candidate_covector_retained_fraction_mean=float(retained_energy.mean()),
        candidate_covector_retained_fraction_min=float(retained_energy.min()),
        candidate_covector_retained_fraction_max=float(retained_energy.max()))
    folder = OUT/'ports'/port
    folder.mkdir(parents=True, exist_ok=True)
    records, traces, coefficients, energy = [], [], [], []
    max_direct_error = 0.
    for draw in range(5):
        support = np.searchsorted(ids, old['support_rows'][draw])
        ev = old['eval_mask'][draw]
        labels = frame.class_id.to_numpy()[old['support_rows'][draw]]
        beta = old['coefficients'][draw]
        gradients = raw_readout_gradients(model, adapted['visual_correction'][draw],
            adapted['source_scale'], retained[support], complement[support], beta, stats['discarded_scale'])
        latent, contribution = score_contributions(raw, rc, rs, q, gradients)
        traces.append(np.stack([contribution[k] for k in COMPONENTS]))
        coefficients.append(np.stack([q.T@(rs[:, None]*gradients[k]) for k in COMPONENTS]))
        for name, g in gradients.items():
            centered = contribution[name][ev]-contribution[name][ev].mean(axis=1, keepdims=True)
            energy.append(dict(port=port, draw=draw, component=name,
                classifier_gradient_fraction=centered_energy_fraction(rs[:, None]*g, q),
                centered_score_contribution_rms=float(np.sqrt(np.square(centered).mean()))))
        # Independent feature-space replay checks the analytic gradient paths.
        n = min(17, len(ids))
        raw_delta = (latent[:n]@q.T)*rs
        dxs = (raw_delta@model['pca'].components_.T)/model['scale']
        dr = np.zeros_like(retained[:n])
        dr[:, :xs.shape[1]] = dxs/model['expert_scale'][:xs.shape[1]]/np.sqrt(retained.shape[1])
        dc = np.zeros_like(complement[:n])
        dc[:, :raw.shape[1]] = ((raw_delta-(raw_delta@model['pca'].components_.T)@model['pca'].components_)
            /stats['discarded_scale'][:raw.shape[1]]/np.sqrt(complement.shape[1]))
        direct = dict(base_source=model['base'].decision_function(xs[:n])-model['base'].decision_function(xs[:n]-dxs),
            base_adaptation=(dxs/adapted['source_scale']/np.sqrt(xs.shape[1]))@adapted['visual_correction'][draw, :-1],
            expert_retained=.5*(dr@retained[support].T)@beta,
            expert_complement=.5*(dc@complement[support].T)@beta)
        # Recover prior inputs directly, including unchanged R18 coordinates.
        xe = retained[:n]*np.sqrt(retained.shape[1])
        direct['expert_source'] = model['expert'].decision_function(xe)-model['expert'].decision_function(
            xe-dr*np.sqrt(retained.shape[1]))
        for name, value in direct.items():
            error = float(np.max(np.abs(value-contribution[name][:n])))
            max_direct_error = max(max_direct_error, error)
            np.testing.assert_allclose(value, contribution[name][:n], atol=2e-6, rtol=2e-6)
        draw_records = []
        for arm, removed in ARMS.items():
            b, e = old['base_scores'][draw].copy(), old['expert_scores'][draw].copy()
            for name in removed:
                destination = b if name.startswith('base_') else e
                destination[ev] -= contribution[name][ev]
            result = route(b, e, labels, ev, ids, model['gate'], old['cap'][draw])
            if arm == 'reference':
                np.testing.assert_array_equal(result['selected'], old['selected'][draw])
                np.testing.assert_array_equal(result['predictions'], old['predictions'][draw])
            draw_records.append(result)
        records.append(draw_records)
    geometry['max_direct_gradient_error'] = max_direct_error
    np.savez_compressed(folder/'trace.npz', query_rows=ids, source_rows=source, support_rows=old['support_rows'],
        eval_mask=old['eval_mask'], q=q, singular_values=singular, latent=latent,
        component_names=np.asarray(COMPONENTS), contributions=np.stack(traces),
        component_coefficients=np.stack(coefficients), arm_names=np.asarray(list(ARMS)),
        **{k: np.stack([[r[k] for r in draw] for draw in records]) for k in records[0][0]})
    pd.DataFrame(energy).to_csv(folder/'component_usage.csv', index=False)
    write_json(folder/'FROZEN.json', geometry)
    print(f'FROZEN {port}: rank={q.shape[1]}, direct error={max_direct_error:.2g}', flush=True)
    return geometry


def evaluate(inventory, frame):
    rows, components = [], []
    for item in inventory:
        port = item['port']
        p = archive(OUT/'ports'/port/'trace.npz')
        y = frame.class_id.to_numpy()[p['query_rows']]
        components.append(pd.read_csv(OUT/'ports'/port/'component_usage.csv'))
        for draw, ev in enumerate(p['eval_mask']):
            original = p['predictions'][draw, 0]
            for j, arm in enumerate(p['arm_names']):
                full = metrics(y[ev], p['predictions'][draw, j, ev], 8)
                b = metrics(y[ev], p['base_predictions'][draw, j, ev], 8)
                e = metrics(y[ev], p['expert_predictions'][draw, j, ev], 8)
                rows.append(dict(port=port, draw=draw, arm=arm, acc=full['acc'], ba=full['ba'],
                    base_acc=b['acc'], base_ba=b['ba'], expert_acc=e['acc'], expert_ba=e['ba'],
                    full_prediction_change=float(np.mean(original[ev] != p['predictions'][draw,j,ev])),
                    base_prediction_change=float(np.mean(p['base_predictions'][draw,0,ev] != p['base_predictions'][draw,j,ev])),
                    expert_prediction_change=float(np.mean(p['expert_predictions'][draw,0,ev] != p['expert_predictions'][draw,j,ev])),
                    admission_change=float(np.mean(p['selected'][draw,0,ev] != p['selected'][draw,j,ev]))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT/'metrics.csv', index=False, encoding='utf-8-sig')
    pm = df.groupby(['port', 'arm']).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT/'port_means.csv', index=False, encoding='utf-8-sig')
    means = pm.groupby('arm').mean(numeric_only=True).reset_index()
    means.to_csv(OUT/'equal_port_means.csv', index=False, encoding='utf-8-sig')
    usage = pd.concat(components, ignore_index=True)
    usage.to_csv(OUT/'component_usage.csv', index=False, encoding='utf-8-sig')
    um = usage.groupby(['port', 'component']).mean(numeric_only=True).groupby('component').mean().drop(columns='draw')
    um.to_csv(OUT/'component_usage_means.csv', encoding='utf-8-sig')
    write_json(OUT/'COMPLETE.json', dict(ports=23, draws=5, arms=len(ARMS), rows=len(df),
        main_model_unchanged=True, fits=0, reference_policy_exact=True, summaries=means.to_dict('records')))
    print(means.to_string(index=False), flush=True)
    print(um.to_string(), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'COMPLETE.json').exists():
        raise ValueError('Preserve completed attribution')
    write_json(OUT/'PROTOCOL.json', dict(date='2026-10-08', task='Locate pre-existing use, not add a module',
        candidate=str(CANDIDATE), baseline='Original full joint, not candidate-augmented model',
        scope='23 source-purged outer ports; five existing supports; frozen R50 paths only',
        operation='x=(h-source_mean)/source_std; Q=SVD candidate basis; remove xQQT from query readout paths',
        training='None; source heads, support coefficients, support features, R18, gate and cap frozen',
        paths=list(COMPONENTS), arms=list(ARMS),
        gate='Recompute margins/entropy, eligibility and fixed budget ranking from intervened scores',
        geometry='Parent saved PCA is a numerically recovered map to cached Xs; report replay error',
        interpretation='Computational sensitivity of current readout; not causal ship factors or retrain-without-information performance',
        redundancy='R18 may encode correlated information; unchanged here, so not an exhaustive information ablation',
        energy='Class-centered classifier-gradient squared projection, NOT accuracy contribution percentage'))
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    geometries = []
    with threadpool_limits(limits=2):
        for item in inventory:
            marker = OUT/'ports'/item['port']/'FROZEN.json'
            geometries.append(json.loads(marker.read_text(encoding='utf-8')) if marker.exists()
                              else freeze(item['port'], base, frame))
        pd.DataFrame(geometries).to_csv(OUT/'geometry.csv', index=False)
        write_json(OUT/'ALL_FROZEN.json', dict(ports=23))
        evaluate(inventory, frame)


def verify_completed_readouts():
    inventory = json.loads((ROOT/'oof_base_adapt_results_20261007/POLICIES_SAVED.json').read_text(encoding='utf-8'))
    base = np.load(SOURCE/'pooled_features.npy', mmap_mode='r')
    records = []
    with threadpool_limits(limits=2):
        for item in inventory:
            port = item['port']
            parent = PARENT/port
            old = archive(parent/'matched_joint/joint_policies.npz')
            adapted = archive(ROOT/'oof_base_adapt_results_20261007'/port/'policies.npz')
            with (parent/'source_model.pkl').open('rb') as f:
                model = pickle.load(f)
            stats = archive(parent/'deployment_complement.npz')
            with np.load(ROOT/'pred_cache2'/f'{port}.npz') as saved:
                xs = saved['Xs'][old['query_rows']]
            raw = np.asarray(base[old['query_rows']], dtype=float)
            extra = np.load(parent/'matched_joint/joint_extra.npy', mmap_mode='r')
            bd = ((discarded_coordinates(raw, model['pca'])-stats['discarded_center'][:raw.shape[1]])/
                  stats['discarded_scale'][:raw.shape[1]])
            prior, retained, complement = matched_views(xs, bd, extra, model, stats, True)
            result = verify_readouts(model, xs, retained, complement, prior, old, adapted)
            records.append(dict(port=port, **result))
    write_json(OUT/'READOUT_REPLAY_VERIFICATION.json', dict(ports=len(records), results=records,
        max_expert_error=max(r['max_expert_score_replay_error'] for r in records),
        max_base_error=max(r['max_base_score_replay_error'] for r in records),
        note='Validates fixed support/readout reconstruction, not the unsaved original raw-to-Xs transform'))
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    verify_completed_readouts() if '--verify-readouts' in sys.argv else main()
