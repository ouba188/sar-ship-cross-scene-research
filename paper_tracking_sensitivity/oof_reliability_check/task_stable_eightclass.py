"""Eight-output sensitivity with all target queries and recomputed handoff."""

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '2'

import itertools
import json
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import predict_gate, route_scores
from run_protected_drift_activation import gate_features
from run_generic_transfer import metrics
from sar_joint_completion import write_json
from task_stable_sensitivity import (ARMS, PARENT, ROOT, SOURCE, make_views,
                                      pair_geometry, readout_gradient)

OUT = ROOT / 'task_stable_eightclass_20261009'
STRENGTHS = (.25, 1.)


def predicted_pairs(base):
    # Stable ties use class column order, independent of target truth.
    return np.sort(np.argsort(-base, axis=1, kind='stable')[:, :2], axis=1)


def direction_delta(z, geo):
    u = geo['direction']
    return np.outer((z - geo['midpoint']) @ u, u)


def predicted_delta(z, base, geometry):
    pairs = predicted_pairs(base)
    delta = np.zeros_like(z, dtype=float)
    for key, geo in geometry.items():
        use = np.all(pairs == key, axis=1)
        if use.any():
            delta[use] = direction_delta(z[use], geo)
    return delta


def intervene(base, expert, delta, bg, eg, ev, strength):
    new_base, new_expert = base.copy(), expert.copy()
    new_base[ev] -= strength * delta[ev] @ bg
    new_expert[ev] -= strength * delta[ev] @ eg
    return new_base, new_expert


def score_run(base, expert, gate, support_y, ev, rows, y):
    score = predict_gate(gate, gate_features(base, expert))
    result = route_scores(base, expert, score, support_y, ev, rows)
    records = dict(query_n=int(ev.sum()), selected=int(result['selected'].sum()), cap=result['cap'])
    recalls = {}
    for name, prediction in [('full', result['predictions']), ('base', result['base_predictions']),
                              ('expert', result['expert_predictions'])]:
        m = metrics(y[ev], prediction[ev], 8)
        records[name + '_oa'] = m['acc']
        records[name + '_ba'] = m['ba']
        recalls[name] = m['class_recall']
    return records, recalls, result, score


def run_port(folder, frame, raw):
    with (folder / 'geometry.pkl').open('rb') as f:
        g = pickle.load(f)
    with (folder / 'source_models.pkl').open('rb') as f:
        fit = pickle.load(f)
    with np.load(folder / 'policies.npz') as f:
        p = {k: f[k] for k in f.files}
    srows = np.load(folder / 'source_rows.npy')
    trows = np.load(folder / 'target_rows.npy')
    np.testing.assert_array_equal(trows, p['query_rows'])
    np.testing.assert_array_equal(srows, p['source_rows'])
    sb = np.load(folder / 'source_bank.npy', mmap_mode='r')
    tb = np.load(folder / 'target_bank.npy', mmap_mode='r')
    r = g['stable_rank']
    sz = np.asarray(sb[:, 192:192+r]) * g['stable_scale'] + g['stable_center']
    tz = np.asarray(tb[:, 192:192+r]) * g['stable_scale'] + g['stable_center']
    labels, domains = frame.class_id.to_numpy(), frame.port.to_numpy()
    if (domains[srows] == folder.name).any():
        raise ValueError('Target in source direction fit')
    geometry = {}
    for a, b in itertools.combinations(range(8), 2):
        geo = pair_geometry(sz, labels[srows], domains[srows], a, b)
        if geo is not None:
            geometry[a, b] = geo
    fixed_deltas = {f'fixed_pair_{a}_{b}': direction_delta(tz, geo)
                    for (a, b), geo in geometry.items()}
    extra = np.load(ROOT / 'module_completion_20261008/ports_projection' / folder.name
                    / 'matched_joint/joint_extra.npy', mmap_mode='r')
    records, class_rows = [], []
    max_gate_error, ref_runs = 0., 0
    for draw, ev in enumerate(p['eval_mask']):
        support = np.searchsorted(trows, p['support_rows'][draw])
        np.testing.assert_array_equal(trows[support], p['support_rows'][draw])
        if ev[support].any():
            raise ValueError('Supports included in query metric')
        support_views = make_views(tb[support], raw[trows[support]], extra[support], g)
        support_y, y = labels[trows[support]], labels[trows]
        for arm in ARMS:
            j = list(p['arm_names']).index(arm)
            base, expert = p['base_scores'][draw, j], p['expert_scores'][draw, j]
            bg = readout_gradient(g, fit['heads'], p, support_views, arm, draw, 'base')
            eg = readout_gradient(g, fit['heads'], p, support_views, arm, draw, 'expert')
            rec, recall, route, scores = score_run(base, expert, fit['gates'][arm], support_y, ev, trows, y)
            np.testing.assert_array_equal(route['predictions'], p['predictions'][draw, j])
            np.testing.assert_array_equal(route['selected'], p['selected'][draw, j])
            error = float(np.max(np.abs(scores - p['gate_scores'][draw, j])))
            np.testing.assert_allclose(scores, p['gate_scores'][draw, j], atol=1e-9)
            max_gate_error, ref_runs = max(max_gate_error, error), ref_runs + 1
            key = dict(port=folder.name, draw=draw, arm=arm)
            reference = dict(rec)
            records.append(dict(**key, selector='reference', attenuation=0.,
                full_oa_delta_pp=0., full_ba_delta_pp=0., **rec))
            for stage, values in recall.items():
                for c, value in enumerate(values):
                    class_rows.append(dict(**key, selector='reference', attenuation=0.,
                                           stage=stage, class_id=c, recall=value, n=int(sum(y[ev] == c))))
            deltas = dict(fixed_deltas, predicted_top2=predicted_delta(tz, base, geometry))
            for selector, delta in deltas.items():
                for strength in STRENGTHS:
                    b, e = intervene(base, expert, delta, bg, eg, ev, strength)
                    np.testing.assert_array_equal(b[~ev], base[~ev])
                    np.testing.assert_array_equal(e[~ev], expert[~ev])
                    rec, recall, route, _ = score_run(b, e, fit['gates'][arm], support_y, ev, trows, y)
                    if route['selected'][~ev].any() or rec['selected'] > rec['cap']:
                        raise ValueError('Invalid post-intervention handoff')
                    records.append(dict(**key, selector=selector, attenuation=strength,
                        full_oa_delta_pp=100 * (rec['full_oa'] - reference['full_oa']),
                        full_ba_delta_pp=100 * (rec['full_ba'] - reference['full_ba']), **rec))
                    if selector == 'predicted_top2':
                        for stage, values in recall.items():
                            for c, value in enumerate(values):
                                class_rows.append(dict(**key, selector=selector, attenuation=strength,
                                    stage=stage, class_id=c, recall=value, n=int(sum(y[ev] == c))))
    check = dict(port=folder.name, reference_policy_replays=ref_runs,
                 max_gate_replay_error=max_gate_error, source_pairs=len(geometry))
    print(json.dumps(check), flush=True)
    return records, class_rows, check


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'COMPLETE.json').exists():
        raise ValueError('Preserve completed eight-class diagnostic')
    write_json(OUT / 'PROTOCOL.json', dict(date='2026-10-09', new_fits=0,
        parent=str(PARENT), main_model_unchanged=True, arms=list(ARMS),
        task='All query samples, argmax over all8 columns, then recomputed existing5percent gate',
        geometry='Source-only common-port pairmean directions in fixed Qs; same previous sensitivity probe',
        selectors='Each source pair direction applied to ALLqueries separately; plus unperturbed base-predicted top2 perquery',
        attenuation=list(STRENGTHS), intervention='Query only, both base/expert R50 scorepaths; supports/R18/coefficients fixed',
        target_query_labels='Metrics only; NOT used to select pair, modify scores, route, or tune strength',
        frozen_gate='Refitted parent gate stays fixed; margin/entropy inputs, ranking/eligibility recomputed after change',
        ba='Mean recall over classes with targetquery support; then five draws within port, then23 equal ports',
        interpretation='Readout computational dependency, not a new trained model, causal invariance, or task improvement',
        development=True))
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['class_id', 'port'], keep_default_na=False)
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    folders = [p for p in sorted((PARENT / 'ports').iterdir()) if (p / 'policies.npz').exists()]
    records, classes, checks = [], [], []
    with threadpool_limits(limits=2):
        for folder in folders:
            r, c, v = run_port(folder, frame, raw)
            records.extend(r)
            classes.extend(c)
            checks.append(v)
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'metrics.csv', index=False)
    keys = ['arm', 'selector', 'attenuation']
    pm = df.groupby(['port', *keys]).mean(numeric_only=True).drop(columns='draw').reset_index()
    pm.to_csv(OUT / 'port_means.csv', index=False)
    means = pm.groupby(keys).mean(numeric_only=True).reset_index()
    means.to_csv(OUT / 'equal_port_means.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT / 'class_recalls.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT / 'verification.csv', index=False)
    write_json(OUT / 'COMPLETE.json', dict(ports=len(folders), draws=5, records=len(df),
        reference_policy_replays=sum(c['reference_policy_replays'] for c in checks),
        max_gate_replay_error=max(c['max_gate_replay_error'] for c in checks),
        main_model_unchanged=True, new_fits=0))
    print(means[means.selector.isin(['reference', 'predicted_top2'])].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
