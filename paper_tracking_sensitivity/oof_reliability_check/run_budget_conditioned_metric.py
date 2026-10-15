"""Prepare and evaluate a fixed-capacity budget-conditioned metric pilot."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'

import argparse
import gc
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from evidence_aware_handoff import fit_gate, predict_gate, route_scores
from projection_residual_expert import discarded_coordinates, fit_kernel_correction
from run_stable_guided_spatial_pooling import (
    ROOT, SOURCE, PARENT, EVIDENCE, PORTS, archive, read_json, gate_features, collect_record)
from sar_joint_completion import write_json


OUT = ROOT / 'budget_conditioned_metric_followup'
ARMS = ('original', 'matched_gate', 'classification_metric', 'budget_metric')


def episode_views(ep, extra):
    z = ((extra - ep['ssl_mean']) @ ep['ssl_components'].T).astype(np.float32)
    xe = (np.column_stack((ep['xs'], z)) - ep['expert_center']) / ep['expert_scale']
    prior = (xe @ ep['expert_coef'].T + ep['expert_intercept']).astype(np.float32).astype(float)
    centered = extra.astype(float) - ep['ssl_mean']
    discarded = centered - (centered @ ep['ssl_components'].T) @ ep['ssl_components']
    discarded = (discarded - ep['discarded_center'][2048:]) / ep['discarded_scale'][2048:]
    complement = np.column_stack((ep['base_discarded'], discarded)) / np.sqrt(2560)
    return prior, xe.astype(float) / np.sqrt(192), complement


def predict(prior, retained, complement, support, labels, weights):
    kernel = .5 * (retained @ retained[support].T) + .5 * ((complement * weights) @ complement[support].T) + 1
    return prior + kernel @ fit_kernel_correction(kernel[support], labels, prior[support])


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(ports=list(PORTS), arms=list(ARMS), support=64, draws=5, budget=.05,
        metric='2560 positive complement-channel weights; separately unit-mean in R50 2048 and R18 512 blocks',
        parameterization='w=(1+.5*tanh(theta))/blockmean; theta0=0 gives exact original kernel',
        fixed='Backbone, source prior, retained kernel, base64 scores, ridge8, and once-fitted four-confidence source gate',
        capacity='No extra feature dimensions or new expert; same2560 trainable scalars in both learned arms',
        training_split='Existing source episode index modulo3==0 held for diagnostics, others trained',
        epochs=6, optimizer='AdamW lr.01 weight_decay.001, gradientclip5, seed271828, no sweeps',
        classification_loss='Class-balanced full-query CE after the actual64-support adaptation',
        budget_loss='.5full-query CE +.5CE on actual selected handoffs; preserve full-query class weights within tail',
        selection='Full source episode, floor(.05*Nquery), actual frozen source gate, observed-class and disagreement guards; selection detached',
        selection_gradient='No gradient through discrete top-k or argmax; this is a selected-set surrogate, not exact differentiation of BA',
        checkpoint='Last epoch only; held-source diagnostics do not select checkpoint or parameters',
        gate='Fit once on original source episode predictions and keep same parameters during learning and target deployment',
        boundary='Geometry and encoder use all outer source; gate also uses allsource including diagnostic-held episodes. Not fully nested OOF.',
        target_labels='Only same64 supports; query labels accessed after all six targets frozen',
        gate_diagnostic='Held source excludes metric optimizer but not upstream geometry or gate fit; not independent validation',
        main_model_changed=False, development_only=True, independent_confirmation=False)
    if (OUT / 'PROTOCOL.json').exists():
        assert read_json(OUT / 'PROTOCOL.json') == protocol
    write_json(OUT / 'PROTOCOL.json', protocol)
    for port in PORTS:
        folder = OUT / 'ports' / port
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / 'PREPARED.json').exists():
            continue
        source = np.load(EVIDENCE / 'source_features' / port / 'rows.npy')
        extra = np.load(EVIDENCE / 'source_features' / port / 'features.npy', mmap_mode='r')
        descriptions = read_json(PARENT / port / 'EPISODES_COMPLETE.json')['episodes']
        receipts = {e['port']: e for e in read_json(EVIDENCE / 'source_episodes' / port / 'COMPLETE.json')['episodes']}
        groups, records, replay_error, fp32_error = [], [], 0., 0.
        for index, description in enumerate(descriptions):
            ep = archive(description['path'])
            saved = archive(receipts[description['port']]['path'])
            rows, support, query = ep['rows'], ep['support'], ep['query']
            local = np.searchsorted(source, rows)
            np.testing.assert_array_equal(source[local], rows)
            np.testing.assert_array_equal(rows, saved['rows'])
            np.testing.assert_array_equal(support, saved['support'])
            np.testing.assert_array_equal(query, np.flatnonzero(saved['query']))
            assert len(support) == 64 and not np.intersect1d(support, query).size
            prior, retained, complement = episode_views(ep, extra[local])
            labels, base, expert = ep['labels'], saved['base_scores'], saved['expert_scores']
            replay = predict(prior, retained, complement, support, labels[support], np.ones(2560))
            replay_error = max(replay_error, float(np.max(np.abs(replay - expert))))
            np.testing.assert_allclose(replay, expert, atol=3e-6, rtol=3e-6)
            rounded = predict(prior, retained.astype(np.float32).astype(float), complement.astype(np.float32).astype(float),
                support, labels[support], np.ones(2560))
            fp32_error = max(fp32_error, float(np.max(np.abs(rounded - expert))))
            np.testing.assert_allclose(rounded, expert, atol=2e-5, rtol=2e-5)
            filename = f'episode_{index:02d}.npz'
            np.savez(folder / filename, prior=prior, retained=retained.astype(np.float32),
                complement=complement.astype(np.float32), rows=rows, labels=labels, base=base,
                expert=expert, support=support, query=query)
            records.append(dict(port=description['port'], path=str(folder / filename),
                split='held' if index % 3 == 0 else 'train'))
            _, inverse, counts = np.unique(labels[query], return_inverse=True, return_counts=True)
            eligible = (base.argmax(1) != expert.argmax(1)) & np.isin(expert.argmax(1), labels[support])
            groups.append(dict(base=base[query], expert=expert[query], labels=labels[query],
                features=gate_features(base, expert)[query], eligible=eligible[query],
                weights=1 / (len(descriptions) * len(counts) * counts[inverse])))
        joined = {key: np.concatenate([g[key] for g in groups]) for key in groups[0]}
        gate = fit_gate(**joined)
        with (folder / 'gate.pkl').open('wb') as stream:
            pickle.dump(gate, stream)
        np.savez(folder / 'gate.npz', **{key: gate[key] for key in ('center', 'scale', 'coef', 'intercept')})
        write_json(folder / 'PREPARED.json', dict(port=port, episodes=records, original_expert_replay_error=replay_error,
            fp32_feature_replay_error=fp32_error, target_query_labels_used=0))
        print(f'PREPARED {port}, source episodes={len(records)}, replay={replay_error:.3g}, FP32={fp32_error:.3g}', flush=True)
        gc.collect()


def freeze_target(port, raw, support_truth):
    folder = OUT / 'ports' / port
    if (folder / 'TARGET_FROZEN.json').exists():
        return
    old = archive(PARENT / port / 'matched_joint/joint_policies.npz')
    rows = old['query_rows']
    with (folder / 'gate.pkl').open('rb') as stream:
        gate = pickle.load(stream)
    with (PARENT / port / 'source_model.pkl').open('rb') as stream:
        model = pickle.load(stream)
    with np.load(ROOT / 'pred_cache2' / f'{port}.npz') as saved:
        xs = saved['Xs'][rows]
    extra = np.load(PARENT / port / 'matched_joint/joint_extra.npy', mmap_mode='r')
    z = model['pca_ssl'].transform(extra).astype(np.float32)
    xe = (np.column_stack((xs, z)) - model['expert_center']) / model['expert_scale']
    prior = model['expert'].decision_function(xe).astype(np.float32).astype(float)
    retained = xe.astype(float) / np.sqrt(192)
    stats = archive(PARENT / port / 'deployment_complement.npz')
    raw_complement = np.column_stack((discarded_coordinates(raw[rows], model['pca']),
                                      discarded_coordinates(extra, model['pca_ssl'])))
    complement = (raw_complement - stats['discarded_center']) / stats['discarded_scale'] / np.sqrt(2560)
    weights = archive(folder / 'learned_weights.npz')
    outputs = {key: [] for key in ('expert_scores', 'gate_scores', 'predictions', 'selected')}
    maximum_error = 0.
    for draw, ev in enumerate(old['eval_mask']):
        support = np.searchsorted(rows, old['support_rows'][draw])
        np.testing.assert_array_equal(rows[support], old['support_rows'][draw])
        assert len(support) == 64 and not ev[support].any()
        y, base = support_truth[draw], old['base_scores'][draw]
        replay = predict(prior, retained, complement, support, y, np.ones(2560))
        maximum_error = max(maximum_error, float(np.max(np.abs(replay - old['expert_scores'][draw]))))
        np.testing.assert_allclose(replay, old['expert_scores'][draw], atol=3e-6, rtol=3e-6)
        experts = {'matched_gate': old['expert_scores'][draw]}
        for arm in ARMS[2:]:
            experts[arm] = predict(prior, retained, complement, support, y, weights[arm])
        values = {key: [old[key][draw]] for key in outputs}
        for arm in ARMS[1:]:
            score = predict_gate(gate, gate_features(base, experts[arm]))
            routed = route_scores(base, experts[arm], score, y, ev, rows)
            assert routed['cap'] == old['cap'][draw]
            for key, value in dict(expert_scores=experts[arm], gate_scores=score,
                predictions=routed['predictions'], selected=routed['selected']).items():
                values[key].append(value)
        for key in outputs:
            outputs[key].append(np.stack(values[key]))
    np.savez_compressed(folder / 'policies.npz', arms=np.array(ARMS), query_rows=rows,
        eval_mask=old['eval_mask'], support_rows=old['support_rows'], cap=old['cap'],
        base_scores=old['base_scores'], **{key: np.stack(value) for key, value in outputs.items()})
    write_json(folder / 'TARGET_FROZEN.json', dict(port=port, draws=5, target_query_labels_used=0,
        unchanged_base=True, original_expert_replay_error=maximum_error, gate_refitted_after_learning=False))
    print('TARGET FROZEN ' + port, flush=True)


def evaluate():
    frame = pd.read_csv(SOURCE / 'manifest.csv', usecols=['class_id'])
    raw = np.load(SOURCE / 'pooled_features.npy', mmap_mode='r')
    for port in PORTS:
        with np.load(PARENT / port / 'matched_joint/joint_policies.npz') as old:
            truth = frame.class_id.to_numpy()[old['support_rows']]
        freeze_target(port, raw, truth)
    write_json(OUT / 'ALL_FROZEN.json', dict(ports=list(PORTS), query_labels_read=False))
    records = []
    for port in PORTS:
        saved = archive(OUT / 'ports' / port / 'policies.npz')
        y = frame.class_id.to_numpy()[saved['query_rows']]
        for draw, ev in enumerate(saved['eval_mask']):
            for index, arm in enumerate(ARMS):
                r = collect_record(port, '', arm, y[ev], saved['base_scores'][draw, ev],
                    saved['expert_scores'][draw, index, ev], saved['predictions'][draw, index, ev],
                    saved['selected'][draw, index, ev])
                r['draw'] = draw
                records.append(r)
    df = pd.DataFrame(records)
    df.to_csv(OUT / 'target_draw_metrics.csv', index=False)
    columns = ['oa', 'ba', 'base_ba', 'expert_ba', 'rescues', 'harms', 'admissions']
    ports = df.groupby(['port', 'arm'], as_index=False)[columns].mean()
    means = ports.groupby('arm', as_index=False)[columns].mean()
    ports.to_csv(OUT / 'target_port_means.csv', index=False)
    means.to_csv(OUT / 'target_equal_port_means.csv', index=False)
    write_json(OUT / 'COMPLETE.json', dict(ports=list(PORTS), draws=5, arms=list(ARMS),
        status='complete', metrics=means.to_dict('records'), main_model_changed=False))
    print(means.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        prepare() if args.stage == 'prepare' else evaluate()
