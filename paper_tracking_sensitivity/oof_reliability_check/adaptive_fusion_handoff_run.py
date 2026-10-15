"""Matched latest-dual source-calibrated base/fusion selection experiment."""

import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from compact_fusion_source import ROOT, SOURCE, load_npz, load_source_pairs
from compact_fusion_policy import balanced_weights
from adaptive_fusion_handoff_policy import (FEATURES, PENALTY, THRESHOLDS, accounting,
    features, probability, select, train_policy)

OUT = ROOT / 'adaptive_fusion_handoff_20261010'
TASKS = ROOT / 'learned_frequency_residual_20261009/tasks'
REFERENCE = ROOT / 'backbone_ablation_20261010'
OLD = ROOT / 'compact_fusion_20261010'
MODES = ('base', 'full_fusion', 'adaptive', 'fixed_half', 'source_constant')
BATCH = 512


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def metadata(folder):
    info = json.loads((folder / 'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def tasks():
    result = [p.parent for p in sorted(TASKS.rglob('PREPARED.json'))]
    if len(result) != 25:
        raise ValueError('Expected 23 port tasks + F0921 + OSS2')
    return result


def freeze_protocol():
    path = OUT / 'PROTOCOL.json'
    if path.exists():
        return
    write(path, dict(date='2026-10-10', frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        reference=str(REFERENCE), reference_mode='dual_refit_late', tasks=25, support_draws=5,
        target_labels_per_draw=64, modes=MODES, primary_mode='adaptive', primary_metric='BA',
        actions='Base argmax or equal softmax fusion argmax; NEVER expert-only substitution',
        source='Reuse exact port pairs; recompute external held-port-out priors on matched joint-tail coordinates',
        selector='Eight confidence/comparison features plus intercept, sigmoid linear score',
        features=FEATURES, slope_l2=PENALTY, coefficient_bounds=[-8., 8.],
        objective='Exclusive fusion-rescue versus fusion-harm log loss; retain full-population port/class weights before filtering',
        threshold_grid=THRESHOLDS.tolist(), threshold_selection='Three source-port OOF folds, maximize equal-port BA; ties prefer lower threshold',
        source_evaluation='Nested 3-fold source-port evaluation including inner threshold selection; inherited representation is not held out',
        constant_control='Source port/class weighted choice of always base versus always fusion',
        diagnostic_fixed_half='Same fitted selector, threshold .5 fixed before target evaluation',
        diagnostic_random='Analytic expected accuracy of uniform random selection among disagreements at matched selected count',
        adaptive_budget='Fraction of all queries on which base/fusion disagree and fusion is selected; no forced cap',
        inference='Both feature extractors and heads already computed; NOT compute saving',
        target_selection=False, target_predictions_frozen_before_readout=True,
        independent_confirmation=False, retrain_network=False))


def source_task(folder):
    info, suffix = metadata(folder)
    dst = OUT / 'source' / suffix
    if (dst / 'MODEL.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if info['dataset'] == 'self24ports':
        source = load_npz(OLD / 'source' / suffix / 'source_pairs_final.npz')
        b, e, y, episode, rows = [source[k] for k in ('base', 'expert', 'labels', 'episode', 'rows')]
        provenance = json.loads((OLD / 'source' / suffix / 'FUSION.json').read_text())['provenance']
    else:
        pairs, provenance = load_source_pairs(folder, refit_joint=True)
        b, e, y, rows = [np.concatenate([p[k] for p in pairs]) for k in ('base', 'expert', 'labels', 'rows')]
        episode = np.concatenate([np.full(len(p['labels']), p['episode']) for p in pairs])
    mapping = {}
    for ei in np.unique(episode):
        with np.load(folder / 'episodes' / f'{ei:02d}.npz') as z:
            mapping[int(ei)] = str(z['held_port'])
    groups = np.array([mapping[int(ei)] for ei in episode])
    pb, pe = softmax(b, axis=1), softmax(e, axis=1)
    x = features(pb, pe)
    base, fusion = pb.argmax(1), ((pb + pe) * .5).argmax(1)
    model, diagnostic = train_policy(x, base, fusion, y, groups)
    model.update(dataset=info['dataset'], task=info['task'], source_ports=len(np.unique(groups)),
                 source_queries=len(y), provenance=provenance, source_query_labels_only=True)
    np.savez_compressed(dst / 'source_pairs.npz', base=b, expert=e, labels=y, episode=episode,
        rows=rows, groups=groups, features=x, **diagnostic)
    write(dst / 'MODEL.json', model)
    w = balanced_weights(y, groups)
    delta = 100 * w @ ((diagnostic['nested_prediction'] == y).astype(float) - (fusion == y))
    print(f'SOURCE {suffix}: t={model["threshold"]:.2f} nested dBA={delta:+.4f}', flush=True)


def target_task(folder):
    from dual_evidence_core import EvidenceContext, base_cross, load_task

    info, suffix = metadata(folder)
    dst = OUT / 'targets' / suffix
    if (dst / 'COMPLETE.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = json.loads((OUT / 'source' / suffix / 'MODEL.json').read_text())
    data = load_task(folder)
    reference = load_npz(REFERENCE / 'targets' / suffix / 'predictions.npz')
    for key in ('rows', 'ids', 'support', 'eval_masks'):
        np.testing.assert_array_equal(data[key], reference[key])
    saved = load_npz(REFERENCE / 'targets' / suffix / 'coefficients.npz')
    n = len(data['rows'])
    pred = np.empty((5, len(MODES), n), np.int8)
    gate = np.empty((5, n), float)
    trace = np.empty((5, n, len(FEATURES)), np.float32)
    accepted = np.empty((5, n), bool)
    union = np.unique(data['support'])
    columns = [np.searchsorted(union, s) for s in data['support']]
    if info['dataset'] == 'self24ports':
        with np.load(folder / 'target.npz') as z:
            expert_scores = z['expert']
        ctx = None
    else:
        ctx = EvidenceContext(folder, data)
        ref = ctx.features(union)
    for lo in range(0, n, BATCH):
        indices = np.arange(lo, min(lo + BATCH, n))
        if ctx is not None:
            current = ctx.features(indices)
            kernel = ctx.cross(current, ref)['D']
        for draw, support in enumerate(data['support']):
            kb = base_cross(data, indices, support, info['eta'])
            b = data['prior'][indices] + kb @ saved[f'{draw}_base']
            if ctx is None:
                e = expert_scores[draw, indices]
            else:
                e = current['priors']['dual'] + kernel[:, columns[draw]] @ saved[f'{draw}_dual_refit_expert']
            pb, pe = softmax(b, axis=1), softmax(e, axis=1)
            base, fusion = pb.argmax(1), ((pb + pe) * .5).argmax(1)
            x = features(pb, pe)
            p = probability(model, x)
            routed, use = select(base, fusion, p, model['threshold'])
            fixed, _ = select(base, fusion, p, .5)
            values = (base, fusion, routed, fixed, fusion if model['constant_use_fusion'] else base)
            for mi, value in enumerate(values):
                pred[draw, mi, indices] = value
            for mi, name in ((0, 'base'), (1, 'dual_refit_late')):
                np.testing.assert_array_equal(pred[draw, mi, indices],
                    reference['predictions'][draw, list(reference['modes']).index(name), indices],
                    err_msg=f'{suffix} draw{draw} {name} replay')
            gate[draw, indices], trace[draw, indices], accepted[draw, indices] = p, x, use
        if lo % (BATCH * 40) == 0:
            print(f'TARGET {suffix}: {indices[-1] + 1}/{n}; reference exact', flush=True)
    np.savez_compressed(dst / 'predictions.npz', predictions=pred, modes=np.array(MODES),
        gate_probability=gate, features=trace, accepted=accepted, rows=data['rows'], ids=data['ids'],
        support=data['support'], support_labels=data['support_labels'], eval_masks=data['eval_masks'])
    write(dst / 'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        threshold=model['threshold'], original_predictions_exact=True, target_query_labels_read=False,
        seconds=time.perf_counter() - started))
    print(f'FROZEN {suffix}: {time.perf_counter()-started:.1f}s', flush=True)


def scores(y, pred):
    weight = balanced_weights(y, np.zeros(len(y), int))
    return dict(oa=float(100 * np.mean(y == pred)), ba=float(100 * weight @ (y == pred)))


def auc(y, base, fusion, probability):
    exclusive = (base == y) != (fusion == y)
    target = fusion[exclusive] == y[exclusive]
    return float(roc_auc_score(target, probability[exclusive])) if len(np.unique(target)) == 2 else np.nan


def report():
    from strong_single_multidata_explain import truth_for

    folders = tasks()
    if not all((OUT / 'targets' / metadata(f)[1] / 'COMPLETE.json').exists() for f in folders):
        raise RuntimeError('Freeze every target prediction before evaluation')
    records, details, class_rows, cases, sources, threshold_rows, reliability = [], [], [], [], [], [], []
    for folder in folders:
        info, suffix = metadata(folder)
        model = json.loads((OUT / 'source' / suffix / 'MODEL.json').read_text())
        p = load_npz(OUT / 'targets' / suffix / 'predictions.npz')
        truth, _ = truth_for(info, p)
        source = load_npz(OUT / 'source' / suffix / 'source_pairs.npz')
        sb, se = softmax(source['base'], axis=1), softmax(source['expert'], axis=1)
        sbase, sfused = sb.argmax(1), ((sb + se) * .5).argmax(1)
        for group in np.unique(source['groups']):
            ix = source['groups'] == group
            sy = source['labels'][ix]
            for name, value in [('base', sbase), ('full_fusion', sfused),
                                ('adaptive_nested', source['nested_prediction']),
                                ('source_constant_nested', source['nested_constant'])]:
                sources.append(dict(dataset=info['dataset'], task=info['task'], source_port=group,
                    mode=name, **scores(sy, value[ix]),
                    unique_auc=auc(sy, sbase[ix], sfused[ix], source['oof_probability'][ix])))
        threshold_rows.append(dict(dataset=info['dataset'], task=info['task'], threshold=model['threshold'],
            source_events=model['events'], source_ports=model['source_ports'],
            source_constant_use_fusion=model['constant_use_fusion']))
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            predictions = p['predictions'][draw, :, ev].T
            base, fusion = predictions[0], predictions[1]
            ref = scores(y, fusion)
            probability = p['gate_probability'][draw, ev]
            different = base != fusion
            weight = balanced_weights(y, np.zeros(len(y), int))
            for mi, mode in enumerate(MODES):
                value = predictions[mi]
                met = scores(y, value)
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    **met, delta_oa=met['oa']-ref['oa'], delta_ba=met['ba']-ref['ba'], query_n=len(y)))
                if mode not in ('adaptive', 'fixed_half'):
                    continue
                threshold = model['threshold'] if mode == 'adaptive' else .5
                _, selected = select(base, fusion, probability, threshold)
                np.testing.assert_array_equal(np.where(selected, fusion, base), value)
                ratio = selected.sum() / different.sum() if different.any() else 0.
                expected_correct = (base == y).astype(float) + ratio * ((fusion == y).astype(float) - (base == y))
                facts = accounting(y, base, fusion, value)
                details.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    threshold=threshold, selection_rate=100*selected.mean(),
                    disagreement_rate=100*different.mean(), acceptance_of_disagreements=100*ratio,
                    unique_auc=auc(y, base, fusion, probability),
                    random_matched_expected_oa=float(100*expected_correct.mean()),
                    random_matched_expected_ba=float(100*weight@expected_correct),
                    delta_ba_vs_random=float(met['ba']-100*weight@expected_correct),
                    oracle_base_or_fusion_ba=float(100*weight@((base == y) | (fusion == y))), **facts))
                for c in np.unique(y):
                    ci = y == c
                    class_rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                        class_id=int(c), n=int(ci.sum()), base_recall=100*np.mean(base[ci] == y[ci]),
                        fusion_recall=100*np.mean(fusion[ci] == y[ci]), selected_recall=100*np.mean(value[ci] == y[ci]),
                        **accounting(y[ci], base[ci], fusion[ci], value[ci])))
                if mode == 'adaptive':
                    exclusive = (base == y) != (fusion == y)
                    for lo in np.arange(0., 1., .1):
                        mask = exclusive & (probability >= lo) & (probability < lo+.1)
                        if mask.any():
                            reliability.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                                lo=float(lo), n=int(mask.sum()), mean_score=float(probability[mask].mean()),
                                fusion_win_fraction=float(np.mean(fusion[mask] == y[mask])),
                                ba_fusion_win_fraction=float(weight[mask]@(fusion[mask] == y[mask])/weight[mask].sum())))
                    if draw == 0:
                        types = dict(avoided_harm=(base == y)&(fusion != y)&~selected,
                                     lost_rescue=(base != y)&(fusion == y)&~selected,
                                     retained_rescue=(base != y)&(fusion == y)&selected,
                                     retained_harm=(base == y)&(fusion != y)&selected)
                        evrows = np.flatnonzero(ev)
                        theta = np.asarray(model['theta'])
                        for event, mask in types.items():
                            for j in np.flatnonzero(mask)[:2]:
                                row = evrows[j]
                                terms = (p['features'][draw, row]-model['center'])/model['scale']*theta[1:]
                                reconstructed = float(1/(1+np.exp(-(theta[0]+terms.sum()))))
                                np.testing.assert_allclose(reconstructed, probability[j], atol=1e-6, rtol=0)
                                cases.append(dict(dataset=info['dataset'], task=info['task'], event=event,
                                    sample_id=p['ids'][row], truth=int(y[j]), base=int(base[j]), fusion=int(fusion[j]),
                                    selected=int(value[j]), gate_probability=float(probability[j]), threshold=threshold,
                                    intercept=float(theta[0]), **dict(zip(FEATURES, terms))))
    metrics = pd.DataFrame(records)
    fields = ['oa', 'ba', 'delta_oa', 'delta_ba']
    task = metrics.groupby(['dataset', 'task', 'mode'])[fields].mean().reset_index()
    summary = task.groupby(['dataset', 'mode'])[fields].mean().reset_index()
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    task.to_csv(OUT / 'task_means.csv', index=False)
    summary.to_csv(OUT / 'summary.csv', index=False)
    metrics.groupby(['dataset', 'draw', 'mode'])[fields].mean().to_csv(OUT / 'draw_metrics.csv')
    diagnostics = pd.DataFrame(details)
    diagnostics.to_csv(OUT / 'diagnostics.csv', index=False)
    diagnostics.groupby(['dataset', 'mode']).mean(numeric_only=True).to_csv(OUT / 'diagnostic_summary.csv')
    pd.DataFrame(class_rows).to_csv(OUT / 'class_metrics.csv', index=False)
    pd.DataFrame(cases).to_csv(OUT / 'cases.csv', index=False)
    pd.DataFrame(sources).to_csv(OUT / 'source_nested.csv', index=False)
    pd.DataFrame(sources).groupby(['dataset', 'task', 'mode'])[['oa', 'ba']].mean().groupby(['dataset', 'mode']).mean().to_csv(OUT / 'source_summary.csv')
    pd.DataFrame(threshold_rows).to_csv(OUT / 'thresholds.csv', index=False)
    pd.DataFrame(reliability).to_csv(OUT / 'reliability.csv', index=False)
    old = pd.read_csv(REFERENCE / 'summary.csv')
    for mode, oldmode in [('base', 'base'), ('full_fusion', 'dual_refit_late')]:
        a = summary[summary['mode'] == mode].sort_values('dataset')
        b = old[old['mode'] == oldmode].sort_values('dataset')
        np.testing.assert_allclose(a[['oa', 'ba']], b[['oa', 'ba']], atol=1e-9, rtol=0)
    write(OUT / 'COMPLETE.json', dict(tasks=25, support_draws=125, modes=MODES,
        rows=len(metrics), latest_dual_replay_exact=True, gain_accounting_exact=True,
        source_only_selection=True, independent_confirmation=False))
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    args = parser.parse_args()
    freeze_protocol()
    with threadpool_limits(limits=2):
        if args.stage in ('source', 'all'):
            for folder in tasks():
                source_task(folder)
                gc.collect()
        if args.stage in ('target', 'all'):
            if not all((OUT / 'source' / metadata(f)[1] / 'MODEL.json').exists() for f in tasks()):
                raise RuntimeError('Freeze all source selectors before predicting targets')
            for folder in tasks():
                target_task(folder)
                gc.collect()
        if args.stage in ('report', 'all'):
            report()


if __name__ == '__main__':
    main()
