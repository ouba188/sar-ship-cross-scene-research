"""Freeze compact source-only fusion policies before target-label readout."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from evaluate_learned_frequency import predict
from frozen_comparison import OUT as PREVIOUS, folders, write, ba
from compact_fusion_policy import (FEATURES, GRID, PENALTY, adaptive_weight, balanced_weights,
                                   features, fit, mix, probabilities)

OUT = ROOT/'compact_fusion_20261010'
MODES = ('base', 'expert', 'hard_uncapped', 'equal', 'fixed_source', 'adaptive',
         'adaptive_nll', 'constant_nll', 'utility_constant', 'adaptive_mean')


def metadata(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    return info, f"{info['dataset']}/{info['task']}"


def readouts(model, pb, pe, eval_mask=None):
    w = adaptive_weight(model, pb, pe)
    mean = float(w.mean() if eval_mask is None else w[eval_mask].mean())
    nll_weight = adaptive_weight(dict(model, theta=model['nll_theta']), pb, pe)
    return dict(equal=mix(pb, pe, .5), fixed_source=mix(pb, pe, model['fixed_weight']),
                adaptive=mix(pb, pe, w), constant_nll=mix(pb, pe, model['constant_nll_weight']),
                adaptive_nll=mix(pb, pe, nll_weight), utility_constant=mix(pb, pe, model['utility_constant']),
                adaptive_mean=mix(pb, pe, mean)), w


def source_fit(folder):
    from compact_fusion_source import load_source_pairs
    info, suffix = metadata(folder)
    dst = OUT/'source'/suffix
    if (dst/'FUSION.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if (dst/'MODEL.json').exists():
        old = load_npz(dst/'source_pairs.npz')
        b, e, y, group = (old[k] for k in ('base', 'expert', 'labels', 'episode'))
        source_rows = old['rows']
        provenance = json.loads((dst/'MODEL.json').read_text())['provenance']
    else:
        episodes, provenance = load_source_pairs(folder)
        b, e, y, group = [np.concatenate([item[key] if key != 'episode' else
            np.full(len(item['labels']), item['episode']) for item in episodes])
            for key in ('base', 'expert', 'labels', 'episode')]
        source_rows = np.concatenate([item['rows'] for item in episodes])
    pb, pe = probabilities(b, e)
    groups = np.unique(group)
    if len(groups) < 3:
        raise ValueError('At least three source episodes needed for grouped evaluation')
    fold = np.searchsorted(groups, group) % 3
    oof, w_oof, records = {}, np.empty(len(y)), []
    for k in range(3):
        train, val = fold != k, fold == k
        model = fit(pb[train], pe[train], y[train], group[train])
        result, w_oof[val] = readouts(model, pb[val], pe[val])
        for mode, p in result.items():
            if mode not in oof:
                oof[mode] = np.empty_like(pb)
            oof[mode][val] = p
        for g in groups[np.isin(groups, group[val])]:
            ix = group == g
            for mode in result:
                pred = oof[mode][ix].argmax(1)
                weights = balanced_weights(y[ix], group[ix])
                nll = -weights @ np.log(np.maximum(oof[mode][ix, y[ix]], 1e-15))
                records.append(dict(episode=int(g), fold=k, mode=mode,
                    oa=100*np.mean(pred == y[ix]), ba=ba(y[ix], pred), balanced_nll=float(nll)))
    model = fit(pb, pe, y, group)
    model.update(dataset=info['dataset'], task=info['task'], source_episodes=len(groups),
                 source_queries=len(y), provenance=provenance,
                 source_oof='Three grouped episode folds; final fit uses all source query pairs',
                 no_target_query_labels=True)
    pd.DataFrame(records).to_csv(dst/'source_oof_final.csv', index=False)
    np.savez_compressed(dst/'source_pairs_final.npz', base=b, expert=e, labels=y, episode=group,
        oof_weight=w_oof, rows=source_rows)
    write(dst/'FUSION.json', model)
    print(f'SOURCE {suffix}: fixed={model["fixed_weight"]:.1f}; '
          f'NLL-constant={model["constant_nll_weight"]:.3f}; episodes={len(groups)}', flush=True)


def target_predict(folder):
    info, suffix = metadata(folder)
    dst = OUT/'targets'/suffix
    if (dst/'COMPLETE.json').exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    model = json.loads((OUT/'source'/suffix/'FUSION.json').read_text())
    data = load_npz(folder/'target.npz')
    embedding = load_npz(folder/'selective_embedding.npz')
    previous = load_npz(PREVIOUS/'targets'/suffix/'policies.npz')
    np.testing.assert_array_equal(data['rows'], embedding['rows'])
    np.testing.assert_array_equal(data['ids'], previous['ids'])
    np.testing.assert_array_equal(data['support'], previous['support'])
    np.testing.assert_array_equal(data['eval_masks'], previous['eval_masks'])
    indices = {m: list(previous['modes']).index(m) for m in
               ('ours_base', 'expert_all', 'budget_100', 'softmax_average_all')}
    predictions = np.empty((5, len(MODES), len(data['rows'])), np.int8)
    weights, traces, rows = [], [], []
    start = time.perf_counter()
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support/query mismatch')
        score = predict(data, embedding['embedding'].astype(float), support, sy, info['eta'])
        pb, pe = probabilities(score, data['expert'][draw])
        result, w = readouts(model, pb, pe, ev)
        pred = {mode: p.argmax(1) for mode, p in result.items()}
        pred.update(base=pb.argmax(1), expert=pe.argmax(1),
                    hard_uncapped=previous['predictions'][draw, indices['budget_100']])
        for mode, old_mode in [('base', 'ours_base'), ('expert', 'expert_all'),
                               ('equal', 'softmax_average_all')]:
            np.testing.assert_array_equal(pred[mode], previous['predictions'][draw, indices[old_mode]])
        for mi, mode in enumerate(MODES):
            predictions[draw, mi] = pred[mode]
        weights.append(w.astype(np.float32))
        x = features(pb, pe)
        traces.append(x.astype(np.float32))
        rows.append(dict(draw=draw, weight_mean=float(w[ev].mean()), weight_std=float(w[ev].std()),
            weight_q05=float(np.quantile(w[ev], .05)), weight_q95=float(np.quantile(w[ev], .95)),
            fixed_weight=model['fixed_weight'], constant_nll_weight=model['constant_nll_weight'],
            adaptive_vs_fixed_changes=int(np.sum(pred['adaptive'][ev] != pred['fixed_source'][ev])),
            adaptive_vs_mean_changes=int(np.sum(pred['adaptive'][ev] != pred['adaptive_mean'][ev]))))
        print(f'TARGET {suffix} draw{draw}: mean w={w[ev].mean():.3f}; sd={w[ev].std():.3f}', flush=True)
    np.savez_compressed(dst/'policies.npz', modes=np.array(MODES), predictions=predictions,
        weight=np.array(weights), features=np.array(traces), rows=data['rows'], ids=data['ids'],
        support=data['support'], eval_masks=data['eval_masks'])
    pd.DataFrame(rows).to_csv(dst/'weight_statistics.csv', index=False)
    write(dst/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        target_query_labels_read=False, original_predictions_exact=True, seconds=time.perf_counter()-start))


def diagnostic(y, base, expert, w):
    bc, ec = base == y, expert == y
    unique = bc != ec
    auc = float(roc_auc_score(ec[unique], w[unique])) if len(np.unique(ec[unique])) == 2 else np.nan
    states = dict(expert_only=(~bc & ec), base_only=(bc & ~ec), both_correct=bc & ec, both_wrong=~bc & ~ec)
    result = dict(unique_correct_auc=auc, unique_correct_n=int(unique.sum()))
    for name, mask in states.items():
        result[name+'_n'] = int(mask.sum())
        result[name+'_weight'] = float(w[mask].mean()) if mask.any() else np.nan
    return result


def report():
    tasks = folders()
    if len(tasks) != 25 or not all((OUT/'targets'/metadata(f)[1]/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Complete all frozen predictions before target-label readout')
    records, classes, causes, source_records, source_causes, examples, wstats = [], [], [], [], [], [], []
    for folder in tasks:
        info, suffix = metadata(folder)
        dst = OUT/'targets'/suffix
        p = load_npz(dst/'policies.npz')
        data = load_npz(folder/'target.npz')
        source_frame = pd.read_csv(OUT/'source'/suffix/'source_oof_final.csv')
        source_records.append(source_frame.assign(dataset=info['dataset'], task=info['task']))
        sp = load_npz(OUT/'source'/suffix/'source_pairs_final.npz')
        source_causes.append(dict(dataset=info['dataset'], task=info['task'],
            **diagnostic(sp['labels'], sp['base'].argmax(1), sp['expert'].argmax(1), sp['oof_weight'])))
        wstats.append(pd.read_csv(dst/'weight_statistics.csv').assign(dataset=info['dataset'], task=info['task']))
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        np.testing.assert_array_equal(truth[p['support']], data['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            pred = {mode: p['predictions'][draw, mi, ev] for mi, mode in enumerate(MODES)}
            fixed, equal, dynamic = pred['fixed_source'], pred['equal'], pred['adaptive']
            causes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                **diagnostic(y, pred['base'], pred['expert'], p['weight'][draw, ev])))
            for mode, value in pred.items():
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    oa=100*np.mean(value == y), ba=ba(y, value),
                    delta_oa_vs_equal=100*np.mean((value == y).astype(float)-(equal == y)),
                    delta_ba_vs_equal=ba(y, value)-ba(y, equal),
                    delta_oa_vs_fixed=100*np.mean((value == y).astype(float)-(fixed == y)),
                    delta_ba_vs_fixed=ba(y, value)-ba(y, fixed),
                    rescues_vs_fixed=int(np.sum((value == y) & (fixed != y))),
                    harms_vs_fixed=int(np.sum((value != y) & (fixed == y))), query_n=len(y)))
                for c in np.unique(y):
                    cm = y == c
                    classes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                        class_id=int(c), n=int(cm.sum()), support_n=int(np.sum(data['support_labels'][draw] == c)),
                        recall=100*np.mean(value[cm] == c), delta_vs_fixed=100*np.mean(
                            (value[cm] == c).astype(float)-(fixed[cm] == c))))
            if draw == 0:
                evrows = np.flatnonzero(ev)
                for state, mask in [('rescue', (dynamic == y) & (fixed != y)),
                                    ('harm', (dynamic != y) & (fixed == y))]:
                    for j in np.flatnonzero(mask)[:3]:
                        row = evrows[j]
                        examples.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                            case=state, sample_id=p['ids'][row], truth=int(y[j]), base=int(pred['base'][j]),
                            expert=int(pred['expert'][j]), fixed=int(fixed[j]), adaptive=int(dynamic[j]),
                            expert_weight=float(p['weight'][draw, row]),
                            **dict(zip(FEATURES, p['features'][draw, row].astype(float)))))
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    fields = ['oa', 'ba', 'delta_oa_vs_equal', 'delta_ba_vs_equal', 'delta_oa_vs_fixed', 'delta_ba_vs_fixed']
    task = metrics.groupby(['dataset', 'task', 'mode'])[fields].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'mode'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    metrics.groupby(['dataset', 'draw', 'mode'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    pd.DataFrame(causes).to_csv(OUT/'diagnostics.csv', index=False)
    pd.DataFrame(source_causes).to_csv(OUT/'source_diagnostics.csv', index=False)
    pd.concat(source_records).to_csv(OUT/'source_oof.csv', index=False)
    pd.concat(wstats).to_csv(OUT/'weight_statistics.csv', index=False)
    pd.DataFrame(examples).to_csv(OUT/'cases.csv', index=False)
    previous_summary = pd.read_csv(PREVIOUS/'summary.csv')
    for mode, oldmode in [('equal', 'softmax_average_all'), ('base', 'ours_base'),
                          ('expert', 'expert_all'), ('hard_uncapped', 'budget_100')]:
        actual = summary[summary['mode'] == mode].sort_values('dataset')
        old = previous_summary[previous_summary['mode'] == oldmode].sort_values('dataset')
        np.testing.assert_allclose(actual[['oa', 'ba']], old[['oa', 'ba']], atol=1e-10, rtol=0)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, modes=MODES, rows=len(metrics),
        source_only_fusion=True, equal_and_hard_replay_exact=True, target_prediction_freeze_before_readout=True,
        independent_confirmation=False))
    print(summary[['dataset', 'mode', 'oa', 'ba']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(date='2026-10-10', task='Compact frozen two-model fusion comparison',
            primary_modes=['equal', 'fixed_source', 'adaptive'], modes=MODES, tasks=25, support_draws=5,
            target_support_labels=64, fixed_weight_grid=GRID, fixed_criterion='Equal source episode BA',
            adaptive='Sigmoid linear gate; five permutation-invariant confidence features and intercept',
            adaptive_objective='Episode/class weighted exclusive-correctness binary log loss + .01/2 times slope squared norm',
            features=FEATURES, slope_penalty=PENALTY, parameter_bounds=[-8, 8],
            initialization='Source exclusive-correctness frequency; one run; no parameter search',
            source_only_pilot='Mixture NLL collapsed toward expert in source-only pilot before target readout; retained as adaptive_nll',
            source_evaluation='Three grouped episode folds for fusion only; no independent representation validation',
            diagnostic_constant='NLL scalar optimum and relative-correctness scalar frequency',
            diagnostic_mean='Unlabelled target-query mean adaptive weight; diagnostic, not a trained deployment arm',
            no_target_query_selection=True, no_new_network_training=True, independent_confirmation=False,
            cost='Both model outputs computed for all queries; no end-to-end compute savings claim'))
    tasks = folders()
    if args.stage in ('source', 'all'):
        for folder in tasks:
            source_fit(folder)
    if args.stage in ('target', 'all'):
        if not all((OUT/'source'/metadata(f)[1]/'FUSION.json').exists() for f in tasks):
            raise RuntimeError('Freeze all source policies first')
        for folder in tasks:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
