"""Matched baselines and routing replay for the frozen frequency model."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from prepare_learned_frequency import ROOT, OUT as EXP, SOURCE, PARENT, DATASETS, load_npz, load_pickle
from evaluate_learned_frequency import predict
from frequency_band_core import old_kernel
from receiver_consistent_handoff import confidence_parts, gate_scores
from evidence_aware_handoff import predict_gate
from budgeted_residual_handoff import ranked_mask

OUT = ROOT / 'frozen_comparison_20261010'
HEADS = ROOT / 'frequency_preserve_20261010'
RIDGES = (.01, .1, 1., 8., 64.)
CS = (.1, 1., 10., 100.)
BUDGETS = (0., .05, .1, .2, .5, 1.)
CLASSIFIERS = ('source_prior', 'ridge_mean64', 'ridge_phi64', 'ridge_full_kernel64',
               'prototype64', 'logistic_mean64', 'previous_full', 'ours_base',
               'ours_5pct', 'expert_all', 'softmax_average_all')
ROUTERS = tuple(f'budget_{int(b*100):03d}' for b in BUDGETS) + (
    'gate_common05', 'random_common05', 'margin_common05', 'entropy_common05',
    'random_positive_matched', 'margin_positive_matched')
MODES = CLASSIFIERS + ROUTERS


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def folders():
    return [p.parent for p in sorted((EXP / 'tasks').rglob('EVALUATED.json'))]


def weights(y):
    _, inv, count = np.unique(y, return_inverse=True, return_counts=True)
    return len(y) / (len(count) * count[inv])


def ridge_scores(kernel, support, y, strength):
    root = np.sqrt(weights(y))
    target = 2*np.eye(8)[y]-1
    alpha = root[:, None]*np.linalg.solve(
        root[:, None]*kernel[support]*root[None, :]+strength*np.eye(len(y)), root[:, None]*target)
    score = kernel @ alpha
    score[:, ~np.isin(np.arange(8), y)] = -np.inf
    return score


def prototypes(x, support, y, metric):
    classes = np.unique(y)
    centers = np.stack([x[support][y == c].mean(0) for c in classes])
    if metric == 'cosine':
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
        values = x @ centers.T
    else:
        values = 2*x@centers.T - np.sum(centers**2, axis=1)
    return classes[values.argmax(1)]


def logistic(x, support, y, c):
    if len(np.unique(y)) == 1:
        return np.full(len(x), y[0])
    model = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=500, tol=1e-7)
    model.fit(x[support], y)
    if np.max(model.n_iter_) >= 500:
        raise RuntimeError('Logistic baseline did not converge')
    return model.predict(x)


def kernels(data, embedding, support, eta):
    old, cond = old_kernel(data, support, eta)
    return dict(ridge_mean64=data['mean']@data['mean'][support].T+1,
                ridge_phi64=data['phi']@data['phi'][support].T,
                ridge_full_kernel64=old+.25*(embedding@embedding[support].T)*cond)


def ba(y, pred):
    return 100*float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))


def source_select(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    dest = OUT/'source'/suffix
    if (dest/'SELECTION.json').exists():
        return
    episodes = sorted((folder/'episodes').glob('*.npz'))
    records = []
    for path in sorted((HEADS/'source'/suffix).glob('episode_*.npz')):
        z = load_npz(path)
        ep = load_npz(episodes[int(path.stem.split('_')[-1])])
        ix = z['indices']
        data = {k: ep[k][ix] for k in ('phi', 'mean', 'conditions')}
        s, sy, truth = np.arange(64), ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        for name, kernel in kernels(data, z['learned'].astype(float), s, info['eta']).items():
            for strength in RIDGES:
                pred = ridge_scores(kernel, s, sy, strength).argmax(1)
                records.append(dict(episode=path.stem, mode=name, parameter=str(strength), ba=ba(truth, pred[64:])))
        for metric in ('cosine', 'euclidean'):
            pred = prototypes(data['mean'], s, sy, metric)
            records.append(dict(episode=path.stem, mode='prototype64', parameter=metric, ba=ba(truth, pred[64:])))
        for c in CS:
            pred = logistic(data['mean'], s, sy, c)
            records.append(dict(episode=path.stem, mode='logistic_mean64', parameter=str(c), ba=ba(truth, pred[64:])))
    frame = pd.DataFrame(records)
    means = frame.groupby(['mode', 'parameter'], sort=False).ba.mean()
    selected = {mode: str(means.loc[mode].idxmax()) for mode in frame['mode'].unique()}
    write(dest/'SELECTION.json', dict(**info, selected=selected, criterion='Equal-source-episode BA',
          target_query_labels_used=False, source_episodes=int(frame.episode.nunique())))
    frame.to_csv(dest/'source_validation.csv', index=False)
    print('SOURCE '+suffix+' '+str(selected), flush=True)


def route_inputs(info, data, score, expert, sy, ev):
    if info['dataset'] == 'self24ports':
        gate = load_pickle(PARENT/info['task']/'source_models.pkl')['gates']['joint']
        em, eh = confidence_parts(expert)
        bm, bh = confidence_parts(score)
        gain = predict_gate(gate, np.c_[em, bm, eh, bh])
    else:
        gate = load_pickle(Path(info['model_path']))['gate']
        gain = gate_scores(score, expert, gate)
    b, e = score.argmax(1), expert.argmax(1)
    common = ev & (b != e) & np.isin(e, sy)
    return gain, common, common & (gain > 0)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    dest = OUT/'targets'/suffix
    if (dest/'COMPLETE.json').exists():
        return
    selection = json.loads((OUT/'source'/suffix/'SELECTION.json').read_text())['selected']
    data, old = load_npz(folder/'target.npz'), load_npz(folder/'policies.npz')
    z = load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(z['rows'], data['rows'])
    np.testing.assert_array_equal(old['sample_ids'], data['ids'])
    embedding = z['embedding'].astype(float)
    j = list(old['modes']).index('selective')
    n = len(data['rows'])
    predictions = np.empty((5, len(MODES), n), np.int8)
    selected = np.zeros_like(predictions, dtype=bool)
    traces = []
    gains, commons, positives, ranks = [], [], [], []
    start = time.perf_counter()
    for draw, s in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(s)) != 64 or ev[s].any():
            raise ValueError('Support/query overlap')
        # Recompute float64 scores to avoid changing near-boundary routing through storage rounding.
        score = predict(data, embedding, s, sy, info['eta'])
        expert = data['expert'][draw]
        b, e = score.argmax(1), expert.argmax(1)
        np.testing.assert_array_equal(b, old['base_predictions'][draw, j])
        gain, common, positive = route_inputs(info, data, score, expert, sy, ev)
        cap = int(.05*ev.sum())
        main_sel = ranked_mask(gain, positive, data['route_ids'], cap)
        np.testing.assert_array_equal(main_sel, old['selected'][draw, j])
        np.testing.assert_array_equal(np.where(main_sel, e, b), old['predictions'][draw, j])
        out = dict(source_prior=data['prior'].argmax(1), previous_full=old['predictions'][draw, 0],
                   ours_base=b, ours_5pct=old['predictions'][draw, j], expert_all=e)
        for name, kernel in kernels(data, embedding, s, info['eta']).items():
            out[name] = ridge_scores(kernel, s, sy, float(selection[name])).argmax(1)
        out['prototype64'] = prototypes(data['mean'], s, sy, selection['prototype64'])
        out['logistic_mean64'] = logistic(data['mean'], s, sy, float(selection['logistic_mean64']))
        def softmax(v):
            p = np.exp(v-v.max(1, keepdims=True))
            return p/p.sum(1, keepdims=True)
        out['softmax_average_all'] = (softmax(score)+softmax(expert)).argmax(1)
        masks = dict(ours_5pct=main_sel, expert_all=ev, previous_full=old['selected'][draw, 0])
        for budget in BUDGETS:
            mode = f'budget_{int(budget*100):03d}'
            masks[mode] = ranked_mask(gain, positive, data['route_ids'], int(budget*ev.sum()))
        bm, entropy = confidence_parts(score)
        random = np.random.default_rng(20261010+draw).random(n)
        for name, utility in (('gate', gain), ('random', random), ('margin', -bm), ('entropy', entropy)):
            masks[name+'_common05'] = ranked_mask(utility, common, data['route_ids'], cap)
        for name, utility in (('random', random), ('margin', -bm)):
            masks[name+'_positive_matched'] = ranked_mask(utility, positive, data['route_ids'], int(main_sel.sum()))
        for mode in ROUTERS:
            out[mode] = np.where(masks[mode], e, b)
        for mi, mode in enumerate(MODES):
            predictions[draw, mi] = out[mode]
            if mode in masks:
                selected[draw, mi] = masks[mode]
            if selected[draw, mi, ~ev].any():
                raise ValueError('Handoff outside query set')
        candidates = np.flatnonzero(positive)
        ordered = candidates[np.lexsort((data['route_ids'][candidates], -gain[candidates]))]
        rank = np.zeros(n, np.int32)
        rank[ordered] = np.arange(1, len(ordered)+1)
        gains.append(gain)
        commons.append(common)
        positives.append(positive)
        ranks.append(rank)
        chosen = ordered[:cap]
        traces.append(dict(draw=draw, cap=cap, common=int(common.sum()), positive=int(positive.sum()),
            admitted=len(chosen), cutoff_score=float(gain[chosen[-1]]) if len(chosen) else None,
            tie_break='route_ids ascending', replay_exact=True))
        print(f'TARGET {suffix} draw{draw}: {len(MODES)} modes frozen', flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest/'policies.npz', modes=np.array(MODES), predictions=predictions, selected=selected,
        rows=data['rows'], ids=data['ids'], support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(dest/'routing_trace.npz', gate_score=np.array(gains), common=np.array(commons),
        positive=np.array(positives), rank=np.array(ranks), route_ids=data['route_ids'])
    write(dest/'ROUTING.json', traces)
    write(dest/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5, modes=MODES,
        target_query_labels_read=False, original_exact=True, seconds=time.perf_counter()-start))


def report():
    tasks = folders()
    if not all((OUT/'targets'/json.loads((f/'PREPARED.json').read_text())['dataset']/
                json.loads((f/'PREPARED.json').read_text())['task']/'COMPLETE.json').exists() for f in tasks):
        raise RuntimeError('Freeze all predictions before query-label readout')
    records, classes = [], []
    for folder in tasks:
        info = json.loads((folder/'PREPARED.json').read_text())
        data = load_npz(folder/'target.npz')
        p = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        manifest, idcol = ((SOURCE/'manifest.csv', 'instance_id') if info['dataset'] == 'self24ports'
            else (ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', 'sample_id'))
        frame = pd.read_csv(manifest, usecols=[idcol, 'class_id'], keep_default_na=False).iloc[p['rows']]
        np.testing.assert_array_equal(frame[idcol].to_numpy(str), p['ids'])
        truth = frame.class_id.to_numpy()
        np.testing.assert_array_equal(truth[p['support']], data['support_labels'])
        for draw, ev in enumerate(p['eval_masks']):
            y = truth[ev]
            ref = p['predictions'][draw, MODES.index('ours_5pct'), ev]
            base = p['predictions'][draw, MODES.index('ours_base'), ev]
            for mi, mode in enumerate(MODES):
                pred = p['predictions'][draw, mi, ev]
                records.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                    oa=100*np.mean(pred == y), ba=ba(y, pred),
                    macro_f1_8=100*f1_score(y, pred, labels=np.arange(8), average='macro', zero_division=0),
                    delta_oa_vs_ours=100*(np.mean(pred == y)-np.mean(ref == y)),
                    delta_ba_vs_ours=ba(y, pred)-ba(y, ref),
                    handoff_rate=(np.nan if mode == 'softmax_average_all' else 100*p['selected'][draw, mi, ev].mean()),
                    expert_decision_use_rate=(100. if mode == 'softmax_average_all' else 100*p['selected'][draw, mi, ev].mean()),
                    rescues=int(np.sum((base != y) & (pred == y))),
                    harms=int(np.sum((base == y) & (pred != y))), query_n=len(y)))
                for c in np.unique(y):
                    mask = y == c
                    classes.append(dict(dataset=info['dataset'], task=info['task'], draw=draw, mode=mode,
                        class_id=int(c), recall=100*np.mean(pred[mask] == c), count=int(mask.sum()),
                        support_count=int(np.sum(data['support_labels'][draw] == c))))
    m = pd.DataFrame(records)
    m.to_csv(OUT/'metrics.csv', index=False)
    pd.DataFrame(classes).to_csv(OUT/'class_metrics.csv', index=False)
    fields = ['oa', 'ba', 'macro_f1_8', 'delta_oa_vs_ours', 'delta_ba_vs_ours', 'handoff_rate',
              'expert_decision_use_rate', 'rescues', 'harms']
    task = m.groupby(['dataset', 'task', 'mode'])[fields].mean().reset_index()
    task.to_csv(OUT/'task_means.csv', index=False)
    summary = task.groupby(['dataset', 'mode'])[fields].mean().reset_index()
    summary.to_csv(OUT/'summary.csv', index=False)
    m.groupby(['dataset', 'draw', 'mode'])[fields].mean().to_csv(OUT/'draw_metrics.csv')
    original = pd.read_csv(EXP/'summary.csv').query("stage == 'full' and mode == 'selective'")
    for row in original.itertuples():
        actual = summary[(summary.dataset == row.dataset) & (summary['mode'] == 'ours_5pct')].iloc[0]
        np.testing.assert_allclose([actual.oa, actual.ba], [row.oa, row.ba], atol=1e-10, rtol=0)
    write(OUT/'COMPLETE.json', dict(tasks=25, draws=125, rows=len(m), modes=len(MODES), original_exact=True,
        prediction_freeze_before_readout=True, public_method_reproduction_complete=False,
        scope='Matched lightweight baselines, original-model budget sweep and routing controls'))
    print(summary[['dataset', 'mode', 'oa', 'ba', 'handoff_rate']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(date='2026-10-10', primary=str(EXP), primary_mode='selective',
            datasets=['self24ports', 'f0921', 'oss2'], tasks=25, draws=5, labels=64,
            modes=MODES, ridge_grid=RIDGES, logistic_C_grid=CS, prototype_metrics=['cosine','euclidean'],
            selection='Mean source episode BA; fixed order ties', no_target_query_selection=True,
            source_episode_limit='Existing held query caches; source representations not independently held',
            feature_geometry='mean: normalized C4 PCA128; phi: original dataset-specific geometry; full_kernel: our learned representation',
            missing_support_classes='Scratch baselines predict observed support classes only; source-anchored model retains eight outputs',
            source_prior_target_labels=0, random_seed_base=20261010,
            common_eligibility='query, disagreement, expert class observed among support; no gain-sign filter',
            positive_matched='Same positive-gain eligibility and same realized count as primary',
            budget_grid=BUDGETS, full_expert_separate=True, neural_training=False,
            random_replicates_per_support_draw=1, independent_confirmation=False,
            cost='Cached-model replay walltime is not end-to-end inference cost'))
    tasks = [f for f in folders() if not args.tasks or f.name in args.tasks]
    if args.stage in ('source', 'all'):
        for folder in tasks:
            source_select(folder)
    if args.stage in ('target', 'all'):
        for folder in tasks:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        report()


if __name__ == '__main__':
    main()
