"""Source-held-port ranking selection and frozen visual candidate decisions."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import pickle
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_reliability_core import kernels, coefficients
from frequency_reliability_run import router, statistics
from frequency_decision_run import report
from frequency_candidate_core import class_evidence, candidate_difference, fit_ranker, rank_margin, candidate_scores

OUT = ROOT/'frequency_candidate_20261009'
CANDIDATES = ('pooled_band', 'local_band')
WIDTHS = (4, 12)
MODES = ('baseline', 'learned', *CANDIDATES, 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def source_fit(folder, output=OUT, feature_indices=None):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = output/'source'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'SELECTION.json').exists():
        return
    d = load_npz(OUT/'source'/info['dataset']/info['task']/'examples.npz')
    indices = feature_indices or {mode: list(range(width)) for mode, width in zip(CANDIDATES, WIDTHS)}
    y, ep = d['labels'], d['episode']
    old, new = d['old'], d['new']
    a, b = old.argmax(1), new.argmax(1)
    utility = (b == y).astype(float)-(a == y)
    oof, records = np.zeros((len(y), 2)), []
    for held in np.unique(ep):
        train, valid = (ep != held) & d['train_valid'], ep == held
        _, baseline_ba, _ = statistics(y[valid], b[valid])
        for j, mode in enumerate(CANDIDATES):
            cols = indices[mode]
            model = fit_ranker(d['pairs'][train][:, cols], d['weights'][train])
            margin = rank_margin(model, d['candidate'][valid][:, cols])
            oof[valid, j] = margin
            score = candidate_scores(old[valid], new[valid], margin, d['eligible'][valid])
            _, ba, _ = statistics(y[valid], score.argmax(1))
            pair_valid = valid & d['train_valid']
            pair_margin = rank_margin(model, d['pairs'][pair_valid][:, cols])
            pair_accuracy = float(np.average(pair_margin > 0, weights=d['weights'][pair_valid])) if pair_valid.any() else None
            records.append(dict(episode=int(held), mode=mode, n=int(valid.sum()),
                train_pairs=int(train.sum()), eligible=int(d['eligible'][valid].sum()),
                ba=ba, learned_ba=baseline_ba, delta_ba=ba-baseline_ba,
                held_pair_accuracy=pair_accuracy,
                rejected=int(np.sum(score.argmax(1) != b[valid]))))
    validation = pd.DataFrame(records)
    gains = validation.groupby('mode').delta_ba.mean().to_dict()
    best = max(('learned', *CANDIDATES), key=lambda m: 0. if m == 'learned' else gains[m])
    models = {mode: fit_ranker(d['pairs'][d['train_valid']][:, indices[mode]], d['weights'][d['train_valid']])
        for mode in CANDIDATES}
    with (out/'models.pkl').open('wb') as handle:
        pickle.dump(models, handle)
    np.savez_compressed(out/'oof.npz', benefit=oof, utility=utility, eligible=d['eligible'],
        episode=ep, labels=y, old_predictions=a, new_predictions=b)
    validation.to_csv(out/'validation.csv', index=False)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=best,
        source_delta_ba=gains, source_queries=len(y), source_conflicts=int(np.sum(a != b)),
        source_rescues=int(np.sum(utility > 0)), source_harms=int(np.sum(utility < 0)),
        source_ports=len(np.unique(ep)), training_pairs=int(d['train_valid'].sum()),
        eligible_conflicts=int(d['eligible'].sum()), no_target_labels=True,
        feature_indices=indices,
        fit_pair_accuracy={mode: float(np.average(rank_margin(models[mode], d['pairs'][d['train_valid']][:, indices[mode]]) > 0,
            weights=d['weights'][d['train_valid']])) for mode in CANDIDATES},
        models={m: None if model is None else {k: v.tolist() for k, v in model.items()} for m, model in models.items()}))
    print(f'VISUAL FIT {info["task"]}: {best}, BA gains {gains}', flush=True)


def target_predict(folder, output=OUT):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = output/'targets'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    src = output/'source'/info['dataset']/info['task']
    selection = json.loads((src/'SELECTION.json').read_text())
    indices = selection.get('feature_indices', {mode: list(range(width)) for mode, width in zip(CANDIDATES, WIDTHS)})
    with (src/'models.pkl').open('rb') as handle:
        models = pickle.load(handle)
    data, emb = load_npz(folder/'target.npz'), load_npz(folder/'selective_embedding.npz')
    np.testing.assert_array_equal(data['rows'], emb['rows'])
    embedding = emb['embedding'].astype(float)
    previous = load_npz(ROOT/'frequency_decision_20261009/targets'/info['dataset']/info['task']/'policies.npz')
    original = load_npz(folder/'policies.npz')
    arm = list(original['modes']).index('selective')
    routing = router(info, data)
    shape = (5, len(MODES), len(data['rows']))
    base, full, frozen = [np.empty(shape, np.int8) for _ in range(3)]
    selected = np.zeros(shape, bool)
    benefits = np.empty((5, 2, len(data['rows'])), np.float32)
    eligible_all = np.zeros((5, len(data['rows'])), bool)
    pair_archive = {}
    cached_path = OUT/'targets'/info['dataset']/info['task']/'contrasts.npz'
    cached = load_npz(cached_path) if cached_path.exists() else None
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        np.testing.assert_array_equal(support, previous['support'][draw])
        if ev[support].any() or len(np.unique(support)) != 64:
            raise ValueError('Support split or budget mismatch')
        k0, k1 = kernels(data, embedding, support, info['eta'])
        s0 = data['prior']+k0@coefficients(k0[support], sy, data['prior'][support])
        s1 = data['prior']+k1@coefficients(k1[support], sy, data['prior'][support])
        conflict_rows = np.flatnonzero(ev & (s0.argmax(1) != s1.argmax(1)))
        if cached is None:
            evidence, counts = class_evidence(emb['embedding'][conflict_rows], emb['embedding'][support],
                data['conditions'][conflict_rows], data['conditions'][support], sy)
            difference, valid = candidate_difference(evidence, counts,
                s0[conflict_rows].argmax(1), s1[conflict_rows].argmax(1))
        else:
            np.testing.assert_array_equal(conflict_rows, cached[f'rows_{draw}'])
            difference, counts = cached[f'difference_{draw}'], cached[f'counts_{draw}']
            valid = (counts[s0[conflict_rows].argmax(1)] >= 2) & (counts[s1[conflict_rows].argmax(1)] >= 2)
        eligible = np.zeros(len(s0), bool)
        eligible[conflict_rows] = valid
        eligible_all[draw] = eligible
        pair_archive[f'rows_{draw}'] = conflict_rows
        pair_archive[f'difference_{draw}'] = difference
        pair_archive[f'counts_{draw}'] = counts
        scores = dict(baseline=s0, learned=s1)
        for j, mode in enumerate(CANDIDATES):
            margin = np.zeros(len(s0))
            margin[conflict_rows] = rank_margin(models[mode], difference[:, indices[mode]])
            benefits[draw, j] = margin
            scores[mode] = candidate_scores(s0, s1, margin, eligible)
        scores['source_selected'] = scores[selection['rule']]
        for j, mode in enumerate(MODES):
            bp = scores[mode].argmax(1)
            pred, sel = routing(scores[mode], data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Expert handoff cap exceeded')
            if mode in ('baseline', 'learned'):
                ref = list(previous['modes']).index(mode)
                np.testing.assert_array_equal(bp, previous['base_predictions'][draw, ref])
                np.testing.assert_array_equal(pred, previous['predictions'][draw, ref])
                np.testing.assert_array_equal(sel, previous['selected'][draw, ref])
            base[draw, j], full[draw, j], selected[draw, j] = bp, pred, sel
            frozen[draw, j] = np.where(original['selected'][draw, arm], data['expert'][draw].argmax(1), bp)
        print(f'VISUAL TARGET {info["task"]} draw{draw}: {valid.sum()}/{len(valid)} eligible conflicts', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, benefit=benefits,
        eligible=eligible_all, rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(out/'contrasts.npz', **pair_archive)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], rule=selection['rule'],
        draws=5, labels=64, handoff_fraction=.05, target_eval_labels_read=False,
        rank_margin_not_probability=True, original_paths_replayed=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target', 'report', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_fit(folder)
    if args.stage in ('target', 'all'):
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        report(folders, output=OUT, candidates=CANDIDATES)
