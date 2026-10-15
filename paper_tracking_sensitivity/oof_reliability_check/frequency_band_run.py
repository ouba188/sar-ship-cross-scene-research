"""Source-held-port band selection; freeze every target before oracle readout."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_reliability_run import router, statistics
from frequency_band_core import (BANDS, CANDIDATES, unit_bands, old_kernel, score_embedding,
    choice_features, choose_scores, fit_benefit, predict_benefit)

OUT = ROOT/'frequency_band_complementarity_20261009'
DYNAMIC = ('dynamic_scores', 'dynamic_support')
MODES = (*CANDIDATES, *(f'selective_{b}' for b in BANDS), 'source_static', *DYNAMIC, 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def fit_choice(d, keep, width):
    predictions = d['scores'].argmax(2)
    reference = predictions[:, 0]
    utility = (predictions[:, 2:] == d['labels'][:, None]).astype(float) - (reference == d['labels'])[:, None]
    conflict = predictions[:, 2:] != reference[:, None]
    use = keep[:, None] & conflict
    weight = np.broadcast_to(d['weights'][:, None], utility.shape)
    return fit_benefit(d['features'][:, :, :width][use], utility[use], weight[use])


def predict_choice(model, scores, features, width):
    conflict = scores[:, 2:].argmax(2) != scores[:, 0].argmax(1)[:, None]
    value = np.zeros(conflict.shape, dtype=float)
    if conflict.any():
        value[conflict] = predict_benefit(model, features[:, :, :width][conflict])
    chosen, index = choose_scores(scores, value)
    return chosen, index, value


def source_fit(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'source'/info['dataset']/info['task']
    if (out/'SELECTION.json').exists():
        return
    prep = json.loads((out/'PREPARED.json').read_text())
    d = load_npz(out/'examples.npz')
    widths = dict(dynamic_scores=prep['score_width'], dynamic_support=prep['feature_width'])
    y, ep = d['labels'], d['episode']
    records, oof = [], np.empty((len(y), 2), dtype=np.int8)
    started = time.perf_counter()
    for held in np.unique(ep):
        valid, train = ep == held, ep != held
        ba = statistics(y[valid], d['scores'][valid, 0].argmax(1))[1]
        for j, mode in enumerate(CANDIDATES):
            acc = statistics(y[valid], d['scores'][valid, j].argmax(1))[1]
            records.append(dict(episode=int(held), mode=mode, ba=acc, delta_ba=acc-ba))
        for j, mode in enumerate(DYNAMIC):
            model = fit_choice(d, train, widths[mode])
            score, _, _ = predict_choice(model, d['scores'][valid], d['features'][valid], widths[mode])
            oof[valid, j] = score.argmax(1)
            acc = statistics(y[valid], score.argmax(1))[1]
            records.append(dict(episode=int(held), mode=mode, ba=acc, delta_ba=acc-ba))
    validation = pd.DataFrame(records)
    gains = validation.groupby('mode').delta_ba.mean().to_dict()
    static = max(CANDIDATES, key=lambda m: gains[m])
    rule = max((*CANDIDATES, *DYNAMIC), key=lambda m: gains[m])
    models = {m: fit_choice(d, np.ones(len(y), bool), widths[m]) for m in DYNAMIC}
    with (out/'models.pkl').open('wb') as handle:
        pickle.dump(models, handle)
    validation.to_csv(out/'validation.csv', index=False)
    np.savez_compressed(out/'oof.npz', predictions=oof, labels=y, episode=ep)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=rule,
        static_rule=static, source_delta_ba=gains, source_queries=len(y), widths=widths,
        source_ports=len(np.unique(ep)), seconds=time.perf_counter()-started,
        label_budget=64, target_query_labels_used=False,
        limitation='Selector leaves source episode port out; representation training shares source ports',
        architecture='Source HGB hard band-score selector, not retrained neural attention'))
    print(f'BAND FIT {info["task"]}: {rule}, static={static}, gains={gains}, {time.perf_counter()-started:.1f}s', flush=True)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'targets'/info['dataset']/info['task']
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    src = OUT/'source'/info['dataset']/info['task']
    selection = json.loads((src/'SELECTION.json').read_text())
    with (src/'models.pkl').open('rb') as handle:
        models = pickle.load(handle)
    data = load_npz(folder/'target.npz')
    embeddings = {}
    for mode in ('fixed', 'selective'):
        e = load_npz(folder/f'{mode}_embedding.npz')
        np.testing.assert_array_equal(e['rows'], data['rows'])
        embeddings[mode] = e['embedding'].astype(float)
    fixed_bands, selective_bands = unit_bands(embeddings['fixed']), unit_bands(embeddings['selective'])
    original = load_npz(folder/'policies.npz')
    original_arm = list(original['modes']).index('selective')
    routing, n = router(info, data), len(data['rows'])
    shape = (5, len(MODES), n)
    base, full, frozen = [np.empty(shape, np.int8) for _ in range(3)]
    selected = np.empty(shape, bool)
    choices = np.empty((5, 2, n), np.int8)
    benefits = np.empty((5, 2, n, 4), np.float32)
    selection_rows = []
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if ev[support].any() or len(np.unique(support)) != 64:
            raise ValueError('Support split or budget mismatch')
        k0, condition = old_kernel(data, support, info['eta'])
        scores = np.empty((n, 6, 8), dtype=float)
        scores[:, 0] = score_embedding(data, embeddings['selective'], support, sy, k0, condition)[0]
        scores[:, 1] = score_embedding(data, embeddings['fixed'], support, sy, k0, condition)[0]
        features = np.empty((n, 4, selection['widths']['dynamic_support']), np.float32)
        for band in range(4):
            score, sim = score_embedding(data, fixed_bands[:, band], support, sy, k0, condition)
            scores[:, band+2] = score
            for lo in range(0, n, 8192):
                f, width = choice_features(scores[lo:lo+8192, 0], score[lo:lo+8192], sim[lo:lo+8192],
                    sy, data['conditions'][lo:lo+8192], band)
                features[lo:lo+8192, band] = f
            if width != selection['widths']['dynamic_scores']:
                raise ValueError('Feature layout mismatch')
        all_scores = {mode: scores[:, j] for j, mode in enumerate(CANDIDATES)}
        for band, name in enumerate(BANDS):
            all_scores[f'selective_{name}'] = score_embedding(data, selective_bands[:, band], support, sy, k0, condition)[0]
        for j, mode in enumerate(DYNAMIC):
            score, index, benefit = predict_choice(models[mode], scores, features, selection['widths'][mode])
            all_scores[mode], choices[draw, j], benefits[draw, j] = score, index, benefit
            for selected_index in (0, 2, 3, 4, 5):
                selection_rows.append(dict(dataset=info['dataset'], task=info['task'], draw=draw,
                    mode=mode, band=CANDIDATES[selected_index], n=int(np.sum(ev & (index == selected_index)))))
        all_scores['source_static'] = all_scores[selection['static_rule']]
        all_scores['source_selected'] = all_scores[selection['rule']]
        for j, mode in enumerate(MODES):
            score = all_scores[mode]
            bp = score.argmax(1)
            pred, sel = routing(score, data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Expert handoff cap exceeded')
            if mode in ('learned', 'fixed_equal'):
                reference = list(original['modes']).index('selective' if mode == 'learned' else 'fixed')
                for actual, key in ((bp, 'base_predictions'), (pred, 'predictions'), (sel, 'selected')):
                    np.testing.assert_array_equal(actual, original[key][draw, reference])
            base[draw, j], full[draw, j], selected[draw, j] = bp, pred, sel
            frozen[draw, j] = np.where(original['selected'][draw, original_arm], data['expert'][draw].argmax(1), bp)
        print(f'BAND TARGET {info["task"]} draw{draw}: both reference paths replay exact', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected,
        choices=choices, benefit=benefits, rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    pd.DataFrame(selection_rows).to_csv(out/'choice_counts.csv', index=False)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        target_eval_labels_read=False, reference_paths_replayed=True, support_n=64, handoff_fraction=.05,
        unit_band_strength=.25, source_rule=selection['rule'], source_static=selection['static_rule']))


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
        from frequency_band_report import report
        report()
