"""Matched conditional utility study with frozen thresholds and frequency heads."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_margin_run import OUT as PREV, PREV as HEADS, THRESHOLDS, write
from frequency_band_core import old_kernel, score_embedding, unit_bands
from frequency_specialist_policy import loo_scores, local_choice
from frequency_pair_policy import conditional_utilities
from frequency_reliability_run import router, statistics

OUT = ROOT / 'frequency_pair_selection_20261010'
MODES = ('learned', 'previous_selected', 'smooth_parent', 'class_local', 'pair_local', 'source_selected')
OPTIONS = ('previous_selected', 'class_local', 'pair_local')


def candidates(data, learned, preserved, support, sy, eta, cached_loo=None):
    kernel, condition = old_kernel(data, support, eta)
    reference, sim = score_embedding(data, learned, support, sy, kernel, condition)
    loo_ref = (cached_loo[:, 0] if cached_loo is not None else
               loo_scores(kernel[support] + .25*sim[support], data['prior'][support], sy))
    scores, unit = [], unit_bands(preserved)
    utilities = np.empty((3, len(reference), 4))
    coverage = np.empty((len(reference), 4))
    for bi in range(4):
        score, sim = score_embedding(data, unit[:, bi], support, sy, kernel, condition)
        scores.append(score)
        loo_band = (cached_loo[:, bi+1] if cached_loo is not None else
                    loo_scores(kernel[support] + .25*sim[support], data['prior'][support], sy))
        a, b = reference.argmax(1), score.argmax(1)
        for lo in range(0, len(reference), 8192):
            hi = lo+8192
            u, rho = conditional_utilities(sim[lo:hi], loo_ref, loo_band, sy, a[lo:hi], b[lo:hi])
            utilities[:, lo:hi, bi], coverage[lo:hi, bi] = u.T, rho
    return reference, np.stack(scores, axis=1), utilities, coverage


def readout(reference, bands, utilities, threshold):
    return {mode: local_choice(reference, bands, utilities[i] - threshold)
            for i, mode in enumerate(('smooth_parent', 'class_local', 'pair_local'))}


def source_validate(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    out = OUT/'source'/suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out/'SELECTION.json').exists():
        return
    previous = json.loads((PREV/'source'/suffix/'SELECTION.json').read_text())
    old_validation = pd.read_csv(PREV/'source'/suffix/'validation.csv')
    parent = max(('smooth', 'smooth02', 'smooth05'), key=lambda m: previous['source_delta_ba'][m])
    threshold = THRESHOLDS[parent]
    episodes = sorted((folder/'episodes').glob('*.npz'))
    records = []
    for path in sorted((HEADS/'source'/suffix).glob('episode_*.npz')):
        ei = int(path.stem.split('_')[-1])
        z, ep = load_npz(path), load_npz(episodes[ei])
        ix = z['indices']
        data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
        sy, y = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        ref, bands, utility, coverage = candidates(data, z['learned'].astype(float),
            z['preserved'].astype(float), np.arange(64), sy, info['eta'])
        np.testing.assert_allclose(ref[64:], z['expected'], atol=1e-10, rtol=1e-10)
        rba = statistics(y, ref[64:].argmax(1))[1]
        rows = old_validation[old_validation.episode == ei].set_index('mode')
        old = rows.loc[previous['rule']]
        records.append(dict(episode=ei, mode='previous_selected', oa=old.oa, ba=old.ba,
                            delta_ba=old.delta_ba, switches=int(old.switches)))
        for mode, (score, choice) in readout(ref, bands, utility, threshold).items():
            oa, ba, _ = statistics(y, score[64:].argmax(1))
            if mode == 'smooth_parent':
                np.testing.assert_allclose([oa, ba], rows.loc[parent, ['oa', 'ba']].to_numpy(float), atol=1e-10, rtol=0)
            records.append(dict(episode=ei, mode=mode, oa=oa, ba=ba, delta_ba=ba-rba,
                                switches=int(np.sum(choice[64:] != 0))))
    frame = pd.DataFrame(records)
    gains = frame.groupby('mode').delta_ba.mean().to_dict()
    chosen = max(OPTIONS, key=lambda m: gains[m])
    frame.to_csv(out/'validation.csv', index=False)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=chosen,
        previous_rule=previous['rule'], parent_rule=parent, threshold=threshold, source_delta_ba=gains,
        target_query_labels_used=False, source_episodes=len(records)//4,
        threshold_rule='Best previous smooth threshold by already-frozen source episode base BA; no new threshold search',
        criterion='Mean source episode base BA among previous_selected/class_local/pair_local, ties keep previous',
        limitation='Development data; source validation rows held out of head fitting but source ports shared'))
    print(f'SOURCE {suffix}: parent={parent}, selected={chosen}, gains={gains}', flush=True)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    out = OUT/'targets'/suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    selection = json.loads((OUT/'source'/suffix/'SELECTION.json').read_text())
    data, previous = load_npz(folder/'target.npz'), load_npz(PREV/'targets'/suffix/'policies.npz')
    cached = load_npz(HEADS/'targets'/suffix/'support_loo.npz')
    learned = load_npz(folder/'selective_embedding.npz')
    preserved = load_npz(HEADS/'tasks'/suffix/'preserved_embedding.npz')
    for z in (learned, preserved, previous):
        np.testing.assert_array_equal(z['rows'], data['rows'])
    old_modes, n = list(previous['modes']), len(data['rows'])
    oldi, parenti = old_modes.index('source_selected'), old_modes.index(selection['parent_rule'])
    shape = (5, len(MODES), n)
    base, full, frozen, choices = [np.empty(shape, np.int8) for _ in range(4)]
    selected = np.empty(shape, bool)
    utilities = np.empty((5, 3, n, 4), np.float32)
    coverage_all = np.empty((5, n, 4), np.float32)
    band_predictions = np.empty((5, n, 4), np.int8)
    routing = router(info, data)
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support budget or split changed')
        ref, bands, utility, coverage = candidates(data, learned['embedding'].astype(float),
            preserved['embedding'].astype(float), support, sy, info['eta'], cached[f'preserved_{draw}'])
        np.testing.assert_array_equal(utility[0].astype(np.float32), previous['utilities'][draw, 1])
        np.testing.assert_array_equal(bands.argmax(2), previous['candidates'][draw])
        rules = readout(ref, bands, utility, selection['threshold'])
        utilities[draw], coverage_all[draw], band_predictions[draw] = utility, coverage, bands.argmax(2)
        for j, mode in enumerate(MODES[:-1]):
            if mode in ('learned', 'previous_selected'):
                oi = old_modes.index('learned') if mode == 'learned' else oldi
                base[draw, j], full[draw, j] = previous['base_predictions'][draw, oi], previous['predictions'][draw, oi]
                selected[draw, j], choices[draw, j] = previous['selected'][draw, oi], previous['choices'][draw, oi]
            else:
                score, choice = rules[mode]
                bp, (pred, sel) = score.argmax(1), routing(score, data['expert'][draw], sy, ev)
                if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                    raise ValueError('Expert budget exceeded')
                base[draw, j], full[draw, j], selected[draw, j], choices[draw, j] = bp, pred, sel, choice
                if mode == 'smooth_parent':
                    for actual, key in ((bp, 'base_predictions'), (pred, 'predictions'), (sel, 'selected'), (choice, 'choices')):
                        np.testing.assert_array_equal(actual, previous[key][draw, parenti])
            frozen[draw, j] = np.where(previous['selected'][draw, oldi], data['expert'][draw].argmax(1), base[draw, j])
        for array in (base, full, frozen, selected, choices):
            array[draw, -1] = array[draw, MODES.index(selection['rule'])]
        print(f'TARGET {suffix} draw{draw}: generic/parent exact; rule={selection["rule"]}', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, choices=choices,
        utilities=utilities, coverage=coverage_all, candidates=band_predictions,
        rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        source_rule=selection['rule'], parent_rule=selection['parent_rule'],
        labels=64, handoff_fraction=.05, target_eval_labels_read=False, parent_replay_exact=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(tasks=25, support_draws=125, support=64, handoff=.05,
            temperature=.1, neighbors=8, new_training=False, new_threshold_search=False,
            parent='Best previous smooth threshold selected only from previous source validation',
            arms=['class_local: same generic margin, pair-label relevance',
                  'pair_local: same relevance, true-v-other-query-class pair margin'],
            shrinkage='rho=min(effective_n_a,effective_n_b)/(min(effective_n_a,effective_n_b)+2)',
            fallback='Either predicted class absent locally -> exact generic utility; singleton rho<=1/3',
            selection=list(OPTIONS), target_labels_used_for_selection=False,
            data_status='Historical development corpora; support draws not independent training seeds'))
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_validate(folder)
    if args.stage in ('target', 'all'):
        if len(list((OUT/'source').glob('*/*/SELECTION.json'))) != 25:
            raise ValueError('Freeze all source rules first')
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        from frequency_pair_report import report
        report()


if __name__ == '__main__':
    main()
