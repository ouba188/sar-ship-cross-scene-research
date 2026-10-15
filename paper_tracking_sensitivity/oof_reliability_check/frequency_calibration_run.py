"""Final bounded calibration trial. No further candidates after this run."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import numpy as np
import pandas as pd
from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_margin_run import PREV as HEADS, OUT as MARGIN, write
from frequency_pair_run import OUT as PAIR
from frequency_band_core import old_kernel, score_embedding, unit_bands
from frequency_specialist_policy import loo_scores, local_choice
from frequency_calibration_policy import fit_correctness, predict_correctness, fit_metadata
from frequency_reliability_run import router, statistics

OUT = ROOT/'frequency_closeout_20261010'
MODES = ('learned', 'standalone_local', 'preserved_hard', 'margin_selected', 'pair_local', 'calibrated', 'source_selected')


def score_heads(data, learned, preserved, support, sy, eta, cached=None):
    kernel, cond = old_kernel(data, support, eta)
    embeddings = [learned, *unit_bands(preserved).transpose(1, 0, 2)]
    scores, loo = [], []
    for z in embeddings:
        s, sim = score_embedding(data, z, support, sy, kernel, cond)
        scores.append(s)
        if cached is None:
            loo.append(loo_scores(kernel[support]+.25*sim[support], data['prior'][support], sy))
    return np.stack(scores, axis=1), np.stack(loo, axis=1) if cached is None else cached


def choose(scores, loo, sy):
    fitted = fit_correctness(loo, sy)
    probability = predict_correctness(fitted, scores)
    gain = probability[:, 1:]-probability[:, :1]
    selected_scores, choices = local_choice(scores[:, 0], scores[:, 1:], gain)
    return selected_scores, choices, probability, fitted


def source_validate(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    out = OUT/'source'/suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out/'SELECTION.json').exists():
        return
    episodes = sorted((folder/'episodes').glob('*.npz'))
    records = []
    for path in sorted((HEADS/'source'/suffix).glob('episode_*.npz')):
        ei = int(path.stem.split('_')[-1])
        z, ep = load_npz(path), load_npz(episodes[ei])
        ix = z['indices']
        data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
        sy, y = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        scores, loo = score_heads(data, z['learned'].astype(float), z['preserved'].astype(float),
                                 np.arange(64), sy, info['eta'])
        np.testing.assert_allclose(scores[64:, 0], z['expected'], atol=1e-10, rtol=1e-10)
        selected, choice, probability, fitted = choose(scores, loo, sy)
        ref_ba = statistics(y, scores[64:, 0].argmax(1))[1]
        for mode, s in (('learned', scores[:, 0]), ('calibrated', selected)):
            oa, ba, _ = statistics(y, s[64:].argmax(1))
            records.append(dict(episode=ei, mode=mode, oa=oa, ba=ba, delta_ba=ba-ref_ba))
    frame = pd.DataFrame(records)
    gains = frame.groupby('mode').delta_ba.mean().to_dict()
    rule = 'calibrated' if gains['calibrated'] > 0 else 'learned'
    frame.to_csv(out/'validation.csv', index=False)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=rule,
        source_delta_ba=gains, criterion='Mean source episode base BA; one new arm vs retained original main',
        target_query_labels_used=False, source_episodes=len(frame)//2))
    print(f'SOURCE {suffix} {rule} gain={gains["calibrated"]:.6f}', flush=True)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = f"{info['dataset']}/{info['task']}"
    out = OUT/'targets'/suffix
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    rule = json.loads((OUT/'source'/suffix/'SELECTION.json').read_text())['rule']
    data = load_npz(folder/'target.npz')
    old, margin, pair = [load_npz(root/'targets'/suffix/'policies.npz') for root in (HEADS, MARGIN, PAIR)]
    cached = load_npz(HEADS/'targets'/suffix/'support_loo.npz')
    learned = load_npz(folder/'selective_embedding.npz')
    preserved = load_npz(HEADS/'tasks'/suffix/'preserved_embedding.npz')
    for p in (old, margin, pair, learned, preserved):
        np.testing.assert_array_equal(p['rows'], data['rows'])
    shape = (5, len(MODES), len(data['rows']))
    base, full, frozen, choices = [np.empty(shape, np.int8) for _ in range(4)]
    selected = np.empty(shape, bool)
    probabilities = np.empty((5, len(data['rows']), 5), np.float32)
    candidates = np.empty((5, len(data['rows']), 4), np.int8)
    routing, fits = router(info, data), []
    controls = dict(learned=(margin, 'learned'), standalone_local=(margin, 'standalone_local'),
        preserved_hard=(margin, 'hard'), margin_selected=(margin, 'source_selected'), pair_local=(pair, 'pair_local'))
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('64-label split mismatch')
        scores, loo = score_heads(data, learned['embedding'].astype(float), preserved['embedding'].astype(float),
                                 support, sy, info['eta'], cached[f'preserved_{draw}'])
        s, choice, probability, fitted = choose(scores, loo, sy)
        probabilities[draw], candidates[draw] = probability, scores[:, 1:].argmax(2)
        np.testing.assert_array_equal(candidates[draw], pair['candidates'][draw])
        learned_mask = margin['selected'][draw, list(margin['modes']).index('learned')]
        for j, mode in enumerate(MODES[:-1]):
            if mode in controls:
                previous, old_mode = controls[mode]
                oi = list(previous['modes']).index(old_mode)
                base[draw, j], full[draw, j], selected[draw, j], choices[draw, j] = (
                    previous[k][draw, oi] for k in ('base_predictions', 'predictions', 'selected', 'choices'))
            else:
                pred, sel = routing(s, data['expert'][draw], sy, ev)
                if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                    raise ValueError('Expert handoff budget exceeded')
                base[draw, j], full[draw, j], selected[draw, j], choices[draw, j] = s.argmax(1), pred, sel, choice
            frozen[draw, j] = np.where(learned_mask, data['expert'][draw].argmax(1), base[draw, j])
        ref_i = MODES.index('learned')
        np.testing.assert_array_equal(scores[:, 0].argmax(1), base[draw, ref_i])
        for array in (base, full, frozen, choices, selected):
            array[draw, -1] = array[draw, MODES.index(rule)]
        fits.append(dict(draw=draw, **fit_metadata(fitted)))
        print(f'TARGET {suffix} draw{draw}: frozen; rule={rule}', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, choices=choices,
        probabilities=probabilities, candidates=candidates, rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    write(out/'CALIBRATORS.json', fits)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5, rule=rule,
        target_query_labels_read=False, target_support_labels=64, neural_training=False,
        calibrator_parameters=7, handoff_fraction=.05))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'PROTOCOL.json').exists():
        write(OUT/'PROTOCOL.json', dict(tasks=25, support_draws=125, target_labels=64,
            final_round=True, further_candidate_search=False,
            method='Pooled target-LOO correctness logistic regression; two common margin slopes, four shrunk head offsets, intercept',
            parameters=7, C=1, total_fit_weight=64, head_records_per_label=5,
            gain='weighted correctness probability band minus main; switch only if positive and predicted class differs',
            formal_calibration_guarantee=False, probability_target='class-balanced support distribution',
            threshold_search=False, source_options=['learned', 'calibrated'], handoff=.05,
            limitation='Historical development data; labels reused for adaptation and LOO meta-fit, not an independent calibration set'))
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_validate(folder)
    if args.stage in ('target', 'all'):
        if len(list((OUT/'source').glob('*/*/SELECTION.json'))) != 25:
            raise ValueError('Freeze all source choices first')
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        from frequency_calibration_report import report
        report()


if __name__ == '__main__':
    main()
