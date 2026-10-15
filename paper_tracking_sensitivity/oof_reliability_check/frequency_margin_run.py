"""Frozen preserved bands: hard votes versus continuous LOO margin evidence."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_band_core import BANDS, old_kernel, score_embedding, unit_bands
from frequency_specialist_policy import loo_scores, local_choice
from frequency_margin_policy import support_utilities
from frequency_reliability_run import router, statistics

PREV = ROOT / 'frequency_preserve_20261010'
OUT = ROOT / 'frequency_margin_selection_20261010'
RULES = ('hard', 'smooth', 'smooth02', 'smooth05')
THRESHOLDS = {'hard': 0., 'smooth': 0., 'smooth02': .02, 'smooth05': .05}
MODES = ('learned', 'standalone_local', *RULES, 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def candidates(data, learned, preserved, support, sy, eta, cached_loo=None):
    kernel, condition = old_kernel(data, support, eta)
    reference, sim = score_embedding(data, learned, support, sy, kernel, condition)
    loo_ref = (cached_loo[:, 0] if cached_loo is not None else
               loo_scores(kernel[support] + .25 * sim[support], data['prior'][support], sy))
    scores, unit = [], unit_bands(preserved)
    utilities = np.empty((2, len(reference), 4))
    for bi in range(4):
        score, sim = score_embedding(data, unit[:, bi], support, sy, kernel, condition)
        scores.append(score)
        loo_band = (cached_loo[:, bi+1] if cached_loo is not None else
                    loo_scores(kernel[support] + .25 * sim[support], data['prior'][support], sy))
        for lo in range(0, len(reference), 8192):
            hi = lo + 8192
            utilities[:, lo:hi, bi] = support_utilities(sim[lo:hi], loo_ref, loo_band, sy).T
    return reference, np.stack(scores, axis=1), utilities


def readout(reference, bands, utilities):
    return {name: local_choice(reference, bands,
                utilities[0 if name == 'hard' else 1] - THRESHOLDS[name])
            for name in RULES}


def source_validate(folder):
    info = json.loads((folder / 'PREPARED.json').read_text())
    suffix = (info['dataset'], info['task'])
    out = OUT / 'source' / suffix[0] / suffix[1]
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'SELECTION.json').exists():
        return
    previous = PREV / 'source' / suffix[0] / suffix[1]
    episodes = sorted((folder / 'episodes').glob('*.npz'))
    records = []
    for path in sorted(previous.glob('episode_*.npz')):
        ei = int(path.stem.split('_')[-1])
        z, ep = load_npz(path), load_npz(episodes[ei])
        ix = z['indices']
        data = {k: ep[k][ix] for k in ('phi', 'prior', 'mean', 'conditions')}
        sy, y = ep['labels'][ix[:64]], ep['labels'][ix[64:]]
        ref, bands, utility = candidates(data, z['learned'].astype(float),
            z['preserved'].astype(float), np.arange(64), sy, info['eta'])
        np.testing.assert_allclose(ref[64:], z['expected'], atol=1e-10, rtol=1e-10)
        rules = readout(ref, bands, utility)
        old = load_npz(previous / f'query_scores_{ei:02d}.npz')
        oldi = list(old['modes']).index('preserved_support_local')
        np.testing.assert_allclose(rules['hard'][0][64:].astype(np.float32), old['scores'][:, oldi], atol=0, rtol=0)
        rba = statistics(y, ref[64:].argmax(1))[1]
        for name, (score, choice) in rules.items():
            oa, ba, _ = statistics(y, score[64:].argmax(1))
            records.append(dict(episode=ei, mode=name, oa=oa, ba=ba, delta_ba=ba-rba,
                                switches=int(np.sum(choice[64:] != 0))))
    frame = pd.DataFrame(records)
    gain = frame.groupby('mode').delta_ba.mean().to_dict()
    chosen = max(RULES, key=lambda k: gain[k])
    frame.to_csv(out / 'validation.csv', index=False)
    write(out / 'SELECTION.json', dict(dataset=suffix[0], task=suffix[1], rule=chosen, source_delta_ba=gain,
        target_query_labels_used=False, source_episodes=len(frame)//len(RULES),
        criterion='Mean frozen source episode base BA, hard first on ties',
        limitation='Source validation query rows excluded from neural fitting; source ports shared'))
    print(f'SOURCE {suffix}: {chosen} gains={gain}', flush=True)


def target_predict(folder):
    info = json.loads((folder / 'PREPARED.json').read_text())
    ds, task = info['dataset'], info['task']
    out = OUT / 'targets' / ds / task
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'COMPLETE.json').exists():
        return
    rule = json.loads((OUT / 'source' / ds / task / 'SELECTION.json').read_text())['rule']
    data = load_npz(folder / 'target.npz')
    learned = load_npz(folder / 'selective_embedding.npz')
    preserved = load_npz(PREV / 'tasks' / ds / task / 'preserved_embedding.npz')
    for z in (learned, preserved):
        np.testing.assert_array_equal(z['rows'], data['rows'])
    previous = load_npz(PREV / 'targets' / ds / task / 'policies.npz')
    cached = load_npz(PREV / 'targets' / ds / task / 'support_loo.npz')
    oldm = list(previous['modes'])
    hardi = oldm.index('preserved_support_local')
    n, routing = len(data['rows']), router(info, data)
    shape = (5, len(MODES), n)
    base, full, frozen, choices = [np.zeros(shape, np.int8) for _ in range(4)]
    selected = np.zeros(shape, bool)
    utilities = np.empty((5, 2, n, 4), np.float32)
    band_predictions = np.empty((5, n, 4), np.int8)
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('64-support query split changed')
        ref, bands, utility = candidates(data, learned['embedding'].astype(float),
            preserved['embedding'].astype(float), support, sy, info['eta'], cached[f'preserved_{draw}'])
        rules = readout(ref, bands, utility)
        utilities[draw], band_predictions[draw] = utility, bands.argmax(2)
        np.testing.assert_allclose(utility[0].astype(np.float32), previous['benefit'][draw, 3], atol=0, rtol=0)
        for bi, band in enumerate(BANDS):
            np.testing.assert_array_equal(band_predictions[draw, :, bi],
                previous['base_predictions'][draw, oldm.index('preserved_'+band)])
        for j, mode in enumerate(MODES[:-1]):
            if mode == 'standalone_local':
                oi = oldm.index('standalone_support_local')
                base[draw, j] = previous['base_predictions'][draw, oi]
                full[draw, j] = previous['predictions'][draw, oi]
                selected[draw, j] = previous['selected'][draw, oi]
                choices[draw, j] = previous['choices'][draw, 2]
            else:
                score, choice = (ref, np.zeros(n, int)) if mode == 'learned' else rules[mode]
                bp, (pred, sel) = score.argmax(1), routing(score, data['expert'][draw], sy, ev)
                if sel[~ev].any() or sel.sum() > int(.05 * ev.sum()):
                    raise ValueError('Expert budget exceeded')
                base[draw, j], full[draw, j], selected[draw, j], choices[draw, j] = bp, pred, sel, choice
                if mode in ('learned', 'hard'):
                    oi = oldm.index('learned') if mode == 'learned' else hardi
                    for actual, key in ((bp, 'base_predictions'), (pred, 'predictions'), (sel, 'selected')):
                        np.testing.assert_array_equal(actual, previous[key][draw, oi])
            frozen[draw, j] = np.where(previous['selected'][draw, hardi],
                data['expert'][draw].argmax(1), base[draw, j])
        rulei = MODES.index(rule)
        for array in (base, full, frozen, selected, choices):
            array[draw, -1] = array[draw, rulei]
        print(f'TARGET {ds}/{task} draw {draw}: old paths exact; rule={rule}', flush=True)
    np.savez_compressed(out / 'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, choices=choices,
        utilities=utilities, candidates=band_predictions, rows=data['rows'],
        support=data['support'], eval_masks=data['eval_masks'])
    write(out / 'COMPLETE.json', dict(dataset=ds, task=task, draws=5, source_rule=rule,
        labels=64, handoff_fraction=.05, frozen_head=True, baseline_replay_exact=True,
        target_eval_labels_read=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('source', 'target', 'report', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'PROTOCOL.json').exists():
        write(OUT / 'PROTOCOL.json', dict(temperature=.1, thresholds=THRESHOLDS,
            new_training=False, families=['frozen_preserved'], support=64, neighbors=8,
            utility='weighted band minus main sigmoid(true-minus-maxwrong margin / .1)',
            caveat='Smooth correctness surrogate, not a calibrated probability of rescue',
            source_selection='Mean source episode BA, four prespecified choices including hard',
            target_selection=False, data_status='Previously used development corpora',
            tasks=25, support_draws=125, five_draws_not_training_seeds=True))
    folders = [p.parent for p in sorted((EXP / 'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_validate(folder)
    if args.stage in ('target', 'all'):
        if len(list((OUT / 'source').glob('*/*/SELECTION.json'))) != 25:
            raise ValueError('Freeze all source selections before target predictions')
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        from frequency_margin_report import report
        report()


if __name__ == '__main__':
    main()
