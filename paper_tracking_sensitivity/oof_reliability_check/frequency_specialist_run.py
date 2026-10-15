"""Matched branch readout and support-only switching without target query truth."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import json
import numpy as np
import pandas as pd

from prepare_learned_frequency import ROOT, OUT as EXP, load_npz
from frequency_band_core import BANDS, old_kernel, score_embedding, unit_bands
from frequency_reliability_run import router, statistics
from frequency_specialist_policy import loo_scores, support_static, support_utility, local_choice

OUT = ROOT/'frequency_specialists_20261009'
FAMILIES = ('old', 'continued', 'independent', 'specialist')
MODES = ('learned', *(f'{f}_{b}' for f in FAMILIES for b in ('equal', *BANDS)),
         *(f'{f}_support_{s}' for f in FAMILIES for s in ('static', 'local')), 'source_selected')


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def family_predictions(data, z, reference, loo_ref, support, sy, kernel, condition):
    scores = {}
    scores['equal'] = score_embedding(data, z, support, sy, kernel, condition)[0]
    unit = unit_bands(z)
    loo = [loo_ref]
    utility = np.empty((len(reference), 4), dtype=float)
    for b, band in enumerate(BANDS):
        score, sim = score_embedding(data, unit[:, b], support, sy, kernel, condition)
        scores[band] = score
        l = loo_scores(kernel[support]+.25*sim[support], data['prior'][support], sy)
        loo.append(l)
        for lo in range(0, len(reference), 8192):
            utility[lo:lo+8192, b] = support_utility(sim[lo:lo+8192], loo_ref, l, sy)
    loo = np.stack(loo, axis=1)
    candidate = np.stack([scores[b] for b in BANDS], axis=1)
    static = support_static(loo, sy)
    scores['support_static'] = reference if static == 0 else scores[BANDS[static-1]]
    scores['support_local'], choice = local_choice(reference, candidate, utility)
    return scores, loo, static, choice, utility


def source_validate(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    out = OUT/'source'/info['dataset']/info['task']
    if (out/'SELECTION.json').exists():
        return
    if not (out/'ENCODED.json').exists():
        raise ValueError('Source embeddings incomplete')
    eps = sorted((folder/'episodes').glob('*.npz'))
    records, diagnostics = [], []
    for path in sorted(out.glob('episode_*.npz')):
        ei = int(path.stem.split('_')[-1])
        z = load_npz(path)
        ep = load_npz(eps[ei])
        ix = z['indices']
        data = {key: ep[key][ix] for key in ('phi', 'prior', 'mean', 'conditions')}
        sy, y, support = ep['labels'][ix[:64]], ep['labels'][ix[64:]], np.arange(64)
        k0, condition = old_kernel(data, support, info['eta'])
        ref, sim = score_embedding(data, z['learned'].astype(float), support, sy, k0, condition)
        np.testing.assert_allclose(ref[64:], z['expected'], atol=1e-10, rtol=1e-10)
        loo_ref = loo_scores(k0[support]+.25*sim[support], data['prior'][support], sy)
        all_scores = dict(learned=ref)
        for family in FAMILIES:
            scores, loo, static, choice, value = family_predictions(data, z[family].astype(float), ref, loo_ref, support, sy, k0, condition)
            all_scores.update({family+'_'+key: v for key, v in scores.items()})
            diagnostics.append(dict(episode=ei, family=family, static_choice=int(static),
                local_switches=int(np.sum(choice[64:] != 0)),
                loo_change_events=int(np.sum(loo[:, 1:].argmax(2) != loo_ref.argmax(1)[:, None]))))
        rba = statistics(y, ref[64:].argmax(1))[1]
        for mode, score in all_scores.items():
            oa, ba, _ = statistics(y, score[64:].argmax(1))
            records.append(dict(episode=ei, mode=mode, oa=oa, ba=ba, delta_ba=ba-rba))
        np.savez_compressed(out/f'query_scores_{ei:02d}.npz', modes=np.array(list(all_scores)),
            scores=np.stack([score[64:] for score in all_scores.values()], axis=1).astype(np.float32),
            labels=y, rows=ep['rows'][ix[64:]])
    frame = pd.DataFrame(records)
    gains = frame.groupby('mode').delta_ba.mean().to_dict()
    best = max(MODES[:-1], key=lambda m: gains[m])
    frame.to_csv(out/'validation.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
    write(out/'SELECTION.json', dict(dataset=info['dataset'], task=info['task'], rule=best,
        source_delta_ba=gains, source_episodes=len(records)//(len(MODES)-1),
        target_query_labels_used=False, source_query_plan='Original frozen queries unused by original or continuation neural fit',
        limitation='Representation training shares source ports; not independent representation-held evaluation'))
    print(f'SPECIALIST SOURCE SELECT {info["task"]}: {best} BA+{gains[best]:.4f}', flush=True)


def target_predict(folder):
    info = json.loads((folder/'PREPARED.json').read_text())
    suffix = (info['dataset'], info['task'])
    out = OUT/'targets'/suffix[0]/suffix[1]
    out.mkdir(parents=True, exist_ok=True)
    if (out/'COMPLETE.json').exists():
        return
    selection = json.loads((OUT/'source'/suffix[0]/suffix[1]/'SELECTION.json').read_text())
    data = load_npz(folder/'target.npz')
    embeds = {'learned': load_npz(folder/'selective_embedding.npz'), 'old': load_npz(folder/'fixed_embedding.npz')}
    dst = OUT/'tasks'/suffix[0]/suffix[1]
    embeds.update({f: load_npz(dst/f'{f}_embedding.npz') for f in FAMILIES[1:]})
    for z in embeds.values():
        np.testing.assert_array_equal(z['rows'], data['rows'])
    embeds = {f: z['embedding'].astype(float) for f, z in embeds.items()}
    original = load_npz(folder/'policies.npz')
    prior_band = load_npz(ROOT/'frequency_band_complementarity_20261009/targets'/suffix[0]/suffix[1]/'policies.npz')
    original_arm = list(original['modes']).index('selective')
    n, routing = len(data['rows']), router(info, data)
    shape = (5, len(MODES), n)
    base, full, frozen = [np.empty(shape, np.int8) for _ in range(3)]
    selected = np.empty(shape, bool)
    choices = np.empty((5, 4, n), np.int8)
    benefits = np.empty((5, 4, n, 4), np.float32)
    diagnostics, loo_archive = [], {}
    for draw, support in enumerate(data['support']):
        sy, ev = data['support_labels'][draw], data['eval_masks'][draw]
        if len(np.unique(support)) != 64 or ev[support].any():
            raise ValueError('Support budget/split mismatch')
        k0, condition = old_kernel(data, support, info['eta'])
        ref, sim = score_embedding(data, embeds['learned'], support, sy, k0, condition)
        loo_ref = loo_scores(k0[support]+.25*sim[support], data['prior'][support], sy)
        all_scores = dict(learned=ref)
        for fi, family in enumerate(FAMILIES):
            scores, loo, static, choice, value = family_predictions(data, embeds[family], ref, loo_ref, support, sy, k0, condition)
            all_scores.update({family+'_'+key: v for key, v in scores.items()})
            choices[draw, fi], benefits[draw, fi] = choice, value
            loo_archive[f'{family}_{draw}'] = loo
            diagnostics.append(dict(draw=draw, family=family, static_choice=int(static),
                local_switches=int(np.sum(ev & (choice != 0))),
                loo_change_events=int(np.sum(loo[:, 1:].argmax(2) != loo_ref.argmax(1)[:, None]))))
        all_scores['source_selected'] = all_scores[selection['rule']]
        for j, mode in enumerate(MODES):
            score = all_scores[mode]
            bp = score.argmax(1)
            pred, sel = routing(score, data['expert'][draw], sy, ev)
            if sel[~ev].any() or sel.sum() > int(.05*ev.sum()):
                raise ValueError('Expert handoff budget exceeded')
            if mode in ('learned', 'old_equal'):
                original_index = list(original['modes']).index('selective' if mode == 'learned' else 'fixed')
                for actual, key in ((bp, 'base_predictions'), (pred, 'predictions'), (sel, 'selected')):
                    np.testing.assert_array_equal(actual, original[key][draw, original_index])
            if mode in tuple('old_'+b for b in BANDS):
                old_index = list(prior_band['modes']).index('fixed_'+mode[4:])
                for actual, key in ((bp, 'base_predictions'), (pred, 'predictions'), (sel, 'selected')):
                    np.testing.assert_array_equal(actual, prior_band[key][draw, old_index])
            base[draw, j], full[draw, j], selected[draw, j] = bp, pred, sel
            frozen[draw, j] = np.where(original['selected'][draw, original_arm], data['expert'][draw].argmax(1), bp)
        print(f'SPECIALIST TARGET {info["task"]} draw{draw}: main/equal/four-band replay exact', flush=True)
    np.savez_compressed(out/'policies.npz', modes=np.array(MODES), base_predictions=base,
        predictions=full, frozen_predictions=frozen, selected=selected, choices=choices, benefit=benefits,
        rows=data['rows'], support=data['support'], eval_masks=data['eval_masks'])
    np.savez_compressed(out/'support_loo.npz', **loo_archive)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
    write(out/'COMPLETE.json', dict(dataset=info['dataset'], task=info['task'], draws=5,
        target_eval_labels_read=False, reference_paths_replayed=True, labels=64, handoff_fraction=.05,
        families=FAMILIES, source_rule=selection['rule']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['source', 'target', 'report', 'all'], default='all')
    args = parser.parse_args()
    folders = [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]
    if args.stage in ('source', 'all'):
        for folder in folders:
            source_validate(folder)
    if args.stage in ('target', 'all'):
        for folder in folders:
            target_predict(folder)
    if args.stage in ('report', 'all'):
        from frequency_specialist_report import report
        report()
