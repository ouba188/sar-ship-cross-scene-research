"""Post-freeze partition of realized changes and origin of smooth evidence."""

from frequency_margin_run import OUT, PREV, EXP, MODES
import json
import numpy as np
import pandas as pd
from prepare_learned_frequency import ROOT, SOURCE, DATASETS, load_npz
from frequency_band_core import unit_bands
from frequency_margin_policy import margin_evidence
from frequency_reliability_core import balanced_weights


def evidence_parts(similarity, loo, labels):
    main, band = loo[:, 0], loo[:, 1]
    main_ok, band_ok = main.argmax(1) == labels, band.argmax(1) == labels
    masks = (main_ok & band_ok, ~main_ok & ~band_ok, ~main_ok & band_ok, main_ok & ~band_ok)
    delta = margin_evidence(band, labels) - margin_evidence(main, labels)
    positive = np.maximum(similarity, 0.)
    top = np.argsort(-positive, axis=1, kind='stable')[:, :8]
    weight = np.take_along_axis(positive, top, axis=1) * balanced_weights(labels)[top]
    denominator = np.maximum(weight.sum(1), 1e-12)
    return np.stack([np.sum(weight * (delta * m)[top], axis=1)/denominator
                     for m in masks], axis=1)


def main():
    if len(list((OUT/'targets').glob('*/*/COMPLETE.json'))) != 25:
        raise ValueError('Freeze all predictions before diagnosis')
    transitions, evidence = [], []
    names = ('both_correct', 'both_wrong', 'rescue', 'harm')
    for folder in [p.parent for p in sorted((EXP/'tasks').rglob('EVALUATED.json'))]:
        info = json.loads((folder/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        suffix = f"{meta['dataset']}/{meta['task']}"
        p = load_npz(OUT/'targets'/suffix/'policies.npz')
        data = load_npz(folder/'target.npz')
        loofile = load_npz(PREV/'targets'/suffix/'support_loo.npz')
        unit = unit_bands(load_npz(PREV/'tasks'/suffix/'preserved_embedding.npz')['embedding'])
        manifest = SOURCE/'manifest.csv' if meta['dataset'] == 'self24ports' else ROOT/DATASETS[meta['dataset']][0]/'target_manifest.csv'
        y_all = pd.read_csv(manifest, usecols=['class_id']).class_id.to_numpy()[p['rows']]
        for draw, ev in enumerate(p['eval_masks']):
            y, support, sy = y_all[ev], data['support'][draw], data['support_labels'][draw]
            hard = p['base_predictions'][draw, MODES.index('hard'), ev]
            oldchoice = p['choices'][draw, MODES.index('hard'), ev]
            _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
            ba_weight = 100 / (len(counts) * counts[inverse])
            comp = np.empty((len(y), 4, 4))
            condition = np.exp(-np.sum((data['conditions'][ev, None] - data['conditions'][support][None])**2, axis=2)/.5)
            for bi in range(4):
                sim = (unit[ev, bi] @ unit[support, bi].T) * condition
                loo = loofile[f'preserved_{draw}'][:, [0, bi+1]]
                for lo in range(0, len(y), 8192):
                    comp[lo:lo+8192, bi] = evidence_parts(sim[lo:lo+8192], loo, sy)
            np.testing.assert_allclose(comp.sum(2), p['utilities'][draw, 1, ev], atol=1e-7, rtol=1e-5)
            for mode in ('smooth', 'smooth02', 'smooth05', 'source_selected'):
                mi = MODES.index(mode)
                pred, choice = p['base_predictions'][draw, mi, ev], p['choices'][draw, mi, ev]
                groups = dict(extra_switch=(oldchoice == 0) & (choice > 0),
                    stop_switch=(oldchoice > 0) & (choice == 0),
                    different_band=(oldchoice > 0) & (choice > 0) & (oldchoice != choice),
                    same_choice=oldchoice == choice)
                delta = (pred == y).astype(float) - (hard == y)
                for group, mask in groups.items():
                    transitions.append(dict(**meta, draw=draw, mode=mode, group=group,
                        n=int(mask.sum()), rescues=int(np.sum(mask & (delta > 0))),
                        harms=int(np.sum(mask & (delta < 0))),
                        delta_oa=100*float(delta[mask].sum())/len(y),
                        delta_ba=float((delta[mask] * ba_weight[mask]).sum())))
                for outcome, mask in dict(rescue=(hard != y) & (pred == y),
                        harm=(hard == y) & (pred != y), wrong_to_wrong=(hard != y) & (pred != y) & (hard != pred)).items():
                    mask &= choice > 0
                    take = np.flatnonzero(mask)
                    chosen_parts = comp[take, choice[take]-1]
                    chosen_hard = p['utilities'][draw, 0, ev][take, choice[take]-1]
                    row = dict(**meta, draw=draw, mode=mode, outcome=outcome, n=len(take),
                        hard_zero_n=int(np.sum(chosen_hard == 0)))
                    for ci, name in enumerate(names):
                        row[name+'_sum'] = float(chosen_parts[:, ci].sum())
                        row[name+'_dominant_n'] = int(np.sum(chosen_parts.argmax(1) == ci)) if len(take) else 0
                    evidence.append(row)
        print('DIAGNOSE '+suffix, flush=True)
    t, e = pd.DataFrame(transitions), pd.DataFrame(evidence)
    t.to_csv(OUT/'transition_diagnosis.csv', index=False)
    e.to_csv(OUT/'evidence_origin.csv', index=False)
    totals = t.groupby(['dataset', 'mode', 'group'])[['n', 'rescues', 'harms']].sum()
    means = t.groupby(['dataset', 'task', 'mode', 'group'])[['delta_oa', 'delta_ba']].mean().groupby(['dataset', 'mode', 'group']).mean()
    totals.join(means).to_csv(OUT/'transition_summary.csv')
    cols = [c for c in e if c not in ('dataset', 'task', 'draw', 'mode', 'outcome')]
    es = e.groupby(['dataset', 'mode', 'outcome'])[cols].sum()
    for name in names:
        es[name+'_mean'] = es[name+'_sum']/es.n.replace(0, np.nan)
    es.to_csv(OUT/'evidence_origin_summary.csv')
    print(totals.join(means).to_string(), flush=True)


if __name__ == '__main__':
    main()
