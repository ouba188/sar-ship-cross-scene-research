"""Post-freeze diagnosis of endpoint utility generalization, not policy tuning."""

import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frequency_decision_run import OUT, EXP, ROOT, SOURCE, DATASETS, CANDIDATES
from frequency_decision_core import predict_benefit
from prepare_learned_frequency import load_npz


def auc(y, score, weights=None):
    event = y != 0
    if len(np.unique(y[event])) < 2:
        return np.nan
    return float(roc_auc_score(y[event] > 0, score[event],
        sample_weight=None if weights is None else weights[event]))


def action_record(utility, benefit, weights):
    reject = benefit < 0
    event = utility != 0
    return dict(n=len(utility), events=int(event.sum()), auc=auc(utility, benefit),
        balanced_auc=auc(utility, benefit, weights), rejected=int(reject.sum()),
        lost_rescues=int(np.sum(reject & (utility > 0))),
        repaired_harms=int(np.sum(reject & (utility < 0))),
        rejected_predicted_utility=float(np.average(benefit[reject], weights=weights[reject])) if reject.any() else np.nan,
        rejected_actual_utility=float(np.average(utility[reject], weights=weights[reject])) if reject.any() else np.nan,
        delta_ba=-100*float(np.sum(weights[reject]*utility[reject])))


def main():
    if not json.loads((OUT/'STATUS.json').read_text())['complete']:
        raise ValueError('Requires frozen and evaluated experiment')
    source_records, bins, target_records, transitions = [], [], [], []
    ys = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    edges = [-np.inf, -.2, 0., .2, np.inf]
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        info = json.loads((marker.parent/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        src = OUT/'source'/info['dataset']/info['task']
        d, oof = load_npz(src/'examples.npz'), load_npz(src/'oof.npz')
        selection = json.loads((src/'SELECTION.json').read_text())
        with (src/'models.pkl').open('rb') as handle:
            models = pickle.load(handle)
        conflict = oof['disagreement']
        y, weights = oof['utility'][conflict], d['weights'][conflict]/selection['source_ports']
        names = selection['features']
        empty_neighbor = np.mean((d['features'][conflict, names.index('neighbor_rescue_mass')]
            + d['features'][conflict, names.index('neighbor_harm_mass')]) == 0)
        for j, mode in enumerate(CANDIDATES):
            f = d['features'][:, :selection['decision_width']] if mode == 'decision_only' else d['features']
            for split, value in [('fit', predict_benefit(models[mode], f)), ('held_port', oof['benefit'][:, j])]:
                source_records.append(dict(**meta, mode=mode, split=split,
                    **action_record(y, value[conflict], weights), no_effective_neighbor_fraction=empty_neighbor))
        p = load_npz(OUT/'targets'/info['dataset']/info['task']/'policies.npz')
        labels = ys[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        for draw, ev in enumerate(p['eval_masks']):
            a, b = p['base_predictions'][draw, 0], p['base_predictions'][draw, 1]
            mask = ev & (a != b)
            utility = (b[mask] == labels[mask]).astype(float)-(a[mask] == labels[mask])
            present, counts = np.unique(labels[ev], return_counts=True)
            weight_map = {c: 1./(len(present)*n) for c, n in zip(present, counts)}
            w = np.array([weight_map[c] for c in labels[mask]])
            for j, mode in enumerate(CANDIDATES):
                v = p['benefit'][draw, j, mask]
                target_records.append(dict(**meta, draw=draw, mode=mode, **action_record(utility, v, w)))
                for lower, upper in zip(edges[:-1], edges[1:]):
                    ix = (v >= lower) & (v < upper)
                    if ix.any():
                        bins.append(dict(**meta, draw=draw, mode=mode, lower=lower, upper=upper,
                            n=int(ix.sum()), weight=float(w[ix].sum()),
                            predicted_sum=float(np.sum(w[ix]*v[ix])),
                            actual_sum=float(np.sum(w[ix]*utility[ix]))))
                for old, new in np.unique(np.c_[a[mask], b[mask]], axis=0):
                    ix = (a[mask] == old) & (b[mask] == new) & (v < 0)
                    if ix.any():
                        transitions.append(dict(**meta, draw=draw, mode=mode,
                            old_class=int(old), new_class=int(new), n=int(ix.sum()),
                            lost_rescues=int(np.sum(utility[ix] > 0)),
                            repaired_harms=int(np.sum(utility[ix] < 0)),
                            delta_ba=-100*float(np.sum(w[ix]*utility[ix]))))
    s, t, b = pd.DataFrame(source_records), pd.DataFrame(target_records), pd.DataFrame(bins)
    s.to_csv(OUT/'diagnosis_source.csv', index=False)
    t.to_csv(OUT/'diagnosis_target.csv', index=False)
    b.to_csv(OUT/'diagnosis_bins.csv', index=False)
    pd.DataFrame(transitions).to_csv(OUT/'diagnosis_transitions.csv', index=False)
    keys = ['auc', 'balanced_auc', 'delta_ba', 'rejected_actual_utility', 'rejected_predicted_utility']
    print('SOURCE FIT / HELD-PORT\n'+s.groupby(['dataset', 'mode', 'split'])[keys].mean().to_string())
    print('TARGET\n'+t.groupby(['dataset', 'mode'])[keys].mean().to_string())
    print('SOURCE NEIGHBOR EVIDENCE\n'+s.groupby('dataset').no_effective_neighbor_fraction.mean().to_string())
    draws = pd.read_csv(OUT/'draw_metrics.csv')
    print('SELECTED DRAW EFFECTS\n'+draws.loc[(draws['mode'] == 'source_selected') & (draws.stage == 'full'),
        ['dataset', 'draw', 'delta_oa', 'delta_ba']].to_string(index=False))
    ports = pd.read_csv(OUT/'task_means.csv')
    print('SELECTED PORT EFFECTS\n'+ports.loc[(ports['mode'] == 'source_selected') & (ports.stage == 'full'),
        ['dataset', 'task', 'delta_oa', 'delta_ba']].to_string(index=False))
    c = pd.read_csv(OUT/'class_metrics.csv')
    c = c.loc[(c['mode'] == 'source_selected') & (c.stage == 'base')]
    print('SELECTED BASE CLASS EFFECTS\n'+c.groupby(['dataset', 'class_id']).delta_recall.mean().to_string())


if __name__ == '__main__':
    main()
