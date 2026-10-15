"""Explain candidate decisions after prediction freeze; never change policy."""

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frequency_candidate_run import OUT, EXP, ROOT, CANDIDATES, WIDTHS
from prepare_learned_frequency import SOURCE, DATASETS, load_npz
from frequency_candidate_core import rank_margin


def auc(utility, value):
    mask = utility != 0
    return float(roc_auc_score(utility[mask] > 0, value[mask])) if len(np.unique(utility[mask])) == 2 else np.nan


def main():
    if not json.loads((OUT/'STATUS.json').read_text())['complete']:
        raise ValueError('Freeze and evaluate before diagnosis')
    ys = pd.read_csv(SOURCE/'manifest.csv', usecols=['class_id']).class_id.to_numpy()
    sources, targets, cases, contributions = [], [], [], []
    names = [f'{group}_{band}' for group in ('global', 'aligned', 'matched') for band in ('LL', 'LH', 'HL', 'HH')]
    for marker in sorted((EXP/'tasks').rglob('EVALUATED.json')):
        info = json.loads((marker.parent/'PREPARED.json').read_text())
        meta = dict(dataset=info['dataset'], task=info['task'])
        src = OUT/'source'/info['dataset']/info['task']
        d, oof = load_npz(src/'examples.npz'), load_npz(src/'oof.npz')
        selection = json.loads((src/'SELECTION.json').read_text())
        validation = pd.read_csv(src/'validation.csv')
        with (src/'models.pkl').open('rb') as handle:
            models = pickle.load(handle)
        eligible = d['eligible']
        u = (d['new'].argmax(1) == d['labels']).astype(float)-(d['old'].argmax(1) == d['labels'])
        weights = d['weights']/selection['source_ports']
        for j, (mode, width) in enumerate(zip(CANDIDATES, WIDTHS)):
            coef = models[mode]['coef']/models[mode]['scale'] if models[mode] is not None else np.zeros(width)
            for name, value in zip(names, coef):
                contributions.append(dict(**meta, mode=mode, feature=name, effective_coefficient=float(value)))
            for split, value in [('fit', rank_margin(models[mode], d['candidate'][:, :width])), ('held_port', oof['benefit'][:, j])]:
                reject = eligible & (value < 0)
                sources.append(dict(**meta, mode=mode, split=split, queries=len(u),
                    train_pairs=int(d['train_valid'].sum()), eligible=int(eligible.sum()),
                    auc=auc(u[eligible], value[eligible]),
                    delta_ba=-100*float(np.sum(weights[reject]*u[reject])),
                    pair_accuracy=selection['fit_pair_accuracy'][mode] if split == 'fit' else
                        float(validation.loc[validation['mode'].eq(mode), 'held_pair_accuracy'].mean()),
                    lost=int(np.sum(reject & (u > 0))), repaired=int(np.sum(reject & (u < 0)))))
        dst = OUT/'targets'/info['dataset']/info['task']
        p, contrasts = load_npz(dst/'policies.npz'), load_npz(dst/'contrasts.npz')
        y = ys[p['rows']] if info['dataset'] == 'self24ports' else pd.read_csv(
            ROOT/DATASETS[info['dataset']][0]/'target_manifest.csv', usecols=['class_id']).class_id.to_numpy()
        for draw, ev in enumerate(p['eval_masks']):
            a, b = p['base_predictions'][draw, 0], p['base_predictions'][draw, 1]
            conflict, eligible = ev & (a != b), ev & p['eligible'][draw]
            u = (b == y).astype(float)-(a == y)
            labels, counts = np.unique(y[ev], return_counts=True)
            class_weight = np.zeros(8)
            class_weight[labels] = 1./(len(labels)*counts)
            for j, (mode, width) in enumerate(zip(CANDIDATES, WIDTHS)):
                v = p['benefit'][draw, j]
                reject = eligible & (v < 0)
                targets.append(dict(**meta, draw=draw, mode=mode,
                    conflicts=int(conflict.sum()), eligible=int(eligible.sum()), rejected=int(reject.sum()),
                    eligible_auc=auc(u[eligible], v[eligible]),
                    eligible_old_correct=int(np.sum(eligible & (a == y))),
                    eligible_new_correct=int(np.sum(eligible & (b == y))),
                    eligible_neither_correct=int(np.sum(eligible & (a != y) & (b != y))),
                    delta_ba=-100*float(np.sum(class_weight[y[reject]]*u[reject])),
                    lost=int(np.sum(reject & (u > 0))), repaired=int(np.sum(reject & (u < 0)))))
                rows = contrasts[f'rows_{draw}']
                diff = contrasts[f'difference_{draw}'][:, :width]
                model = models[mode]
                terms = np.zeros_like(diff) if model is None else diff/model['scale']*model['coef']
                # Exact linear score decomposition, not additive accuracy attribution.
                np.testing.assert_allclose(terms.sum(1), v[rows], atol=1e-5, rtol=1e-5)
                if draw == 0:
                    for kind, event in [('repair', reject & (u < 0)), ('lost_rescue', reject & (u > 0)),
                                        ('kept_rescue', eligible & ~reject & (u > 0))]:
                        ii = np.flatnonzero(event[rows])
                        if len(ii):
                            chosen = ii[np.argmax(np.abs(v[rows[ii]]))]
                            row = int(rows[chosen])
                            cases.append(dict(**meta, draw=draw, mode=mode, event=kind,
                                local_row=row, manifest_row=int(p['rows'][row]), true_class=int(y[row]),
                                old_class=int(a[row]), new_class=int(b[row]), margin=float(v[row]),
                                **{name: float(value) for name, value in zip(names, terms[chosen])}))
    s, t = pd.DataFrame(sources), pd.DataFrame(targets)
    s.to_csv(OUT/'diagnosis_source.csv', index=False)
    t.to_csv(OUT/'diagnosis_target.csv', index=False)
    pd.DataFrame(cases).to_csv(OUT/'decision_cases.csv', index=False)
    pd.DataFrame(contributions).to_csv(OUT/'coefficients.csv', index=False)
    print('SOURCE\n'+s.groupby(['dataset', 'mode', 'split'])[['auc', 'delta_ba', 'pair_accuracy']].mean().to_string())
    print('TARGET\n'+t.groupby(['dataset', 'mode'])[['eligible_auc', 'delta_ba']].mean().to_string())
    print('TARGET EVENTS\n'+t.groupby(['dataset', 'mode'])[['conflicts', 'eligible', 'rejected', 'lost', 'repaired']].sum().to_string())
    m = pd.read_csv(OUT/'summary.csv')
    print('BA BY STAGE\n'+m.pivot_table(index=['dataset', 'mode'], columns='stage', values='delta_ba').to_string())


if __name__ == '__main__':
    main()
