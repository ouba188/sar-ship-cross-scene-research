"""Small matched-cohort SAR/knowledge reader comparison, not neural training."""

import json
import numpy as np
import pandas as pd
from check_teacher_weight_strength import BASE, ROOT, SEEDS, load_bundle, apply_quality, evaluate
from check_translation_teacher_feasibility import pairs_for, margins, rank_metrics
from incidence_transport import pair_weights
from scene_training import sample_weights

OUT = ROOT/'input_probe'
ARMS = ('sar', 'knowledge', 'joint', 'sar_capacity')


def raw_inputs(bundle, descriptors):
    ids = bundle['predictions']['obs_ids']
    physical = np.array([descriptors[ids[ix]].mean(0) for ix in bundle['members']])
    scores = bundle['values']['scores'].astype(float)
    centered = scores-scores.mean(1, keepdims=True)
    sar = np.c_[physical, centered]
    knowledge = bundle['values']['x'][:, 1:5]
    assert sar.shape[1] == 14 and knowledge.shape[1] == 4
    assert np.isfinite(sar).all()
    return sar, knowledge


def transform(sar, knowledge, center, scale, arm):
    z = np.tanh((sar-center)/scale)
    if arm == 'sar':
        return z
    if arm == 'knowledge':
        return knowledge
    if arm == 'joint':
        return np.c_[z, knowledge]
    if arm == 'sar_capacity':
        # Four fixed nonlinear SAR terms, matching the joint reader's dimension.
        return np.c_[z, z[:, 0]**2, z[:, 3]**2, z[:, 5]**2, np.mean(z[:, 6:]**2, axis=1)]
    raise ValueError(arm)


def fit_reader(bundle, sar, knowledge, pairs, arm):
    ids = np.unique(pairs)
    cells = bundle['cells']; cs = bundle['values']['scores']
    wobs = sample_weights(cells.iloc[ids]); wobs /= wobs.sum()
    center = wobs@sar[ids]
    scale = np.sqrt(wobs@(sar[ids]-center)**2).clip(1e-5)
    x = transform(sar, knowledge, center, scale, arm)
    a, b = pairs.T; dx = x[b]-x[a]
    y = margins(cs, cells.class_id.to_numpy())
    w = pair_weights(cells, pairs)
    assert np.isfinite(dx).all()
    beta = np.linalg.solve(dx.T@(w[:, None]*dx)+.1*np.eye(dx.shape[1]), dx.T@(w*(y[b]-y[a])))
    return dict(beta=beta, center=center, scale=scale, arm=arm, train_pairs=pairs)


def predict(model, sar, knowledge):
    return transform(sar, knowledge, model['center'], model['scale'], model['arm'])@model['beta']


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT/'PROTOCOL.json').write_text(json.dumps(dict(new_neural_fits=0, target_evaluations=0,
        arms=dict(sar=14, knowledge=4, joint=18, sar_capacity=18), ridge=.1,
        sar='Six existing frozen CNN spatial summaries plus eight centered teacher scores. Not measured physical scattering.',
        transform='SAR weighted mean/std from training paired cells only, tanh bound; native bounded knowledge unchanged',
        knowledge='u,u^2,u^2cos2theta,u^2sin2theta, no truth/class/identity as predictors',
        target='Same-vessel different-product true-class margin difference, zero intercept; class/vessel/pair weighting',
        matched='All arms use previous identical known-knowledge cells and donors; SAR-only not given wider coverage',
        evaluation='Both previously fixed minmax and absolute_gap weighting, all arms reported, no held-best selection',
        fold='Same source vessel3fold readers plus full; upstream teacher not nested retrained',
        decision='Check task-level teacher gains vs equal, knowledge, SAR and capacity across seeds/sourceCV/vessels before any student module',
        limitations='Repeated historical development cohorts, frozen representation, no causal identification or student gain claim')),
        encoding='utf-8')
    meta = pd.read_csv(BASE/'observations.csv')
    descriptors = np.full((len(meta), 6), np.nan)
    with np.load(BASE/'condition_evidence_20261011/evidence_features.npz') as f:
        descriptors[f['obs_ids']] = f['features']
    models, bundles, inputs, qualities, ranking = {}, {}, {}, {}, []
    for seed in SEEDS:
        bundle = load_bundle('fit', seed); bundles[seed] = bundle
        sar, knowledge = raw_inputs(bundle, descriptors); inputs[seed] = (sar, knowledge)
        cells = bundle['cells']; pairs = pairs_for(cells)
        for arm in ARMS:
            q = np.full(len(cells), np.nan)
            for k in range(3):
                train = pairs_for(cells, cells.fold.to_numpy() != k)
                model = fit_reader(bundle, sar, knowledge, train, arm)
                val = cells.fold.to_numpy() == k
                assert not set(cells.iloc[train.ravel()].mmsi) & set(cells.loc[val].mmsi)
                q[val] = predict(model, sar[val], knowledge[val])
                np.savez_compressed(OUT/f'fold{k}_{arm}_{seed}.npz', **model)
            qualities[seed, arm] = q
            ranking.append(rank_metrics(cells, bundle['values']['scores'], pairs, q, seed, arm, 'source_cv'))
            model = fit_reader(bundle, sar, knowledge, pairs, arm)
            models[seed, arm] = model
            np.savez_compressed(OUT/f'full_{arm}_{seed}.npz', **model)
    (OUT/'FROZEN.json').write_text(json.dumps(dict(all_models_fitted=True,
        held_teacher_loaded_by_this_script=False, historical_held_exposure=True)), encoding='utf-8')
    rows, events, spans = [], [], []
    for role in ('fit', 'held'):
        for seed in SEEDS:
            bundle = bundles[seed] if role == 'fit' else load_bundle(role, seed)
            sar, knowledge = inputs[seed] if role == 'fit' else raw_inputs(bundle, descriptors)
            if role == 'held':
                assert not set(bundle['cells'].mmsi) & set(bundles[seed]['cells'].mmsi)
            old = bundle['predictions']['equal']; outputs = {'equal': old}; qq = {}
            rows.append(evaluate(bundle, old, 'equal', role, seed))
            for arm in ARMS:
                q = qualities[seed, arm] if role == 'fit' else predict(models[seed, arm], sar, knowledge)
                qq[arm] = q
                if role == 'held':
                    ranking.append(rank_metrics(bundle['cells'], bundle['values']['scores'],
                        pairs_for(bundle['cells']), q, seed, arm, role))
                for mode in ('minmax', 'absolute_gap'):
                    pred, detail = apply_quality(bundle, q, mode)
                    name = arm+'_'+mode; outputs[name] = pred
                    rows.append(evaluate(bundle, pred, name, role, seed))
                    detail['role'] = role; detail['seed'] = seed; detail['arm'] = name
                    spans.append(detail)
                    mm = bundle['meta']; y = mm.class_id.to_numpy()
                    changed = bundle['predictions']['multiview'] & (old.argmax(1) != pred.argmax(1))
                    for i in np.flatnonzero(changed):
                        events.append(dict(role=role, seed=seed, arm=name,
                            obs_id=int(bundle['predictions']['obs_ids'][i]), mmsi=int(mm.iloc[i].mmsi),
                            class_id=int(y[i]), before=int(old[i].argmax()), after=int(pred[i].argmax()),
                            rescue=bool(old[i].argmax() != y[i] and pred[i].argmax() == y[i]),
                            harm=bool(old[i].argmax() == y[i] and pred[i].argmax() != y[i])))
            np.savez_compressed(OUT/f'{role}_{seed}_predictions.npz', obs_ids=bundle['predictions']['obs_ids'], **outputs)
            np.savez_compressed(OUT/f'{role}_{seed}_quality.npz', **qq)
    pd.DataFrame(rows).to_csv(OUT/'results.csv', index=False)
    pd.DataFrame(events).to_csv(OUT/'changed_predictions.csv', index=False)
    pd.concat(spans).to_csv(OUT/'weight_spans.csv', index=False)
    pd.DataFrame(ranking).to_csv(OUT/'ranking_results.csv', index=False)
    summary = pd.DataFrame(rows).groupby(['role', 'arm'])[['OA', 'BA', 'CE', 'rescue', 'harm', 'rescue_ships', 'harm_ships']].mean()
    summary.to_csv(OUT/'summary.csv')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(completed=True, ridge_fits=32,
        neural_fits=0, target_evaluations=0, training_held_vessels_disjoint=True)), encoding='utf-8')
    print(summary.to_string())
    print(pd.DataFrame(ranking).groupby(['role', 'arm'])[['MSE', 'zero_MSE', 'pair_ranking_accuracy']].mean().to_string())


if __name__ == '__main__':
    main()
