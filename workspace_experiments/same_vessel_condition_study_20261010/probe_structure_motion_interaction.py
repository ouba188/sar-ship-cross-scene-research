"""Fixed small response regressions, not a new neural classifier or causal proof."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
from pathlib import Path
import numpy as np
import pandas as pd
from incidence_transport import pair_weights
from scene_training import sample_weights, metrics

BASE = Path(__file__).resolve().parent
OUT = BASE/'interaction_feasibility_20261012'
ARMS = ('anchor', 'additive', 'pairwise', 'triple', 'capacity', 'sham')
TARGETS = ('center_signed_change', 'center_absolute_change', 'class_margin_change', 'next_error')


def block(v):
    s, d, m = v[:, :2], v[:, 2:5], v[:, 5:6]
    pair = np.c_[(s[:, :, None]*d[:, None, :]).reshape(len(v), -1), s*m, d*m]
    triple = (s[:, :, None]*d[:, None, :]*m[:, None, :]).reshape(len(v), -1)
    extra = np.c_[d[:, :2]**3, d[:, 2:3]**2, d[:, 2:3]**3, m**2, m**3]
    return pair, triple, extra


def design(context, state, directed, arm):
    a, b = directed.T
    x = np.c_[context[a], state[a]]
    if arm == 'anchor':
        return x
    x = np.c_[x, state[b, 2:]-state[a, 2:]]
    if arm == 'additive':
        return x
    pa, ta, ca = block(state[a]); pb, tb, cb = block(state[b])
    x = np.c_[x, pb-pa]
    if arm in ('triple', 'sham'):
        x = np.c_[x, tb-ta]
    elif arm == 'capacity':
        x = np.c_[x, cb-ca]
    return x


def shuffle_state(meta, state, ids):
    result = state.copy(); rng = np.random.default_rng(51012)
    for _, g in meta.iloc[ids].groupby(['mmsi', 'port', 'platform', 'pass_name'], sort=True):
        ix = g.index.to_numpy()
        result[ix, 2:] = state[rng.permutation(ix), 2:]
    return result


def targets(meta, evidence, scores, directed):
    a, b = directed.T
    truth = meta.class_id.to_numpy()
    competitor = scores.copy(); competitor[np.arange(len(scores)), truth] = -np.inf
    margin = scores[np.arange(len(scores)), truth]-competitor.max(1)
    delta = evidence[b, 5]-evidence[a, 5]
    return np.c_[delta, np.abs(delta), margin[b]-margin[a], (scores[b].argmax(1) != truth[b]).astype(float)]


def context_features(evidence, scores):
    sorted_score = np.sort(scores, axis=1)
    return np.c_[evidence[:, 5], evidence[:, 2]-evidence[:, 0], sorted_score[:, -1]-sorted_score[:, -2]]


def fit_response(meta, state, context, evidence, scores, pairs, arm):
    directed = np.r_[pairs, pairs[:, ::-1]]
    ids = np.unique(pairs)
    w = np.tile(pair_weights(meta, pairs)/2, 2)
    ow = sample_weights(meta.iloc[ids]); ow /= ow.sum()
    center = ow@state[ids]
    scale = np.sqrt(ow@(state[ids]-center)**2).clip(1e-5)
    norm = (state-center)/scale
    if arm == 'sham':
        norm = shuffle_state(meta, norm, ids)
    x = design(context, norm, directed, arm)
    xm = w@x; xs = np.sqrt(w@(x-xm)**2).clip(1e-5)
    xn = (x-xm)/xs
    y = targets(meta, evidence, scores, directed); ym = w@y
    gram = xn.T@(w[:, None]*xn)
    coefficient = np.linalg.solve(gram+.1*np.eye(x.shape[1]), xn.T@(w[:, None]*(y-ym)))
    return dict(center=center, scale=scale, xcenter=xm, xscale=xs, coefficient=coefficient,
                target_mean=ym, arm=arm, n_parameters=x.shape[1]+1, fit_ids=ids)


def predict(model, state, context, directed):
    norm = (state-model['center'])/model['scale']
    x = design(context, norm, directed, model['arm'])
    result = (x-model['xcenter'])/model['xscale']@model['coefficient']+model['target_mean']
    result[:, 1] = np.maximum(result[:, 1], 0.)
    result[:, 3] = np.clip(result[:, 3], 0., 1.)
    return result


def evaluate(meta, evidence, scores, pairs, prediction, model, tag):
    directed = np.r_[pairs, pairs[:, ::-1]]
    truth = targets(meta, evidence, scores, directed)
    w = np.tile(pair_weights(meta, pairs)/2, 2)
    errors = (prediction-truth)**2
    reference = (truth-model['target_mean'])**2
    rows = []
    for j, task in enumerate(TARGETS):
        mse = float(w@errors[:, j]); base = float(w@reference[:, j])
        rows.append(dict(**tag, task=task, n_parameters=model['n_parameters'], pairs=len(pairs),
            ships=int(meta.iloc[pairs[:, 0]].mmsi.nunique()), MSE=mse,
            constant_MSE=base, skill_vs_constant=100*(1-mse/base) if base else np.nan))
    detail = pd.DataFrame(dict(a=directed[:, 0], b=directed[:, 1], mmsi=meta.iloc[directed[:, 0]].mmsi.to_numpy(),
                              class_id=meta.iloc[directed[:, 0]].class_id.to_numpy()))
    for j, task in enumerate(TARGETS):
        detail[task+'_squared_error'] = errors[:, j]
        detail[task+'_truth'] = truth[:, j]
        detail[task+'_prediction'] = prediction[:, j]
    return rows, detail


def teacher_selection(meta, scores, directed, prediction, tag):
    variants = ('equal', 'confidence', 'response_soft', 'response_hard', 'label_oracle')
    saved = {key: [] for key in variants}; ids = []
    for a in np.unique(directed[:, 0]):
        ix = np.flatnonzero(directed[:, 0] == a)
        if len(ix) < 2:
            continue
        b = directed[ix, 1]; donors = scores[b]
        y = int(meta.iloc[a].class_id)
        mean = donors.mean(0)
        confidence = np.sort(donors, axis=1)
        gap = confidence[:, -1]-confidence[:, -2]
        risk = prediction[ix, 3]
        weights = 2-risk
        options = np.vstack([mean, donors]); rivals = options.copy(); rivals[:, y] = -np.inf
        saved['equal'].append(mean)
        saved['confidence'].append(np.average(donors, axis=0, weights=1+gap/(1+gap)))
        saved['response_soft'].append(np.average(donors, axis=0, weights=weights))
        saved['response_hard'].append(donors[np.isclose(risk, risk.min(), rtol=0, atol=1e-12)].mean(0))
        saved['label_oracle'].append(options[np.argmax(options[:, y]-rivals.max(1))])
        ids.append(int(a))
    ids = np.array(ids, int)
    if not len(ids):
        return [], ids, {}
    y = meta.iloc[ids].class_id.to_numpy()
    values = {k: np.stack(v) for k, v in saved.items()}
    old = values['equal'].argmax(1)
    rows = [dict(**tag, selector=k, n=len(ids), ships=int(meta.iloc[ids].mmsi.nunique()),
            **metrics(v.argmax(1), y), rescue=int(((old != y)&(v.argmax(1) == y)).sum()),
            harm=int(((old == y)&(v.argmax(1) != y)).sum())) for k, v in values.items()]
    return rows, ids, values


def self_test():
    rng = np.random.default_rng(51)
    state = rng.normal(size=(8, 6)); context = rng.normal(size=(8, 3))
    edges = np.array([[0, 1], [2, 3]])
    d = design(context, state, edges, 'triple')
    rev = design(context, state, edges[:, ::-1], 'triple')
    np.testing.assert_allclose(d[:, 9:], -rev[:, 9:])
    assert d.shape[1] == design(context, state, edges, 'capacity').shape[1] == 30
    altered = context.copy(); altered[[1, 3]] += 1000
    np.testing.assert_array_equal(d, design(altered, state, edges, 'triple'))


def main():
    self_test()
    results = OUT/'response_probe'
    results.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    with np.load(OUT/'data.npz') as f:
        data = {k: f[k] for k in f.files}
    state = data['values'][:, :6]
    evidence = np.full((len(meta), 6), np.nan)
    with np.load(BASE/'condition_evidence_20261011/evidence_features.npz') as f:
        evidence[f['obs_ids']] = f['features']
    with np.load(BASE/'crossfit_evidence_20261011/source_folds.npz') as f:
        fold = np.full(len(meta), -1, int)
        np.testing.assert_array_equal(f['fit_obs_ids'], data['fit_obs_ids'])
        fold[f['fit_obs_ids']] = f['assignment']
    for key in ('fit_pairs', 'held_pairs'):
        assert np.isfinite(state[np.unique(data[key])]).all()
        assert np.isfinite(evidence[np.unique(data[key])]).all()
    assert not set(meta.iloc[data['fit_pairs'].ravel()].mmsi) & set(meta.iloc[data['held_pairs'].ravel()].mmsi)
    protocol = dict(date='2026-10-12', neural_fits=0, target_ports_used=False,
        targets=list(TARGETS), arms=list(ARMS), seeds=[31415, 27182], ridge=.1,
        structure='AIS logL/logW; partial geometry proxy, not full scattering structure.',
        direction='sin2/cos2 AIS heading relative native ground-range axis plus incidence.',
        motion='log1p SOG; does NOT test angular-velocity rotation model in cited paper.',
        design='A-image context plus Ametadata; add B-A main/pairwise/triple functions; Bimage is response only.',
        controls='Capacity adds6univariate nonlinear changes instead of6tripleterms. Sham jointpermutes D/M in training vessels/ports/platform/pass; evaluates real metadata.',
        cv='Original vessel3fold response-only CV; upstream OOF models not nestedrefit. Full fits frozen before held response and teacher selection.',
        selection='No best-arm or best-task promotion from held metrics; report all predefined arms/tasks. Teacher softweights2-predictederror, hard risk minimum diagnostic.',
        limitation='20heldships, historicalcohort, no posture timeseries; no deployedSAR-onlyclassifier change or claim of causal identification.')
    (results/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows, teacher_rows, models, scores_by_seed = [], [], {}, {}
    for seed in (31415, 27182):
        scores = np.full((len(meta), 8), np.nan)
        with np.load(BASE/'geometry_train_teacher_20261011'/f'teachers_{seed}.npz') as f:
            scores[data['fit_obs_ids']] = f['original_oof']
        with np.load(BASE/'crossfit_evidence_20261011'/f'full_{seed}_outputs.npz') as f:
            scores[data['held_obs_ids']] = f['prior'][data['held_obs_ids']]
        scores_by_seed[seed] = scores
        context = context_features(evidence, scores)
        pairs = data['fit_pairs']; pfold = fold[pairs[:, 0]]
        np.testing.assert_array_equal(pfold, fold[pairs[:, 1]])
        for arm in ARMS:
            for k in range(3):
                train, val = pairs[pfold != k], pairs[pfold == k]
                model = fit_response(meta, state, context, evidence, scores, train, arm)
                directed = np.r_[val, val[:, ::-1]]
                prediction = predict(model, state, context, directed)
                tag = dict(seed=seed, arm=arm, cohort='source_cv', fold=k)
                rr, detail = evaluate(meta, evidence, scores, val, prediction, model, tag); rows.extend(rr)
                detail.to_csv(results/f'cv_{k}_{arm}_{seed}.csv', index=False)
            model = fit_response(meta, state, context, evidence, scores, pairs, arm)
            models[seed, arm] = model
            np.savez_compressed(results/f'{arm}_{seed}.npz', **model)
    (results/'FROZEN.json').write_text(json.dumps(dict(all_response_models_fitted=True, held_metrics_read=False)), encoding='utf-8')
    for seed in (31415, 27182):
        scores = scores_by_seed[seed]; context = context_features(evidence, scores)
        for arm in ARMS:
            model = models[seed, arm]
            for role in ('fit', 'held'):
                pairs = data[role+'_pairs']; directed = np.r_[pairs, pairs[:, ::-1]]
                prediction = predict(model, state, context, directed)
                tag = dict(seed=seed, arm=arm, cohort=role, fold=-1)
                rr, detail = evaluate(meta, evidence, scores, pairs, prediction, model, tag); rows.extend(rr)
                detail.to_csv(results/f'{role}_{arm}_{seed}.csv', index=False)
                if role == 'held':
                    tr, ids, pv = teacher_selection(meta, scores, directed, prediction, tag)
                    teacher_rows.extend(tr)
                    np.savez_compressed(results/f'teacher_{arm}_{seed}.npz', obs_ids=ids, **pv)
    frame = pd.DataFrame(rows); frame.to_csv(results/'response_results.csv', index=False)
    summary = frame.groupby(['cohort', 'task', 'arm'])[['MSE', 'skill_vs_constant']].mean()
    summary.to_csv(results/'response_summary.csv')
    pd.DataFrame(teacher_rows).to_csv(results/'teacher_selection.csv', index=False)
    (results/'COMPLETE.json').write_text(json.dumps(dict(completed=True, neural_fits=0,
        response_fits=len(ARMS)*2*4, synthetic_checks_passed=True)), encoding='utf-8')
    print(summary.to_string(), flush=True)
    print('TEACHER', pd.DataFrame(teacher_rows).groupby(['arm', 'selector'])[['BA', 'OA', 'rescue', 'harm']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
