"""Fixed-size source-only decision-supervised pair experiment; no target fitting."""

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from explicit_candidate_projection import identity_episode
from incidence_transport import known_condition, shuffled_angles
from condition_support_matching import metrics, decision_accounting
from task_pair_relation import (ARMS, AMPLITUDE, observation_coordinates, eligible, pair_design,
                                feature_names, normalize_design, episode_base, predict, balanced_ce)

BASE = Path(__file__).resolve().parent
OUT = BASE/'task_pair_relation_20261011'
PARENT = BASE/'incidence_transport_20261011'
SEEDS = (31415, 27182)
EPISODES = 180
STEPS = 360


def load(path):
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_episode(meta, bundle, s, q, coordinates, angles, sham, limits):
    if set(meta.iloc[s].mmsi) & set(meta.iloc[q].mmsi):
        raise ValueError('Support/query ships overlap')
    y = meta.class_id.to_numpy()
    base, op, residual = episode_base(bundle['prior'].astype(float), bundle['kernel'].astype(float), s, q, y[s])
    valid = eligible(meta, s, q, limits)
    designs = {arm: pair_design(coordinates, sham if arm == 'sham' else angles, s, q, arm) for arm in ARMS}
    if not all(np.isfinite(x[valid]).all() for x in designs.values()):
        raise ValueError('Missing image evidence on eligible edges')
    return dict(s=s, q=q, y=y[q], base=base, op=op, residual=residual, valid=valid, designs=designs)


def apply(ep, arm, model, state='selected'):
    x = normalize_design(ep['designs'][arm], ep['valid'], model['center'], model['scale'])
    return predict(ep['base'], ep['op'], ep['residual'], x, model[state])


def fit(bank, validation, arm):
    raw = np.concatenate([ep['designs'][arm][ep['valid']] for ep in bank])
    center, scale = raw.mean(0), np.maximum(raw.std(0), 1e-6)
    samples = []
    for ep in bank:
        x = normalize_design(ep['designs'][arm], ep['valid'], center, scale)
        y = ep['y']; counts = np.bincount(y, minlength=8)
        w = len(y)/((counts > 0).sum()*counts[y])
        samples.append(tuple(torch.as_tensor(a) for a in (x, ep['base'], ep['op'], ep['residual'], y, w)))
    theta = torch.zeros(samples[0][0].shape[-1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=.01)
    selected = theta.detach().numpy().copy()
    best_ba, chosen = -np.inf, 0
    trace = []
    def evaluate(step):
        nonlocal selected, best_ba, chosen
        model = dict(center=center, scale=scale, selected=theta.detach().numpy())
        scores = [apply(ep, arm, model)[0] for ep in validation]
        ba = np.mean([metrics(p, ep['y'])['BA'] for p, ep in zip(scores, validation)])
        ce = np.mean([balanced_ce(p, ep['y']) for p, ep in zip(scores, validation)])
        trace.append(dict(step=step, validation_BA=float(ba), validation_CE=float(ce)))
        if ba > best_ba+1e-10:
            selected, best_ba, chosen = theta.detach().numpy().copy(), float(ba), step
    evaluate(0)
    losses = []
    for step in range(1, STEPS+1):
        x, base, op, residual, y, w = samples[(step-1)%len(samples)]
        change = AMPLITUDE*torch.tanh(x@theta)
        score = base+(op*change)@residual
        ce = (w*F.cross_entropy(score, y, reduction='none')).mean()
        loss = ce+.001*theta.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(float(ce.detach()))
        if step%60 == 0:
            evaluate(step)
    model = dict(center=center, scale=scale, selected=selected, last=theta.detach().numpy().copy(),
                 chosen_step=np.array(chosen), validation_BA=np.array(best_ba))
    diagnostics = []
    for state in ('selected', 'last'):
        for name, episodes in [('train', bank), ('validation', validation)]:
            values = []
            for ep in episodes:
                score, _ = apply(ep, arm, model, state)
                values.append(dict(**metrics(score, ep['y']), CE=balanced_ce(score, ep['y']),
                                   baseline_CE=balanced_ce(ep['base'], ep['y']),
                                   baseline_BA=metrics(ep['base'], ep['y'])['BA']))
            diagnostics.append(dict(state=state, split=name, **pd.DataFrame(values).mean().to_dict()))
    return model, trace, diagnostics


def main():
    start = time.perf_counter()
    torch.set_num_threads(1)
    OUT.mkdir(exist_ok=False)
    meta = pd.read_csv(BASE/'observations.csv')
    cohorts = load(PARENT/'source_cohorts.npz')
    fit_ids, held = cohorts['fit_obs_ids'], cohorts['held_obs_ids']
    folds = load(BASE/'crossfit_evidence_20261011/source_folds.npz')
    np.testing.assert_array_equal(fit_ids, folds['fit_obs_ids'])
    e = load(BASE/'condition_evidence_20261011/evidence_features.npz')
    features = np.full((len(meta), 6), np.nan); features[e['obs_ids']] = e['features']
    coordinates, norm = observation_coordinates(features, fit_ids)
    angles = meta.incidence_deg.to_numpy(float)
    sham = shuffled_angles(meta, fit_ids)
    good = fit_ids[known_condition(meta.iloc[fit_ids])]
    limits = np.array([angles[good].min(), angles[good].max()])
    np.savez_compressed(OUT/'normalization.npz', **norm, limits=limits, fit_obs_ids=fit_ids,
                        sham_angles=sham[fit_ids], assignment=folds['assignment'])
    write_json(OUT/'PROTOCOL.json', dict(
        question='Does decision-supervised cross-vessel pair calibration exploit spatial evidence and real incidence better than matched image-only and within-ship angle-sham controls?',
        scope='Source-side eight-class cached R18-derived32 prior/case experiment. No target ports, new datasets, backbone training, or main-paper changes. Historically reused development cohorts, NOT independent confirmation.',
        mechanism='Original signed ridge support op and residual unchanged. score=base+(op*.25*tanh(standardized_pair_features@theta))@residual. This is bounded post-solve influence calibration, not probability attention, new PSD kernel, or causal factor separation.',
        arms='constant1 parameter; evidence/condition/sham8 each. Shared squared differences of center-border ratio, total branch energy, middle share. Image adds means and ratio-distance interactions; condition/sham replaces four extra terms with mean angle/gap and interactions.',
        training='180 fixed 64support/96query episodes within native excluded vessel fold, repeated twice. Each episode upstream trained without those vessels; normalization of new calibrator uses source-fit only. Fixed Adam .01, CE class-balanced within query episode, L2 .001, bound .25; no search.',
        selection='Step0/60/120/180/240/300/360 by mean BA on original3new-ship validation episodes, earliest ties. Last checkpoint is diagnostic only. No held labels used until selection frozen.',
        matching='Same platform/pass and both known source-range incidence for ALL arms; other edges exact multiplier1. Unlike old within-vessel volatility transfer, this relation is trained on DIFFERENT vessels including cross-port pairs; port is not an input. More coverage is a changed protocol, NOT attributed to the loss alone.',
        sham='Existing within-MMSI/port/platform/pass source angle permutation; real angle at validation/held. Preserves identity-level angle confounding. It tests sample correspondence only, not all angle association.',
        boundaries='CNN spatial energy proxies are not physical bands, ratio not vessel segmentation. Historical backbone exposure and reused upstream-selected validation persist. No causal identification claim.',
        fit_observations=len(fit_ids), held_observations=len(held),
        shuffled_known_rows=int(np.sum(np.isfinite(angles[fit_ids]) & (sham[fit_ids] != angles[fit_ids])))))
    v = load(BASE/'crossfit_evidence_20261011/validation_episodes.npz')
    rows, traces, diagnostics, accounting, classes, coverage, choices, coefs = [], [], [], [], [], [], [], []
    for seed in SEEDS:
        bundles = {name: load(PARENT/f'{name}_{seed}_inputs.npz') for name in ('full', 'fold0', 'fold1', 'fold2')}
        bank, saved_episodes = [], {}
        for j in range(EPISODES):
            fold = j%3; ids = fit_ids[folds['assignment'] == fold]
            bundle = bundles[f'fold{fold}']
            np.testing.assert_array_equal(np.sort(ids), np.sort(bundle['excluded_obs_ids']))
            if set(meta.iloc[ids].mmsi) & set(meta.iloc[bundle['fit_obs_ids']].mmsi):
                raise ValueError('OOF upstream identity leakage')
            s, q = identity_episode(meta.iloc[ids].reset_index(drop=True), np.random.default_rng(seed+70000+j))
            s, q = ids[s], ids[q]
            bank.append(make_episode(meta, bundle, s, q, coordinates, angles, sham, limits))
            saved_episodes[f'{j}_support'], saved_episodes[f'{j}_query'] = s, q
        np.savez_compressed(OUT/f'{seed}_training_episodes.npz', **saved_episodes)
        validation = [make_episode(meta, bundles['full'], v[f'val_new_ship_{j}_support'],
                      v[f'val_new_ship_{j}_query'], coordinates, angles, angles, limits) for j in range(3)]
        models = {}
        for arm in ARMS:
            model, trace, diag = fit(bank, validation, arm)
            models[arm] = model
            np.savez_compressed(OUT/f'{seed}_{arm}_model.npz', **model)
            traces.extend(dict(seed=seed, arm=arm, **x) for x in trace)
            diagnostics.extend(dict(seed=seed, arm=arm, **x) for x in diag)
            choices.append(dict(seed=seed, arm=arm, step=int(model['chosen_step']), BA=float(model['validation_BA'])))
            for state in ('selected', 'last'):
                coefs.extend(dict(seed=seed, arm=arm, state=state, term=n, coefficient=float(c))
                             for n, c in zip(feature_names(arm), model[state]))
            print('FITTED', seed, arm, choices[-1], flush=True)
        write_json(OUT/f'SELECTION_FROZEN_{seed}.json', dict(choices=choices[-4:], held_metrics_read=False))
        for draw in range(3):
            old = load(PARENT/f'{seed}_source_{draw}_predictions.npz')
            s, q = old['support_obs_ids'], old['query_obs_ids']
            if not np.isin(np.r_[s, q], held).all():
                raise ValueError('Held episode membership changed')
            ep = make_episode(meta, bundles['full'], s, q, coordinates, angles, angles, limits)
            np.testing.assert_array_equal(ep['base'].argmax(1), old['baseline'].argmax(1))
            saved = dict(support_obs_ids=s, query_obs_ids=q, baseline=ep['base'], valid=ep['valid'])
            rows.append(dict(seed=seed, draw=draw, arm='baseline', state='selected',
                             **metrics(ep['base'], ep['y']), CE=balanced_ce(ep['base'], ep['y'])))
            for arm in ARMS:
                for state in ('selected', 'last'):
                    score, multiplier = apply(ep, arm, models[arm], state)
                    saved[f'{arm}_{state}'], saved[f'{arm}_{state}_multiplier'] = score, multiplier
                    rows.append(dict(seed=seed, draw=draw, arm=arm, state=state,
                                     **metrics(score, ep['y']), CE=balanced_ce(score, ep['y'])))
                    accounting.append(dict(seed=seed, draw=draw, arm=arm, state=state,
                        **decision_accounting(ep['base'], score, ep['op'], ep['residual'], multiplier,
                                              np.ones_like(multiplier), ep['y'], ep['valid'])))
                    for c in np.unique(ep['y']):
                        use = ep['y'] == c
                        classes.append(dict(seed=seed, draw=draw, arm=arm, state=state, class_id=int(c),
                            n=int(use.sum()), baseline_recall=float(100*np.mean(ep['base'][use].argmax(1) == c)),
                            recall=float(100*np.mean(score[use].argmax(1) == c))))
            np.savez_compressed(OUT/f'{seed}_source_{draw}_predictions.npz', **saved)
            coverage.append(dict(seed=seed, draw=draw, queries=len(q), edges=ep['valid'].size,
                                 eligible_edges=int(ep['valid'].sum()), active_queries=int(ep['valid'].any(1).sum())))
        del bank, bundles
    for name, data in [('results', rows), ('trace', traces), ('diagnostics', diagnostics), ('accounting', accounting),
                       ('classes', classes), ('coverage', coverage), ('selection', choices), ('coefficients', coefs)]:
        pd.DataFrame(data).to_csv(OUT/f'{name}.csv', index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['arm', 'state'])[['OA', 'BA', 'MSE', 'CE']].mean()
    summary.to_csv(OUT/'summary.csv')
    frame.groupby(['arm', 'state', 'seed'])[['OA', 'BA', 'MSE', 'CE']].mean().to_csv(OUT/'seed_summary.csv')
    write_json(OUT/'COMPLETE.json', dict(seconds=time.perf_counter()-start, calibrator_fits=8, backbone_fits=0,
                                        held_episodes=6, target_evaluations=0, main_model_modified=False))
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
