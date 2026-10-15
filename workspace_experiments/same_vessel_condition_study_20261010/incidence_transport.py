"""Anchored feature-change prediction using observed incidence, never B features."""

import itertools
import numpy as np
import pandas as pd

ARMS = ('no_condition', 'angle_shared', 'angle_context', 'angle_sham')


def known_condition(meta):
    return (np.isfinite(meta.incidence_deg.to_numpy(float)) &
            meta.pass_name.isin(['Ascending', 'Descending']).to_numpy() &
            meta.platform.notna().to_numpy())


def matched_pairs(meta, ids):
    """Global row indices, no repeated product, fixed known orbit/platform/port."""
    sub = meta.iloc[ids]
    pairs = []
    for _, group in sub.loc[known_condition(sub)].groupby(['mmsi', 'port', 'platform', 'pass_name'], sort=True):
        group = group.sort_values(['acq_time', 'obs_id'])
        for a, b in itertools.combinations(group.index, 2):
            if meta.loc[a, 'product'] != meta.loc[b, 'product']:
                pairs.append((a, b))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def pair_weights(meta, pairs):
    table = meta.iloc[pairs[:, 0]][['mmsi', 'class_id']].reset_index(drop=True)
    weight = np.zeros(len(pairs))
    for _, cls in table.groupby('class_id', sort=True):
        for _, ship in cls.groupby('mmsi', sort=True):
            weight[ship.index] = 1/(table.class_id.nunique()*cls.mmsi.nunique()*len(ship))
    return weight


def shuffled_angles(meta, fitting, seed=20261011):
    """Training-only within-vessel/port/orbit permutation, not global shuffle."""
    result = meta.incidence_deg.to_numpy(float).copy()
    sub = meta.iloc[fitting]
    rng = np.random.default_rng(seed)
    for _, group in sub.loc[known_condition(sub)].groupby(['mmsi', 'port', 'platform', 'pass_name'], sort=True):
        ids = group.index.to_numpy()
        if len(ids) > 1:
            result[ids] = result[rng.permutation(ids)]
    return result


def design(context_a, angle_a, angle_b, arm):
    context_a = np.asarray(context_a, np.float64)
    da = ((np.asarray(angle_b)-np.asarray(angle_a))/5.).reshape(-1, 1)
    base = np.c_[np.ones(len(context_a)), context_a]
    if arm == 'no_condition':
        return base
    if arm == 'angle_shared':
        return da
    return base*da


def fit_response(z, context, meta, fitting, arm, pairs=None):
    """Only A's frozen class context and A/B scalar angles predict B-A."""
    pairs = matched_pairs(meta, fitting) if pairs is None else np.asarray(pairs)
    if not len(pairs):
        raise ValueError('No matched source pairs for incidence response')
    if not np.isin(pairs, fitting).all():
        raise ValueError('Response fitting pairs leave the fitting cohort')
    directed = np.r_[pairs, pairs[:, ::-1]]
    a, b = directed.T
    angles = shuffled_angles(meta, fitting) if arm == 'angle_sham' else meta.incidence_deg.to_numpy(float)
    x = design(context[a], angles[a], angles[b], arm)
    target = z[b]-z[a]
    weight = np.tile(pair_weights(meta, pairs)/2, 2)
    gram = x.T@(weight[:, None]*x)
    ridge = max(.05*float(np.trace(gram))/x.shape[1], 1e-10)
    coefficient = np.linalg.solve(gram+ridge*np.eye(x.shape[1]), x.T@(weight[:, None]*target))
    valid_fit = np.asarray(fitting)[known_condition(meta.iloc[fitting])]
    observed = meta.incidence_deg.to_numpy(float)
    return dict(coefficient=coefficient, arm=arm, ridge=ridge, pairs=pairs,
        fitting=np.asarray(fitting), angle_reference=float(np.median(observed[valid_fit])),
        angle_min=float(observed[valid_fit].min()), angle_max=float(observed[valid_fit].max()),
        angle_delta_cap=float(np.quantile(np.abs(observed[pairs[:, 1]]-observed[pairs[:, 0]]), .95)),
        gram_rank=int(np.linalg.matrix_rank(gram)))


def predict_delta(context_a, angle_a, angle_b, model):
    return design(context_a, angle_a, angle_b, str(model['arm']))@model['coefficient']


def canonical_delta(context, meta, model):
    """Single-observation step toward the SOURCE reference; missing is unchanged."""
    angle = meta.incidence_deg.to_numpy(float)
    valid = known_condition(meta) & (angle >= model['angle_min']) & (angle <= model['angle_max'])
    shift = np.clip(model['angle_reference']-angle, -model['angle_delta_cap'], model['angle_delta_cap'])
    safe_a = np.where(valid, angle, model['angle_reference'])
    safe_b = np.where(valid, angle+shift, model['angle_reference'])
    result = predict_delta(context, safe_a, safe_b, model)
    result[~valid] = 0
    return result, valid


def response_metrics(z, context, meta, pairs, model):
    if not len(pairs):
        return dict(pairs=0, ships=0, explained_percent=np.nan)
    directed = np.r_[pairs, pairs[:, ::-1]]
    a, b = directed.T
    angle = meta.incidence_deg.to_numpy(float)
    predicted = predict_delta(context[a], angle[a], angle[b], model)
    target = z[b]-z[a]
    weight = np.tile(pair_weights(meta, pairs)/2, 2)
    zero = float(weight@np.square(target).mean(1))
    error = float(weight@np.square(target-predicted).mean(1))
    amplitude = float(weight@np.square(predicted).mean(1))
    alignment = float(weight@(predicted*target).mean(1))
    return dict(pairs=len(pairs), ships=int(meta.iloc[pairs[:, 0]].mmsi.nunique()),
        classes=int(meta.iloc[pairs[:, 0]].class_id.nunique()), zero_mse=zero, mse=error,
        explained_percent=100*(1-error/max(zero, 1e-12)), prediction_energy=amplitude,
        signed_alignment=alignment, diagnostic_optimal_amplitude=alignment/max(amplitude, 1e-12))
