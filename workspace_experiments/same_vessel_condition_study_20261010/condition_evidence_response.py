"""Small anchored response models: condition information versus anchor-only fits."""

import numpy as np
from incidence_transport import pair_weights, shuffled_angles

ARMS = ('anchor', 'capacity', 'condition', 'sham')


def design(anchor, context, angle_a, angle_b, mode, arm):
    u, v = np.asarray(anchor), np.asarray(context)
    a, b = (np.asarray(angle_a)-38.)/5., (np.asarray(angle_b)-38.)/5.
    d, mid = b-a, (a+b)/2
    columns = [u, v]
    if arm == 'capacity':
        columns += [u*u, v*v, u*v, u*u*u]
    elif arm in ('condition', 'sham'):
        if mode == 'signed':
            columns += [d, d*mid, d*u, d*v]
        elif mode in ('magnitude', 'deterioration'):
            columns += [np.abs(d), d*d, mid, np.abs(d)*u]
        elif mode == 'error':
            columns += [a, b, b*b, d*v]
        else:
            raise ValueError(mode)
    elif arm != 'anchor':
        raise ValueError(arm)
    return np.column_stack(columns)


def response_target(values, directed, mode):
    a, b = directed.T
    if mode == 'error':
        return values[b]
    delta = values[b]-values[a]
    if mode == 'signed':
        return delta
    if mode == 'magnitude':
        return np.abs(delta)
    if mode == 'deterioration':
        return np.maximum(-delta, 0.)
    raise ValueError(mode)


def clip_prediction(prediction, mode):
    if mode == 'error':
        return np.clip(prediction, 0., 1.)
    if mode != 'signed':
        return np.maximum(prediction, 0.)
    return prediction


def fit_model(meta, fitting, pairs, anchor, context, values, mode, arm):
    if not len(pairs) or not np.isin(pairs, fitting).all():
        raise ValueError('Training pairs must be nonempty and within fitting rows')
    directed = np.concatenate([pairs, pairs[:, ::-1]])
    a, b = directed.T
    weight = np.tile(pair_weights(meta, pairs)/2, 2)
    raw = np.column_stack([anchor[a], context[a]])
    center = weight@raw
    scale = np.sqrt(weight@((raw-center)**2)).clip(1e-6)
    norm = (raw-center)/scale
    angle = (shuffled_angles(meta, fitting) if arm == 'sham'
             else meta.incidence_deg.to_numpy(float))
    x = design(norm[:, 0], norm[:, 1], angle[a], angle[b], mode, arm)
    xcenter = weight@x
    xscale = np.sqrt(weight@((x-xcenter)**2)).clip(1e-6)
    x = np.column_stack([np.ones(len(x)), (x-xcenter)/xscale])
    y = response_target(values, directed, mode)
    penalty = np.eye(x.shape[1])*.05
    penalty[0, 0] = 0.
    coefficient = np.linalg.solve(x.T@(weight[:, None]*x)+penalty, x.T@(weight*y))
    return dict(coefficient=coefficient, center=center, scale=scale,
                xcenter=xcenter, xscale=xscale, arm=arm, mode=mode,
                constant=float(weight@y), fit_obs_ids=np.asarray(fitting), pairs=pairs)


def predict_model(anchor_a, context_a, angle_a, angle_b, model):
    raw = np.column_stack([anchor_a, context_a])
    norm = (raw-model['center'])/model['scale']
    mode, arm = str(model['mode']), str(model['arm'])
    x = design(norm[:, 0], norm[:, 1], angle_a, angle_b, mode, arm)
    x = np.column_stack([np.ones(len(x)), (x-model['xcenter'])/model['xscale']])
    return clip_prediction(x@model['coefficient'], mode)


def evaluate(meta, pairs, truth, prediction, constant):
    if not len(pairs):
        return dict(pairs=0, ships=0, classes=0, mse=np.nan, constant_mse=np.nan,
                    skill_vs_constant=np.nan, target_mean=np.nan)
    weight = np.tile(pair_weights(meta, pairs)/2, 2)
    mse = float(weight@((truth-prediction)**2))
    base = float(weight@((truth-constant)**2))
    return dict(pairs=len(pairs), ships=meta.iloc[pairs[:, 0]].mmsi.nunique(),
                classes=meta.iloc[pairs[:, 0]].class_id.nunique(), mse=mse,
                constant_mse=base, skill_vs_constant=100*(1-mse/max(base, 1e-12)),
                target_mean=float(weight@truth))
