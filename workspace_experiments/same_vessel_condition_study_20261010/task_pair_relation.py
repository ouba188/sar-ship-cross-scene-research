"""Task-supervised, bounded calibration of signed support influences."""

import numpy as np
from incidence_transport import known_condition
from paired_evidence_factors import support_operator

ARMS = ('constant', 'evidence', 'condition', 'sham')
AMPLITUDE = .25


def observation_coordinates(features, fitting, normalization=None):
    energy = np.full(len(features), np.nan)
    available = np.isfinite(features[:, :3]).all(1)
    energy[available] = np.logaddexp.reduce(features[available, :3], axis=1)
    raw = np.c_[features[:, 5], energy, features[:, 4]]
    if normalization is None:
        normalization = dict(center=raw[fitting].mean(0), scale=np.maximum(raw[fitting].std(0), 1e-6))
    return (raw-normalization['center'])/normalization['scale'], normalization


def eligible(meta, supports, queries, limits):
    angles = meta.incidence_deg.to_numpy(float)
    good = known_condition(meta) & (angles >= limits[0]) & (angles <= limits[1])
    valid = good[queries, None] & good[supports][None, :]
    for field in ('platform', 'pass_name'):
        values = meta[field].to_numpy()
        valid &= values[queries, None] == values[supports][None, :]
    return valid


def pair_design(coordinates, angles, supports, queries, arm):
    a, b = coordinates[queries, None, :], coordinates[supports][None, :, :]
    distance = np.square(a-b)
    if arm == 'constant':
        return np.ones((*distance.shape[:2], 1))
    mid = (a+b)/2
    if arm == 'evidence':
        extra = np.stack([mid[..., 0], mid[..., 1], distance[..., 0]*mid[..., 0],
                          distance[..., 0]*mid[..., 1]], axis=-1)
    else:
        aq, ass = angles[queries, None], angles[supports][None, :]
        gap, mean = np.abs(aq-ass)/5, ((aq+ass)/2-38)/5
        extra = np.stack([gap, mean, distance[..., 0]*gap, distance[..., 0]*mean], axis=-1)
    return np.concatenate([distance, extra], axis=-1)


def feature_names(arm):
    if arm == 'constant':
        return ['intercept']
    shared = ['ratio_squared_difference', 'energy_squared_difference', 'middle_share_squared_difference']
    extra = (['mean_ratio', 'mean_energy', 'ratio_difference_x_mean_ratio', 'ratio_difference_x_mean_energy']
             if arm == 'evidence' else
             ['angle_gap', 'mean_angle', 'ratio_difference_x_angle_gap', 'ratio_difference_x_mean_angle'])
    return ['intercept']+shared+extra


def normalize_design(raw, valid, center, scale):
    if raw.shape[-1] == 1:
        return valid[..., None].astype(float)
    x = np.clip((raw-center)/scale, -5, 5)
    x = np.concatenate([np.ones((*x.shape[:2], 1)), x], axis=-1)
    # Missing metadata must contribute exactly zero, including the intercept.
    return np.where(valid[..., None], x, 0.)


def episode_base(prior, kernel, supports, queries, support_labels):
    op, present = support_operator(kernel[supports], kernel[queries], support_labels)
    residual = (2*np.eye(8)[support_labels]-1-prior[supports])*present[None, :]
    return prior[queries]+op@residual, op, residual


def predict(base, op, residual, design, coefficient):
    change = AMPLITUDE*np.tanh(design@coefficient)
    return base+(op*change)@residual, 1+change


def balanced_ce(scores, labels):
    shifted = scores-scores.max(1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)), labels]
    return float(np.mean([loss[labels == c].mean() for c in np.unique(labels)]))
