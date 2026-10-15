"""Calibrate existing support influence using predicted spatial-change tolerance."""

import numpy as np
from incidence_transport import known_condition, pair_weights
from condition_evidence_response import predict_model
from paired_evidence_factors import support_operator, adapted_scores

ARMS = ('fixed', 'anchor', 'capacity', 'condition', 'sham', 'gated')


def incremental_factors(factors):
    """Remove the generic center-based attenuation; retain each model's increment."""
    return {arm: 1+(value-factors['anchor']) for arm, value in factors.items()}


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind='stable')
    return float(np.asarray(values)[order][np.searchsorted(np.cumsum(np.asarray(weights)[order]), quantile, side='left')])


def source_limits(meta, fitting, pairs, ratio):
    a, b = pairs.T
    changes = np.abs(ratio[b]-ratio[a])
    w = pair_weights(meta, pairs)
    angles = meta.incidence_deg.to_numpy(float)
    valid = fitting[known_condition(meta.iloc[fitting])]
    return dict(floor=max(weighted_quantile(changes, w, .25), 1e-6),
                fixed=max(weighted_quantile(changes, w, .5), 1e-6),
                angle_min=float(angles[valid].min()), angle_max=float(angles[valid].max()),
                max_gap=float(np.max(np.abs(angles[b]-angles[a]))))


def comparable_pairs(meta, supports, queries, limits):
    qs = np.repeat(queries, len(supports))
    ss = np.tile(supports, len(queries))
    valid_obs = known_condition(meta)
    angles = meta.incidence_deg.to_numpy(float)
    valid_obs &= (angles >= limits['angle_min']) & (angles <= limits['angle_max'])
    valid = valid_obs[qs] & valid_obs[ss]
    for key in ('port', 'platform', 'pass_name'):
        values = meta[key].to_numpy()
        valid &= values[qs] == values[ss]
    valid &= np.abs(angles[qs]-angles[ss]) <= limits['max_gap']
    return qs, ss, valid


def multipliers(meta, supports, queries, ratio, context, models, gate, limits):
    from condition_applicability import gate_inputs, predict_gate
    qs, ss, valid = comparable_pairs(meta, supports, queries, limits)
    shape = (len(queries), len(supports))
    result = {arm: np.ones(len(qs)) for arm in ARMS}
    details = dict(valid=valid.reshape(shape), predicted={}, gate=np.zeros(shape))
    if not valid.any():
        return {k: v.reshape(shape) for k, v in result.items()}, details
    q, s = qs[valid], ss[valid]
    angle = meta.incidence_deg.to_numpy(float)
    forecast = {}
    for arm in ('anchor', 'capacity', 'condition', 'sham'):
        forward = predict_model(ratio[q], context[q], angle[q], angle[s], models[arm])
        reverse = predict_model(ratio[s], context[s], angle[s], angle[q], models[arm])
        forecast[arm] = (forward+reverse)/2
    gq = predict_gate(gate_inputs(ratio[q], context[q], angle[q], angle[s]), gate)
    gs = predict_gate(gate_inputs(ratio[s], context[s], angle[s], angle[q]), gate)
    # Mix the directed forecasts first, then symmetrize the pair comparison.
    af = predict_model(ratio[q], context[q], angle[q], angle[s], models['anchor'])
    ar = predict_model(ratio[s], context[s], angle[s], angle[q], models['anchor'])
    cf = predict_model(ratio[q], context[q], angle[q], angle[s], models['condition'])
    cr = predict_model(ratio[s], context[s], angle[s], angle[q], models['condition'])
    forecast['gated'] = ((1-gq)*af+gq*cf+(1-gs)*ar+gs*cr)/2
    forecast['fixed'] = np.full(len(q), limits['fixed'])
    delta = ratio[q]-ratio[s]
    for arm in ARMS:
        tolerance = np.maximum(forecast[arm], limits['floor'])
        result[arm][valid] = np.exp(-.5*np.square(delta/tolerance))
        details['predicted'][arm] = forecast[arm]
    details['gate'].ravel()[valid] = (gq+gs)/2
    return {k: v.reshape(shape) for k, v in result.items()}, details


def score_episode(prior, kernel, supports, queries, support_labels, factors):
    op, present = support_operator(kernel[supports], kernel[queries], support_labels)
    residual = (2*np.eye(8)[support_labels]-1-prior[supports])*present[None, :]
    base = adapted_scores(prior[supports], prior[queries], op, present, support_labels)
    deltas = {arm: (op*(factor-1))@residual for arm, factor in factors.items()}
    return base, deltas, op, residual


def metrics(scores, labels):
    prediction = scores.argmax(1)
    recalls = [np.mean(prediction[labels == c] == c) for c in np.unique(labels)]
    counts = np.bincount(labels, minlength=8)
    w = len(labels)/((counts > 0).sum()*counts[labels])
    return dict(OA=float(100*np.mean(prediction == labels)), BA=float(100*np.mean(recalls)),
                MSE=float(np.mean(w[:, None]*np.square(scores-(2*np.eye(8)[labels]-1)))))


def decision_accounting(base, result, op, residual, multiplier, fixed, labels, valid):
    rival_scores = base.copy()
    rival_scores[np.arange(len(labels)), labels] = -np.inf
    rivals = rival_scores.argmax(1)
    contrast = residual[:, labels].T-residual[:, rivals].T
    contribution = op*contrast
    delta = (multiplier-1)*contribution
    relative = (multiplier-fixed)*contribution
    correct_old, correct_new = base.argmax(1) == labels, result.argmax(1) == labels
    return dict(queries=len(labels), active_queries=int(valid.any(1).sum()),
                rescues=int((~correct_old & correct_new).sum()), harms=int((correct_old & ~correct_new).sum()),
                helpful_removed=float(-np.minimum(delta*(contribution > 0), 0).sum()/len(labels)),
                harmful_removed=float(np.maximum(delta*(contribution < 0), 0).sum()/len(labels)),
                helpful_added=float(np.maximum(delta*(contribution > 0), 0).sum()/len(labels)),
                harmful_added=float(-np.minimum(delta*(contribution < 0), 0).sum()/len(labels)),
                margin_delta=float(delta.sum()/len(labels)),
                helpful_retained_vs_fixed=float(relative[contribution > 0].sum()/len(labels)),
                harmful_retained_vs_fixed=float(relative[contribution < 0].sum()/len(labels)))
