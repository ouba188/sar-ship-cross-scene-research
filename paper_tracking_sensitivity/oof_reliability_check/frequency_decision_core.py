"""Decision-conditioned endpoint utility; no target-query labels at inference."""

import numpy as np


def crossing_threshold(old, new):
    winner = new.argmax(1)
    rr = np.arange(len(old))
    change = new-old
    gap = old-old[rr, winner, None]
    slope = change[rr, winner, None]-change
    ratio = np.divide(gap, slope, out=np.zeros_like(gap, dtype=float), where=(gap > 0) & (slope > 0))
    return np.clip(ratio.max(1), 0., 1.)


def probabilities(score):
    p = np.exp(score-score.max(1, keepdims=True))
    return p/p.sum(1, keepdims=True)


def point_ce(score, labels):
    z = score-score.max(1, keepdims=True)
    return np.log(np.exp(z).sum(1))-z[np.arange(len(labels)), labels]


def decision_features(old, new, similarity, support_labels, loo_old, loo_new):
    a, b = old.argmax(1), new.argmax(1)
    rr = np.arange(len(old))
    p0, p1 = probabilities(old), probabilities(new)
    c0, c1 = old-old.mean(1, keepdims=True), new-new.mean(1, keepdims=True)
    gap0, gap1 = old[rr, a]-old[rr, b], new[rr, b]-new[rr, a]
    top0, top1 = np.sort(old, axis=1)[:, -2:], np.sort(new, axis=1)[:, -2:]
    columns = [top0[:, 1]-top0[:, 0], top1[:, 1]-top1[:, 0], gap0, gap1,
        crossing_threshold(old, new), np.sqrt(np.mean((c1-c0)**2, axis=1)),
        -np.sum(p0*np.log(np.maximum(p0, 1e-12)), axis=1),
        -np.sum(p1*np.log(np.maximum(p1, 1e-12)), axis=1),
        p0[rr, a], p1[rr, b], p1[rr, a]-p0[rr, a], p1[rr, b]-p0[rr, b],
        np.sqrt(np.mean(c0**2, axis=1)), np.sqrt(np.mean(c1**2, axis=1))]
    names = ['old_margin', 'new_margin', 'old_pair_gap', 'new_pair_gap', 'crossing_gamma',
        'centered_change_rms', 'old_entropy', 'new_entropy', 'old_top_probability',
        'new_top_probability', 'old_class_probability_change', 'new_class_probability_change',
        'old_score_scale', 'new_score_scale']
    for prefix, pred in [('old', a), ('new', b)]:
        for cl in range(old.shape[1]):
            columns.append((pred == cl).astype(float))
            names.append(f'{prefix}_class_{cl}')
    width = len(columns)
    sy = np.asarray(support_labels)
    lo, ln = loo_old.argmax(1), loo_new.argmax(1)
    utility = (ln == sy).astype(float)-(lo == sy)
    ce_utility = point_ce(loo_old, sy)-point_ce(loo_new, sy)
    s = np.maximum(similarity, 0.)
    top = np.argsort(-s, axis=1, kind='stable')[:, :min(8, s.shape[1])]
    w = np.take_along_axis(s, top, axis=1)
    mass = w.sum(1)
    norm = w/np.maximum(mass[:, None], 1e-12)
    old_match, new_match = sy[top] == a[:, None], sy[top] == b[:, None]
    pair = (lo[top] == a[:, None]) & (ln[top] == b[:, None])
    pair_mass = np.sum(w*pair, axis=1)
    positive = np.sum(w*np.maximum(ce_utility[top], 0), axis=1)
    absolute = np.sum(w*np.abs(ce_utility[top]), axis=1)
    support_columns = [mass/8., s.max(1), np.sum(norm*old_match, axis=1),
        np.sum(norm*new_match, axis=1), np.mean(sy[None, :] == a[:, None], axis=1),
        np.mean(sy[None, :] == b[:, None], axis=1), np.sum(norm*utility[top], axis=1),
        np.sum(norm*ce_utility[top], axis=1), positive/np.maximum(absolute, 1e-12),
        pair_mass/np.maximum(mass, 1e-12),
        np.sum(w*pair*utility[top], axis=1)/np.maximum(pair_mass, 1e-12),
        np.sum(norm*(utility[top] > 0), axis=1), np.sum(norm*(utility[top] < 0), axis=1)]
    columns.extend(support_columns)
    names.extend(['neighbor_mean_similarity', 'max_similarity', 'old_class_neighbor_mass',
        'new_class_neighbor_mass', 'old_class_support_fraction', 'new_class_support_fraction',
        'neighbor_hard_utility', 'neighbor_ce_utility', 'neighbor_positive_ce_ratio',
        'same_transition_mass', 'same_transition_hard_utility', 'neighbor_rescue_mass', 'neighbor_harm_mass'])
    return np.column_stack(columns), names, width


def endpoint_scores(old, new, benefit):
    reject = (old.argmax(1) != new.argmax(1)) & (np.asarray(benefit) < 0)
    return np.where(reject[:, None], old, new)


def fit_benefit(features, utility, weights):
    from sklearn.ensemble import HistGradientBoostingRegressor
    if not len(features):
        return None
    model = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, max_depth=3,
        min_samples_leaf=20, learning_rate=.05, l2_regularization=10.,
        early_stopping=False, random_state=20261009)
    model.fit(features, utility, sample_weight=weights/np.mean(weights))
    return model


def predict_benefit(model, features):
    return np.zeros(len(features)) if model is None else model.predict(features)
