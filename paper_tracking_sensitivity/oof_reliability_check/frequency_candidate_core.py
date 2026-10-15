"""Class-balanced support comparisons retaining band and local-position evidence."""

import numpy as np


def band_vectors(embedding):
    bands = np.asarray(embedding, np.float32).reshape(-1, 4, 64)
    return bands/np.maximum(np.linalg.norm(bands, axis=2, keepdims=True), 1e-12)


def class_evidence(query_embedding, support_embedding, query_conditions,
                   support_conditions, support_labels, classes=8):
    q, s = band_vectors(query_embedding), band_vectors(support_embedding)
    # Original flatten order is channel, y, x, not position, channel.
    qt = q.reshape(-1, 4, 16, 4).transpose(0, 1, 3, 2)
    st = s.reshape(-1, 4, 16, 4).transpose(0, 1, 3, 2)
    qt = qt/np.maximum(np.linalg.norm(qt, axis=3, keepdims=True), 1e-12)
    st = st/np.maximum(np.linalg.norm(st, axis=3, keepdims=True), 1e-12)
    sy = np.asarray(support_labels)
    counts = np.bincount(sy, minlength=classes)
    out = np.zeros((len(q), classes, 12), np.float32)
    for lo in range(0, len(q), 512):
        hi = min(lo+512, len(q))
        condition = np.sum((query_conditions[lo:hi, None]-support_conditions[None])**2, axis=2)/.5
        for band in range(4):
            global_sim = q[lo:hi, band]@s[:, band].T
            local = (qt[lo:hi, band].reshape(-1, 16)@st[:, band].reshape(-1, 16).T)
            local = local.reshape(hi-lo, 4, len(s), 4).transpose(0, 2, 1, 3)
            aligned = np.diagonal(local, axis1=2, axis2=3).mean(2)
            matched = .5*(local.max(2).mean(2)+local.max(3).mean(2))
            for cl in np.flatnonzero(counts):
                mask = sy == cl
                logits = global_sim[:, mask]/.2-condition[:, mask]
                attention = np.exp(logits-logits.max(1, keepdims=True))
                attention /= attention.sum(1, keepdims=True)
                for offset, value in [(0, global_sim), (4, aligned), (8, matched)]:
                    out[lo:hi, cl, offset+band] = np.sum(attention*value[:, mask], axis=1)
    return out, counts


def candidate_difference(evidence, counts, old_pred, new_pred):
    rr = np.arange(len(evidence))
    difference = evidence[rr, new_pred]-evidence[rr, old_pred]
    eligible = (old_pred != new_pred) & (counts[old_pred] >= 2) & (counts[new_pred] >= 2)
    return difference, eligible


def training_pairs(evidence, counts, old_scores, new_scores, labels):
    rr = np.arange(len(evidence))
    score = np.maximum(old_scores, new_scores).copy()
    score[:, counts < 2] = -np.inf
    score[rr, labels] = -np.inf
    negative = score.argmax(1)
    valid = (counts[labels] >= 2) & np.isfinite(score[rr, negative])
    return evidence[rr, labels]-evidence[rr, negative], valid, negative


def fit_ranker(features, weights):
    from sklearn.linear_model import Ridge
    if not len(features):
        return None
    scale = np.maximum(np.sqrt(np.average(features**2, axis=0, weights=weights)), 1e-6)
    # With no intercept, including mirrored (-x,-1) pairs has the same optimum
    # when their combined weight equals the original pair's weight.
    model = Ridge(alpha=10., fit_intercept=False).fit(features/scale, np.ones(len(features)),
        sample_weight=weights/weights.mean())
    return dict(scale=scale, coef=model.coef_)


def rank_margin(model, features):
    return np.zeros(len(features)) if model is None else (features/model['scale'])@model['coef']


def candidate_scores(old, new, margin, eligible):
    return np.where((eligible & (margin < 0))[:, None], old, new)
