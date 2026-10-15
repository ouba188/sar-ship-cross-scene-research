"""Target support-only hard switching, with strict leave-one-out refits."""

import numpy as np
from frequency_reliability_core import coefficients, balanced_weights


def loo_scores(gram, prior, labels):
    rows = np.arange(len(labels))
    out = np.empty_like(prior)
    for held in rows:
        keep = rows[rows != held]
        alpha = coefficients(gram[np.ix_(keep, keep)], labels[keep], prior[keep])
        out[held] = prior[held]+gram[held, keep]@alpha
    return out


def support_static(loo, labels):
    correct = loo.argmax(2) == labels[:, None]
    value = np.average(correct, axis=0, weights=balanced_weights(labels))
    return int(np.argmax(value))


def support_utility(similarity, loo_main, loo_band, labels):
    utility = (loo_band.argmax(1) == labels).astype(float)-(loo_main.argmax(1) == labels)
    positive = np.maximum(similarity, 0.)
    top = np.argsort(-positive, axis=1, kind='stable')[:, :8]
    weight = np.take_along_axis(positive, top, axis=1)*balanced_weights(labels)[top]
    return np.sum(weight*utility[top], axis=1)/np.maximum(weight.sum(1), 1e-12)


def local_choice(reference, candidates, utilities):
    value = np.array(utilities, copy=True)
    value[candidates.argmax(2) == reference.argmax(1)[:, None]] = -np.inf
    choice = np.argmax(np.c_[np.zeros(len(reference)), value], axis=1)
    all_scores = np.concatenate([reference[:, None], candidates], axis=1)
    return all_scores[np.arange(len(reference)), choice], choice
