"""Class-mass-preserving emphasis on the adapted base's held-out errors."""

import numpy as np
from sklearn.linear_model import Ridge


def error_weights(labels, base_scores):
    difficulty = 1. + (base_scores.argmax(1) != labels)
    mass = np.bincount(labels, weights=difficulty, minlength=8)
    return len(labels) * difficulty / (np.count_nonzero(mass) * mass[labels])


def fit_error_focused(features, labels, source_prior, base_scores):
    weights = error_weights(labels, base_scores)
    target = 2 * np.eye(8)[labels] - 1 - source_prior
    model = Ridge(alpha=8., fit_intercept=False, solver='cholesky')
    model.fit(features, target, sample_weight=weights)
    return model.coef_.T, weights
