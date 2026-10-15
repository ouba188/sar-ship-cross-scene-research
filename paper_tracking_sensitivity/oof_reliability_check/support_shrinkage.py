"""Support-only cross-validated scaling of anchored residual adaptation."""

import numpy as np

from residual_expert import K, fit_correction


def loo_residuals(features, labels, prior):
    """Refit each fold, including class weights, without its held-out label."""
    residuals = np.empty_like(prior, dtype=float)
    rows = np.arange(len(labels))
    for heldout in rows:
        train = rows != heldout
        coefficient = fit_correction(features[train], labels[train], prior[train])
        residuals[heldout] = features[heldout] @ coefficient
    return residuals


def select_strength(labels, prior, residuals):
    """Minimize balanced held-out squared loss along the residual direction."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels) / (len(counts) * counts[inverse])
    desired = 2 * np.eye(K)[labels] - 1 - prior
    numerator = float(np.sum(weights[:, None] * desired * residuals))
    denominator = float(np.sum(weights[:, None] * residuals**2))
    alpha = float(np.clip(numerator / denominator, 0., 1.)) if denominator > 0 else 0.
    return alpha, numerator, denominator
