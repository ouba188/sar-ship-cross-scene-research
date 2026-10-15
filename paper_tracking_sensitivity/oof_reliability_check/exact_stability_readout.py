"""Exact empirical class-conditional shift for frozen pair projections."""

import numpy as np
from scipy.stats import wasserstein_distance


def exact_pair_shift(source_z, source_y, source_weights, target_z, target_y,
                     c, d, source_variance):
    """Compare weighted source classes with uniformly weighted target classes.

    Inputs are frozen one-dimensional projections and aligned labels/weights.
    Only positive-weight source rows in each class participate. Both class
    distances use the same class-balanced source standard deviation, floored
    at 1e-8. No direction, threshold, or other parameter is fitted here.
    """
    source_z, source_y = np.asarray(source_z), np.asarray(source_y)
    source_weights = np.asarray(source_weights)
    target_z, target_y = np.asarray(target_z), np.asarray(target_y)
    sigma = max(float(np.sqrt(0.5 * np.sum(source_variance))), 1e-8)
    distances = []
    for label in (c, d):
        source_mask = (source_y == label) & (source_weights > 0)
        target_mask = target_y == label
        distance = wasserstein_distance(source_z[source_mask], target_z[target_mask],
                                        u_weights=source_weights[source_mask])
        distances.append(float(distance / sigma))
    return dict(class_c_w1=distances[0], class_d_w1=distances[1],
                conditional_w1=0.5 * sum(distances))
