"""Continuous support evidence; no query labels or learned parameters."""

import numpy as np
from scipy.special import expit
from frequency_reliability_core import balanced_weights


def margin_evidence(scores, labels, temperature=.1):
    other = np.array(scores, dtype=float, copy=True)
    rows = np.arange(len(labels))
    other[rows, labels] = -np.inf
    margin = scores[rows, labels] - other.max(axis=1)
    return expit(margin / temperature)


def support_utilities(similarity, loo_main, loo_band, labels):
    main_ok, band_ok = loo_main.argmax(1) == labels, loo_band.argmax(1) == labels
    hard = band_ok.astype(float) - main_ok
    smooth = margin_evidence(loo_band, labels) - margin_evidence(loo_main, labels)
    positive = np.maximum(similarity, 0.)
    top = np.argsort(-positive, axis=1, kind='stable')[:, :8]
    weight = np.take_along_axis(positive, top, axis=1) * balanced_weights(labels)[top]
    denominator = np.maximum(weight.sum(1), 1e-12)
    return np.stack([np.sum(weight * v[top], axis=1) / denominator
                     for v in (hard, smooth)], axis=1)
