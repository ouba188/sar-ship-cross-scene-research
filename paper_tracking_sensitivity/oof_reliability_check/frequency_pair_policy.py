"""Coverage-shrunk class-context and pair-margin support evidence."""

import numpy as np
from scipy.special import expit
from frequency_margin_policy import margin_evidence
from frequency_reliability_core import balanced_weights


def conditional_utilities(similarity, loo_main, loo_band, labels, main_class, band_class):
    positive = np.maximum(similarity, 0.)
    top = np.argsort(-positive, axis=1, kind='stable')[:, :8]
    weight = np.take_along_axis(positive, top, axis=1) * balanced_weights(labels)[top]
    generic_delta = margin_evidence(loo_band, labels) - margin_evidence(loo_main, labels)
    generic = np.sum(weight * generic_delta[top], axis=1) / np.maximum(weight.sum(1), 1e-12)
    a, b = main_class[:, None], band_class[:, None]
    near_labels = labels[top]
    wa, wb = weight * (near_labels == a), weight * (near_labels == b)
    na = wa.sum(1)**2 / np.maximum((wa**2).sum(1), 1e-24)
    nb = wb.sum(1)**2 / np.maximum((wb**2).sum(1), 1e-24)
    coverage = np.minimum(na, nb)
    rho = coverage / (coverage + 2.)
    rho[main_class == band_class] = 0.
    relevant = weight * ((near_labels == a) | (near_labels == b))
    denominator = np.maximum(relevant.sum(1), 1e-12)
    class_local = np.sum(relevant * generic_delta[top], axis=1) / denominator
    sign = np.where(near_labels == a, 1., -1.)
    main_margin = sign * (loo_main[top, a] - loo_main[top, b])
    band_margin = sign * (loo_band[top, a] - loo_band[top, b])
    pair_delta = expit(band_margin / .1) - expit(main_margin / .1)
    pair_local = np.sum(relevant * pair_delta, axis=1) / denominator
    # The generic evidence is retained when either side has no local support.
    return np.stack([generic, generic + rho * (class_local - generic),
                     generic + rho * (pair_local - generic)], axis=1), rho
