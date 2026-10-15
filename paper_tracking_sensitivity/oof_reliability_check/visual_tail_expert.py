"""Recover source-orthogonal raw-feature tails for a fixed residual expert."""

import numpy as np

from residual_expert import fit_correction


def fit_tail(raw, retained, source, query):
    raw_source = np.asarray(raw[source], dtype=np.float64)
    source_center = raw_source.mean(0)
    raw_source -= source_center
    low = np.asarray(retained[source], dtype=np.float64)
    original_energy = float(np.mean(np.sum(low**2, axis=1)))
    retained_center = low.mean(0)
    low -= retained_center
    decoder = np.linalg.solve(low.T@low, low.T@raw_source)
    raw_source -= low@decoder
    tail_energy = float(np.mean(np.sum(raw_source**2, axis=1)))
    if not np.isfinite(tail_energy) or tail_energy <= 0:
        raise ValueError('source omitted-feature energy must be positive')
    multiplier = np.sqrt(original_energy/tail_energy)
    orthogonality = float(np.abs(low.T@raw_source/len(source)).max())
    raw_query = np.asarray(raw[query], dtype=np.float64)-source_center
    low_query = np.asarray(retained[query], dtype=np.float64)-retained_center
    tail = (raw_query-low_query@decoder)*multiplier
    return tail, dict(source_center=source_center, retained_center=retained_center, decoder=decoder,
        multiplier=np.asarray(multiplier), source_retained_energy=np.asarray(original_energy),
        source_tail_energy=np.asarray(tail_energy), source_orthogonality_max=np.asarray(orthogonality))


def predict(features, prior, support, support_labels):
    phi = np.column_stack((features, np.ones(len(features))))
    correction = fit_correction(phi[support], support_labels, prior[support])
    scores = prior+phi@correction
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite visual tail scores')
    return scores, correction
