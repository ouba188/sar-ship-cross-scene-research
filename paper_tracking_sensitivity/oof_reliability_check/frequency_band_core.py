"""Matched single-band probes and source-trained hard band selection."""

import numpy as np

from frequency_reliability_core import coefficients
from frequency_decision_core import probabilities, fit_benefit, predict_benefit

BANDS = ('LL', 'LH', 'HL', 'HH')
CANDIDATES = ('learned', 'fixed_equal', *(f'fixed_{b}' for b in BANDS))


def unit_bands(embedding):
    blocks = np.asarray(embedding, dtype=float).reshape(-1, 4, 64)
    norms = np.linalg.norm(blocks, axis=2, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError('Cannot recover a zero or nonfinite band')
    return blocks / norms


def old_kernel(data, support, eta):
    cond = np.exp(-np.sum((data['conditions'][:, None]
        - data['conditions'][support][None]) ** 2, axis=2) / .5)
    kernel = data['phi'] @ data['phi'][support].T
    kernel += eta * (data['mean'] @ data['mean'][support].T) * cond
    return kernel, cond


def score_embedding(data, embedding, support, labels, kernel, condition):
    similarity = (embedding @ embedding[support].T) * condition
    total = kernel + .25 * similarity
    score = data['prior'] + total @ coefficients(total[support], labels, data['prior'][support])
    return score, similarity


def choice_features(reference, candidate, similarity, support_labels, conditions, band):
    """Inputs contain model outputs and support labels, never query truth."""
    n, classes = reference.shape
    rr = np.arange(n)
    a, b = reference.argmax(1), candidate.argmax(1)
    p, q = probabilities(reference), probabilities(candidate)
    rtop, ctop = np.sort(reference, axis=1)[:, -2:], np.sort(candidate, axis=1)[:, -2:]
    centered = candidate - reference
    centered -= centered.mean(1, keepdims=True)
    columns = [rtop[:, 1]-rtop[:, 0], ctop[:, 1]-ctop[:, 0],
        reference[rr, a]-reference[rr, b], candidate[rr, b]-candidate[rr, a],
        p[rr, a], q[rr, b], q[rr, a]-p[rr, a], q[rr, b]-p[rr, b],
        -(p*np.log(np.maximum(p, 1e-12))).sum(1),
        -(q*np.log(np.maximum(q, 1e-12))).sum(1),
        np.sqrt(np.mean(centered**2, axis=1))]
    columns.extend([(a == c).astype(float) for c in range(classes)])
    columns.extend([(b == c).astype(float) for c in range(classes)])
    columns.extend([np.full(n, float(j == band)) for j in range(4)])
    columns.extend(conditions.T)
    width = len(columns)
    sy = np.asarray(support_labels)
    counts = np.bincount(sy, minlength=classes)
    positive = np.maximum(similarity, 0.)
    top = np.argsort(-positive, axis=1, kind='stable')[:, :8]
    weight = np.take_along_axis(positive, top, axis=1)
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
    for pred in (a, b):
        match = sy[None, :] == pred[:, None]
        columns.extend([counts[pred]/len(sy),
            np.sum(similarity*match, axis=1)/np.maximum(counts[pred], 1),
            np.max(np.where(match, positive, 0.), axis=1),
            np.sum(weight*(sy[top] == pred[:, None]), axis=1)])
    columns.extend([positive.max(1), positive.mean(1)])
    return np.column_stack(columns).astype(np.float32), width


def choose_scores(scores, benefits):
    """Keep the reference on nonpositive utility or same-class candidates."""
    utility = np.array(benefits, dtype=float, copy=True)
    conflict = scores[:, 2:].argmax(2) != scores[:, 0].argmax(1)[:, None]
    utility[~conflict] = -np.inf
    chosen = np.argmax(np.c_[np.zeros(len(scores)), utility], axis=1)
    index = np.where(chosen == 0, 0, chosen + 1)
    return scores[np.arange(len(scores)), index], index


def oracle_predictions(reference, alternatives, labels):
    """Label-assisted diagnostic only; never a deployment policy."""
    can_rescue = (alternatives == labels[:, None]).any(1)
    return np.where(can_rescue, labels, reference)

