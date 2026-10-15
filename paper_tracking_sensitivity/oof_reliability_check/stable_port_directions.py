"""Source-only cross-port class contrasts, not causal identification."""

from itertools import combinations

import numpy as np


def _project(x, basis):
    # Bound mixed-dtype matmul temporaries without copying the full feature bank.
    projected = np.empty((len(x), basis.shape[1]), dtype=np.float64)
    for lo in range(0, len(x), 1024):
        projected[lo:lo+1024] = x[lo:lo+1024]@basis
    return projected


def fit_direction_bank(x_source, labels_source, ports_source,
                       min_class_count=5, min_pair_ports=3):
    """Fit equal-port pooled and coordinate-agreement-weighted contrasts.

    The caller supplies source-standardized features and owns preprocessing.
    Cosines leave one port out of the contrast mean, but condition on the
    caller's preprocessing; they are not a fully held-out pipeline evaluation.
    Projection centers/scales use all source rows with equal row weights.
    Undefined cosines are omitted, or reported as NaN if none remain.
    """
    x = np.asarray(x_source)
    labels, ports = np.asarray(labels_source), np.asarray(ports_source)
    if x.ndim != 2 or labels.shape != (len(x),) or ports.shape != (len(x),):
        raise ValueError('Expected features (N,D) and source labels/ports (N,)')
    if min_class_count < 1 or min_pair_ports < 2:
        raise ValueError('Need positive class counts and at least two ports')

    means = {}
    for port in np.unique(ports):
        rows = np.flatnonzero(ports == port)
        port_labels = labels[rows]
        means[port] = {}
        for label, count in zip(*np.unique(port_labels, return_counts=True)):
            if count >= min_class_count:
                class_rows = rows[port_labels == label]
                total = np.zeros(x.shape[1], dtype=np.float64)
                for lo in range(0, len(class_rows), 1024):
                    total += x[class_rows[lo:lo+1024]].sum(axis=0, dtype=np.float64)
                means[port][label] = total/count

    pairs, pooled, stable = [], [], []
    port_counts, cosine_means, positive_fractions, coherences = [], [], [], []
    tiny = 1e-12
    for c, d in combinations(np.unique(labels), 2):
        contrasts = [value[c]-value[d] for value in means.values()
                     if c in value and d in value]
        if len(contrasts) < min_pair_ports:
            continue
        delta = np.stack(contrasts)
        mean = delta.mean(axis=0)
        second_moment = np.square(delta).mean(axis=0)
        agreement = np.divide(np.square(mean), second_moment,
                              out=np.zeros_like(mean), where=second_moment > tiny)
        agreement = np.clip(agreement, 0., 1.)
        weighted = mean*agreement
        norm, weighted_norm = np.linalg.norm(mean), np.linalg.norm(weighted)
        if norm <= tiny or weighted_norm <= tiny:
            continue

        # Each port is compared with the direction obtained from other ports.
        others = (delta.sum(axis=0)-delta)/(len(delta)-1)
        norms_delta = np.linalg.norm(delta, axis=1)
        norms_others = np.linalg.norm(others, axis=1)
        valid = (norms_delta > tiny) & (norms_others > tiny)
        cosines = np.sum(delta[valid]*others[valid], axis=1)/(
            norms_delta[valid]*norms_others[valid])
        cosines = np.clip(cosines, -1., 1.)

        pairs.append((c, d))
        pooled.append(mean/norm)
        stable.append(weighted/weighted_norm)
        port_counts.append(len(delta))
        cosine_means.append(cosines.mean() if len(cosines) else np.nan)
        positive_fractions.append(np.mean(cosines > 0.) if len(cosines) else np.nan)
        coherences.append(np.dot(np.square(mean), agreement)/norm**2)

    if not pairs:
        raise ValueError('No nondegenerate class pairs meet the source-port counts')
    bank = dict(pairs=np.asarray(pairs, dtype=labels.dtype),
                pooled_basis=np.column_stack(pooled),
                stable_basis=np.column_stack(stable),
                pair_ports=np.asarray(port_counts, dtype=np.int64),
                source_cosine_mean=np.asarray(cosine_means),
                source_cosine_positive_fraction=np.asarray(positive_fractions),
                coherence_mean=np.asarray(coherences))
    for kind in ('pooled', 'stable'):
        projection = _project(x, bank[kind+'_basis'])
        bank[kind+'_center'] = projection.mean(axis=0)
        bank[kind+'_scale'] = np.maximum(projection.std(axis=0), 1e-6)
    return bank


def transform_direction(x, bank, kind):
    """Apply fixed source statistics, with no label or target-fitting step."""
    if kind not in ('pooled', 'stable'):
        raise ValueError('kind must be pooled or stable')
    projected = _project(np.asarray(x), bank[kind+'_basis'])
    projected -= bank[kind+'_center']
    projected /= bank[kind+'_scale']
    projected /= np.sqrt(len(bank['pairs']))
    return projected
