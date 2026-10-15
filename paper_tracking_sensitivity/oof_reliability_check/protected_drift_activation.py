"""Source-only protected drift directions and shared-moment ridge heads."""

import numpy as np
from scipy.linalg import solve


def protected_drift_basis(cell_means, cell_labels, stable_q, rank=16):
    """Fit at most min(rank, 16) drift directions outside the stable span.

    Cell means are already source-standardized. The caller filters classes to
    at least three cells each and supplies an orthonormal (D, r) stable basis.
    Classes and their cells receive equal weight; returned u preserves row order.
    singular_values contains the full residual spectrum. Fractions use squared
    Frobenius norms, with zero for an empty denominator or an empty selection.
    """
    means = np.asarray(cell_means, dtype=np.float64)
    labels = np.asarray(cell_labels)
    qs = np.asarray(stable_q, dtype=np.float64)
    if means.ndim != 2 or labels.shape != (len(means),):
        raise ValueError('Expected cell_means (J,D) and cell_labels (J,)')
    if qs.ndim != 2 or qs.shape[0] != means.shape[1]:
        raise ValueError('Expected stable_q (D,r)')
    if rank < 0 or int(rank) != rank:
        raise ValueError('rank must be a nonnegative integer')

    classes, inverse, counts = np.unique(labels, return_inverse=True,
                                         return_counts=True)
    u = np.empty_like(means)
    for index, count in enumerate(counts):
        rows = inverse == index
        u[rows] = (means[rows] - means[rows].mean(axis=0)) / np.sqrt(
            len(classes) * count)
    stable = u @ qs
    residual = u - stable @ qs.T
    _, singular_values, vt = np.linalg.svd(residual, full_matrices=False)
    total_energy = np.sum(u * u)
    residual_energy = np.sum(singular_values ** 2)

    # Relative residual rank alone would promote roundoff in a contained U.
    roundoff = 64 * np.finfo(np.float64).eps * np.sqrt(total_energy)
    cutoff = max(1e-6 * singular_values[0], roundoff) if singular_values.size else 0.
    selected = min(int(rank), 16, int(np.count_nonzero(singular_values > cutoff)))
    qd = vt[:selected].T.copy()
    if selected and qs.shape[1] and np.max(np.abs(qs.T @ qd)) > 1e-12:
        qd -= qs @ (qs.T @ qd)
        qd = np.linalg.qr(qd, mode='reduced')[0]
    overlap = np.max(np.abs(qs.T @ qd)) if selected and qs.shape[1] else 0.
    return dict(q_drift=qd, singular_values=singular_values,
                protected_overlap_max=float(overlap),
                selected_energy_fraction=float(np.sum(singular_values[:selected] ** 2)
                                               / residual_energy)
                if selected and residual_energy else 0.,
                nuisance_energy_in_stable_fraction=float(np.sum(stable * stable)
                                                         / total_energy)
                if total_energy else 0.,
                u=u)


def fit_source_heads(bank, labels, transforms, strength=1., classes=8):
    """Fit balanced ridge heads using one source Gram and label crossmoment.

    Labels index the fixed output columns 0..classes-1. Class balancing uses
    only present classes: N / (C * n_c). Targets are 2*onehot(labels)-1.
    Each (D, M) transform has its own ridge penalty in transformed coordinates;
    intercepts are unpenalized. Predict with bank @ coef + intercept, where
    coef is (D, classes). A (D, 0) transform gives the intercept-only head.
    The float32 bank is accumulated in float64 blocks without a full copy.
    """
    x, labels = np.asarray(bank), np.asarray(labels)
    if x.ndim != 2 or labels.shape != (len(x),) or not len(x):
        raise ValueError('Expected nonempty source bank (N,D) and labels (N,)')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(
            (labels < 0) | (labels >= classes)):
        raise ValueError('Labels must be integer output indices in [0, classes)')
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('strength must be finite and positive')

    present, inverse, counts = np.unique(labels, return_inverse=True,
                                         return_counts=True)
    weights = len(x) / (len(present) * counts[inverse].astype(np.float64))
    weight_sum = weights.sum()
    mean_x = np.zeros(x.shape[1], dtype=np.float64)
    for lo in range(0, len(x), 1024):
        block = np.asarray(x[lo:lo+1024], dtype=np.float64)
        mean_x += weights[lo:lo+1024] @ block
    mean_x /= weight_sum
    mean_y = 2 * np.bincount(labels, weights=weights, minlength=classes) / weight_sum - 1

    gram = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    cross = np.zeros((x.shape[1], classes), dtype=np.float64)
    for lo in range(0, len(x), 1024):
        centered = np.asarray(x[lo:lo+1024], dtype=np.float64) - mean_x
        desired = np.full((len(centered), classes), -1., dtype=np.float64)
        desired[np.arange(len(centered)), labels[lo:lo+1024]] = 1.
        weighted = weights[lo:lo+1024, None] * centered
        gram += centered.T @ weighted
        cross += weighted.T @ (desired - mean_y)

    heads = {}
    for name, transform in transforms.items():
        transform = np.asarray(transform, dtype=np.float64)
        if transform.ndim != 2 or transform.shape[0] != x.shape[1]:
            raise ValueError('Expected each transform to have shape (D,M)')
        if transform.shape[1]:
            system = transform.T @ gram @ transform
            system.flat[::len(system)+1] += strength
            coef = transform @ solve(system, transform.T @ cross, assume_a='pos')
        else:
            coef = np.zeros_like(cross)
        heads[name] = dict(coef=coef, intercept=mean_y - mean_x @ coef)
    return heads
