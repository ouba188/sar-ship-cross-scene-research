"""Source-only matched controls in [pooled, candidate] projection planes."""

import numpy as np


def equal_domain_pair_moments(z, y, domains, c, d, min_count=5):
    """Return equal-class, equal-domain population moments for source rows.

    Eligibility is per class, not the intersection of the two domain sets.
    Each class needs at least three domains with ``min_count`` rows. The
    caller must supply source rows only; no target fitting is performed.

    The result has mean_c/mean_d (2,), covariance (2,2), weights (N,), and
    domain_counts (2,) in (c,d) order. Covariance is half the sum of class
    covariances about their class means, including between-domain variation.
    Weights sum to one, with mass 0.5 per class and zero on excluded rows.
    Storage is O(N) for weights/indices, with fixed-size numerical blocks.
    """
    z, y, domains = np.asarray(z), np.asarray(y), np.asarray(domains)
    if z.ndim != 2 or z.shape[1] != 2:
        raise ValueError('Expected source projections (N,2)')
    if y.shape != (len(z),) or domains.shape != (len(z),):
        raise ValueError('Expected source labels and domains (N,)')
    if c == d or not isinstance(min_count, (int, np.integer)) or min_count < 1:
        raise ValueError('Need distinct classes and a positive integer min_count')

    weights = np.zeros(len(z), dtype=np.float64)
    covariance = np.zeros((2, 2), dtype=np.float64)
    means, domain_counts = [], []
    for label in (c, d):
        rows = np.flatnonzero(y == label)
        _, inverse, counts = np.unique(domains[rows], return_inverse=True,
                                       return_counts=True)
        eligible = counts >= min_count
        n_domains = int(eligible.sum())
        if n_domains < 3:
            raise ValueError('Each class needs at least three eligible source domains')
        valid = eligible[inverse]
        cell_counts = counts[inverse[valid]]
        rows = rows[valid]
        weights[rows] = 0.5 / n_domains / cell_counts
        domain_counts.append(n_domains)

        # Shift before summing to retain precision under common translations.
        anchor = np.asarray(z[rows[0]], dtype=np.float64)
        offset = np.zeros(2, dtype=np.float64)
        for lo in range(0, len(rows), 4096):
            block = rows[lo:lo + 4096]
            values = np.asarray(z[block], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError('Eligible source projections must be finite')
            offset += (2 * weights[block]) @ (values - anchor)
        means.append(anchor + offset)
        for lo in range(0, len(rows), 4096):
            block = rows[lo:lo + 4096]
            centered = (np.asarray(z[block], dtype=np.float64) - anchor) - offset
            covariance += centered.T @ (weights[block, None] * centered)

    return dict(mean_c=means[0], mean_d=means[1], covariance=covariance,
                weights=weights, domain_counts=np.asarray(domain_counts))


def reflect_with_matched_moments(mean_difference, covariance, reference=None):
    """Reflect a reference about the covariance-metric optimal axis.

    The default reference is [0,1], the candidate coordinate. The returned
    coefficients preserve the signed mean gap and within-class variance.
    Errors are absolute residuals evaluated against the supplied covariance.

    Eigenvalues <= 1e-10 times the largest eigenvalue define the nullspace;
    no ridge is added. A significant nullspace mean difference, or zero
    difference, returns the reference with degenerate=True and NaN errors:
    no matched alternative was constructed. An already optimal reference
    (including covariance-metric collinearity for singular matrices) also
    returns unchanged and degenerate=True, but with measured zero errors.
    """
    delta = np.asarray(mean_difference, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)
    ref = np.array([0., 1.] if reference is None else reference,
                   dtype=np.float64, copy=True)
    if delta.shape != (2,) or sigma.shape != (2, 2) or ref.shape != (2,):
        raise ValueError('Expected mean_difference/reference (2,) and covariance (2,2)')
    if not (np.isfinite(delta).all() and np.isfinite(sigma).all()
            and np.isfinite(ref).all()):
        raise ValueError('Moments and reference must be finite')
    scale = np.max(np.abs(sigma))
    if np.max(np.abs(sigma - sigma.T)) > 1e-10 * scale:
        raise ValueError('Covariance must be symmetric')
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * sigma + 0.5 * sigma.T)
    tolerance = 1e-10 * np.max(np.abs(eigenvalues))
    if eigenvalues[0] < -tolerance:
        raise ValueError('Covariance must be positive semidefinite')
    positive = eigenvalues > tolerance

    delta_scale = np.max(np.abs(delta))
    fallback = dict(coefficients=ref, mean_gap_error=float('nan'),
                    variance_error=float('nan'), degenerate=True)
    if delta_scale == 0:
        return fallback
    # Normalizing these scales leaves the reflection unchanged.
    normalized_delta = delta / delta_scale
    coordinates = eigenvectors.T @ normalized_delta
    if np.linalg.norm(coordinates[~positive]) > 1e-10 * np.linalg.norm(coordinates):
        return fallback
    basis = eigenvectors[:, positive]
    spectrum = eigenvalues[positive] / eigenvalues[-1]
    v = basis @ (coordinates[positive] / spectrum)
    a = (normalized_delta @ ref) / (normalized_delta @ v) * v
    residual_norm = np.linalg.norm(np.sqrt(spectrum) * (basis.T @ (ref - a)))
    reference_norm = np.linalg.norm(np.sqrt(spectrum) * (basis.T @ ref))
    degenerate = bool(residual_norm <= 1e-10 * reference_norm)
    control = ref if degenerate else 2 * a - ref
    return dict(coefficients=control,
                mean_gap_error=float(abs(delta @ control - delta @ ref)),
                variance_error=float(abs(control @ sigma @ control - ref @ sigma @ ref)),
                degenerate=degenerate)
