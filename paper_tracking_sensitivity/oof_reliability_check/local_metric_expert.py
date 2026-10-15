"""Source-only within-class metric and support-local residual correction."""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel

from residual_expert import K, STRENGTH, source_geometry


def fit_geometry(raw, source, source_labels):
    phi, center, scale = source_geometry(raw, source)
    x = phi[:, :-1]
    within = x[source].copy()
    class_means = np.zeros((K, x.shape[1]))
    for c in np.unique(source_labels):
        use = source_labels == c
        class_means[c] = within[use].mean(0)
        within[use] -= class_means[c]
    estimate = LedoitWolf(assume_centered=True).fit(within)
    eigenvalues, eigenvectors = np.linalg.eigh(estimate.covariance_)
    if np.any(eigenvalues <= 0):
        raise ValueError('source within-class covariance must be positive definite')
    transform = (eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
    projected = x@transform
    energy_scale = np.sqrt(np.sum(projected[source]**2)/np.sum(x[source]**2))
    transform /= energy_scale
    metric_x = projected/energy_scale
    size = min(1024, 2*(len(source)//2))
    pairs = np.random.default_rng(1707).choice(source, size, replace=False).reshape(-1, 2)
    bandwidths = np.array([np.median(np.sum((features[pairs[:, 0]]-features[pairs[:, 1]])**2, axis=1))
                           for features in (x, metric_x)])
    if not np.all(np.isfinite(bandwidths) & (bandwidths > 0)):
        raise ValueError('nonpositive source bandwidth')
    metadata = dict(source_center=center, source_scale=scale, source_class_means=class_means,
        covariance=estimate.covariance_, shrinkage=np.asarray(estimate.shrinkage_), transform=transform,
        energy_scale=np.asarray(energy_scale), bandwidths=bandwidths, bandwidth_pairs=pairs)
    return x, metric_x, metadata


def kernel_matrix(query, support, bandwidth=None):
    if bandwidth is None:
        return 1.+query@support.T
    return 1.+rbf_kernel(query, support, gamma=1./bandwidth)


def fit_predict(features, support, support_labels, prior, bandwidth=None):
    """Only support labels enter this predictor; query truth is not accepted."""
    _, inverse, counts = np.unique(support_labels, return_inverse=True, return_counts=True)
    weights = len(support)/(len(counts)*counts[inverse])
    target = 2*np.eye(K)[support_labels]-1-prior[support]
    train_kernel = kernel_matrix(features[support], features[support], bandwidth)
    fit = KernelRidge(alpha=STRENGTH, kernel='precomputed').fit(train_kernel, target, sample_weight=weights)
    scores = prior+kernel_matrix(features, features[support], bandwidth)@fit.dual_coef_
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite local expert scores')
    return scores, fit.dual_coef_
