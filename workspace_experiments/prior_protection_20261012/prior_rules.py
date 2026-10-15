"""Class-wise source anchoring for the existing squared-loss residual solver."""

import numpy as np
from scipy.special import softmax


def penalties(labels, prior, base_strength, mode):
    classes = prior.shape[1]
    counts = np.bincount(labels, minlength=classes)
    present = counts > 0
    lam = np.full(classes, float(base_strength))
    if mode == 'raw':
        return lam
    lam[~present] = np.inf
    if mode == 'guard':
        return lam
    if mode in ('count', 'count_common'):
        lam[present] *= (len(labels)/classes)/counts[present]
        if mode == 'count_common':
            lam[present] = lam[present].mean()
    elif mode == 'confidence':
        probability = softmax(prior, axis=1)
        means = np.array([probability[labels==cls, cls].mean() for cls in np.flatnonzero(present)])
        if not np.all(means > 0):
            raise ValueError('Support confidence numerically zero; inspect prior scale before solving')
        lam[present] *= means/means.mean()
    else:
        raise ValueError(mode)
    return lam


def solve_weights(features, prior, labels, lambdas):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(counts)*counts[inverse]))
    kernel = features@features.T
    gram = root[:, None]*kernel*root[None, :]
    target = 2*np.eye(prior.shape[1])[labels]-1-prior
    rhs = root[:, None]*target
    alpha = np.zeros_like(prior, dtype=float)
    for strength in np.unique(lambdas[np.isfinite(lambdas)]):
        selected = lambdas==strength
        alpha[:, selected] = root[:, None]*np.linalg.solve(
            gram+strength*np.eye(len(labels)), rhs[:, selected])
    return features.T@alpha
