"""Support-trained directional residual with an optional condition kernel."""

import numpy as np

from residual_expert import fit_correction


CONDITION_COLUMNS = ('elongation_proxy', 'center_excess_proxy')
BANDWIDTH = .5


def rank_conditions(frame):
    """Unlabelled cohort ranks; missing descriptors receive the median rank."""
    columns = []
    for key in CONDITION_COLUMNS:
        x = frame[key].replace([np.inf, -np.inf], np.nan)
        n = x.notna().sum()
        columns.append(((x.rank(method='average')-.5)/max(n, 1)).fillna(.5).to_numpy())
    return np.column_stack(columns)


def normalize_direction(delta):
    delta = np.asarray(delta, dtype=float)
    norm = np.linalg.norm(delta, axis=1, keepdims=True)
    return delta/np.maximum(norm, 1e-12)


def direction_kernel(direction, conditions, support, conditional=False):
    kernel = direction @ direction[support].T
    if conditional == 'uniform':
        local = conditions[support]
        distance = np.sum((local[:, None, :]-local[None, :, :])**2, axis=2)
        weights = np.exp(-distance/(2*BANDWIDTH**2))
        n = len(support)
        rho = (weights.sum()-n)/(n*(n-1))
        diagonal = kernel[support, np.arange(n)].copy()
        kernel *= rho
        kernel[support, np.arange(n)] = diagonal
    elif conditional:
        distance = np.sum((conditions[:, None, :]-conditions[support][None, :, :])**2, axis=2)
        kernel *= np.exp(-distance/(2*BANDWIDTH**2))
    return kernel


def fit_predict(phi, prior, direction, conditions, support, labels_support,
                eta, conditional=False, strength=8., return_parts=False):
    """Fit one joint ridge correction, not a second fit to in-sample residuals.

    Only support labels enter the solve. The conditional kernel changes case
    similarity while preserving the directional kernel's diagonal and scale.
    """
    support = np.asarray(support, dtype=int)
    labels = np.asarray(labels_support, dtype=int)
    if eta == 0:
        base = phi @ fit_correction(phi[support], labels, prior[support], strength)
        scores = prior+base
        return (scores, dict(base=base, direction=np.zeros_like(base))) if return_parts else scores
    kernel = direction_kernel(direction, conditions, support, conditional)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    root = np.sqrt(len(labels)/(len(counts)*counts[inverse]))
    desired = 2*np.eye(prior.shape[1])[labels]-1
    support_kernel = phi[support]@phi[support].T+eta*kernel[support]
    dual = np.linalg.solve(root[:, None]*support_kernel*root[None, :]
                           +strength*np.eye(len(support)),
                           root[:, None]*(desired-prior[support]))
    alpha = root[:, None]*dual
    base = phi @ (phi[support].T @ alpha)
    extra = eta*kernel @ alpha
    scores = prior+base+extra
    return (scores, dict(base=base, direction=extra)) if return_parts else scores
