"""Support-frozen ridge response to a change in the source prior."""

import numpy as np


def _balanced_weights(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return len(labels)/(len(counts)*counts[inverse])


def fit_component_response(phi, support, support_labels, component, strength=8.):
    """Return coefficient and residual_component for a fixed prior component.

    phi includes its regularized bias column; support contains local row indices.
    For a prior change alpha*component, update saved scores by
    alpha*residual_component and saved coefficients by -alpha*coefficient.
    Only support rows determine the response coefficient.
    """
    phi = np.asarray(phi, dtype=float)
    component = np.asarray(component, dtype=float)
    labels = np.asarray(support_labels)
    root = np.sqrt(_balanced_weights(labels))
    weighted = phi[support]*root[:, None]
    dual = np.linalg.solve(weighted@weighted.T + strength*np.eye(len(labels)),
                           root[:, None]*component[support])
    coefficient = weighted.T@dual
    return dict(coefficient=coefficient,
                residual_component=component-phi@coefficient)


def ridge_objective(scores_support, labels, coefficient, strength=8.):
    """Weighted squared error to full-K +/-1 labels plus ridge penalty.

    scores_support includes the prior. All coefficient rows, including the
    explicit bias, are regularized, matching residual_expert.fit_correction.
    """
    scores = np.asarray(scores_support, dtype=float)
    labels = np.asarray(labels)
    coefficient = np.asarray(coefficient, dtype=float)
    desired = 2*np.eye(scores.shape[1])[labels]-1
    return float(np.sum(_balanced_weights(labels)[:, None]*(scores-desired)**2)
                 + strength*np.sum(coefficient**2))
