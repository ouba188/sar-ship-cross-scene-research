"""Source-prior multiclass residual ridge, distinct from pairwise routing."""

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier


K = 8
STRENGTH = 8.


def source_geometry(raw, source):
    center = raw[source].mean(0)
    scale = raw[source].std(0) + 1e-6
    standardized = (raw-center)/scale
    phi = np.column_stack((standardized/np.sqrt(raw.shape[1]), np.ones(len(raw))))
    return phi, center, scale


def source_replay(raw, labels, source, center, scale):
    x = (raw-center)/scale
    model = RidgeClassifier(alpha=1., class_weight='balanced').fit(x[source], labels[source])
    if not np.array_equal(model.classes_, np.arange(K)):
        raise ValueError('source class-column mismatch')
    return model.decision_function(x)


def fit_correction(phi, labels, prior, strength=STRENGTH):
    """The caller provides support rows only, including their prior scores."""
    labels = np.asarray(labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = len(labels)/(len(counts)*counts[inverse])
    desired = 2*np.eye(prior.shape[1])[labels]-1
    # Dual solve stays at 64x64 rather than the 193-coordinate primal system.
    root = np.sqrt(weights)
    weighted = phi*root[:, None]
    dual = np.linalg.solve(weighted@weighted.T + strength*np.eye(len(labels)),
                           root[:, None]*(desired-prior))
    return weighted.T@dual


def predict(phi, source_scores, support, support_labels, raw):
    corrections, scores = {}, {}
    for name, x, prior in (
            ('target64', phi, np.zeros_like(source_scores)),
            ('anchor_bias64', phi[:, -1:], source_scores),
            ('anchor_visual64', phi, source_scores)):
        correction = fit_correction(x[support], support_labels, prior[support])
        corrections[name] = correction
        scores[name] = prior+x@correction
    _, inverse, counts = np.unique(support_labels, return_inverse=True, return_counts=True)
    legacy = Ridge(alpha=1.).fit(raw[support], np.eye(K)[support_labels],
                                sample_weight=1./(len(counts)*counts[inverse]))
    scores['legacy_target64'] = legacy.predict(raw)
    corrections['legacy_coef'] = legacy.coef_
    corrections['legacy_intercept'] = legacy.intercept_
    if not all(np.isfinite(s).all() for s in scores.values()):
        raise ValueError('nonfinite expert scores')
    return scores, corrections
