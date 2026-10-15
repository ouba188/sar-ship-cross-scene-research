"""One source-anchored ridge classifier over the exact union of two feature maps."""

import numpy as np
from scipy.special import softmax

STRENGTHS = (8., 1., 64.)
RATIOS = (.5, .25, .75)
GRID = tuple((r, strength) for r in RATIOS for strength in STRENGTHS)


def class_weights(labels):
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    return len(labels)/(len(counts)*counts[inverse])


def coefficients(kernel_support, prior_support, labels, strength):
    root = np.sqrt(class_weights(labels))
    target = 2*np.eye(prior_support.shape[1])[labels]-1.
    return root[:, None]*np.linalg.solve(
        root[:, None]*kernel_support*root[None, :]+strength*np.eye(len(labels)),
        root[:, None]*(target-prior_support))


def score(kernel, prior, support, labels, strength=8.):
    alpha = coefficients(kernel[support], prior[support], labels, strength)
    return prior+kernel@alpha, alpha


def combined(kb, ke, pb, pe, support, labels, ratio=.5, strength=8., anchored=True):
    # The sum kernel is exactly the concatenated, square-root-scaled feature map.
    kernel = (1-ratio)*kb+ratio*ke
    prior = (1-ratio)*pb+ratio*pe if anchored else np.zeros_like(pb)
    return score(kernel, prior, support, labels, strength)


def late(base, expert, ratio=.5, probability=True):
    if probability:
        base, expert = softmax(base, axis=1), softmax(expert, axis=1)
    return (1-ratio)*base+ratio*expert


def fit_diagnostics(kb, ke, pb, pe, support, labels):
    b, ab = score(kb, pb, support, labels)
    e, ae = score(ke, pe, support, labels)
    joint, aj = combined(kb, ke, pb, pe, support, labels)
    y = 2*np.eye(pb.shape[1])[labels]-1.
    weights = class_weights(labels)[:, None]
    def norm(k, a):
        return float(np.sum(a*(k[support]@a)))
    raw = late(b, e, probability=False)
    j_loss = float(np.sum(weights*(joint[support]-y)**2))
    l_loss = float(np.sum(weights*(raw[support]-y)**2))
    j_reg = 8*(.5*norm(kb, aj)+.5*norm(ke, aj))
    l_reg = 8*(.5*norm(kb, ab)+.5*norm(ke, ae))
    correction = .5*kb@(aj-ab)+.5*ke@(aj-ae)
    np.testing.assert_allclose(joint-raw, correction, atol=1e-10, rtol=1e-10)
    if j_loss+j_reg > l_loss+l_reg+1e-7:
        raise AssertionError('Joint solution is not the minimum of its stated objective')
    return dict(unified_support_loss=j_loss, late_support_loss=l_loss,
        unified_objective=j_loss+j_reg, late_objective=l_loss+l_reg,
        joint_vs_late_rms=float(np.sqrt(np.mean((joint-raw)**2))))
