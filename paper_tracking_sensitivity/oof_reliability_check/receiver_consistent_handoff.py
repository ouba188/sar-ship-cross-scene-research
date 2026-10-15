"""Apply fixed source gate parameters to each actual receiving expert."""

import numpy as np


def confidence_parts(logits, allowed=None):
    x = np.asarray(logits, dtype=float)
    if allowed is not None:
        if np.count_nonzero(allowed) < 2:
            raise ValueError('observed-class gate requires two scored classes')
        x = x[:, allowed]
    if not np.isfinite(x).all():
        raise ValueError('nonfinite receiving expert scores')
    ordered = np.sort(x, axis=1)
    p = np.exp(x-x.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return ordered[:, -1]-ordered[:, -2], -(p*np.log(np.maximum(p, 1e-300))).sum(1)


def gate_scores(base_logits, expert_logits, prior, allowed=None):
    bm, bh = confidence_parts(base_logits)
    em, eh = confidence_parts(expert_logits, allowed)
    x = np.column_stack((em, bm, eh, bh))
    return ((x-prior['center'])/prior['scale'])@prior['lr'][:-1]+prior['lr'][-1]
