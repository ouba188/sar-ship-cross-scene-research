"""Fixed budget admission masks shared across unchanged expert candidates."""

import numpy as np


BUDGETS = (.05, .10, .20)
SELECTORS = ('old_lr_cap', 'base_uncertainty')
EXPERTS = ('fusion', 'fusion_observed64', 'anchor_bias64', 'anchor_visual64')


def ranked_mask(score, eligible, rows, cap):
    candidates = np.flatnonzero(eligible)
    order = np.lexsort((rows[candidates], -score[candidates]))
    selected = np.zeros(len(rows), dtype=bool)
    selected[candidates[order[:cap]]] = True
    return selected


def admission_masks(base_logits, fusion_prediction, old_lr_score, support_labels, rows, evaluation):
    if not np.isfinite(base_logits).all() or not np.isfinite(old_lr_score).all():
        raise ValueError('nonfinite selector inputs')
    observed = np.bincount(support_labels, minlength=8) > 0
    base = base_logits.argmax(1)
    eligible = evaluation & (base != fusion_prediction) & observed[fusion_prediction] & (old_lr_score > 0)
    uncertainty = -np.diff(np.sort(base_logits, axis=1)[:, -2:], axis=1).ravel()
    caps = np.floor(np.asarray(BUDGETS)*evaluation.sum()).astype(int)
    masks = np.stack([
        np.stack([ranked_mask(old_lr_score, eligible, rows, cap) for cap in caps]),
        np.stack([ranked_mask(uncertainty, evaluation, rows, cap) for cap in caps])])
    return masks, caps


def handoff(base, experts, selected):
    return np.where(selected[:, :, None, :], experts[None, None, :, :], base[None, None, None, :])


def score_policy(prediction, base, labels, evaluation):
    truth, p, b = labels[evaluation], prediction[evaluation], base[evaluation]
    classes, inverse, counts = np.unique(truth, return_inverse=True, return_counts=True)
    weights = 1./(len(classes)*counts[inverse])
    correct, base_correct = p == truth, b == truth
    return dict(ba=float(weights[correct].sum()), acc=float(correct.mean()),
        recover_ba=float(weights[correct & ~base_correct].sum()),
        harm_ba=float(weights[~correct & base_correct].sum()),
        class_recall=[float(correct[truth == c].mean()) if c in classes else None for c in range(8)])
