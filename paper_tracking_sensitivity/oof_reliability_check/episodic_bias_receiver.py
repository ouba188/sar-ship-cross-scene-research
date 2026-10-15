"""Support-only bias adaptation for source-held-port episodes."""

import numpy as np


def adapt_bias(prior_scores, support_local, support_labels):
    """Return (N x 8 scores, 8-vector bias) using 64 ordered support labels."""
    prior = np.asarray(prior_scores, dtype=float)
    support = np.asarray(support_local)
    labels = np.asarray(support_labels)
    if prior.ndim != 2 or prior.shape[1] != 8:
        raise ValueError('prior_scores must have shape (N, 8)')
    if (support.shape != (64,) or not np.issubdtype(support.dtype, np.integer)
            or len(np.unique(support)) != 64):
        raise ValueError('support_local must contain exactly 64 distinct integer local IDs')
    if np.any((support < 0) | (support >= len(prior))):
        raise ValueError('support_local IDs must index prior_scores')
    if (labels.shape != (64,) or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= 8))):
        raise ValueError('support_labels must contain 64 integer class IDs in [0, 8)')
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 64 / (len(counts) * counts[inverse])
    desired = 2 * np.eye(8)[labels] - 1
    bias = (weights[:, None] * (desired - prior[support])).sum(axis=0) / (64 + 8)
    return prior + bias, bias


def make_episode(base_logits, expert_logits, labels, support_local):
    """Return adapted scores, bias, evaluation mask, and sorted observed class IDs.

    All arrays use episode-local row order. Only labels[support_local] is read;
    base logits establish row alignment and do not enter the expert correction.
    """
    if np.shape(base_logits) != np.shape(expert_logits):
        raise ValueError('base and expert logits must have the same shape')
    support = np.asarray(support_local)
    support_labels = labels[support]
    adapted, bias = adapt_bias(expert_logits, support, support_labels)
    eval_mask = np.ones(len(adapted), dtype=bool)
    eval_mask[support] = False
    return adapted, bias, eval_mask, np.unique(support_labels)
