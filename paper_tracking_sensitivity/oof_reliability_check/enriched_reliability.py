"""Fixed class-sensitive OOF baselines; not a new routing algorithm."""

import numpy as np
from scipy.linalg import helmert
from sklearn.ensemble import HistGradientBoostingRegressor

import run_oof_reliability as old


TREE_PARAMS = dict(loss='squared_error', max_iter=100, learning_rate=.05,
                   max_leaf_nodes=7, min_samples_leaf=30, l2_regularization=1.,
                   max_bins=63, random_state=0, early_stopping=False)
ARMS = ('conf_ridge0', 'conf_ridge64', 'rich_ridge0', 'rich_ridge64',
        'rich_hgb0', 'rich_hgb64')


def features(base_logits, expert_logits, rich):
    confidence = old.confidence_features(base_logits, expert_logits)
    if not rich:
        return confidence
    h = helmert(old.K, full=False)
    relative = np.column_stack((base_logits @ h.T, expert_logits @ h.T))
    pair = old.K * base_logits.argmax(1) + expert_logits.argmax(1)
    return np.column_stack((confidence, relative, np.eye(old.K ** 2)[pair]))


def source_design(x, weights, continuous):
    center = np.zeros(x.shape[1])
    scale = np.ones(x.shape[1])
    center[:continuous] = np.average(x[:, :continuous], axis=0, weights=weights)
    variance = np.average((x[:, :continuous] - center[:continuous]) ** 2,
                          axis=0, weights=weights)
    scale[:continuous] = np.maximum(np.sqrt(variance), 1e-6)
    return old.design(x, center, scale), center, scale


def fit_source(base_logits, expert_logits, labels, ports, rich):
    x = features(base_logits, expert_logits, rich)
    weights = old.source_weights(labels, ports)
    X, center, scale = source_design(x, weights, 18 if rich else 4)
    base, expert = base_logits.argmax(1), expert_logits.argmax(1)
    use = base != expert
    old.require(use.any(), 'no source disagreements')
    y = old.utility(labels, base, expert)
    w = weights[use] / weights[use].sum()
    ridge = np.linalg.solve(X[use].T @ (w[:, None] * X[use])
                            + old.SOURCE_RIDGE_LAMBDA * np.eye(X.shape[1]),
                            X[use].T @ (w * y[use]))
    tree = None
    if rich:
        tree = HistGradientBoostingRegressor(**TREE_PARAMS).fit(
            X[use, :-1], y[use], sample_weight=w * int(use.sum()))
    return dict(center=center, scale=scale, ridge=ridge, tree=tree, rich=rich,
                source_disagreements=int(use.sum()),
                source_zero_utility=int(np.count_nonzero(use & (y == 0))))


def residual_update(X, source_score, labels, base, expert):
    """Only support rows enter the class counts and residual fit."""
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    w = len(labels) / (len(counts) * counts[inverse])
    use = base != expert
    if not use.any():
        return np.zeros(X.shape[1])
    residual = old.utility(labels, base, expert) - source_score
    return np.linalg.solve(X[use].T @ (w[use, None] * X[use])
                            + old.TARGET_PRIOR_STRENGTH * np.eye(X.shape[1]),
                            X[use].T @ (w[use] * residual[use]))


def predict(model, base_logits, expert_logits, support_local, support_labels):
    old.require(len(support_local) == old.ANNOT
                and len(np.unique(support_local)) == old.ANNOT,
                'expected 64 distinct support rows')
    x = features(base_logits, expert_logits, model['rich'])
    X = old.design(x, model['center'], model['scale'])
    base, expert = base_logits.argmax(1), expert_logits.argmax(1)
    observed = np.bincount(support_labels, minlength=old.K) > 0
    eligible = (base != expert) & observed[expert]
    source_scores = {'rich_ridge' if model['rich'] else 'conf_ridge': X @ model['ridge']}
    if model['tree'] is not None:
        source_scores['rich_hgb'] = model['tree'].predict(X[:, :-1])
    scores, predictions, residuals = {}, {}, {}
    for name, source_score in source_scores.items():
        correction = residual_update(X[support_local], source_score[support_local],
                                     support_labels, base[support_local], expert[support_local])
        residuals[name] = correction
        for suffix, score in (('0', source_score), ('64', source_score + X @ correction)):
            old.require(np.isfinite(score).all(), 'nonfinite gate score')
            scores[name + suffix] = score
            predictions[name + suffix] = np.where(eligible & (score > 0), expert, base)
    return scores, predictions, residuals
