"""Source-episode recalibration and evidence-aware budgeted handoff."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from budgeted_residual_handoff import ranked_mask
from receiver_consistent_handoff import confidence_parts


ALPHAS = (0., -.25, .25, -.5, .5)


def select_recalibration(episodes):
    """Select each alpha by equally averaged source-episode query BA (0..1)."""
    episodes = list(episodes)
    if not episodes:
        raise ValueError('at least one source episode is required')
    result = {}
    for name, component in (('stable', 'stable_source'), ('scale', 'base_prior')):
        curve = []
        best_alpha, best_ba = ALPHAS[0], -np.inf
        for alpha in ALPHAS:
            values = []
            for episode in episodes:
                query = np.asarray(episode['query'], dtype=bool)
                truth = np.asarray(episode['labels'])[query]
                if not len(truth):
                    raise ValueError('each source episode needs query rows')
                base = np.asarray(episode['base_scores'])[query]
                part = np.asarray(episode[component])[query]
                if not all(np.isfinite(x).all() for x in (base, part, truth)):
                    raise ValueError('nonfinite source episode query values')
                scores = base if alpha == 0 else (
                    base + alpha * part)
                correct = scores.argmax(1) == truth
                values.append(np.mean([
                    correct[truth == c].mean() for c in np.unique(truth)]))
            ba = float(np.mean(values))
            curve.append(dict(alpha=alpha, ba=ba))
            if ba > best_ba + 1e-12:
                best_alpha, best_ba = alpha, ba
        result[name + '_alpha'] = best_alpha
        result[name + '_curve'] = curve
    return result


def gate_features(base, expert, stable_base, stable_expert):
    """Return a dict with base_features[N,5] and evidence_features[N,6]."""
    base, expert, stable_base, stable_expert = (
        np.asarray(x, dtype=float)
        for x in (base, expert, stable_base, stable_expert))
    bm, bh = confidence_parts(base)
    em, eh = confidence_parts(expert)
    rows = np.arange(len(base))
    b, e = base.argmax(1), expert.argmax(1)
    total_adv = (expert[rows, e] - expert[rows, b]
                 - base[rows, e] + base[rows, b])
    stable_adv = (stable_expert[rows, e] - stable_expert[rows, b]
                  - stable_base[rows, e] + stable_base[rows, b])
    basic = np.column_stack((em, bm, eh, bh, total_adv))
    return dict(base_features=basic,
                evidence_features=np.column_stack((basic, stable_adv)))


def fit_gate(features, base, expert, labels, eligible, weights):
    """Fit on eligible source rows with nonzero expert-minus-base utility.

    The caller supplies source-only rows and eligibility (including query masks).
    Weights are normalized after filtering, to sum to the training row count.
    """
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    utility = ((np.asarray(expert).argmax(1) == labels).astype(int)
               - (np.asarray(base).argmax(1) == labels).astype(int))
    mask = np.asarray(eligible, dtype=bool) & (utility != 0)
    x = features[mask]
    y = (utility[mask] > 0).astype(int)
    weight = np.asarray(weights, dtype=float)[mask]
    counts = dict(eligible=int(np.count_nonzero(eligible)), training=len(y),
                  positive=int(y.sum()), negative=int(len(y) - y.sum()))
    model = dict(center=np.zeros(features.shape[1]),
                 scale=np.ones(features.shape[1]),
                 coef=np.zeros(features.shape[1]), intercept=np.array([0.]),
                 counts=counts, n_iter=np.array([0], dtype=int))
    if not len(y):
        return model
    if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
        raise ValueError('training weights must be finite, nonnegative, and have positive sum')
    weight = weight / weight.sum() * len(y)
    center = np.average(x, axis=0, weights=weight)
    scale = np.maximum(np.sqrt(np.average((x-center)**2, axis=0, weights=weight)), 1e-6)
    model.update(center=center, scale=scale)
    if len(np.unique(y[weight > 0])) < 2:
        # A single observed utility sign supports only a constant gate.
        model['intercept'] = np.array([
            np.log((weight[y == 1].sum() + .5) / (weight[y == 0].sum() + .5))])
        return model
    lr = LogisticRegression(C=1., solver='lbfgs', max_iter=1000, tol=1e-7)
    lr.fit((x-center)/scale, y, sample_weight=weight)
    model.update(coef=lr.coef_[0].copy(), intercept=lr.intercept_.copy(),
                 n_iter=lr.n_iter_.copy())
    return model


def predict_gate(model, features):
    """Return unthresholded LR decision scores; admission uses score > 0."""
    return ((np.asarray(features) - model['center']) / model['scale']) @ model['coef'] + model['intercept']


def route_scores(base, expert, score, support_labels, eval_mask, row_ids, budget=.05):
    """Route at most floor(budget * query_count) eligible query predictions."""
    base, expert = np.asarray(base), np.asarray(expert)
    score = np.asarray(score)
    evaluation = np.asarray(eval_mask, dtype=bool)
    base_predictions, expert_predictions = base.argmax(1), expert.argmax(1)
    eligible = (evaluation & (base_predictions != expert_predictions)
                & np.isin(expert_predictions, np.asarray(support_labels))
                & (score > 0))
    cap = int(np.floor(budget * evaluation.sum()))
    selected = ranked_mask(score, eligible, np.asarray(row_ids), cap)
    return dict(predictions=np.where(selected, expert_predictions, base_predictions),
                selected=selected, cap=cap, expert_predictions=expert_predictions,
                base_predictions=base_predictions)
