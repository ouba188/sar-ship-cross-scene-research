"""Matched binary odds and three-outcome expected gain from source episodes."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_models(features, utility, eligible, weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.) or not np.all(weights > 0):
        raise ValueError('source episode weights must be positive and sum to one')
    center = np.average(features, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average((features-center)**2, axis=0, weights=weights)), 1e-6)
    x = (features-center)/scale
    models = {}
    for name, use in (('matched_binary', eligible & (utility != 0)), ('gain_3way', eligible)):
        required = np.array([-1, 1]) if name == 'matched_binary' else np.array([-1, 0, 1])
        if not np.array_equal(np.unique(utility[use]), required):
            raise ValueError(name+' source outcomes missing')
        fit = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-9)
        fit.fit(x[use], utility[use], sample_weight=len(features)*weights[use])
        if fit.n_iter_.max() >= fit.max_iter:
            raise ValueError(name+' did not converge')
        models[name] = dict(coef=fit.coef_, intercept=fit.intercept_, classes=fit.classes_,
                            training_rows=int(use.sum()), iterations=fit.n_iter_.tolist())
    return dict(center=center, scale=scale, models=models)


def predict_scores(model, features):
    x = (features-model['center'])/model['scale']
    scores, probabilities = {}, {}
    for name, fit in model['models'].items():
        logits = x@fit['coef'].T+fit['intercept']
        if name == 'matched_binary':
            scores[name] = logits[:, 0]
        else:
            p = np.exp(logits-logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            scores[name] = p[:, 2]-p[:, 0]
            probabilities[name] = p
        if not np.isfinite(scores[name]).all():
            raise ValueError('nonfinite gain score')
    return scores, probabilities
