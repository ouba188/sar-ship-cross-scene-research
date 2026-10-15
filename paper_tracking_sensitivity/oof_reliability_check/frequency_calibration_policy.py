"""Seven-parameter, support-only pooled correctness calibration."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from frequency_reliability_core import balanced_weights


def confidence_features(scores):
    ordered = np.sort(scores, axis=-1)
    return np.stack([ordered[..., -1]-ordered[..., -2],
                     ordered[..., -1]-scores.mean(axis=-1)], axis=-1)


def design(features):
    n, models, _ = features.shape
    identity = np.broadcast_to(np.eye(models)[:, 1:], (n, models, models-1))
    return np.concatenate([features, identity], axis=-1).reshape(n*models, -1)


def fit_correctness(loo_scores, labels):
    n, models, _ = loo_scores.shape
    target = (loo_scores.argmax(2) == labels[:, None]).astype(int).reshape(-1)
    weight = np.repeat(balanced_weights(labels)/models, models)
    features = confidence_features(loo_scores)
    scaler = StandardScaler().fit(features.reshape(-1, 2), sample_weight=weight)
    x = design(scaler.transform(features.reshape(-1, 2)).reshape(n, models, 2))
    if len(np.unique(target)) == 1:
        model = float(target[0])
    else:
        model = LogisticRegression(C=1., solver='lbfgs', max_iter=200, tol=1e-7)
        model.fit(x, target, sample_weight=weight)
    return scaler, model


def predict_correctness(fitted, scores):
    scaler, model = fitted
    n, models, _ = scores.shape
    if isinstance(model, float):
        return np.full((n, models), model)
    feat = confidence_features(scores)
    x = design(scaler.transform(feat.reshape(-1, 2)).reshape(n, models, 2))
    return model.predict_proba(x)[:, 1].reshape(n, models)


def fit_metadata(fitted):
    scaler, model = fitted
    return dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
        coefficients=[] if isinstance(model, float) else model.coef_[0].tolist(),
        intercept=model if isinstance(model, float) else float(model.intercept_[0]),
        constant=isinstance(model, float),
        iterations=0 if isinstance(model, float) else int(model.n_iter_[0]))
