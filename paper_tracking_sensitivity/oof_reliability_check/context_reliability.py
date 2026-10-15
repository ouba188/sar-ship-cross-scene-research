"""Matched coverage controls for the value of actual local facility features."""

import numpy as np
from sklearn.linear_model import LogisticRegression

import run_oof_reliability as old


ARMS = ('oof_lr0', 'covered_conf', 'covered_mask', 'covered_values')


def fit_context_scaler(values, known, weights):
    mass = (weights[:, None] * known).sum(0)
    active = mass > 0
    denominator = np.where(active, mass, 1.)
    center = (weights[:, None] * np.where(known, values, 0.)).sum(0) / denominator
    variance = (weights[:, None] * np.where(known, (values-center)**2, 0.)).sum(0) / denominator
    return dict(center=center, scale=np.maximum(np.sqrt(variance), 1e-6), active=active)


def context_design(values, known, scaler):
    usable = known & scaler['active']
    x = np.where(usable, (values - scaler['center']) / scaler['scale'], 0.)
    return x, usable.astype(float), usable.any(1)


def designs(confidence, values, known, model):
    x = (confidence-model['prior']['center']) / model['prior']['scale']
    z, mask, covered = context_design(values, known, model['scaler'])
    return {'covered_conf': x, 'covered_mask': np.column_stack((x, mask)),
            'covered_values': np.column_stack((x, mask, z))}, covered


def fit_source(base_logits, expert_logits, labels, ports, values, known):
    prior = old.fit_reliability(base_logits, expert_logits, labels, ports)
    weights = old.source_weights(labels, ports)
    model = dict(prior=prior, scaler=fit_context_scaler(values, known, weights), coefficients={})
    blocks, covered = designs(old.confidence_features(base_logits, expert_logits), values, known, model)
    delta = old.utility(labels, base_logits.argmax(1), expert_logits.argmax(1))
    use = covered & (delta != 0)
    model['supported_source'] = int(covered.sum())
    model['informative_source'] = int(use.sum())
    model['fallback'] = np.unique(delta[use]).size != 2
    if not model['fallback']:
        for name, x in blocks.items():
            fit = LogisticRegression(C=old.SOURCE_LR_C, max_iter=2000, solver='lbfgs', tol=1e-9)
            fit.fit(x[use], (delta[use] > 0).astype(int), sample_weight=len(labels)*weights[use])
            old.require(fit.n_iter_.max() < fit.max_iter, name + ' did not converge')
            model['coefficients'][name] = np.r_[fit.coef_[0], fit.intercept_[0]]
    return model


def predict_scores(model, base_logits, expert_logits, values, known):
    confidence = old.confidence_features(base_logits, expert_logits)
    blocks, covered = designs(confidence, values, known, model)
    original = old.design(confidence, model['prior']['center'], model['prior']['scale']) @ model['prior']['lr']
    scores = {'oof_lr0': original}
    for name, x in blocks.items():
        coefficient = model['coefficients'].get(name)
        scores[name] = original.copy()
        if coefficient is not None:
            scores[name][covered] = x[covered] @ coefficient[:-1] + coefficient[-1]
        old.require(np.isfinite(scores[name]).all(), 'nonfinite gate score')
    return scores, covered


def route(scores, base, expert, support_labels):
    observed = np.bincount(support_labels, minlength=old.K) > 0
    eligible = (base != expert) & observed[expert]
    return {name: np.where(eligible & (score > 0), expert, base) for name, score in scores.items()}
