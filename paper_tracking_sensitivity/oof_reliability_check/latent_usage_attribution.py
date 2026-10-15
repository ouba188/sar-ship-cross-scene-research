"""Fixed-readout attribution to a candidate span in source-standardized R50."""

import numpy as np


COMPONENTS = ('base_source', 'base_adaptation', 'expert_source',
              'expert_retained', 'expert_complement')


def span_basis(directions):
    u, s, _ = np.linalg.svd(np.asarray(directions, dtype=float), full_matrices=False)
    rank = int(np.sum(s > s[0]*1e-6))
    return u[:, :rank], s


def raw_readout_gradients(model, base_adaptation, base_residual_scale,
                          retained_support, complement_support, beta, discarded_scale):
    """R50 raw-coordinate gradients with trained support coefficients fixed."""
    p = np.asarray(model['pca'].components_, dtype=float)
    d, k = p.shape[1], p.shape[0]
    dxs = p.T / np.asarray(model['scale'])[None, :]
    base_source = dxs @ model['base'].coef_.T
    base_adapt = dxs @ (base_adaptation[:-1]/base_residual_scale[:, None]/np.sqrt(k))
    expert_source = dxs @ (model['expert'].coef_.T[:k]/model['expert_scale'][:k, None])
    retained_weights = .5*retained_support.T @ beta
    expert_retained = dxs @ (retained_weights[:k]/model['expert_scale'][:k, None]
                             /np.sqrt(retained_support.shape[1]))
    complement_weights = .5*complement_support.T @ beta
    g = complement_weights[:d]/discarded_scale[:d, None]/np.sqrt(complement_support.shape[1])
    expert_complement = g-p.T@(p@g)
    return dict(zip(COMPONENTS, (base_source, base_adapt, expert_source,
                               expert_retained, expert_complement)))


def centered_energy_fraction(gradient, q):
    centered = gradient-gradient.mean(axis=1, keepdims=True)
    total = np.square(centered).sum()
    return float(np.square(q.T@centered).sum()/total) if total else 0.


def score_contributions(raw, raw_center, raw_scale, q, gradients):
    latent = ((raw-raw_center)/raw_scale)@q
    result = {name: latent@(q.T@(raw_scale[:, None]*g)) for name, g in gradients.items()}
    return latent, result
