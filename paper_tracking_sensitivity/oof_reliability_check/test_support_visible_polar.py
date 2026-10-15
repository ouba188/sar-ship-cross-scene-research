"""The new evidence must preserve source scores yet affect support adaptation."""

import numpy as np
import torch

from aligned_residual_ops import residual_scores
from support_visible_polar import SupportVisibleGeometry


def geometry():
    rng = np.random.default_rng(41)
    basis, _ = np.linalg.qr(rng.normal(size=(512, 64)))
    weights = rng.normal(size=(8, 192))
    weights[-1] = weights[:-1].sum(0)
    return SupportVisibleGeometry(dict(ssl_mean=rng.normal(size=512),
        ssl_components=basis.T.copy(), expert_center=rng.normal(size=192),
        expert_scale=rng.uniform(.5, 2., 192), expert_coef=weights,
        expert_intercept=rng.normal(size=8)))


def test_prior_is_invariant_but_residual_geometry_changes():
    torch.manual_seed(9)
    g = geometry()
    xs, base, delta = [torch.randn(73, size, dtype=torch.float64) for size in (128, 512, 512)]
    added = g.project_evidence(delta)
    prior, phi = g(xs, base)
    new_prior, new_phi = g(xs, base+added)
    torch.testing.assert_close(new_prior, prior, atol=1e-10, rtol=1e-10)
    assert (new_phi-phi).norm() > .1
    y = torch.arange(64)%8
    before = residual_scores(phi[:64], prior[:64], y, phi[64:], prior[64:])
    after = residual_scores(new_phi[:64], new_prior[:64], y, new_phi[64:], new_prior[64:])
    assert (after-before).norm() > .01


def test_support_adaptation_loss_reaches_new_evidence():
    torch.manual_seed(12)
    g = geometry()
    xs, base = [torch.randn(73, size, dtype=torch.float64) for size in (128, 512)]
    delta = torch.randn(73, 512, dtype=torch.float64, requires_grad=True)
    prior, phi = g(xs, base+g.project_evidence(delta))
    prediction = residual_scores(phi[:64], prior[:64], torch.arange(64)%8, phi[64:], prior[64:])
    prediction.square().mean().backward()
    assert torch.isfinite(delta.grad).all()
    assert delta.grad[:64].norm() > 1e-8
    assert delta.grad[64:].norm() > 1e-8
