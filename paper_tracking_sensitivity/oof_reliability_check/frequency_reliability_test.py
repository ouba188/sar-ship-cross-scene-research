"""Focused checks for deletion, CE selection and unchanged endpoint behavior."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from frequency_reliability_core import (coefficients, strict_loo, ce_strength,
    balanced_weights, select_rule, local_strength)


def test_strict_loo_matches_independent_primal_singleton():
    rng = np.random.default_rng(901)
    x = rng.normal(size=(9, 5))
    prior = rng.normal(size=(9, 3))
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2])
    gram = x @ x.T
    old, new = strict_loo(gram, gram * 1.25, prior, y)
    for held in range(len(y)):
        keep = np.arange(len(y)) != held
        for factor, actual in ((1., old), (1.25, new)):
            z = x * np.sqrt(factor)
            counts = {c: int(np.sum(y[keep] == c)) for c in np.unique(y[keep])}
            w = np.array([keep.sum() / (len(counts)*counts[c]) for c in y[keep]])
            design = np.vstack([z[keep]*np.sqrt(w[:, None]), np.sqrt(8)*np.eye(5)])
            target = np.vstack([(2*np.eye(3)[y[keep]]-1-prior[keep])*np.sqrt(w[:, None]), np.zeros((5, 3))])
            coef = np.linalg.lstsq(design, target, rcond=None)[0]
            np.testing.assert_allclose(actual[held], prior[held] + z[held] @ coef, atol=1e-12)


def test_omitted_label_cannot_change_its_prediction():
    rng = np.random.default_rng(23)
    x, prior = rng.normal(size=(7, 4)), rng.normal(size=(7, 3))
    y = np.array([0, 0, 0, 1, 1, 1, 2])
    gram = x @ x.T
    original = strict_loo(gram, 1.25*gram, prior, y)
    y[-1] = 0
    changed = strict_loo(gram, 1.25*gram, prior, y)
    for a, b in zip(original, changed):
        np.testing.assert_array_equal(a[-1], b[-1])


def test_ce_strength_against_independent_optimizer():
    rng = np.random.default_rng(512)
    for _ in range(20):
        old, new = rng.normal(size=(13, 3)), rng.normal(size=(13, 3))
        y = rng.integers(0, 3, 13)
        def loss(g):
            z = old + g*(new-old)
            return np.average(logsumexp(z, axis=1)-z[np.arange(13), y], weights=balanced_weights(y))
        numerical = minimize_scalar(loss, method='bounded', bounds=(0, 1), options={'xatol': 1e-12})
        expected = min(loss(0.), loss(1.), numerical.fun)
        assert abs(loss(ce_strength(old, new, y))-expected) < 1e-10


def test_identical_scores_and_source_tie_preserve_learned():
    scores = np.arange(15).reshape(5, 3).astype(float)
    assert ce_strength(scores, scores, np.array([0, 1, 2, 1, 0])) == 1.
    assert select_rule([{'loo_ce_ba': 45., 'learned_ba': 45.}]) == ('learned', 0.)
    assert select_rule([{'loo_ce_ba': 46., 'learned_ba': 45.}]) == ('loo_ce', 1.)


def test_local_top8_signed_utility_and_zero_evidence():
    sim = np.array([[10., 9., 8., 7., 6., 5., 4., 3., 2., 1.], [-1.]*10, [0.]*10])
    utility = np.array([1., -2., 1., -2., 1., -2., 1., -2., 1000., 1000.])
    gamma = local_strength(sim, utility)
    np.testing.assert_allclose(gamma, [28./76., 0., 0.])
    np.testing.assert_array_equal(local_strength(sim, np.zeros(10)), np.zeros(3))
