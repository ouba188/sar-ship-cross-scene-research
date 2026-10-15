import numpy as np
from scipy.optimize import check_grad

from compact_fusion_policy import (adaptive_weight, balanced_weights, features, fit, mix,
                                   objective, probabilities, select_fixed, utility_objective)


def test_mixture_endpoints_and_equal():
    b, e = probabilities(np.array([[3., 1., 0.], [0., 2., 1.]]),
                         np.array([[0., 3., 1.], [2., 1., 0.]]))
    np.testing.assert_array_equal(mix(b, e, 0), b)
    np.testing.assert_array_equal(mix(b, e, 1), e)
    np.testing.assert_allclose(mix(b, e, .5), (b+e)/2)
    np.testing.assert_allclose(mix(b, e, [.2, .8]).sum(1), 1.)


def test_equal_episode_and_class_mass():
    y = np.array([0, 0, 1, 0, 1, 1, 1])
    ep = np.array([0, 0, 0, 1, 1, 1, 1])
    w = balanced_weights(y, ep)
    for group in (0, 1):
        for c in (0, 1):
            np.testing.assert_allclose(w[(ep == group) & (y == c)].sum(), .25)


def test_analytic_gradient_against_finite_difference():
    rng = np.random.default_rng(8)
    design = np.column_stack((np.ones(25), rng.normal(size=(25, 5))))
    b, e = rng.uniform(.01, .8, size=(2, 25))
    w = np.full(25, 1/25)
    theta = rng.normal(size=6)
    error = check_grad(lambda t: objective(t, design, b, e, w)[0],
                       lambda t: objective(t, design, b, e, w)[1], theta)
    assert error < 1e-6


def test_adaptive_recovers_complementary_confidence_and_class_permutation():
    # Half the cases have a confident correct base, half a confident correct expert.
    b = np.tile([[.8, .2], [.55, .45]], (100, 1))
    e = np.tile([[.45, .55], [.2, .8]], (100, 1))
    y = np.tile([0, 1], 100)
    episode = np.repeat(np.arange(4), 50)
    model = fit(b, e, y, episode)
    w = adaptive_weight(model, b, e)
    assert w[::2].mean() < .5 < w[1::2].mean()
    assert model['final_objective'] < model['initial_objective']
    assert np.mean(mix(b, e, w).argmax(1) == y) == 1.
    np.testing.assert_allclose(features(b, e), features(b[:, ::-1], e[:, ::-1]))


def test_fixed_tie_prefers_half():
    p = np.tile([[.8, .2]], (10, 1))
    weight, _ = select_fixed(p, p, np.zeros(10, int), np.full(10, .1))
    assert weight == .5


def test_relative_correctness_gradient():
    rng = np.random.default_rng(91)
    design = np.column_stack((np.ones(30), rng.normal(size=(30, 5))))
    target = rng.integers(0, 2, 30)
    weight = np.full(30, 1/30)
    theta = rng.normal(size=6)
    error = check_grad(lambda t: utility_objective(t, design, target, weight)[0],
                       lambda t: utility_objective(t, design, target, weight)[1], theta)
    assert error < 1e-6
