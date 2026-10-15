import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from frequency_pair_policy import conditional_utilities
from frequency_margin_policy import support_utilities


def test_generic_replay_and_no_coverage_fallback():
    rng = np.random.default_rng(11)
    labels = np.arange(64) % 8
    main, band = rng.normal(size=(2, 64, 8))
    sim = np.zeros((3, 64))
    sim[:, labels == 2] = 1.
    value, rho = conditional_utilities(sim, main, band, labels, np.zeros(3, int), np.ones(3, int))
    assert_allclose(value[:, 0], support_utilities(sim, main, band, labels)[:, 1], atol=0, rtol=0)
    assert_array_equal(rho, 0.)
    assert_array_equal(value[:, 0], value[:, 1])
    assert_array_equal(value[:, 0], value[:, 2])


def test_singleton_keeps_at_least_two_thirds_generic():
    labels = np.array([0, 1, 1, 1, 1, 1, 1, 1])
    main, band = np.zeros((8, 3)), np.ones((8, 3))
    value, rho = conditional_utilities(np.ones((1, 8)), main, band, labels, np.array([0]), np.array([1]))
    assert_allclose(rho, 1/3)
    assert_allclose(value, 0.)


def test_pair_margin_distinguishes_same_global_margin():
    labels = np.arange(8) % 2
    main = np.tile([.0, .1, .3], (8, 1))
    band = np.tile([.15, .2, .4], (8, 1))
    main[labels == 0, :2] = [.1, .0]
    band[labels == 0, :2] = [.2, .15]
    sim = np.ones((1, 8))
    value, rho = conditional_utilities(sim, main, band, labels, np.array([0]), np.array([1]))
    assert_allclose(value[0, :2], 0., atol=1e-15)
    assert value[0, 2] < 0
    assert_allclose(rho, 2/3)
    swapped, _ = conditional_utilities(sim, main, band, labels, np.array([1]), np.array([0]))
    assert_allclose(value, swapped)


def test_zero_neighbors_and_same_class_do_not_create_evidence():
    rng = np.random.default_rng(4)
    labels = np.arange(8) % 2
    main, band = rng.normal(size=(2, 8, 3))
    values, rho = conditional_utilities(-np.ones((1, 8)), main, band, labels, np.array([0]), np.array([1]))
    assert_array_equal(values, 0.)
    assert_array_equal(rho, 0.)
    values, rho = conditional_utilities(np.ones((1, 8)), main, band, labels, np.array([0]), np.array([0]))
    assert_array_equal(rho, 0.)
    assert_array_equal(values[:, 0], values[:, 1])
    assert_array_equal(values[:, 0], values[:, 2])
