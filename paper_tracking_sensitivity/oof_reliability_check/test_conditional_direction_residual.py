import numpy as np
import pandas as pd

from conditional_direction_residual import (
    direction_kernel, fit_predict, normalize_direction, rank_conditions)
from residual_expert import fit_correction


def example():
    rng = np.random.default_rng(52)
    phi = np.c_[rng.normal(size=(31, 5)), np.ones(31)]
    prior = rng.normal(size=(31, 3))
    direction = normalize_direction(rng.normal(size=(31, 4)))
    condition = rng.uniform(size=(31, 2))
    support = np.arange(9)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    return phi, prior, direction, condition, support, labels


def test_zero_strength_is_original_correction():
    phi, prior, direction, condition, s, y = example()
    expected = prior+phi@fit_correction(phi[s], y, prior[s])
    np.testing.assert_array_equal(fit_predict(phi, prior, direction, condition, s, y, 0), expected)


def test_unconditional_matches_explicit_feature_ridge():
    phi, prior, direction, condition, s, y = example()
    expanded = np.c_[phi, np.sqrt(.25)*direction]
    expected = prior+expanded@fit_correction(expanded[s], y, prior[s])
    actual = fit_predict(phi, prior, direction, condition, s, y, .25)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.max(np.abs(actual-fit_predict(phi, prior, direction, condition, s, y, 0))) > .001


def test_condition_preserves_diagonal_and_positive_semidefiniteness():
    _, _, direction, condition, _, _ = example()
    s = np.arange(len(direction))
    plain = direction_kernel(direction, condition, s)
    conditioned = direction_kernel(direction, condition, s, True)
    np.testing.assert_allclose(np.diag(plain), np.diag(conditioned))
    assert np.linalg.eigvalsh(conditioned).min() > -1e-10
    assert np.max(np.abs(plain-conditioned)) > .01
    constant = direction_kernel(direction, np.zeros_like(condition), s, True)
    np.testing.assert_array_equal(plain, constant)


def test_missing_ranks_and_zero_direction_are_finite():
    frame = pd.DataFrame({'elongation_proxy': [1., 2., np.nan, 2.],
                          'center_excess_proxy': [np.nan]*4})
    ranks = rank_conditions(frame)
    assert np.isfinite(ranks).all()
    np.testing.assert_array_equal(ranks[:, 1], .5*np.ones(4))
    assert ranks[1, 0] == ranks[3, 0]
    np.testing.assert_array_equal(normalize_direction(np.zeros((3, 4))), np.zeros((3, 4)))


def test_uniform_condition_control_changes_weights_not_diagonal():
    _, _, direction, condition, s, _ = example()
    kernel = direction_kernel(direction, condition, s, 'uniform')
    plain = direction_kernel(direction, condition, s)
    np.testing.assert_allclose(kernel[s, np.arange(len(s))], plain[s, np.arange(len(s))])
    ratio = kernel[len(s):]/plain[len(s):]
    np.testing.assert_allclose(ratio, np.full_like(ratio, ratio[0, 0]))
    assert 0 < ratio[0, 0] < 1
    assert np.linalg.eigvalsh(kernel[s]).min() > -1e-10
