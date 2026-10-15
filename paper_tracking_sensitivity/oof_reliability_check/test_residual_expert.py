import numpy as np
from sklearn.linear_model import Ridge

import residual_expert as method


def test_dual_correction_matches_independent_weighted_ridge():
    rng = np.random.default_rng(2)
    x, prior = rng.normal(size=(64, 13)), rng.normal(size=(64, 8))
    labels = rng.integers(0, 5, 64)
    _, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 64/(len(counts)*counts[inv])
    expected = Ridge(alpha=8., fit_intercept=False).fit(x, 2*np.eye(8)[labels]-1-prior,
                                                     sample_weight=weights).coef_.T
    np.testing.assert_allclose(method.fit_correction(x, labels, prior), expected, atol=1e-12)


def test_exact_prior_has_zero_correction():
    rng = np.random.default_rng(1)
    x, labels = rng.normal(size=(64, 8)), rng.integers(0, 8, 64)
    correction = method.fit_correction(x, labels, 2*np.eye(8)[labels]-1)
    np.testing.assert_array_equal(correction, np.zeros((8, 8)))


def test_geometry_does_not_fit_target_rows():
    raw = np.arange(120, dtype=float).reshape(20, 6)
    source = np.arange(10)
    first, center, scale = method.source_geometry(raw, source)
    raw[10:] += 10000
    second, other_center, other_scale = method.source_geometry(raw, source)
    np.testing.assert_array_equal(center, other_center)
    np.testing.assert_array_equal(scale, other_scale)
    np.testing.assert_array_equal(first[source], second[source])


def test_visual_correction_can_recover_class_outside_old_pair():
    x = np.column_stack((np.tile([-1., 1.], 32), np.ones(64)))
    labels = np.where(x[:, 0] < 0, 0, 2)
    prior = np.zeros((64, 8)); prior[:, 1] = .1
    correction = method.fit_correction(x, labels, prior)
    prediction = (prior+x@correction).argmax(1)
    np.testing.assert_array_equal(prediction, labels)
    assert np.all(prediction != prior.argmax(1))
