import numpy as np

from error_focused_expert import error_weights, fit_error_focused
from residual_expert import fit_correction


def test_error_emphasis_preserves_each_class_mass():
    labels = np.array([0, 0, 0, 1, 1, 2])
    scores = np.eye(8)[[0, 1, 0, 1, 2, 3]]
    weights = error_weights(labels, scores)
    np.testing.assert_allclose(np.bincount(labels, weights=weights)[:3], 2.)
    assert weights[1] == 2 * weights[0]
    assert weights[4] == 2 * weights[3]
    assert weights[5] == 2.


def test_constant_difficulty_within_class_recovers_old_fit():
    rng = np.random.default_rng(29)
    x, prior = rng.normal(size=(12, 5)), rng.normal(size=(12, 8))
    labels = np.array([0]*6 + [1]*4 + [2]*2)
    scores = np.eye(8)[[0]*6 + [3]*4 + [2]*2]
    coefficient, _ = fit_error_focused(x, labels, prior, scores)
    np.testing.assert_allclose(coefficient, fit_correction(x, labels, prior), atol=1e-12)


def test_weighted_fit_matches_augmented_svd():
    rng = np.random.default_rng(30)
    x, prior, scores = (rng.normal(size=shape) for shape in ((12, 17), (12, 8), (12, 8)))
    labels = np.arange(12) % 4
    coefficient, weights = fit_error_focused(x, labels, prior, scores)
    root = np.sqrt(weights)[:, None]
    a = np.vstack((root * x, np.sqrt(8) * np.eye(17)))
    b = np.vstack((root * (2*np.eye(8)[labels]-1-prior), np.zeros((17, 8))))
    np.testing.assert_allclose(coefficient, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-12)


def test_score_magnitude_does_not_change_mistake_weights():
    labels = np.array([0, 0, 1, 1])
    scores = np.eye(8)[[0, 2, 1, 3]]
    np.testing.assert_array_equal(error_weights(labels, scores), error_weights(labels, scores*100+7))
