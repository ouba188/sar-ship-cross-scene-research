import numpy as np
from scipy.optimize import minimize_scalar

from support_shrinkage import loo_residuals, select_strength


def test_loo_matches_augmented_svd_with_fold_class_weights():
    rng = np.random.default_rng(10)
    features = np.column_stack((rng.normal(size=(9, 4)), np.ones(9)))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2])
    prior = rng.normal(size=(9, 8))
    actual = loo_residuals(features, labels, prior)
    for i in range(9):
        keep = np.arange(9) != i
        y = labels[keep]
        _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
        root = np.sqrt(8 / (len(counts) * counts[inv]))
        x = np.vstack((root[:, None] * features[keep], np.sqrt(8) * np.eye(5)))
        target = np.vstack((root[:, None] * (2 * np.eye(8)[y] - 1 - prior[keep]), np.zeros((5, 8))))
        coef = np.linalg.lstsq(x, target, rcond=None)[0]
        np.testing.assert_allclose(actual[i], features[i] @ coef, atol=1e-12)


def test_heldout_label_does_not_change_own_loo_prediction():
    rng = np.random.default_rng(12)
    x, p = rng.normal(size=(8, 4)), rng.normal(size=(8, 8))
    y = np.array([0, 0, 0, 1, 1, 1, 2, 7])
    before = loo_residuals(x, y, p)
    y[-1] = 0
    after = loo_residuals(x, y, p)
    np.testing.assert_array_equal(before[-1], after[-1])


def test_strength_matches_independent_scalar_minimization():
    rng = np.random.default_rng(13)
    y = np.array([0] * 8 + [1] * 4 + [2])
    prior = rng.normal(size=(13, 8))
    target = 2 * np.eye(8)[y] - 1
    residual = 2 * (target - prior) + rng.normal(size=(13, 8))
    def loss(alpha):
        squared = np.sum((target - prior - alpha * residual)**2, axis=1)
        return np.mean([squared[y == c].mean() for c in np.unique(y)])
    expected = minimize_scalar(loss, bounds=(0., 1.), method='bounded').x
    np.testing.assert_allclose(select_strength(y, prior, residual)[0], expected, atol=1e-7)


def test_zero_harmful_and_underfit_residuals():
    y = np.arange(8)
    prior = np.zeros((8, 8))
    target = 2 * np.eye(8)[y] - 1
    assert select_strength(y, prior, np.zeros_like(prior))[0] == 0
    assert select_strength(y, prior, -target)[0] == 0
    assert select_strength(y, prior, .5 * target)[0] == 1
