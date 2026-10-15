import numpy as np

from complementary_expert import fit_predict_complement
from residual_expert import fit_correction


def fixture():
    rng = np.random.default_rng(17)
    x = np.column_stack((rng.normal(size=(18, 5)), np.ones(18)))
    support = np.array([0, 2, 3, 5, 7, 9, 11, 13])
    y = np.array([0, 0, 0, 1, 1, 2, 2, 7])
    return x, support, y, rng.normal(size=(8, 8)), rng.normal(size=(18, 8)), rng.normal(size=(18, 8))


def test_complement_matches_independent_augmented_svd():
    x, support, y, training, base, contrast = fixture()
    scores, coef = fit_predict_complement(x, support, y, training, base, contrast)
    counts = np.bincount(y, minlength=8)
    root = np.sqrt(len(y) / (np.count_nonzero(counts) * counts[y]))[:, None]
    target = 2 * np.eye(8)[y] - 1 - training - contrast[support]
    augmented = np.vstack((root * x[support], np.sqrt(8) * np.eye(x.shape[1])))
    response = np.vstack((root * target, np.zeros((x.shape[1], 8))))
    independent = np.linalg.lstsq(augmented, response, rcond=None)[0]
    np.testing.assert_allclose(coef, independent, atol=1e-12)
    np.testing.assert_allclose(scores, base + contrast + x @ independent, atol=1e-12)


def test_source_base_reduction_recovers_old_anchored_expert():
    x, support, y, _, base, contrast = fixture()
    scores, _ = fit_predict_complement(x, support, y, base[support], base, contrast)
    old_prior = base + contrast
    old = old_prior + x @ fit_correction(x[support], y, old_prior[support])
    np.testing.assert_allclose(scores, old, atol=1e-12)


def test_deployment_changes_do_not_retrain_correction():
    x, support, y, training, base, contrast = fixture()
    before, coef = fit_predict_complement(x, support, y, training, base, contrast)
    changed_base = base.copy()
    changed_base[:, 3] += 10
    after, other_coef = fit_predict_complement(x, support, y, training, changed_base, contrast)
    np.testing.assert_array_equal(coef, other_coef)
    np.testing.assert_allclose(after - before, changed_base - base, atol=1e-12)


def test_exact_training_prior_needs_no_extra_correction():
    x, support, y, _, base, contrast = fixture()
    training = 2 * np.eye(8)[y] - 1
    contrast = np.zeros_like(contrast)
    scores, coef = fit_predict_complement(x, support, y, training, base, contrast)
    np.testing.assert_array_equal(coef, np.zeros_like(coef))
    np.testing.assert_array_equal(scores, base)
