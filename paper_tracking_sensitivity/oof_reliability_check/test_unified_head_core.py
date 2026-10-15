import numpy as np

from unified_head_core import class_weights, combined, fit_diagnostics, score


def example():
    rng = np.random.default_rng(43)
    xb, xe = rng.normal(size=(15, 4)), rng.normal(size=(15, 6))
    pb, pe = rng.normal(size=(15, 3)), rng.normal(size=(15, 3))
    support = np.arange(7)
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    return xb, xe, pb, pe, support, labels


def test_union_dual_equals_independent_concatenated_primal():
    xb, xe, pb, pe, s, y = example()
    ratio, strength = .25, 1.7
    x = np.c_[np.sqrt(1-ratio)*xb, np.sqrt(ratio)*xe]
    prior = (1-ratio)*pb+ratio*pe
    w = class_weights(y)
    residual = 2*np.eye(3)[y]-1.-prior[s]
    coefficient = np.linalg.solve(x[s].T@(w[:, None]*x[s])+strength*np.eye(x.shape[1]),
                                  x[s].T@(w[:, None]*residual))
    expected = prior+x@coefficient
    actual, _ = combined(xb@xb[s].T, xe@xe[s].T, pb, pe, s, y, ratio, strength)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_endpoints_preserve_each_original_head():
    xb, xe, pb, pe, s, y = example()
    kb, ke = xb@xb[s].T, xe@xe[s].T
    for ratio, kernel, prior in [(0., kb, pb), (1., ke, pe)]:
        actual, _ = combined(kb, ke, pb, pe, s, y, ratio)
        expected, _ = score(kernel, prior, s, y)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_identical_features_preserve_average_prior_solution():
    xb, _, pb, pe, s, y = example()
    k = xb@xb[s].T
    b, _ = score(k, pb, s, y)
    e, _ = score(k, pe, s, y)
    actual, _ = combined(k, k, pb, pe, s, y)
    np.testing.assert_allclose(actual, .5*(b+e), atol=1e-12)


def test_joint_fit_is_valid_minimum_and_accounting_exact():
    xb, xe, pb, pe, s, y = example()
    result = fit_diagnostics(xb@xb[s].T, xe@xe[s].T, pb, pe, s, y)
    assert result['unified_objective'] <= result['late_objective']+1e-9


def test_scratch_does_not_silently_use_source_prior():
    xb, xe, pb, pe, s, y = example()
    kb, ke = xb@xb[s].T, xe@xe[s].T
    first, _ = combined(kb, ke, pb, pe, s, y, anchored=False)
    second, _ = combined(kb, ke, 9*pb, 17*pe, s, y, anchored=False)
    np.testing.assert_array_equal(first, second)
