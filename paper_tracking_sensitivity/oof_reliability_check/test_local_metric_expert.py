import numpy as np
from sklearn.linear_model import Ridge

from local_metric_expert import fit_geometry, fit_predict, kernel_matrix


def test_linear_kernel_matches_independent_primal_ridge():
    rng = np.random.default_rng(7)
    x, prior = rng.normal(size=(90, 12)), rng.normal(size=(90, 8))
    support = rng.choice(90, 64, replace=False)
    labels = rng.integers(0, 5, 64)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    weights = 64/(len(counts)*counts[inverse])
    phi = np.column_stack((x, np.ones(90)))
    fit = Ridge(alpha=8., fit_intercept=False).fit(phi[support], 2*np.eye(8)[labels]-1-prior[support], sample_weight=weights)
    scores, _ = fit_predict(x, support, labels, prior)
    np.testing.assert_allclose(scores, prior+fit.predict(phi), atol=1e-12)


def test_source_geometry_ignores_target_features_and_preserves_source_energy():
    rng = np.random.default_rng(8)
    raw = rng.normal(size=(150, 12))
    source, labels = np.arange(110), np.arange(110) % 8
    x, metric, first = fit_geometry(raw, source, labels)
    raw[110:] += 10000
    other_x, other_metric, second = fit_geometry(raw, source, labels)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    np.testing.assert_array_equal(metric[source], other_metric[source])
    np.testing.assert_array_equal(x[source], other_x[source])
    np.testing.assert_allclose(np.sum(x[source]**2), np.sum(metric[source]**2), rtol=1e-12)
    assert np.isin(first['bandwidth_pairs'], source).all()


def test_kernel_is_psd_and_local_term_decays():
    x = np.array([[0., 0.], [1., 0.], [0., 1.]])
    gram = kernel_matrix(x, x, 1.)
    np.testing.assert_allclose(np.diag(gram), 2.)
    assert np.linalg.eigvalsh(gram).min() > 0
    np.testing.assert_allclose(kernel_matrix(np.array([[100., 100.]]), x, 1.), 1.)


def test_exact_prior_is_unchanged():
    rng = np.random.default_rng(2)
    x, labels = rng.normal(size=(64, 10)), np.arange(64) % 8
    prior = 2*np.eye(8)[labels]-1
    scores, dual = fit_predict(x, np.arange(64), labels, prior, 1.)
    np.testing.assert_array_equal(scores, prior)
    np.testing.assert_array_equal(dual, np.zeros((64, 8)))


def test_nonlinear_memory_can_recover_xor_outside_old_class():
    x = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (16, 1))
    labels = np.where(x[:, 0]*x[:, 1] > 0, 0, 2)
    prior = np.zeros((64, 8)); prior[:, 1] = .1
    local, _ = fit_predict(x, np.arange(64), labels, prior, 1.)
    linear, _ = fit_predict(x, np.arange(64), labels, prior)
    np.testing.assert_array_equal(local.argmax(1), labels)
    np.testing.assert_allclose(linear[:, 0], linear[:, 2], atol=1e-12)
