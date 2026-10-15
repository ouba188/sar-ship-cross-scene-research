import numpy as np
from sklearn.linear_model import Ridge

from visual_tail_expert import fit_tail, predict


def pca_fixture():
    rng = np.random.default_rng(1707)
    raw = rng.normal(size=(100, 12))
    source, query = np.arange(75), np.arange(75, 100)
    centered = raw-raw[source].mean(0)
    _, _, vh = np.linalg.svd(centered[source], full_matrices=False)
    retained = centered@vh[:4].T/2
    return raw, retained, source, query


def test_tail_matches_orthogonal_pca_remainder_and_source_energy():
    raw, retained, source, query = pca_fixture()
    tail, meta = fit_tail(raw, retained, source, query)
    x = raw-raw[source].mean(0)
    _, _, vh = np.linalg.svd(x[source], full_matrices=False)
    expected = x[query]-x[query]@vh[:4].T@vh[:4]
    np.testing.assert_allclose(tail, expected*meta['multiplier'], atol=1e-12)
    np.testing.assert_allclose(meta['source_tail_energy']*meta['multiplier']**2, meta['source_retained_energy'])
    assert meta['source_orthogonality_max'] < 1e-12


def test_query_changes_do_not_affect_fitted_source_parameters():
    raw, retained, source, query = pca_fixture()
    _, first = fit_tail(raw, retained, source, query)
    raw[query] += 1000
    retained[query] -= 500
    _, second = fit_tail(raw, retained, source, query)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_approximate_retained_projection_uses_source_least_squares():
    raw, _, source, query = pca_fixture()
    retained = raw@np.random.default_rng(3).normal(size=(12, 4))
    tail, meta = fit_tail(raw, retained, source, query)
    low = retained-retained[source].mean(0)
    centered = raw-raw[source].mean(0)
    decoder = np.linalg.lstsq(low[source], centered[source], rcond=None)[0]
    np.testing.assert_allclose(tail, (centered[query]-low[query]@decoder)*meta['multiplier'], atol=1e-12)
    assert meta['source_orthogonality_max'] < 1e-12


def test_predict_matches_weighted_primal_ridge():
    rng = np.random.default_rng(6)
    features, prior = rng.normal(size=(100, 30)), rng.normal(size=(100, 8))
    support, labels = np.arange(64), np.arange(64) % 5
    weights = 64/(5*np.bincount(labels)[labels])
    phi = np.column_stack((features, np.ones(100)))
    ref = Ridge(alpha=8, fit_intercept=False).fit(phi[support], 2*np.eye(8)[labels]-1-prior[support], sample_weight=weights)
    scores, _ = predict(features, prior, support, labels)
    np.testing.assert_allclose(scores, prior+ref.predict(phi), atol=1e-12)


def test_added_component_can_correct_information_absent_from_retained():
    retained = np.zeros((64, 3))
    tail = np.tile([-1., 1.], 32)[:, None]
    labels = np.where(tail[:, 0] < 0, 0, 2)
    prior = np.zeros((64, 8)); prior[:, 1] = .1
    added, _ = predict(np.column_stack((retained, tail)), prior, np.arange(64), labels)
    control, _ = predict(np.sqrt(2)*retained, prior, np.arange(64), labels)
    np.testing.assert_array_equal(added.argmax(1), labels)
    np.testing.assert_allclose(control[:, 0], control[:, 2], atol=1e-12)
