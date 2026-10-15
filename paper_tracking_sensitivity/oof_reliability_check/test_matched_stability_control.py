import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from matched_stability_control import (equal_domain_pair_moments,
                                       reflect_with_matched_moments)


def source_rows():
    # Two equiprobable points per eligible cell, with unequal replication.
    cells = [(0, 'a', 6, (0, 0), (1, 0)),
             (0, 'b', 10, (6, 0), (0, 2)),
             (0, 'c', 8, (0, 6), (1, 1)),
             (1, 'b', 6, (2, 0), (0, 1)),
             (1, 'c', 12, (2, 6), (2, 0)),
             (1, 'd', 10, (8, 0), (1, -1)),
             (0, 'd', 4, (100, 100), (0, 0)),
             (1, 'a', 2, (-100, -100), (0, 0)),
             (7, 'e', 6, (200, 200), (0, 0))]
    z, y, domains = [], [], []
    for label, domain, count, center, offset in cells:
        center, offset = np.asarray(center), np.asarray(offset)
        z.extend([center - offset, center + offset] * (count // 2))
        y.extend([label] * count)
        domains.extend([domain] * count)
    return np.asarray(z, dtype=float), np.asarray(y), np.asarray(domains)


def test_manually_computed_unbalanced_moments_and_excluded_cells():
    z, y, domains = source_rows()
    result = equal_domain_pair_moments(z, y, domains, 0, 1)
    assert_allclose(result['mean_c'], [2, 2])
    assert_allclose(result['mean_d'], [4, 2])
    # Both between-domain covariances are [[8,-4],[-4,8]].
    # Mean within-domain covariance is [[7/6,0],[0,7/6]].
    assert_allclose(result['covariance'], [[55/6, -4], [-4, 55/6]])
    assert_array_equal(result['domain_counts'], [3, 3])
    expected_weights = np.concatenate([
        np.full(count, 1/(6*count)) for count in (6, 10, 8, 6, 12, 10)
    ] + [np.zeros(12)])
    assert_allclose(result['weights'], expected_weights)
    assert_allclose(result['weights'].sum(), 1)
    assert_allclose([result['weights'][y == k].sum() for k in (0, 1)], 0.5)


def test_each_class_requires_three_domains_and_threshold_is_inclusive():
    z, y, domains = source_rows()
    for label, domain in ((0, 'a'), (1, 'd')):
        keep = ~((y == label) & (domains == domain))
        with pytest.raises(ValueError, match='three eligible'):
            equal_domain_pair_moments(z[keep], y[keep], domains[keep], 0, 1)
    assert_array_equal(equal_domain_pair_moments(z, y, domains, 0, 1, 6)
                       ['domain_counts'], [3, 3])
    with pytest.raises(ValueError, match='three eligible'):
        equal_domain_pair_moments(z, y, domains, 0, 1, 7)
    five_rows = np.tile([[0., 0.], [1., 2.]], (15, 1))
    result = equal_domain_pair_moments(five_rows, np.tile([0, 1], 15),
                                     np.repeat([0, 1, 2], 10), 0, 1)
    assert_array_equal(result['domain_counts'], [3, 3])


def test_random_spd_signed_mean_gap_and_variance_matching():
    rng = np.random.default_rng(23)
    for _ in range(50):
        matrix = rng.normal(size=(2, 2))
        sigma = matrix @ matrix.T + 0.2 * np.eye(2)
        delta, reference = rng.normal(size=(2, 2))
        result = reflect_with_matched_moments(delta, sigma, reference)
        control = result['coefficients']
        assert not result['degenerate']
        assert_allclose(delta @ control, delta @ reference, rtol=1e-11, atol=1e-12)
        assert_allclose(control @ sigma @ control, reference @ sigma @ reference,
                        rtol=1e-11, atol=1e-12)
        assert result['mean_gap_error'] < 1e-11
        assert result['variance_error'] < 1e-11


def test_covariance_matches_weighted_actual_projection_variance():
    z, y, domains = source_rows()
    moments = equal_domain_pair_moments(z, y, domains, 0, 1)
    delta = moments['mean_c'] - moments['mean_d']
    result = reflect_with_matched_moments(delta, moments['covariance'])
    assert not result['degenerate']
    variances, gaps = [], []
    for coefficients in (np.array([0., 1.]), result['coefficients'], np.array([2., -3.])):
        projection = z @ coefficients
        means = [np.average(projection[y == k], weights=moments['weights'][y == k])
                 for k in (0, 1)]
        variance = sum(np.sum(moments['weights'][y == k]
                              * (projection[y == k] - means[k])**2) for k in (0, 1))
        assert_allclose(variance, coefficients @ moments['covariance'] @ coefficients)
        gaps.append(means[0] - means[1])
        variances.append(variance)
    assert_allclose(variances[0], variances[1])
    assert_allclose(gaps[0], gaps[1], atol=1e-14)


def test_optimal_axis_collinear_reference_is_degenerate():
    sigma = np.array([[3., 1.], [1., 2.]])
    reference = np.array([0., 1.])
    for perturbation in (0., 1e-12):
        delta = sigma @ (reference + [perturbation, 0.])
        result = reflect_with_matched_moments(delta, sigma)
        assert result['degenerate']
        assert_array_equal(result['coefficients'], reference)
        assert result['mean_gap_error'] == result['variance_error'] == 0


def test_singular_and_nullspace_cases_do_not_claim_a_matched_alternative():
    for sigma, delta in ((np.diag([2., 0.]), [1., 1.]),
                         (np.diag([2., 1e-12]), [1., 1.]),
                         (np.zeros((2, 2)), [1., 0.]),
                         (np.eye(2), [0., 0.])):
        reference = np.array([2., -1.])
        result = reflect_with_matched_moments(delta, sigma, reference)
        assert result['degenerate']
        assert_array_equal(result['coefficients'], reference)
        assert np.isnan(result['mean_gap_error'])
        assert np.isnan(result['variance_error'])
    result = reflect_with_matched_moments([2., 0.], np.diag([2., 0.]), [1., 3.])
    assert result['degenerate']
    assert_array_equal(result['coefficients'], [1., 3.])
    assert result['mean_gap_error'] == result['variance_error'] == 0


def test_relative_eigenvalue_threshold_without_regularization():
    sigma = np.diag([1., 2e-10])
    expected = np.array([4e-10/(1 + 2e-10), (1 - 2e-10)/(1 + 2e-10)])
    for scale in (1e-12, 1., 1e12):
        result = reflect_with_matched_moments([1., 1.], sigma * scale)
        assert not result['degenerate']
        assert_allclose(result['coefficients'], expected, rtol=1e-12)


def test_repeat_is_deterministic_and_api_has_no_target_inputs():
    z, y, domains = source_rows()
    originals = [a.copy() for a in (z, y, domains)]
    first = equal_domain_pair_moments(z, y, domains, 0, 1)
    second = equal_domain_pair_moments(z, y, domains, 0, 1)
    for key in first:
        assert_array_equal(first[key], second[key])
    delta = first['mean_c'] - first['mean_d']
    reference = np.array([0., 1.])
    sigma_before = first['covariance'].copy()
    a = reflect_with_matched_moments(delta, first['covariance'], reference)
    b = reflect_with_matched_moments(delta, first['covariance'], reference)
    for key in a:
        assert_array_equal(a[key], b[key])
    for current, original in zip((z, y, domains), originals):
        assert_array_equal(current, original)
    assert_array_equal(first['covariance'], sigma_before)
    assert_array_equal(reference, [0., 1.])
    assert list(inspect.signature(equal_domain_pair_moments).parameters) == [
        'z', 'y', 'domains', 'c', 'd', 'min_count']
    assert list(inspect.signature(reflect_with_matched_moments).parameters) == [
        'mean_difference', 'covariance', 'reference']


def test_translation_and_common_rescaling_invariance():
    z, y, domains = source_rows()
    base = equal_domain_pair_moments(z, y, domains, 0, 1)
    control = reflect_with_matched_moments(base['mean_c'] - base['mean_d'],
                                          base['covariance'])
    for scale in (-3., 1e-5, 1e5):
        shift = np.array([13., -21.])
        result = equal_domain_pair_moments(scale * z + shift, y, domains, 0, 1)
        assert_allclose(result['mean_c'], scale * base['mean_c'] + shift)
        assert_allclose(result['mean_d'], scale * base['mean_d'] + shift)
        assert_allclose(result['covariance'], scale**2 * base['covariance'], rtol=1e-9)
        reflected = reflect_with_matched_moments(result['mean_c'] - result['mean_d'],
                                                result['covariance'])
        assert_allclose(reflected['coefficients'], control['coefficients'], atol=1e-9)
        assert_array_equal(result['weights'], base['weights'])


def test_coordinate_rescaling_transforms_coefficients_contravariantly():
    sigma = np.array([[4., 1.], [1., 2.]])
    delta, reference = np.array([2., 3.]), np.array([0., 1.])
    base = reflect_with_matched_moments(delta, sigma, reference)
    scales = np.array([-3., 0.2])
    result = reflect_with_matched_moments(scales * delta,
                                         scales[:, None] * sigma * scales[None, :],
                                         reference / scales)
    assert_allclose(result['coefficients'] * scales, base['coefficients'])


def test_float32_chunk_boundaries_preserve_moments():
    z, y, domains = source_rows()
    base = equal_domain_pair_moments(z, y, domains, 0, 1)
    # Replicate only eligible cells, so exclusion semantics do not change.
    eligible = base['weights'] > 0
    repeated = np.repeat(z[eligible].astype(np.float32), 300, axis=0)
    result = equal_domain_pair_moments(repeated, np.repeat(y[eligible], 300),
                                     np.repeat(domains[eligible], 300), 0, 1)
    for key in ('mean_c', 'mean_d', 'covariance', 'domain_counts'):
        assert_allclose(result[key], base[key], atol=1e-12)
    assert repeated.dtype == np.float32


def test_invalid_shapes_and_non_psd_covariance_are_rejected():
    z, y, domains = source_rows()
    with pytest.raises(ValueError, match='projections'):
        equal_domain_pair_moments(z[:, :1], y, domains, 0, 1)
    with pytest.raises(ValueError, match='labels and domains'):
        equal_domain_pair_moments(z, y[:-1], domains, 0, 1)
    with pytest.raises(ValueError, match='distinct classes'):
        equal_domain_pair_moments(z, y, domains, 0, 0)
    for sigma, message in (([[1, 2], [0, 1]], 'symmetric'),
                            ([[1, 0], [0, -1]], 'positive semidefinite')):
        with pytest.raises(ValueError, match=message):
            reflect_with_matched_moments([1, 1], sigma)
