from itertools import combinations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from conditional_invariant_directions import nuisance_penalized_basis


def two_class_source(counts=None):
    x, y, domains = [], [], []
    for c in range(2):
        for d, offset in enumerate((-6., 0., 6.)):
            n = 5 if counts is None else counts[c][d]
            x.extend([[2.*(1-2*c), offset]]*n)
            y.extend([c]*n)
            domains.extend(['port_'+str(d)]*n)
    return np.asarray(x, dtype=np.float32), np.asarray(y), np.asarray(domains)


def test_suppress_within_class_nuisance_preserve_label_direction():
    x, y, domains = two_class_source()
    pooled = np.array([[1.], [1.]])/np.sqrt(2)
    result = nuisance_penalized_basis(x, y, domains, [[0, 1]], pooled)
    # U.T U is diag(0,24); nuisance is reduced 25-fold relative to signal.
    expected = np.array([[1.], [1/25.]])
    expected /= np.linalg.norm(expected)
    assert_allclose(result['basis'], expected, rtol=1e-6)
    assert np.linalg.norm(result['nuisance_rows']@result['basis']) < .1*np.linalg.norm(
        result['nuisance_rows']@pooled)
    label_gap = x[y == 0].mean(axis=0)-x[y == 1].mean(axis=0)
    assert (label_gap@result['basis'])[0] > (label_gap@pooled)[0]
    assert_allclose(np.linalg.norm(result['basis'], axis=0), 1., atol=1e-7)


def test_label_domain_frequency_confounding_does_not_define_nuisance():
    counts = ((1031, 5, 5), (5, 5, 1027))
    x, y, domains = two_class_source(counts)
    pooled = np.eye(2)[:, :1]
    balanced = nuisance_penalized_basis(*two_class_source(), [[0, 1]], pooled)
    confounded = nuisance_penalized_basis(x, y, domains, [[0, 1]], pooled)
    assert abs(x[domains == 'port_0', 0].mean()-
               x[domains == 'port_2', 0].mean()) > 3.
    assert_allclose(confounded['nuisance_rows'], balanced['nuisance_rows'])
    assert_array_equal(confounded['nuisance_rows'][:, 0], 0.)
    assert_array_equal(confounded['basis'], pooled)
    # Renaming labels/domains only reorders rows, not the resulting operator.
    renamed = nuisance_penalized_basis(x, 7-y, np.where(domains == 'port_0',
                                       'z', domains), [[7, 6]], pooled)
    assert_allclose(renamed['basis'], confounded['basis'])


def test_no_nuisance_exact_identity_and_zero_strength():
    x, y, domains = two_class_source()
    x[:, 1] = 0
    pooled = np.eye(2, dtype=np.float32)
    result = nuisance_penalized_basis(x, y, domains, [[0, 1], [1, 0]], pooled)
    assert_array_equal(result['basis'], pooled)
    assert_array_equal(result['nuisance_rows'], np.zeros((6, 2)))
    assert_array_equal(result['retained_norm'], [1., 1.])
    result = nuisance_penalized_basis(*two_class_source(), [[0, 1], [1, 0]],
                                      pooled, strength=0)
    assert_array_equal(result['basis'], pooled)


def test_normalization_exposes_pure_shrinkage_not_false_stability():
    pooled = np.array([[0.], [1.]])
    result = nuisance_penalized_basis(*two_class_source(), [[0, 1]], pooled,
                                      strength=100.)
    assert_allclose(result['retained_norm'], [1/2401], rtol=1e-8)
    assert_array_equal(result['basis'], pooled)
    assert_allclose(result['nuisance_rows']@result['basis'],
                    result['nuisance_rows']@pooled)


def test_cell_threshold_three_domain_requirement_and_no_eligible_classes():
    x, y, domains = two_class_source(((5, 5, 5), (5, 5, 4)))
    pooled = np.array([[1.], [0.]])
    result = nuisance_penalized_basis(x, y, domains, [[0, 1]], pooled)
    assert_array_equal(result['cell_classes'], [0, 0, 0])
    assert_array_equal(result['cell_domains'], ['port_0', 'port_1', 'port_2'])
    expected = np.array([[0., -6.], [0., 0.], [0., 6.]])/np.sqrt(3)
    assert_allclose(result['nuisance_rows'], expected, atol=1e-7)
    empty = nuisance_penalized_basis(x, y, domains, [[0, 1]], pooled, min_count=6)
    assert empty['nuisance_rows'].shape == (0, 2)
    assert empty['cell_classes'].shape == empty['cell_domains'].shape == (0,)
    assert_array_equal(empty['basis'], pooled)


def test_eight_classes_unequal_domains_weights_dense_solution_and_no_mutation():
    rng = np.random.default_rng(12)
    x, y, domains, expected_u = [], [], [], []
    for c in range(8):
        n_domains = 3+c % 3
        means = rng.normal(size=(n_domains, 9)).astype(np.float32)
        expected_u.extend((means.astype(float)-means.mean(axis=0, dtype=float))/
                          np.sqrt(8*n_domains))
        for d, mean in enumerate(means):
            x.extend([mean]*(5+d))
            y.extend([c]*(5+d))
            domains.extend([str(d)]*(5+d))
    x = np.asarray(x, dtype=np.float32)
    original = x.copy()
    pairs = np.asarray(list(combinations(range(8), 2)))
    pooled = rng.normal(size=(9, len(pairs)))
    pooled /= np.linalg.norm(pooled, axis=0)
    original_basis = pooled.copy()
    result = nuisance_penalized_basis(x, y, domains, pairs, pooled, strength=2.5)
    u = np.asarray(expected_u)
    direct = np.linalg.solve(np.eye(9)+2.5*u.T@u, pooled)
    assert_allclose(result['nuisance_rows'], u, rtol=1e-6, atol=1e-7)
    assert_allclose(result['retained_norm'], np.linalg.norm(direct, axis=0))
    assert_allclose(result['basis'], direct/np.linalg.norm(direct, axis=0),
                    rtol=1e-6, atol=1e-7)
    assert result['basis'].dtype == result['nuisance_rows'].dtype == np.float32
    assert result['cell_classes'].dtype.kind == 'i'
    assert result['cell_domains'].dtype.kind == 'U'
    assert_array_equal(x, original)
    assert_array_equal(pooled, original_basis)


def test_invalid_shapes_strength_and_zero_direction():
    args = two_class_source()
    with pytest.raises(ValueError, match='pairs'):
        nuisance_penalized_basis(*args, [[0, 1]], np.eye(2))
    with pytest.raises(ValueError, match='nonnegative'):
        nuisance_penalized_basis(*args, [[0, 1]], np.ones((2, 1)), strength=-1)
    with pytest.raises(ValueError, match='nonzero'):
        nuisance_penalized_basis(*args, [[0, 1]], np.zeros((2, 1)))
