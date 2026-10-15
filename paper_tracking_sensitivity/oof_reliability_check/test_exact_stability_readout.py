import numpy as np
from numpy.testing import assert_allclose

from exact_stability_readout import exact_pair_shift


def test_duplicated_identical_empirical_distributions_have_zero_distance():
    target_z = np.concatenate((np.arange(20), np.arange(20) + 30))
    target_y = np.repeat([0, 1], 20)
    source_z, source_y = np.tile(target_z, 3), np.tile(target_y, 3)
    result = exact_pair_shift(source_z, source_y, np.full(120, 1/120),
                              target_z, target_y, 0, 1, [33.25, 33.25])
    assert_allclose(list(result.values()), 0., atol=1e-14)


def test_pure_translation_uses_shared_source_scale_and_floor():
    z = np.array([0., 1., 4., 10., 12., 15.])
    y = np.repeat([0, 1], 3)
    target_z = z + np.where(y == 0, 3., -6.)
    for variances, sigma in (([4., 14.], 3.), ([0., 0.], 1e-8)):
        result = exact_pair_shift(z, y, np.ones(6), target_z, y, 0, 1, variances)
        assert_allclose(result['class_c_w1'], 3/sigma)
        assert_allclose(result['class_d_w1'], 6/sigma)
        assert_allclose(result['conditional_w1'], 4.5/sigma)


def test_nonuniform_domain_and_class_weights_match_analytic_distance():
    # Class 0: one domain at 0 and a three-row domain at 4, equal domain mass.
    # Target is a point mass at 1: W1 = 0.5*1 + 0.5*3 = 2.
    # Class 1: masses (0.75, 0.25) at (0, 4), target masses (0.5, 0.5).
    # Transport 0.25 mass over distance 4: W1 = 1.
    # Class masses deliberately differ; each class is normalized separately.
    result = exact_pair_shift([0., 4., 4., 4., 0., 4.], [0, 0, 0, 0, 1, 1],
                              [3., 1., 1., 1., 0.75, 0.25],
                              [1., 1., 0., 4.], [0, 0, 1, 1], 0, 1, [1., 1.])
    assert_allclose(result['class_c_w1'], 2.)
    assert_allclose(result['class_d_w1'], 1.)
    assert_allclose(result['conditional_w1'], 1.5)


def test_zero_weight_cells_and_outside_classes_are_excluded():
    result = exact_pair_shift([0., 1000., 2., -1000., 500.], [0, 0, 1, 1, 7],
                              [0.5, 0., 0.5, 0., 1.],
                              [0., 2., -500.], [0, 1, 7], 0, 1, [1., 1.])
    assert_allclose(list(result.values()), 0.)


def test_class_swap_exchanges_distances_and_preserves_aggregate():
    args = ([0., 4., 1., 3.], [2, 2, 7, 7], [1., 3., 2., 1.],
            [1., 2., 5., 6.], [2, 2, 7, 7])
    forward = exact_pair_shift(*args, 2, 7, [2., 6.])
    reverse = exact_pair_shift(*args, 7, 2, [6., 2.])
    assert_allclose(forward['class_c_w1'], reverse['class_d_w1'])
    assert_allclose(forward['class_d_w1'], reverse['class_c_w1'])
    assert_allclose(forward['conditional_w1'], reverse['conditional_w1'])
