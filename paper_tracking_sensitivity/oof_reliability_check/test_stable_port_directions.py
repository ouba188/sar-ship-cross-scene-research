import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from stable_port_directions import fit_direction_bank, transform_direction


def nuisance_source(counts=(5, 5, 5)):
    x, y, ports = [], [], []
    for port, (nuisance, count) in enumerate(zip((8., 8., -8.), counts)):
        delta = np.array([2., nuisance])
        for label, sign in ((0, 1.), (1, -1.)):
            x.extend([np.array([5., -3.])+sign*delta/2]*count)
            y.extend([label]*count)
            ports.extend([port]*count)
    return np.asarray(x), np.asarray(y), np.asarray(ports)


def test_sign_flipping_nuisance_downweighted_and_unseen_port_ranking():
    x, y, ports = nuisance_source()
    bank = fit_direction_bank(x, y, ports)
    assert_array_equal(bank['pairs'], [[0, 1]])
    assert_array_equal(bank['pair_ports'], [3])
    pooled_ratio = bank['pooled_basis'][1, 0]/bank['pooled_basis'][0, 0]
    stable_ratio = bank['stable_basis'][1, 0]/bank['stable_basis'][0, 0]
    assert_allclose(pooled_ratio, 4/3)
    assert_allclose(stable_ratio, 4/27)
    for kind in ('pooled', 'stable'):
        assert_allclose(np.linalg.norm(bank[kind+'_basis'], axis=0), 1.)

    # The new port reverses the nuisance: only the stable direction ranks
    # class 0 above class 1, despite a shared large background offset.
    heldout = np.array([[101., 46.], [99., 54.]])
    pooled = transform_direction(heldout, bank, 'pooled')[:, 0]
    stable = transform_direction(heldout, bank, 'stable')[:, 0]
    assert pooled[0] < pooled[1]
    assert stable[0] > stable[1]
    expected_cosine = (2/np.sqrt(17)-15/17)/3
    assert_allclose(bank['source_cosine_mean'], [expected_cosine])
    assert_allclose(bank['source_cosine_positive_fraction'], [2/3])
    assert_allclose(bank['coherence_mean'], [(4+64/81)/(4+64/9)])


def test_contrasts_equal_port_weight_but_projection_statistics_equal_row_weight():
    x, y, ports = nuisance_source((5, 5, 25))
    bank = fit_direction_bank(x, y, ports)
    balanced_bank = fit_direction_bank(*nuisance_source())
    for kind in ('pooled', 'stable'):
        assert_allclose(bank[kind+'_basis'], balanced_bank[kind+'_basis'])
        projected = x@bank[kind+'_basis']
        assert_allclose(bank[kind+'_center'], projected.mean(axis=0))
        assert_allclose(bank[kind+'_scale'], projected.std(axis=0))


def test_pairs_require_both_class_counts_and_enough_ports():
    x, labels, ports = [], [], []
    for port in range(4):
        for label, count in ((0, 4 if port == 3 else 5),
                             (2, 5), (7, 5 if port < 2 else 4)):
            x.extend([[float(label), float(port)]]*count)
            labels.extend([label]*count)
            ports.extend([port]*count)
    bank = fit_direction_bank(x, labels, ports)
    assert_array_equal(bank['pairs'], [[0, 2]])
    assert_array_equal(bank['pair_ports'], [3])
    with pytest.raises(ValueError, match='No nondegenerate'):
        fit_direction_bank(x, labels, ports, min_pair_ports=4)
    with pytest.raises(ValueError, match='No nondegenerate'):
        fit_direction_bank(x, labels, ports, min_class_count=6)


def test_transform_fitted_statistics_pair_scaling_and_batch_independence():
    centers = np.array([[0., 0.], [2., 0.], [0., 3.]])
    x = np.tile(np.repeat(centers, 5, axis=0), (3, 1))
    labels = np.tile(np.repeat([0, 1, 2], 5), 3)
    ports = np.repeat(['a', 'b', 'c'], 15)
    bank = fit_direction_bank(x, labels, ports)
    assert bank['pairs'].shape == (3, 2)
    assert_allclose(bank['source_cosine_mean'], 1.)
    assert_allclose(bank['coherence_mean'], 1.)
    frozen = {key: value.copy() for key, value in bank.items()}
    target = np.array([[10., -20.], [1000., 2000.]])
    for kind in ('pooled', 'stable'):
        source = transform_direction(x, bank, kind)
        assert_allclose(source.mean(axis=0), 0., atol=1e-14)
        assert_allclose(source.std(axis=0), 1/np.sqrt(3))
        transformed = transform_direction(target, bank, kind)
        expected = ((target@bank[kind+'_basis']-bank[kind+'_center'])/
                    bank[kind+'_scale']/np.sqrt(3))
        assert_allclose(transformed, expected)
        assert_allclose(transform_direction(target[:1], bank, kind), transformed[:1])
    for key in bank:
        assert_array_equal(bank[key], frozen[key])
    assert list(inspect.signature(transform_direction).parameters) == ['x', 'bank', 'kind']


def test_degenerate_directions_rejected_and_undefined_cosines_omitted():
    x, y, ports = nuisance_source()
    with pytest.raises(ValueError, match='No nondegenerate'):
        fit_direction_bank(np.zeros_like(x), y, ports)
    # Only one port has a contrast. Every leave-one-port comparison has
    # a zero vector on one side, so no diagnostic cosine is defined.
    x[ports != 0] = 0.
    bank = fit_direction_bank(x, y, ports)
    assert np.isnan(bank['source_cosine_mean'][0])
    assert np.isnan(bank['source_cosine_positive_fraction'][0])


def test_float32_input_chunk_boundaries_and_no_input_mutation():
    x, y, ports = nuisance_source((1030, 1025, 1027))
    x = x.astype(np.float32)
    original = x.copy()
    bank = fit_direction_bank(x, y, ports)
    for kind in ('pooled', 'stable'):
        direct = x.astype(np.float64)@bank[kind+'_basis']
        assert_allclose(bank[kind+'_center'], direct.mean(axis=0))
        assert_allclose(bank[kind+'_scale'], direct.std(axis=0))
        expected = (direct-bank[kind+'_center'])/bank[kind+'_scale']
        assert_allclose(transform_direction(x, bank, kind), expected)
    assert_array_equal(x, original)
    assert x.dtype == np.float32
