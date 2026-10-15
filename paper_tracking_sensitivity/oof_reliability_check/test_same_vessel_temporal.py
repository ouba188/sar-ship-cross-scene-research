import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from same_vessel_temporal import (
    OBS_COLUMNS, PAIR_COLUMNS, class_port_centers, collect_observations,
    fit_temporal_directions, make_pairs, pair_covariance,
)


def product(day, suffix='chip'):
    return f'S1_202601{day:02d}T120000_{suffix}'


def test_same_acquisition_aggregation_keeps_coordinates_and_first_source_row():
    frame = pd.DataFrame({
        'port': ['a'] * 4, 'mmsi': ['123456789'] * 4, 'class_id': [2] * 4,
        'product': [product(2), product(1, 'first'), product(1, 'second'),
                    product(1, 'excluded')],
    }, index=[50, 10, 30, 90])
    obs, zobs = collect_observations(frame, [50, 30, 10],
                                     [[8., 9.], [2., 4.], [4., 8.]])
    assert list(obs.columns) == OBS_COLUMNS
    assert_array_equal(obs.index, [0, 1])
    assert_array_equal(obs['row_count'], [2, 1])
    assert_array_equal(obs['first_row'], [30, 50])
    assert obs.loc[0, 'product'] == product(1, 'second')
    assert_allclose(zobs, [[3., 6.], [8., 9.]])
    assert obs['acq_time'].is_monotonic_increasing


def test_pairs_stay_within_ports_and_use_raw_adjacent_time_differences():
    frame = pd.DataFrame({
        'port': ['a', 'b', 'a', 'b', 'a'], 'mmsi': ['123456789'] * 5,
        'class_id': [0] * 5,
        'product': [product(day) for day in [5, 2, 1, 4, 3]],
    })
    obs, zobs = collect_observations(frame, frame.index,
                                     [[10.], [100.], [0.], [130.], [4.]])
    pairs, deltas, weights = make_pairs(obs, zobs)
    assert list(pairs.columns) == PAIR_COLUMNS
    assert_array_equal(pairs['port'], ['a', 'a', 'b'])
    assert_allclose(pairs['gap_days'], [2., 2., 2.])
    assert_allclose(deltas, [[4.], [6.], [30.]])
    assert_allclose(weights, [.25, .25, .5])
    assert_array_equal(obs.iloc[pairs['obs_a']]['port'].to_numpy(),
                       obs.iloc[pairs['obs_b']]['port'].to_numpy())


def test_mixed_labels_use_all_supplied_rows_but_never_global_target_labels():
    frame = pd.DataFrame({
        'port': ['a', 'a', 'target', 'a', 'b', 'target'],
        'mmsi': ['123456789'] * 3 + ['234567890'] * 2 + ['345678901'],
        'class_id': [0, 0, 7, 1, 2, 3],
        'product': [product(1), product(2), product(3), product(1),
                    'invalid_time', product(1)],
    }, index=[10, 20, 30, 40, 50, 60])
    obs, zobs = collect_observations(frame, [10, 20, 40, 50],
                                     [[1.], [2.], [3.], [4.]])
    assert_array_equal(obs['mmsi'], ['123456789', '123456789'])
    assert_allclose(zobs, [[1.], [2.]])
    # A source-parent restriction excluding the target ship must not add it back.
    obs, zobs = collect_observations(frame, [40], [[9.]])
    assert_array_equal(obs['first_row'], [40])
    assert_array_equal(obs['mmsi'], ['234567890'])
    assert_allclose(zobs, [[9.]])


@pytest.mark.parametrize('mmsi,timestamp', [
    ('000000000', product(1)), ('12345678', product(1)),
    ('1234567890', product(1)), ('12345678x', product(1)),
    (' 123456789', product(1)), ('123456789', 'S1_20260230T120000_chip'),
    ('123456789', 'S1_20260101T120000'),
])
def test_invalid_identifiers_and_acquisition_times_are_filtered(mmsi, timestamp):
    frame = pd.DataFrame(dict(port=['a'], mmsi=[mmsi], class_id=[0],
                              product=[timestamp]))
    obs, zobs = collect_observations(frame, [0], [[1., 2.]])
    assert obs.empty
    assert zobs.shape == (0, 2)


def hierarchical_case(first_ship_pairs):
    records, coords = [], []
    for label, port, ship, count, delta in [
            (0, 'a', '111111111', first_ship_pairs, 2.),
            (0, 'a', '222222222', 1, 4.),
            (0, 'b', '333333333', 1, 6.),
            (1, 'a', '444444444', 1, 8.),
            (2, 'c', '555555555', 0, 100.)]:
        for day in range(count + 1):
            records.append((port, ship, label, pd.Timestamp('2026-01-01')
                            + pd.Timedelta(days=day)))
            coords.append([day * delta])
    return (pd.DataFrame(records, columns=['port', 'mmsi', 'class_id', 'acq_time']),
            np.asarray(coords))


def test_hierarchical_weights_do_not_give_longer_tracks_more_total_weight():
    for count in [1, 4]:
        pairs, deltas, weights = make_pairs(*hierarchical_case(count))
        assert_allclose(weights.sum(), 1.)
        assert_allclose(weights[:count], np.full(count, 1. / (8 * count)))
        mass = pairs.assign(weight=weights).groupby('mmsi')['weight'].sum()
        assert_allclose(mass.to_numpy(), [.125, .125, .25, .5])
        assert_allclose(pair_covariance(deltas, weights), [[21.75]])


def test_pair_covariance_is_half_second_moment_not_centered_covariance():
    assert_allclose(pair_covariance([[2., 4.]], [1.]), [[2., 4.], [4., 8.]])


def test_class_centers_balance_ports_and_count_observations_equally():
    obs = pd.DataFrame(dict(class_id=[0, 0, 0, 1], port=['a', 'a', 'b', 'c'],
                            row_count=[100, 1, 1, 1]))
    classes, centers = class_port_centers(obs, [[0., 2.], [4., 6.],
                                              [10., 12.], [20., 30.]])
    assert_array_equal(classes, [0, 1])
    assert_allclose(centers, [[6., 8.], [20., 30.]])


def test_known_covariance_protects_only_class_axis_and_keeps_other_mode():
    bank = fit_temporal_directions(np.diag([9., 4., 0.]),
                                    [[-2., 5., 0.], [2., 5., 0.]])
    assert set(bank) == {'temporal', 'class_protected', 'class_basis',
                         'temporal_eigenvalues', 'class_protected_eigenvalues',
                         'class_rank'}
    assert bank['class_rank'] == 1
    assert_allclose(bank['temporal_eigenvalues'], [9., 4.])
    assert_allclose(np.abs(bank['temporal']), [[1., 0.], [0., 1.], [0., 0.]])
    assert_allclose(bank['class_protected_eigenvalues'], [4.])
    assert_allclose(np.abs(bank['class_protected']), [[0.], [1.], [0.]])
    assert_allclose(bank['class_basis'].T @ bank['class_protected'], 0., atol=1e-14)


def test_projection_roundoff_does_not_promote_removed_covariance_to_modes():
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    bank = fit_temporal_directions(9. * np.outer(axis, axis),
                                    np.array([-axis, axis]))
    assert bank['class_rank'] == 1
    assert bank['temporal'].shape == (3, 1)
    assert bank['class_protected'].shape == (3, 0)
    assert bank['class_protected_eigenvalues'].shape == (0,)
    # Repeated non-binary centers can acquire residuals during mean subtraction.
    bank = fit_temporal_directions(np.eye(3), np.full((8, 3), .1))
    assert bank['class_rank'] == 0


def test_eight_mode_cap_and_class_rank_at_most_seven_in_reduced_space():
    centers = np.zeros((8, 43))
    centers[:, :8] = np.eye(8)
    bank = fit_temporal_directions(np.diag(np.arange(43., 0., -1.)), centers)
    assert bank['class_rank'] == 7
    assert bank['temporal'].shape == bank['class_protected'].shape == (43, 8)
    assert_allclose(bank['class_basis'].T @ bank['class_protected'], 0., atol=1e-13)
    assert_allclose(bank['temporal_eigenvalues'], np.arange(43., 35., -1.))


def test_empty_source_and_no_pairs_keep_reduced_dimensions():
    frame = pd.DataFrame(dict(port=['a'], mmsi=['123456789'], class_id=[0],
                              product=[product(1)]))
    for rows, coords in [([], np.empty((0, 3))), ([0], np.ones((1, 3)))]:
        obs, zobs = collect_observations(frame, rows, coords)
        pairs, deltas, weights = make_pairs(obs, zobs)
        assert pairs.empty
        assert deltas.shape == (0, 3)
        cov = pair_covariance(deltas, weights)
        assert_allclose(cov, np.zeros((3, 3)))
        classes, centers = class_port_centers(obs, zobs)
        assert centers.shape == (len(rows), 3)
        bank = fit_temporal_directions(cov, centers)
        assert bank['class_rank'] == 0
        assert bank['temporal'].shape == bank['class_protected'].shape == (3, 0)
