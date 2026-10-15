"""Synthetic checks for the source-only paired condition-response diagnostic."""

import numpy as np
import pandas as pd
import pytest

from conditioned_vessel_drift import (
    condition_differences,
    fit_response,
    incidence_shift,
    response_metrics,
    split_observations,
)
from same_vessel_temporal import make_pairs


@pytest.fixture
def observations():
    records = []
    for ship in range(9):
        for year in (2024, 2025):
            for month in range(1, 13):
                for day in (2, 14):
                    records.append(dict(
                        mmsi=str(200000000 + ship), port=f'port_{ship % 2}',
                        class_id=ship % 3,
                        acq_time=pd.Timestamp(year=year, month=month, day=day),
                        product=f'product_{year}_{month}_{day}',
                    ))
    return pd.DataFrame(records)


def test_split_is_exhaustive_and_identity_month_disjoint(observations):
    split = split_observations(observations)
    assert set(split) == {'fit', 'new_ship', 'new_month', 'new_ship_month'}
    masks = np.stack(list(split.values()))
    assert masks.dtype == np.bool_
    np.testing.assert_array_equal(masks.sum(axis=0), np.ones(len(observations)))
    assert masks.any(axis=1).all()

    groups = {name: observations.loc[mask] for name, mask in split.items()}
    fit_ships = set(groups['fit'].mmsi)
    held_ships = set(groups['new_ship'].mmsi)
    assert fit_ships.isdisjoint(held_ships)
    assert set(groups['new_month'].mmsi) == fit_ships
    assert set(groups['new_ship_month'].mmsi) == held_ships
    assert len(fit_ships) == 6 and len(held_ships) == 3
    for name, group in groups.items():
        expected = {3, 6, 9, 12} if 'month' in name else {1, 2, 4, 5, 7, 8, 10, 11}
        assert set(group.acq_time.dt.month) == expected
        assert set(group.acq_time.dt.year) == {2024, 2025}
        if 'month' in name:
            assert set(group['product']).isdisjoint(groups['fit']['product'])


def test_split_membership_is_stable_under_row_reordering(observations):
    original = split_observations(observations, seed=17)
    shuffled = observations.sample(frac=1, random_state=4)
    reordered = split_observations(shuffled, seed=17)
    for name in original:
        np.testing.assert_array_equal(reordered[name], original[name][shuffled.index])


def test_pairing_after_split_keeps_both_endpoints_in_their_partition(observations):
    split = split_observations(observations)
    for name, mask in split.items():
        obs = observations.loc[mask].reset_index(drop=True)
        original_ids = np.flatnonzero(mask)
        pairs, delta, weights = make_pairs(obs, original_ids[:, None])
        assert not pairs.empty
        a = pairs.obs_a.to_numpy(dtype=int)
        b = pairs.obs_b.to_numpy(dtype=int)
        assert mask[original_ids[a]].all() and mask[original_ids[b]].all(), name
        np.testing.assert_array_equal(obs.mmsi.to_numpy()[a], obs.mmsi.to_numpy()[b])
        np.testing.assert_array_equal(obs.port.to_numpy()[a], obs.port.to_numpy()[b])
        assert (obs.acq_time.to_numpy()[b] > obs.acq_time.to_numpy()[a]).all()
        np.testing.assert_array_equal(delta[:, 0], original_ids[b] - original_ids[a])
        assert weights.sum() == pytest.approx(1.)


def test_nonlinear_feature_is_difference_of_squares_and_reverses_sign():
    obs = pd.DataFrame(index=range(3))
    conditions = np.array([[33., 0., 0.], [43., 0., 0.], [48., 0., 0.]])
    pairs = pd.DataFrame({'obs_a': [0, 1, 1, 2], 'obs_b': [1, 2, 0, 1]})
    actual = condition_differences(obs, conditions, pairs)
    # theta = -1, 1, 2: symmetric angles have equal squares despite nonzero delta.
    expected = [[2., 0., 0., 0.], [1., 3., 0., 0.],
                [-2., 0., 0., 0.], [-1., -3., 0., 0.]]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('latitude,east_km', [(0., 22.239), (60., 11.1195)])
def test_date_line_uses_short_signed_displacement(latitude, east_km):
    obs = pd.DataFrame(index=range(2))
    conditions = np.array([[38., latitude, 179.9], [38., latitude, -179.9]])
    pairs = pd.DataFrame({'obs_a': [0, 1], 'obs_b': [1, 0]})
    actual = condition_differences(obs, conditions, pairs)
    np.testing.assert_allclose(actual[:, 2] * 10., [east_km, -east_km], atol=1e-9)
    np.testing.assert_allclose(actual[:, [0, 1, 3]], 0., atol=1e-12)


def test_displacement_units_and_no_elapsed_time_scaling():
    conditions = np.array([[38., 10., 20.], [38., 11., 20.]])
    pairs = pd.DataFrame({'obs_a': [0], 'obs_b': [1]})
    daily = pd.DataFrame({'acq_time': pd.to_datetime(['2024-01-01', '2024-01-02'])})
    yearly = pd.DataFrame({'acq_time': pd.to_datetime(['2024-01-01', '2025-01-01'])})
    expected = [[0., 0., 0., 11.1195]]
    np.testing.assert_allclose(condition_differences(daily, conditions, pairs), expected)
    np.testing.assert_allclose(condition_differences(yearly, conditions, pairs), expected)


def test_known_multivariate_response_generalizes_to_independent_holdout():
    rng = np.random.default_rng(41)
    train = rng.normal(size=(120, 4))
    held = rng.normal(size=(37, 4))
    coefficients = np.array([[2., -1.], [-.5, 3.], [1.5, 0.], [0., -.75]])
    weights = np.linspace(1., 4., len(train))
    weights /= weights.sum()
    fitted = fit_response(train, train @ coefficients, weights, ridge=0.)
    np.testing.assert_allclose(fitted, coefficients, atol=1e-12)
    actual = held @ coefficients
    predicted = held @ fitted
    np.testing.assert_allclose(predicted, actual, atol=1e-12)
    metrics = response_metrics(actual, predicted, np.full(len(held), 1. / len(held)))
    assert metrics['skill'] == pytest.approx(1.)
    assert metrics['best_scalar'] == pytest.approx(1.)


def test_default_ridge_preserves_weights_and_does_not_fit_an_intercept():
    # Deliberately offset observations distinguish through-origin fitting from centering.
    x = np.array([[1.], [2.]])
    y = np.array([[3., -3.], [5., -5.]])
    weights = np.array([.25, .75])
    expected = 8.25 / 3.26
    fitted = fit_response(x, y, weights)
    np.testing.assert_allclose(fitted, [[expected, -expected]], atol=1e-12)
    np.testing.assert_array_equal(np.zeros((1, 1)) @ fitted, [[0., 0.]])


@pytest.mark.parametrize('factor,expected_skill', [(1., 1.), (0., 0.), (-1., -3.), (3., -3.)])
def test_holdout_skill_is_signed_against_zero_change(factor, expected_skill):
    actual = np.array([[1., 2.], [3., 4.]])
    metrics = response_metrics(actual, factor * actual, np.array([.25, .75]))
    assert metrics['baseline_energy'] == pytest.approx(20.)
    assert metrics['skill'] == pytest.approx(expected_skill)
    assert metrics['predicted_energy'] == pytest.approx(20. * factor**2)
    assert metrics['cross_energy'] == pytest.approx(20. * factor)
    if factor:
        assert metrics['best_scalar'] == pytest.approx(1. / factor)
    else:
        assert np.isnan(metrics['best_scalar'])


def test_holdout_metric_uses_weights_and_does_not_center_actual_change():
    metrics = response_metrics(np.array([[2.], [2.]]), np.array([[2.], [-2.]]),
                               np.array([.75, .25]))
    assert metrics['baseline_energy'] == pytest.approx(4.)
    assert metrics['error_energy'] == pytest.approx(4.)
    assert metrics['skill'] == pytest.approx(0.)


def test_zero_baseline_energy_is_undefined_skill():
    metrics = response_metrics(np.zeros((2, 1)), np.ones((2, 1)), np.array([.5, .5]))
    assert np.isnan(metrics['skill'])
    assert metrics['baseline_energy'] == 0.
    assert metrics['error_energy'] == pytest.approx(1.)


def test_incidence_shift_leaves_missing_and_reference_features_unchanged():
    coefficients = np.array([[2., -1.], [3., 4.], [90., 80.], [70., 60.]])
    angles = np.array([np.nan, 38., np.inf, -np.inf])
    features = np.array([[1., 2.], [-3., 4.], [5., -6.], [7., 8.]])
    shift = incidence_shift(angles, coefficients)
    np.testing.assert_array_equal(shift, np.zeros_like(features))
    np.testing.assert_array_equal(features - shift, features)


def test_incidence_shift_uses_known_angle_coefficients_only():
    coefficients = np.array([[2., -1.], [3., 4.], [90., 80.], [70., 60.]])
    angles = np.array([33., 43., 48.])
    before = coefficients.copy()
    # t = -1, 1, 2; position coefficients must not enter an incidence-only shift.
    expected = np.array([[1., 5.], [5., 3.], [16., 14.]])
    np.testing.assert_allclose(incidence_shift(angles, coefficients), expected)
    np.testing.assert_array_equal(coefficients, before)
    np.testing.assert_array_equal(angles, [33., 43., 48.])


def test_incidence_shift_removes_known_response_without_removing_identity():
    coefficients = np.array([[2., -1.], [3., 4.]])
    angles = np.array([33., 43., 48., 38.])
    identity = np.array([7., -2.])
    observed = identity + np.array([[1., 5.], [5., 3.], [16., 14.], [0., 0.]])
    corrected = observed - incidence_shift(angles, coefficients)
    np.testing.assert_allclose(corrected, np.tile(identity, (4, 1)))


def test_correction_predictor_replays_original_joint_support_fit():
    from run_incidence_correction_trial import predict_joint
    from run_protected_drift_activation import FACTORS, predict_draw

    rng = np.random.default_rng(63)
    bank = rng.normal(size=(11, 5))
    stable = rng.normal(size=(11, 2))
    common = {
        suppress: dict(retained=rng.normal(size=(11, 3)),
                       complement=rng.normal(size=(11, 4)))
        for suppress in (False, True)
    }
    phis = {
        factors: np.column_stack((rng.normal(size=(11, 3)), np.ones(11)))
        for factors in FACTORS.values()
    }
    views = dict(common=common, phis=phis, stable=stable)
    heads = {
        arm + '_' + head: dict(coef=rng.normal(size=(5, 8)),
                               intercept=rng.normal(size=8))
        for arm in FACTORS for head in ('base', 'expert')
    }
    # Unsorted, noncontiguous supports catch accidental row-order assumptions.
    support = np.array([8, 1, 6, 3, 0])
    labels = np.array([2, 0, 2, 1, 2])
    original = predict_draw(bank, views, heads, support, labels)['joint']
    base, expert = predict_joint(bank, views, heads, support, labels)
    np.testing.assert_allclose(base, original['base'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(expert, original['expert'], rtol=1e-12, atol=1e-12)
