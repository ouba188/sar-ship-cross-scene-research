"""Focused tests for cached-pass joins, interactions, and split isolation."""

import numpy as np
import pandas as pd

import explore_additional_conditions as probe


def test_exact_join_rejects_template_and_non_ok_angle():
    frame = pd.DataFrame(dict(instance_id=['a', 'b', 'c'], product=['exact', 'similar', 'other'],
                              port=['port']*3, mmsi=['123456789']*3, class_id=[0]*3))
    side = frame[['instance_id', 'product', 'port']].copy()
    side['status'] = ['ok', 'ok', 'unsupported_coordinate_provenance']
    side['incidence_deg'], side['latitude'], side['longitude'] = 38., 1., 2.
    result = probe.attach_metadata(frame, side.iloc[::-1],
        {'exact': {'pass': 'Ascending'}, 'other': {'pass': 'Descending'},
         'similar.SAFE': {'pass': 'Ascending'}})
    assert result.instance_id.tolist() == ['a', 'b', 'c']
    assert result.pass_value.iloc[0] == 1
    assert np.isnan(result.pass_value.iloc[1])
    assert np.isnan(result.incidence_deg.iloc[2])


def test_interaction_is_difference_of_endpoint_features():
    obs = pd.DataFrame(index=range(2))
    pairs = pd.DataFrame(dict(obs_a=[0], obs_b=[1]))
    # t goes from1 to2 and pass goes from1 to0: delta(p*t)=-1,
    # unlike delta(p)*delta(t)=-1 by coincidence; quadratic separates them.
    conditions = np.array([[43., 0., 0., 1.], [48., 0., 0., 0.]])
    x = probe.design(obs, conditions, pairs)
    np.testing.assert_allclose(x[0, [0, 1, 4, 5, 6]], [1., 3., -1., -1., -1.])


def fixture_observations():
    rows = []
    for ship in range(12):
        for month in [1, 2, 3, 6]:
            rows.append(dict(port='p', mmsi=str(123456780+ship), class_id=ship % 2,
                product=f'S1A_IW_GRDH_1SDV_2025{month:02d}01T000000_end_{ship:06d}'))
    frame = pd.DataFrame(rows)
    values = np.column_stack((np.arange(len(frame)), np.full(len(frame), 38.),
                              np.zeros(len(frame)), np.zeros(len(frame)), np.ones(len(frame))))
    return frame, values


def test_missing_pass_does_not_reassign_source_identity_splits():
    frame, values = fixture_observations()
    original, _, assignment, _ = probe.observations_with_fixed_splits(frame, frame.index, values)
    values[frame.mmsi.eq('123456780'), -1] = np.nan
    filtered, _, changed, _ = probe.observations_with_fixed_splits(frame, frame.index, values)
    pd.testing.assert_series_equal(assignment['split'], changed['split'])
    expected = original[original.mmsi.ne('123456780')].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, filtered)
    fit = filtered[filtered.split.eq('fit')]
    held_ship = filtered[filtered.split.isin(['new_ship', 'new_ship_month'])]
    held_month = filtered[filtered.split.isin(['new_month', 'new_ship_month'])]
    assert not set(fit.mmsi) & set(held_ship.mmsi)
    assert not set(fit['product']) & set(held_month['product'])


def test_duplicate_missing_pass_is_not_averaged_away():
    frame, values = fixture_observations()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    values = np.vstack([values, values[0]])
    values[-1, -1] = np.nan
    obs, _, _, audit = probe.observations_with_fixed_splits(frame, frame.index, values)
    assert audit['rejected_pass_observations'] == 1
    assert len(obs) == len(frame)-2


def test_constant_pass_has_no_main_effect_support():
    pairs = pd.DataFrame(dict(mmsi=['123456789']*20, port=['p']*20, class_id=[0]*20))
    result = probe.switch_coverage(pairs, np.zeros((20, 7)))
    assert result['switch_pairs'] == 0
    assert not result['adequate_switch_coverage']


def test_blank_identity_is_not_a_both_pass_ship():
    frame = pd.DataFrame(dict(port=['p', 'p'], class_id=[0, 0], mmsi=['', ''],
        product=['a', 'b'], incidence_deg=[38., 39.], latitude=[1., 1.],
        longitude=[2., 2.], pass_value=[0., 1.]))
    table = probe.coverage_table(frame)
    assert table.complete_rows.iloc[0] == 2
    assert table.complete_ships.iloc[0] == 0
    assert table.both_pass_ships.iloc[0] == 0


def test_source_observations_ignore_non_source_labels():
    frame, values = fixture_observations()
    source_rows = np.arange(40)
    before = probe.observations_with_fixed_splits(frame, source_rows, values[:40])
    frame.loc[40:, 'class_id'] = 99
    after = probe.observations_with_fixed_splits(frame, source_rows, values[:40])
    pd.testing.assert_frame_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
