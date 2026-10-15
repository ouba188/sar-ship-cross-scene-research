"""Focused tests for the fixed source-only applicability gate."""

import unittest
import numpy as np
import pandas as pd
from condition_applicability import (BASE, FEATURE_NAMES, gate_inputs, predict_gate,
    fit_gate, learn_gate, fit_responses, evidence_arrays)


def synthetic():
    rng = np.random.default_rng(17)
    ships = np.repeat(np.arange(96), 2)
    ids = np.arange(len(ships))
    angle = rng.uniform(30., 45., len(ids))
    features = rng.normal(size=(len(ids), 6))
    features[:, 5] = np.repeat(rng.normal(size=96), 2)+.08*angle
    meta = pd.DataFrame(dict(obs_id=ids, mmsi=ships, class_id=ships % 4,
        port='p', platform='S1A', pass_name='Ascending', product=ids.astype(str),
        acq_time=ids, incidence_deg=angle))
    fit, held = ids[:180], ids[180:]
    assignment = np.full(len(ids), -1)
    assignment[fit] = ships[fit] % 3
    return meta, fit, held, features, assignment


class ApplicabilityTest(unittest.TestCase):
    def test_known_dependence(self):
        x = np.zeros((80, 4))
        x[:, 0] = np.r_[np.zeros(40), np.ones(40)]
        truth = np.zeros(80)
        anchor = np.r_[np.ones(40)*2., np.ones(40)]
        condition = np.r_[np.ones(40), np.ones(40)*2.]
        gate, _ = fit_gate(x, truth, anchor, condition, np.ones(80), np.arange(80))
        self.assertEqual(gate['feature_index'], 0)
        np.testing.assert_allclose(predict_gate(x, gate), np.r_[np.full(40, .6), np.zeros(40)])
        self.assertLess(gate['gain_sse'], gate['root_sse'])

    def test_leaf_ship_minimum_not_pair_count(self):
        x = np.zeros((100, 4))
        x[:50, 0] = 1.
        ships = np.r_[np.zeros(50), np.arange(1, 51)]
        anchor = np.r_[np.full(50, 2.), np.ones(50)]
        condition = anchor[::-1]
        gate, candidates = fit_gate(x, np.zeros(100), anchor, condition, np.ones(100), ships)
        self.assertEqual(gate['feature_index'], -1)
        self.assertTrue(all(not row['eligible'] for row in candidates[1:]))
        self.assertEqual(gate['leaf_mix'].shape, (2,))

    def test_range_and_raw_input_order(self):
        x = gate_inputs([1., 2.], [3., 4.], [30., 42.], [40., 32.])
        np.testing.assert_array_equal(x, [[1., 3., 10., 35.], [2., 4., 10., 37.]])
        model = dict(feature_index=0, threshold=1.5, leaf_mix=np.array([.25, .75]))
        np.testing.assert_array_equal(predict_gate(x, model), [.25, .75])
        x[1, 2] = np.nan
        np.testing.assert_array_equal(predict_gate(x, model), [.25, 0.])
        self.assertEqual(len(FEATURE_NAMES), 4)

    def test_no_held_poison_and_nested_response_isolation(self):
        meta, fit, held, features, assignment = synthetic()
        first = learn_gate(meta, fit, features, assignment)
        first_models = fit_responses(meta, fit, features)
        altered = meta.copy()
        altered.loc[held, 'class_id'] = 99
        altered.loc[held, 'incidence_deg'] = -1000.
        corrupted = features.copy()
        corrupted[held] = 1e6
        second = learn_gate(altered, fit, corrupted, assignment)
        second_models = fit_responses(altered, fit, corrupted)
        for key in first[0]:
            np.testing.assert_array_equal(first[0][key], second[0][key])
        for arm in first_models:
            for key in first_models[arm]:
                np.testing.assert_array_equal(first_models[arm][key], second_models[arm][key])
        training = fit[assignment[fit] != 0]
        outer_before = learn_gate(meta, training, features, assignment)
        poisoned_outer = features.copy()
        poisoned_outer[fit[assignment[fit] == 0]] = 2e6
        outer_after = learn_gate(meta, training, poisoned_outer, assignment)
        for key in outer_before[0]:
            np.testing.assert_array_equal(outer_before[0][key], outer_after[0][key])
        self.assertEqual(len(outer_before[-1]), 2)
        for row in outer_before[-1]:
            self.assertEqual(row['train_ships'], 30)
            self.assertEqual(row['eval_ships'], 30)

    def test_uncovered_features_no_warning(self):
        features = np.full((3, 6), np.nan)
        features[0] = 0.
        with np.errstate(invalid='raise'):
            _, total = evidence_arrays(features)
        self.assertAlmostEqual(total[0], np.log(3.))
        self.assertTrue(np.isnan(total[1:]).all())

    def test_exported_controls_preserve_previous_results(self):
        path = BASE/'condition_applicability_20261011/held_predictions.npz'
        if not path.exists():
            self.skipTest('Run the source-only experiment before export integration check')
        with np.load(path) as current, np.load(BASE/'condition_evidence_20261011/predictions/0_center_border_log_ratio_magnitude.npz') as old:
            np.testing.assert_array_equal(current['directed'], old['directed'])
            np.testing.assert_array_equal(current['truth'], old['truth'])
            for arm in ('anchor', 'capacity', 'condition', 'sham'):
                np.testing.assert_array_equal(current[arm], old[arm])
            with np.load(BASE/'condition_applicability_20261011/gate.npz') as model:
                np.testing.assert_array_equal(predict_gate(current['inputs'], model), current['gate_mix'])


if __name__ == '__main__':
    unittest.main()
