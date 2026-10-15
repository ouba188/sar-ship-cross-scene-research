import unittest
import numpy as np
import pandas as pd
from incidence_factor_chain import fit_slope, evaluate_slope, pair_intervention, decision_metrics, matched_pairs


class IncidenceChainTests(unittest.TestCase):
    def test_slope_has_known_shrinkage_and_reversal_symmetry(self):
        values = np.array([[0., 0.], [2., -1.], [10., 10.], [14., 8.]])
        table = pd.DataFrame(dict(a=[0, 2], b=[1, 3], mmsi=[1, 2], delta_angle=[5., 10.]))
        expected = np.array([2., -1.])/1.05
        np.testing.assert_allclose(fit_slope(values, table), expected)
        reverse = table.copy()
        reverse[['a', 'b']] = table[['b', 'a']].to_numpy()
        reverse['delta_angle'] *= -1
        np.testing.assert_allclose(fit_slope(values, reverse), expected)

    def test_midpoint_intervention_keeps_mean_and_removes_modeled_difference(self):
        values = np.array([[3., 4.], [5., 3.]])
        table = pd.DataFrame(dict(a=[0], b=[1], mmsi=[1], delta_angle=[5.]))
        changed = pair_intervention(values, table, np.array([2., -1.]))
        np.testing.assert_allclose(changed[0, 0], changed[0, 1])
        np.testing.assert_allclose(changed.mean(1), values.mean(0)[None])
        result = evaluate_slope(values, table, np.array([2., -1.]))
        self.assertAlmostEqual(result['explained_percent'], 100.)

    def test_both_wrong_is_not_correct_and_no_pair_is_missing(self):
        table = pd.DataFrame(dict(mmsi=[1]))
        score = np.array([[[0., 2.], [0., 3.]]])
        result = decision_metrics(score, np.array([0]), table)
        self.assertEqual(result['both_correct'], 0.)
        self.assertEqual(result['same_wrong'], 100.)
        self.assertEqual(evaluate_slope(np.zeros((1, 2)), table.iloc[:0], np.zeros(2))['pairs'], 0)

    def test_missing_or_different_conditions_are_not_matched(self):
        meta = pd.DataFrame(dict(mmsi=[1]*4, port=['p']*4, platform=['S1A']*4,
            pass_name=['Ascending', 'Ascending', 'Descending', None],
            incidence_deg=[30., 35., 40., 45.], product=['a', 'b', 'c', 'd'],
            acq_time=['1', '2', '3', '4'], obs_id=np.arange(4)))
        np.testing.assert_array_equal(matched_pairs(meta, np.arange(4)), [[0, 1]])


if __name__ == '__main__':
    unittest.main()
