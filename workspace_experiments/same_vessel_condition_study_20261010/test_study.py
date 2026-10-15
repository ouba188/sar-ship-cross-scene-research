"""Focused checks for angle wrap, group boundaries and repeated-ship weighting."""
import unittest
import numpy as np
import pandas as pd
from study import angular_distance, assign_group, identity_class_weights
from report import truth_margin, hierarchical_summary, PRIMARY


class StudyTests(unittest.TestCase):
    def test_axial_angle_wrap(self):
        np.testing.assert_allclose(angular_distance([89, 0, 10], [-89, 180, 40]), [2, 0, 30])

    def test_groups_do_not_fill_unknowns(self):
        result = assign_group(np.array([1, 1, 0, 0, 0, 1], bool),
            np.ones(6, bool), np.array([1, 0, 1, 1, 1, 1], bool),
            np.array([2, 0, 2, 5, 3, np.nan]), np.array([1, 1, 1, 1, 1, 0], bool))
        self.assertEqual(list(result), ['SP_near', 'SP_far', 'DP_near', 'DP_far', 'unmatched', 'unmatched'])

    def test_training_weights_equal_identity_and_class(self):
        obs = pd.DataFrame({'mmsi': ['a', 'a', 'b', 'c'], 'class_id': [0, 0, 0, 1]})
        w = identity_class_weights(obs)
        np.testing.assert_allclose(w, [.5, .5, 1, 2])

    def test_wrong_consistency_is_not_accuracy(self):
        logits = np.array([[2., 1., 0.], [3., 1., 0.]])
        margin = truth_margin(logits, np.array([1, 1]))
        np.testing.assert_allclose(margin, [-1, -2])

    def test_many_pairs_do_not_overweight_one_ship(self):
        f = pd.DataFrame({'group': ['SP_near']*4, 'class_id': [0]*4,
                          'mmsi': ['a', 'a', 'a', 'b']})
        for key in PRIMARY:
            f[key] = [0., 0., 0., 100.]
        result = hierarchical_summary(f, ['group'])
        self.assertEqual(result.r50_accuracy.iloc[0], 50.)


if __name__ == '__main__':
    unittest.main()
