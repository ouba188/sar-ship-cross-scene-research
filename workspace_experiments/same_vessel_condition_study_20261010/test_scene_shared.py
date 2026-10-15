import unittest
import numpy as np
from scene_shared import balanced_offsets, corrected_scores, choose_control


class SceneTests(unittest.TestCase):
    def test_class_means_cancel_without_removing_class_gap(self):
        reference = np.array([[10., 0.], [-20., 30.]])
        nuisance = np.array([2., -3.])
        same, other = balanced_offsets(reference+nuisance, reference, reference)
        np.testing.assert_allclose(same, nuisance)
        np.testing.assert_allclose(other, 0.)
        corrected = reference+nuisance-same
        np.testing.assert_allclose(corrected[0]-corrected[1], reference[0]-reference[1])

    def test_score_intervention_subtracts_not_adds(self):
        w = np.array([[1., 0.], [0., 1.]])
        score = np.array([[2., 3.]])
        after = corrected_scores(score, np.array([0., 2.]), w, .5)
        np.testing.assert_allclose(after, [[2., 2.]])

    def test_control_matches_donors_and_excludes_both_reference_scenes(self):
        cells = {(s, c): None for s in ('a', 'b', 'c', 'd') for c in (0, 1, 2)}
        other, refs = choose_control('a', 0, [1, 2], cells,
                                     ['a', 'b', 'c', 'd'], {'a': 0, 'b': 3, 'c': 9, 'd': 20})
        self.assertEqual(other, 'b')
        self.assertEqual(refs, {1: ['c', 'd'], 2: ['c', 'd'], 0: ['c', 'd']})


if __name__ == '__main__':
    unittest.main()
