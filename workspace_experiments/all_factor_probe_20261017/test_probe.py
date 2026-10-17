import unittest
import numpy as np
from probe_classifier import standardize, map_at_r


class ReadoutProbe(unittest.TestCase):
    def test_known_retrieval_rank_and_missing_class(self):
        score = np.array([[4., 2., 3., 1.], [0., 1., 2., 3.]])
        ap = map_at_r(score, np.array([0, 7]), np.array([0, 0, 1, 1]))
        self.assertEqual(ap[0], .5)
        self.assertTrue(np.isnan(ap[1]))

    def test_target_changes_cannot_fit_source_scaler(self):
        rng = np.random.default_rng(42)
        raw = rng.uniform(size=(8, 18, 6))
        raw[:, :, [2, 5]] = 1
        raw[1, 3, 2] = 0
        valid = np.ones(8, bool)
        valid[-1] = False
        first = standardize(raw.reshape(8, -1), valid, np.arange(4)).numpy()
        raw[4:] *= 100
        second = standardize(raw.reshape(8, -1), valid, np.arange(4)).numpy()
        np.testing.assert_array_equal(first[:4], second[:4])
        self.assertEqual(first.reshape(8, 18, 6)[1, 3, 0], 0)
        self.assertTrue((second[-1] == 0).all())


if __name__ == '__main__':
    unittest.main()
