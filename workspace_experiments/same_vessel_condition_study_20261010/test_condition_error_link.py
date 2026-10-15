import unittest
import numpy as np
import pandas as pd
from condition_error_link import make_pairs, pair_features, ridge_predict


class PairChecks(unittest.TestCase):
    def test_time_orientation_and_scene_exclusion(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2], class_id=[0, 0, 0, 1],
            product=['b', 'a', 'a', 'c'], acq_time=['2025-02-01', '2025-01-01', '2025-01-01', '2025-01-03'],
            obs_id=[0, 1, 2, 3], port=['p']*4))
        pairs = make_pairs(meta)
        self.assertEqual(set(zip(pairs.a, pairs.b)), {(1, 0), (2, 0)})
        self.assertTrue((pairs.gap_days > 0).all())

    def test_prediction_is_antisymmetric_without_class_input(self):
        z = np.random.default_rng(3).normal(size=(7, 4))
        pairs = pd.DataFrame(dict(a=[0, 1, 2], b=[4, 5, 6]))
        reverse = pairs.rename(columns={'a': 'b', 'b': 'a'})
        x, xr = pair_features(z, pairs), pair_features(z, reverse)
        np.testing.assert_allclose(x, -xr)
        p, pr = ridge_predict(x, np.array([.2, -.5, .9]), np.ones(3), xr)
        np.testing.assert_allclose(p, -pr)
        np.testing.assert_allclose(pair_features(z, pairs, True), pair_features(z, reverse, True))


if __name__ == '__main__':
    unittest.main()
