import unittest
import numpy as np
from run_mechanism import maps, coefficients, retrieval, identity_values
import pandas as pd


class MechanismTests(unittest.TestCase):
    def test_public_identity_column(self):
        values = identity_values(pd.DataFrame({'mmsi_or_ship_id': [354252000., np.nan]}))
        np.testing.assert_array_equal(values, ['354252000', ''])

    def test_cross_paths_match_explicit_kernel_products(self):
        rng = np.random.default_rng(12)
        raw, changed, query = rng.normal(size=(7, 11)), rng.normal(size=(7, 11)), rng.normal(size=(5, 11))
        prior, labels = rng.normal(size=(7, 3)), np.array([0, 0, 1, 1, 1, 2, 2])
        w, _, _ = maps(raw, changed, prior, labels, 2.)
        a0 = coefficients(raw@raw.T, prior, labels, 2.)
        a1 = coefficients(changed@changed.T, prior, labels, 2.)
        np.testing.assert_allclose(query@w['relation_only'], (query@changed.T)@a0, atol=1e-12)
        np.testing.assert_allclose(query@w['coefficient_only'], (query@raw.T)@a1, atol=1e-12)
        np.testing.assert_allclose(query@w['initial'], (query@changed.T)@a1, atol=1e-12)

    def test_unchanged_map_replays_all_arms(self):
        rng = np.random.default_rng(13)
        raw, prior = rng.normal(size=(7, 11)), rng.normal(size=(7, 3))
        w, gamma, gammas = maps(raw, raw.copy(), prior, np.arange(7)%3, 2.)
        for arm in w:
            np.testing.assert_allclose(w[arm], w['raw'], atol=1e-12)
        self.assertAlmostEqual(gamma, 1.)
        np.testing.assert_allclose(gammas, 1.)

    def test_retrieval_excludes_same_ship_and_handles_no_positive(self):
        # First query's nearest case is its own vessel and must be removed.
        result = retrieval(np.array([[9., 4., 1.], [4., 3., 2.]]), np.array([0, 2]),
            np.array([0, 0, 1]), np.array(['a', 'z']), np.array(['a', 'b', 'c']))
        self.assertEqual(result['same_ship_pairs_removed'], 1)
        self.assertEqual(result['covered_n'], 1)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['map_at_r'], 1.)
        self.assertEqual(result['macro_p1'], 1.)


if __name__ == '__main__':
    unittest.main()
