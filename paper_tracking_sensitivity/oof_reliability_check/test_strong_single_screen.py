"""Numerical and split tests for the new source-selected screening readout."""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from strong_single_screen import geometry, source_heads, validation_splits, ALPHAS
from unified_head_core import coefficients


class ScreenTests(unittest.TestCase):
    def test_rescaled_kernel_preserves_selected_readout(self):
        rng = np.random.default_rng(44)
        x = rng.normal(size=(90, 12))
        labels = np.arange(64) % 8
        prior = rng.normal(size=(90, 8))
        kernel = x@x[:64].T+1
        for strength in (.1, 1., 8.):
            a = coefficients(kernel[:64], prior[:64], labels, strength)
            b = coefficients(kernel[:64]/strength, prior[:64], labels, 1.)
            np.testing.assert_allclose(prior+kernel@a, prior+(kernel/strength)@b, atol=1e-10)

    def test_weighted_source_fit_matches_sklearn(self):
        rng = np.random.default_rng(31)
        y = np.repeat(np.arange(8), np.arange(4, 12))
        x = rng.normal(size=(len(y), 13)).astype(np.float32)
        rows = np.arange(len(y))
        center, scale = geometry(x, rows)
        z = (x.astype(float)-center)/scale
        np.testing.assert_allclose(z.mean(0), 0, atol=1e-12)
        self.assertAlmostEqual(np.mean(np.sum(z*z, axis=1)), 1)
        heads = source_heads(x, y, rows, center, scale)
        for alpha in ALPHAS:
            expected = RidgeClassifier(alpha=alpha, class_weight='balanced').fit(z, y)
            coef, intercept = heads[alpha]
            np.testing.assert_allclose(z@coef+intercept, expected.decision_function(z), atol=1e-10)

    def test_validation_removes_cross_port_identity(self):
        n = 160
        frame = pd.DataFrame(dict(port=np.repeat(['a', 'b', 'c', 'd'], n),
                                  mmsi=[str(i) for i in range(4*n)],
                                  class_id=np.arange(4*n) % 8))
        frame.loc[165, 'mmsi'] = frame.loc[0, 'mmsi']
        splits = validation_splits(frame, np.arange(len(frame)))
        train = next(train for port, train, _, _ in splits if port == 'a')
        self.assertNotIn(165, train)
        for port, train, val, supports in splits:
            self.assertFalse(set(train).intersection(val))
            self.assertTrue((frame.iloc[val].port == port).all())
            self.assertFalse((frame.iloc[train].port == port).any())
            for support in supports:
                self.assertEqual(len(np.unique(support)), 64)


if __name__ == '__main__':
    unittest.main()
