"""Verify the changed solver against the existing sklearn implementation."""

import unittest
import numpy as np
from sklearn.linear_model import RidgeClassifier
from run_diagnosis import sufficient, solve, weights


class ReadoutTests(unittest.TestCase):
    def test_unpenalized_intercept_matches_sklearn(self):
        rng = np.random.default_rng(923)
        y = np.repeat(np.arange(8), np.arange(3, 11))
        x = rng.normal(size=(len(y), 13)) + 2.5
        w = weights(y, 37.)
        h = np.c_[x, np.ones(len(x))]
        theta = solve(*sufficient(h, y, w))
        model = RidgeClassifier(alpha=1., solver='cholesky').fit(x, y, sample_weight=w)
        q = rng.normal(size=(17, 13)) + 2.5
        np.testing.assert_allclose(np.c_[q, np.ones(len(q))] @ theta,
                                   model.decision_function(q), atol=1e-11, rtol=1e-11)

    def test_target_innovation_recovers_combined_fit(self):
        rng = np.random.default_rng(817)
        y = np.tile(np.arange(8), 5)
        h = np.c_[rng.normal(size=(len(y), 9)), np.ones(len(y))]
        w = rng.uniform(.1, 2., len(y))
        gs, cs = sufficient(h[:32], y[:32], w[:32])
        gt, ct = sufficient(h[32:], y[32:], w[32:])
        source, combined = solve(gs, cs), solve(gs+gt, cs+ct)
        error = 2.*np.eye(8)[y[32:]]-1.-h[32:] @ source
        delta = solve(gs+gt, h[32:].T @ (w[32:, None]*error))
        np.testing.assert_allclose(source+delta, combined, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
