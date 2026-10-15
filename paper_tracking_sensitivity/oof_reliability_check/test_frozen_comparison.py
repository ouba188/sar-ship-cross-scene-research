import unittest
import numpy as np
from frozen_comparison import ridge_scores, prototypes, weights, ranked_mask


class ComparisonTests(unittest.TestCase):
    def test_dual_matches_independent_primal(self):
        rng = np.random.default_rng(31)
        x = np.c_[rng.normal(size=(13, 4)), np.ones(13)]
        s = np.arange(7)
        y = np.array([0, 0, 0, 1, 1, 2, 3])
        w = weights(y)
        target = 2*np.eye(8)[y]-1
        coef = np.linalg.solve(x[s].T@(w[:, None]*x[s])+1.7*np.eye(5), x[s].T@(w[:, None]*target))
        scores = ridge_scores(x@x[s].T, s, y, 1.7)
        np.testing.assert_allclose(scores[:, :4], (x@coef)[:, :4], atol=1e-12)
        self.assertTrue(np.isneginf(scores[:, 4:]).all())

    def test_missing_class_and_prototype_identity(self):
        x = np.eye(3)
        y = np.array([2, 5])
        for metric in ('cosine', 'euclidean'):
            p = prototypes(x, np.array([0, 1]), y, metric)
            np.testing.assert_array_equal(p[:2], y)
            self.assertTrue(np.isin(p, y).all())

    def test_budget_eligibility_and_tie(self):
        mask = ranked_mask(np.array([1., 1., 9., -1.]), np.array([True, True, False, True]),
                           np.array([8, 2, 1, 3]), 1)
        np.testing.assert_array_equal(mask, [False, True, False, False])
        self.assertFalse(ranked_mask(np.ones(4), np.ones(4, bool), np.arange(4), 0).any())


if __name__ == '__main__':
    unittest.main()
