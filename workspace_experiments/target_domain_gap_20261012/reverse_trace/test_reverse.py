"""Synthetic explanations must remain explanations of the mechanism supplied."""

import unittest
import numpy as np
from reverse_analysis import decompose, shapley, support_decomposition


class ReverseTests(unittest.TestCase):
    def test_bias_cannot_become_evidence(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(80, 12)); w = rng.normal(size=(12, 8)); b = rng.normal(size=8)
        change = rng.normal(size=8)
        d = decompose(x, np.arange(60), w, b, w, b+change)
        np.testing.assert_allclose(d['bias'], np.broadcast_to(change, (80, 8)), atol=1e-12)
        np.testing.assert_allclose(d['weights'], 0, atol=1e-12)
        np.testing.assert_allclose(d['remainder'], 0, atol=1e-12)

    def test_old_score_remix_has_no_new_evidence(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 16)); w = rng.normal(size=(16, 8)); b = rng.normal(size=8)
        change = w@rng.normal(size=(8, 8))
        d = decompose(x, np.arange(70), w, b, w+change, b)
        np.testing.assert_allclose(d['weights'], d['remix'], atol=1e-11)
        np.testing.assert_allclose(d['remainder'], 0, atol=1e-11)
        np.testing.assert_allclose(d['weights']+d['bias'], x@change, atol=1e-11)

    def test_interaction_is_shared(self):
        values = []
        for mask in range(8):
            a, b, c = (bool(mask & (1 << j)) for j in range(3))
            values.append([2*a+4*b+6*c+10*a*b])
        np.testing.assert_allclose(shapley(np.array(values, dtype=float)), [[7., 9., 6.]])

    def test_support_null_is_unobservable_on_support(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(50, 20)); w = rng.normal(size=(20, 8)); b = rng.normal(size=8)
        p = support_decomposition(x, np.arange(10), w, b)
        self.assertEqual(p['rank'], 10)
        self.assertLess(p['null_on_support'], 1e-12)
        self.assertGreater(np.linalg.norm(p['outside'][10:]), 1.)
        np.testing.assert_allclose(p['inside']+p['outside'], x@w+b, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
