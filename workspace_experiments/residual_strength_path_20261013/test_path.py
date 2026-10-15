import unittest
import numpy as np
from run_path import measures, path_logits, fusion


class PathTest(unittest.TestCase):
    def test_class_denominators_and_risk_partition(self):
        y = np.array([0, 0, 0, 1])
        old = np.full((4, 8), .02)
        new = old.copy()
        old[np.arange(4), [0, 1, 0, 1]] = .86
        new[np.arange(4), [0, 0, 0, 0]] = .86
        score, classes = measures(y, new, old)
        self.assertAlmostEqual(score['delta_oa'], 0.)
        self.assertAlmostEqual(score['rescue_ba'], 100 / 6)
        self.assertAlmostEqual(score['harm_ba'], 50.)
        self.assertAlmostEqual(score['delta_ba'], -100 / 3)
        self.assertAlmostEqual(sum(c['rescue_ba'] - c['harm_ba'] for c in classes), -100 / 3)

    def test_actual_logit_path_not_probability_mix(self):
        prior = np.array([[2., -1.]])
        residual = np.array([[1., -2.]])
        gamma = np.array([.2, 1.8])
        raw, end = prior + residual, prior + gamma * residual
        alpha = .25
        actual = path_logits(raw, end, alpha)
        np.testing.assert_allclose(actual, prior + (1 + alpha * (gamma - 1)) * residual)
        self.assertFalse(np.allclose(fusion(prior, actual),
            (1-alpha)*fusion(prior, raw) + alpha*fusion(prior, end), atol=1e-10, rtol=0))


if __name__ == '__main__':
    unittest.main()
