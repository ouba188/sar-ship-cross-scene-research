import unittest
import numpy as np
from followup import hybrid_scores, margin


class FollowupTests(unittest.TestCase):
    def test_hybrid_corners_and_temporal_accounting(self):
        p = np.array([[1., 3.], [2., -1.]])
        q = np.array([[0., 2.], [4., 1.]])
        bias = np.array([[.2, -.3], [.2, -.3]])
        a, b = np.array([0]), np.array([1])
        values = hybrid_scores(p, q, bias, a, b)
        np.testing.assert_allclose(values['actual_a'], (p+q+bias)[a])
        np.testing.assert_allclose(values['actual_b'], (p+q+bias)[b])
        np.testing.assert_allclose(values['r18_previous'], p[b]+q[a]+bias[b])
        y, c = np.array([1]), np.array([0])
        delta = margin(values['actual_b']-values['actual_a'], y, c)
        np.testing.assert_allclose(delta, margin(p[b]-p[a], y, c)+margin(q[b]-q[a], y, c))

    def test_score_gap_includes_intercept(self):
        old_path = np.array([[1., 2.]])
        old_bias = np.array([[.5, -.5]])
        new_path = np.array([[.2, 1.6]])
        added = np.array([[-.3, .8]])
        new_bias = np.array([[-.4, .3]])
        y, c = np.array([1]), np.array([0])
        pieces = margin(new_path-old_path, y, c)+margin(added, y, c)+margin(new_bias-old_bias, y, c)
        difference = margin(new_path+added+new_bias, y, c)-margin(old_path+old_bias, y, c)
        np.testing.assert_allclose(pieces, difference)


if __name__ == '__main__':
    unittest.main()
