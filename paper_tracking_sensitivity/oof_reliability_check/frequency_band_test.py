"""Small contracts that catch band scaling and label-assisted selection leakage."""

import unittest
import numpy as np
from frequency_band_core import unit_bands, old_kernel, score_embedding, choose_scores, oracle_predictions, choice_features
from frequency_reliability_core import coefficients


class BandTest(unittest.TestCase):
    def test_unit_recovery_does_not_keep_suppressed_amplitudes(self):
        rng = np.random.default_rng(3)
        bands = rng.normal(size=(7, 4, 64))
        bands /= np.linalg.norm(bands, axis=2, keepdims=True)
        gates = np.array([1e-20, .7, .2, .1])
        recovered = unit_bands((bands*np.sqrt(gates)[None, :, None]).reshape(7, 256))
        np.testing.assert_allclose(recovered, bands, atol=1e-14)
        equal = (bands*.5).reshape(7, 256)
        expected = sum(bands[:, b]@bands[:, b].T for b in range(4))/4
        np.testing.assert_allclose(equal@equal.T, expected, atol=1e-14)

    def test_band_ridge_matches_independent_primal_solution(self):
        rng = np.random.default_rng(1)
        phi, z = rng.normal(size=(9, 3)), rng.normal(size=(9, 2))
        prior = rng.normal(size=(9, 3))
        data = dict(phi=phi, mean=np.zeros((9, 2)), conditions=np.zeros((9, 2)), prior=prior)
        support, y = np.arange(6), np.array([0, 0, 0, 1, 2, 2])
        k, c = old_kernel(data, support, .2)
        actual, _ = score_embedding(data, z, support, y, k, c)
        x = np.c_[phi, .5*z]
        w = np.array([2/3, 2/3, 2/3, 2, 1, 1])
        target = 2*np.eye(3)[y]-1-prior[support]
        coef = np.linalg.solve(x[support].T@(w[:, None]*x[support])+8*np.eye(5), x[support].T@(w[:, None]*target))
        np.testing.assert_allclose(actual, prior+x@coef, atol=1e-12)

    def test_selection_keeps_reference_on_tie_or_same_class(self):
        scores = np.zeros((3, 6, 3))
        scores[:, 0, 0] = 2
        scores[:, 2:, 1] = 3
        scores[1, 2] = [4, 0, 0]
        chosen, index = choose_scores(scores, np.array([[0, -1, 0, -1], [1, -.1, -.1, -.1], [-1, .3, .1, 0]]))
        np.testing.assert_array_equal(index, [0, 0, 3])
        np.testing.assert_array_equal(chosen.argmax(1), [0, 0, 1])

    def test_oracle_preserves_correct_and_cannot_invent_rescue(self):
        ref = np.array([0, 1, 2, 1])
        alternatives = np.array([[1, 2], [2, 0], [0, 1], [0, 1]])
        y = np.array([0, 0, 2, 2])
        np.testing.assert_array_equal(oracle_predictions(ref, alternatives, y), [0, 0, 2, 1])

    def test_no_conflict_batch_never_calls_estimator(self):
        from frequency_band_run import predict_choice
        class MustNotPredict:
            def predict(self, data):
                raise AssertionError('No candidate changes the answer')
        scores = np.zeros((2, 6, 8))
        scores[:, :, 3] = 1
        chosen, index, _ = predict_choice(MustNotPredict(), scores, np.zeros((2, 4, 33)), 33)
        np.testing.assert_array_equal(index, [0, 0])
        np.testing.assert_array_equal(chosen.argmax(1), [3, 3])

    def test_support_order_invariance_and_missing_class_finite(self):
        old, new = np.array([[3., 0, 0], [0, 3, 0]]), np.array([[0., 0, 3], [0, 0, 3]])
        sim = np.array([[.7, .4, .1], [.3, .8, .2]])
        sy, cond = np.array([0, 1, 1]), np.zeros((2, 2))
        f, width = choice_features(old, new, sim, sy, cond, 1)
        g, _ = choice_features(old, new, sim[:, [2, 0, 1]], sy[[2, 0, 1]], cond, 1)
        np.testing.assert_allclose(f, g)
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(f.shape[1]-width, 10)


if __name__ == '__main__':
    unittest.main()
