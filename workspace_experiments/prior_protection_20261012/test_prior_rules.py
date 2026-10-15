"""Independent small-dimensional mathematical checks; no experiment data."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "2"

import unittest

import numpy as np

from prior_rules import penalties, solve_weights


MODES = ("raw", "guard", "count", "count_common", "confidence")


def primal_oracle(features, prior, labels, lambdas):
    """Solve feature-space normal equations for each residual column."""
    batch, dim = features.shape
    classes = prior.shape[1]
    present = set(labels.tolist())
    weights = np.array([
        batch / (len(present) * sum(int(other == label) for other in labels))
        for label in labels
    ])
    residual = -np.ones((batch, classes)) - prior
    residual[np.arange(batch), labels] += 2.0
    metric = np.diag(weights)
    gram = features.T @ metric @ features
    rhs = features.T @ metric @ residual
    result = np.zeros((dim, classes))
    for cls, strength in enumerate(lambdas):
        if np.isfinite(strength):
            result[:, cls] = np.linalg.solve(
                gram + strength * np.eye(dim), rhs[:, cls]
            )
    return result


def softmax(scores):
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class PriorRuleTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 0, 0, 2, 3, 3])
        self.probabilities = np.array([
            [.9, .02, .03, .05],
            [.8, .05, .05, .1],
            [.7, .05, .1, .15],
            [.4, .1, .2, .3],
            [.2, .2, .2, .4],
            [.1, .1, .2, .6],
        ])
        self.prior = np.log(self.probabilities)
        self.features = np.random.default_rng(120).normal(size=(6, 3))

    def test_penalties_match_hand_calculated_support_cases(self):
        # B=6, C=4, counts=(3, 0, 1, 2); true-class means=(.8, -, .2, .5).
        expected = {
            "raw": [2., 2., 2., 2.],
            "guard": [2., np.inf, 2., 2.],
            "count": [1., np.inf, 3., 1.5],
            "count_common": [11. / 6., np.inf, 11. / 6., 11. / 6.],
            "confidence": [3.2, np.inf, .8, 2.],
        }
        for mode, target in expected.items():
            with self.subTest(mode=mode):
                actual = penalties(self.labels, self.prior, 2., mode)
                self.assertEqual(actual.shape, (4,))
                np.testing.assert_allclose(actual, target, rtol=1e-12, atol=1e-12)

    def test_dual_matches_independent_primal_with_unequal_penalties(self):
        # Both sides of B vs D, plus a dependent feature column.
        rng = np.random.default_rng(121)
        for dim in (3, 9):
            features = rng.normal(size=(6, dim))
            features[:, -1] = 2. * features[:, 0]
            strengths = np.array([.3, np.inf, 1.7, 5.])
            with self.subTest(dim=dim):
                actual = solve_weights(features, self.prior, self.labels, strengths)
                self.assertEqual(actual.shape, (dim, 4))
                np.testing.assert_allclose(
                    actual, primal_oracle(features, self.prior, self.labels, strengths),
                    rtol=1e-10, atol=1e-11,
                )

    def test_each_mode_matches_primal_on_imbalanced_support(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                strengths = penalties(self.labels, self.prior, 2., mode)
                np.testing.assert_allclose(
                    solve_weights(self.features, self.prior, self.labels, strengths),
                    primal_oracle(self.features, self.prior, self.labels, strengths),
                    rtol=1e-10, atol=1e-11,
                )

    def test_balanced_full_coverage_constant_confidence_reduces_old_ridge(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        prior = np.log(np.array([
            [.6, .2, .2], [.2, .6, .2], [.2, .2, .6],
            [.6, .2, .2], [.2, .6, .2], [.2, .2, .6],
        ]))
        signed = 2. * np.eye(3)[labels] - 1.
        old_ridge = np.linalg.solve(
            self.features.T @ self.features + 2. * np.eye(3),
            self.features.T @ (signed - prior),
        )
        for mode in MODES:
            with self.subTest(mode=mode):
                strengths = penalties(labels, prior, 2., mode)
                np.testing.assert_allclose(strengths, [2., 2., 2.], atol=1e-12)
                np.testing.assert_allclose(
                    solve_weights(self.features, prior, labels, strengths),
                    old_ridge, rtol=1e-10, atol=1e-11,
                )

    def test_absent_columns_exact_zero_and_guard_preserves_present_raw_columns(self):
        raw = solve_weights(self.features, self.prior, self.labels, np.full(4, 2.))
        for mode in MODES[1:]:
            with self.subTest(mode=mode):
                strengths = penalties(self.labels, self.prior, 2., mode)
                actual = solve_weights(self.features, self.prior, self.labels, strengths)
                np.testing.assert_array_equal(actual[:, 1], np.zeros(3))
                if mode == "guard":
                    np.testing.assert_allclose(actual[:, [0, 2, 3]], raw[:, [0, 2, 3]],
                                               rtol=1e-10, atol=1e-11)

    def test_support_permutation_invariance(self):
        order = np.array([4, 0, 5, 2, 3, 1])
        for mode in MODES:
            with self.subTest(mode=mode):
                strengths = penalties(self.labels, self.prior, 2., mode)
                reordered = penalties(self.labels[order], self.prior[order], 2., mode)
                np.testing.assert_allclose(reordered, strengths, atol=1e-12)
                np.testing.assert_allclose(
                    solve_weights(self.features[order], self.prior[order],
                                  self.labels[order], reordered),
                    solve_weights(self.features, self.prior, self.labels, strengths),
                    rtol=1e-10, atol=1e-11,
                )

    def test_confidence_uses_softmax_and_is_invariant_to_row_logit_offsets(self):
        offsets = np.array([1000., -1000., 50., -50., 500., -500.])[:, None]
        actual = penalties(self.labels, self.prior + offsets, 2., "confidence")
        np.testing.assert_allclose(actual, [3.2, np.inf, .8, 2.],
                                   rtol=1e-11, atol=1e-11)

    def test_confidence_stronger_prior_means_stronger_anchor(self):
        strengths = penalties(self.labels, self.prior, 2., "confidence")
        self.assertGreater(strengths[0], strengths[3])
        self.assertGreater(strengths[3], strengths[2])
        anchored = solve_weights(self.features, self.prior, self.labels, strengths)
        common = solve_weights(self.features, self.prior, self.labels,
                               np.array([2., np.inf, 2., 2.]))
        # Keep each column's feature matrix and RHS fixed; compare its own norm.
        self.assertLess(np.linalg.norm(anchored[:, 0]), np.linalg.norm(common[:, 0]))
        self.assertGreater(np.linalg.norm(anchored[:, 2]), np.linalg.norm(common[:, 2]))

    def test_only_one_present_class_uses_num_present_in_sample_weights(self):
        labels = np.array([2, 2, 2])
        features = np.array([[1., 0.], [0., 2.], [1., 1.]])
        prior = np.zeros((3, 4))
        strengths = penalties(labels, prior, 2., "count")
        np.testing.assert_allclose(strengths, [np.inf, np.inf, .5, np.inf])
        actual = solve_weights(features, prior, labels, strengths)
        expected_present = np.linalg.solve(
            features.T @ features + .5 * np.eye(2), features.T @ np.ones(3)
        )
        np.testing.assert_allclose(actual[:, 2], expected_present, atol=1e-12)
        np.testing.assert_array_equal(actual[:, [0, 1, 3]], np.zeros((2, 3)))

    def test_zero_absent_residual_does_not_preserve_softmax_probability(self):
        features = np.eye(2)
        prior = np.zeros((2, 3))
        labels = np.array([0, 1])
        strengths = penalties(labels, prior, 1., "guard")
        adjusted = prior + features @ solve_weights(features, prior, labels, strengths)
        np.testing.assert_array_equal(adjusted[:, 2], prior[:, 2])
        # At either row, softmax denominator becomes exp(.5)+exp(-.5)+1 > 3.
        self.assertTrue(np.all(softmax(adjusted)[:, 2] < softmax(prior)[:, 2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
