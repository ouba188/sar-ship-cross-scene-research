"""Focused invariants for the frozen spatial pooling pilot."""

import inspect
import unittest

import numpy as np

from projection_residual_expert import fit_kernel_correction
from stable_guided_spatial_pooling import (
    TokenCache, coordinated_expert, fit_residual_head, source_rms, spatial_residual)


class SpatialPoolingTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_cache_mapping_and_float32_token_mean(self):
        old = self.rng.normal(size=(3, 4, 6)).astype(np.float16)
        new = self.rng.normal(size=(2, 4, 6)).astype(np.float16)
        cache = TokenCache(old, new, [2, 0, 1, -1, -1], boundary=3)
        rows = np.array([4, 0, 3, 2, 1])
        expected = np.stack([new[1], old[2], new[0], old[1], old[0]]).astype(np.float32)
        np.testing.assert_array_equal(cache.read(rows).mean(1), expected.mean(1))

    def test_uniform_attention_is_exact_noop(self):
        projected = self.rng.normal(size=(7, 4, 5))
        residual, weights = spatial_residual(projected, np.zeros((7, 4, 1)),
                                            np.ones((7, 2)), [[0, 1]], [1.])
        np.testing.assert_array_equal(weights, np.full((7, 4), .25))
        np.testing.assert_array_equal(residual, np.zeros((7, 5)))

    def test_centered_residual_matches_weighted_pool_minus_mean(self):
        tokens = self.rng.normal(size=(9, 4, 6))
        projection = self.rng.normal(size=(6, 3))
        centered = tokens - tokens.mean(1, keepdims=True)
        evidence = centered @ self.rng.normal(size=(6, 1))
        residual, weights = spatial_residual(centered @ projection, evidence,
                                            np.ones((9, 2)), [[0, 1]], [2.])
        expected = (np.einsum('nt,ntd->nd', weights, tokens) - tokens.mean(1)) @ projection
        np.testing.assert_allclose(residual, expected, atol=2e-15)

    def test_inference_has_no_truth_input_and_obeys_predicted_pair(self):
        self.assertNotIn('labels', inspect.signature(spatial_residual).parameters)
        projected = self.rng.normal(size=(2, 4, 3))
        projected[1] = projected[0]
        evidence = np.zeros((2, 4, 3))
        evidence[:, 0, 0] = 3
        evidence[:, 3, 2] = 3
        residual, _ = spatial_residual(projected, evidence,
            [[3, 2, 1], [1, 2, 3]], [[0, 1], [0, 2], [1, 2]], np.ones(3))
        self.assertGreater(np.linalg.norm(residual[0] - residual[1]), .01)

    def test_source_scaling_does_not_depend_on_target(self):
        source = self.rng.normal(size=(12, 4, 3))
        scale = source_rms(source)
        target = self.rng.normal(size=(5, 4, 3))
        before = target / scale
        target[1:] *= 1000
        np.testing.assert_array_equal((target / scale)[0], before[0])
        np.testing.assert_array_equal(source_rms(source), scale)

    def test_kernel_readout_matches_explicit_primal_ridge(self):
        core = self.rng.normal(size=(13, 5))
        core = np.column_stack((core, np.ones(13)))
        features = self.rng.normal(size=(13, 3))
        prior = self.rng.normal(size=(13, 3))
        head = self.rng.normal(size=(3, 3))
        support = np.arange(6)
        labels = np.array([0, 0, 1, 1, 1, 2])
        scores, _ = coordinated_expert(prior, core @ core[support].T,
                                       features, head, support, labels)
        phi = np.column_stack((core, .5 * features / np.sqrt(3)))
        weights = 6 / (3 * np.bincount(labels)[labels])
        augmented_prior = prior + features @ head
        desired = 2 * np.eye(3)[labels] - 1 - augmented_prior[support]
        coef = np.linalg.solve(phi[support].T @ (weights[:, None] * phi[support])
                               + 8 * np.eye(phi.shape[1]),
                               phi[support].T @ (weights[:, None] * desired))
        np.testing.assert_allclose(scores, augmented_prior + phi @ coef, atol=2e-14)

    def test_zero_branch_preserves_original_expert(self):
        core = self.rng.normal(size=(9, 4))
        prior = self.rng.normal(size=(9, 3))
        support, labels = np.arange(4), np.array([0, 1, 1, 2])
        kernel = core @ core[support].T + 1
        head = fit_residual_head(np.zeros((9, 2)), np.arange(9) % 3, prior)
        scores, _ = coordinated_expert(prior, kernel, np.zeros((9, 2)), head, support, labels)
        expected = prior + kernel @ fit_kernel_correction(kernel[support], labels, prior[support])
        np.testing.assert_array_equal(scores, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
