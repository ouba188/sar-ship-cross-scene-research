import unittest
import numpy as np
from sklearn.linear_model import RidgeClassifier
from scene_training import sufficient, fit_head, semantic_projector, penalties


class TrainingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(24)
        self.x = rng.normal(size=(160, 12))
        self.y = np.arange(160) % 8
        self.weights = rng.uniform(.5, 1.5, 160)

    def test_baseline_matches_independent_sklearn(self):
        w, b = fit_head(sufficient(self.x, self.y, self.weights), np.eye(12), np.eye(12))
        ref = RidgeClassifier(alpha=1., solver='cholesky').fit(self.x, self.y, sample_weight=self.weights)
        np.testing.assert_allclose(self.x@w+b, ref.decision_function(self.x), atol=1e-10)

    def test_refit_transform_matches_explicit_transformed_fit(self):
        t = np.eye(12); t[:3, :3] *= .5
        w, b = fit_head(sufficient(self.x, self.y, self.weights), t, np.eye(12))
        ref = RidgeClassifier(alpha=1., solver='cholesky').fit(self.x@t, self.y, sample_weight=self.weights)
        np.testing.assert_allclose(self.x@w+b, ref.decision_function(self.x@t), atol=1e-10)

    def test_protected_penalty_does_not_penalize_centroid_span(self):
        sem = semantic_projector(self.x, self.y, self.weights)
        g = sufficient(self.x, self.y, self.weights)[0]
        regs = penalties(g, np.diag(np.arange(1., 13.)), sem)
        added = regs['semantic_protected_consistency']-np.eye(12)
        np.testing.assert_allclose(added@sem, 0., atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(added).min(), -1e-10)
        self.assertAlmostEqual(np.trace(added), np.trace(regs['isotropic_regularized']-np.eye(12)), places=8)


if __name__ == '__main__':
    unittest.main()
