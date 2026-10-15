"""Generic class counts must preserve the SAR policy and weighted ridge objective."""

import unittest

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from residual_expert import fit_correction
from run_generic_transfer import fit_gate, metrics
from run_oof_reliability import fit_reliability
from run_opensarship2_transfer import freeze_draw


class GenericModelTest(unittest.TestCase):
    def test_residual_matches_weighted_primal_ridge_for_7_8_65(self):
        rng = np.random.default_rng(54)
        phi = rng.normal(size=(64, 19))
        for k in (7, 8, 65):
            y = rng.integers(0, k, 64)
            prior = rng.normal(size=(64, k))
            _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
            weights = len(y)/(len(counts)*counts[inv])
            expected = Ridge(alpha=8., fit_intercept=False, solver='svd').fit(
                phi, 2*np.eye(k)[y]-1-prior, sample_weight=weights).coef_.T
            np.testing.assert_allclose(fit_correction(phi, y, prior), expected, atol=1e-12)

    def test_gate_replays_original_eight_class_parameters(self):
        rng = np.random.default_rng(83)
        b, e = rng.normal(size=(400, 8)), rng.normal(size=(400, 8))
        y = rng.integers(0, 8, 400)
        domains = np.array(['a', 'b', 'c', 'd'])[np.arange(400) % 4]
        expected = fit_reliability(b, e, y, domains)
        actual = fit_gate(b, e, y, domains)
        for key in ('center', 'scale', 'lr'):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-12)

    def test_65class_policy_budget_and_observed_guard(self):
        rng = np.random.default_rng(0)
        n, k = 200, 65
        b, e = rng.normal(size=(n, k)), rng.normal(size=(n, k))
        phi = rng.normal(size=(n, 12))
        support = np.arange(64)
        labels = np.arange(64) % 20
        gate = dict(center=np.zeros(4), scale=np.ones(4), lr=np.array([0., 0., 0., 0., 1.]))
        p = freeze_draw(b, e, phi, phi, gate, support, labels, np.arange(n))
        self.assertEqual(p['observed_classes'].shape, (65,))
        self.assertEqual(p['caps'], 6)
        self.assertFalse(p['selected'][:64].any())
        self.assertLessEqual(p['selected'].sum(), 6)
        self.assertTrue(np.isin(p['predictions'][2, p['selected']], np.arange(20)).all())

    def test_metrics_match_library_and_absent_classes(self):
        y = np.array([0, 0, 1, 1, 1, 3])
        pred = np.array([0, 1, 1, 1, 0, 3])
        result = metrics(y, pred, 65)
        self.assertEqual(result['acc'], accuracy_score(y, pred))
        self.assertEqual(result['ba'], balanced_accuracy_score(y, pred))
        self.assertIsNone(result['class_recall'][64])


if __name__ == '__main__':
    unittest.main()
