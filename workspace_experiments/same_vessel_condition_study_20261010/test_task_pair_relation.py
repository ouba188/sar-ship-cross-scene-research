"""Integration checks for the actual signed support calibrator."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from paired_evidence_factors import adapted_scores, support_operator
from task_pair_relation import (AMPLITUDE, eligible, episode_base, normalize_design, pair_design, predict)


class PairRelationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.prior = rng.normal(size=(12, 8))
        self.kernel = rng.normal(size=(12, 4))
        self.s, self.q = np.arange(7), np.arange(7, 12)
        self.y = np.array([0, 1, 1, 3, 3, 3, 0])

    def test_missing_class_zero_matches_original(self):
        base, op, residual = episode_base(self.prior, self.kernel, self.s, self.q, self.y)
        original, present = support_operator(self.kernel[self.s], self.kernel[self.q], self.y)
        expected = adapted_scores(self.prior[self.s], self.prior[self.q], original, present, self.y)
        np.testing.assert_array_equal(base, expected)
        score, m = predict(base, op, residual, np.ones((5, 7, 2)), np.zeros(2))
        np.testing.assert_array_equal(score, expected)
        np.testing.assert_array_equal(m, np.ones((5, 7)))

    def test_nonzero_matches_direct_original_residual(self):
        base, op, residual = episode_base(self.prior, self.kernel, self.s, self.q, self.y)
        score, m = predict(base, op, residual, np.ones((5, 7, 2)), np.array([.1, -.3]))
        expected = self.prior[self.q]+(op*m)@residual
        np.testing.assert_allclose(score, expected, atol=1e-14, rtol=0)
        absent = np.setdiff1d(np.arange(8), self.y)
        np.testing.assert_array_equal(score[:, absent], self.prior[self.q][:, absent])

    def test_missing_edge_has_exact_identity_and_bound(self):
        raw = np.full((5, 7, 7), np.nan); valid = np.zeros((5, 7), bool)
        raw[0, 0] = 100; valid[0, 0] = True
        x = normalize_design(raw, valid, np.zeros(7), np.ones(7))
        base, op, residual = episode_base(self.prior, self.kernel, self.s, self.q, self.y)
        _, m = predict(base, op, residual, x, np.ones(8))
        np.testing.assert_array_equal(m[~valid], np.ones((~valid).sum()))
        self.assertGreaterEqual(m.min(), 1-AMPLITUDE)
        self.assertLessEqual(m.max(), 1+AMPLITUDE)

    def test_symmetric_features_and_cross_port_mask(self):
        rng = np.random.default_rng(1)
        z, angle = rng.normal(size=(12, 3)), rng.uniform(32, 42, 12)
        for arm in ('constant', 'evidence', 'condition', 'sham'):
            a = pair_design(z, angle, self.s, self.q, arm)
            b = pair_design(z, angle, self.q, self.s, arm)
            np.testing.assert_array_equal(a, b.transpose(1, 0, 2))
        meta = pd.DataFrame(dict(incidence_deg=[35., 36., np.nan], platform=['S1A']*3,
                                 pass_name=['Ascending']*3, port=['A', 'B', 'A']))
        np.testing.assert_array_equal(eligible(meta, np.array([0]), np.array([1, 2]), [30, 45]), [[True], [False]])

    def test_actual_decision_gradient_by_finite_difference(self):
        base, op, residual = episode_base(self.prior, self.kernel, self.s, self.q, self.y)
        x = np.random.default_rng(31).normal(size=(5, 7, 3))
        coefficient = np.array([.2, -.1, .05])
        theta = torch.tensor(coefficient, requires_grad=True)
        b, o, r, xt = [torch.tensor(v) for v in (base, op, residual, x)]
        labels = torch.tensor([0, 1, 3, 1, 0])
        loss = F.cross_entropy(b+(o*AMPLITUDE*torch.tanh(xt@theta))@r, labels)
        loss.backward()
        numerical = []
        for j in range(3):
            values = []
            for sign in (-1, 1):
                c = coefficient.copy(); c[j] += sign*1e-6
                score, _ = predict(base, op, residual, x, c)
                values.append(float(F.cross_entropy(torch.tensor(score), labels)))
            numerical.append((values[1]-values[0])/2e-6)
        np.testing.assert_allclose(theta.grad.numpy(), numerical, atol=1e-8, rtol=1e-5)
        self.assertGreater(np.linalg.norm(numerical), 1e-6)


if __name__ == '__main__':
    unittest.main()
