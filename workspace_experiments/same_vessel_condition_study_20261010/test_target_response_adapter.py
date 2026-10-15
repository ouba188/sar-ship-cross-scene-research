import unittest
import numpy as np
import pandas as pd
import torch
from target_response_adapter import residual_head, ResponseAdapter, support_design


class TargetAdapterTests(unittest.TestCase):
    def test_weighted_loo_and_absent_class_prior(self):
        torch.manual_seed(31)
        phi = torch.randn(12, 6, dtype=torch.float64)
        prior = torch.randn(12, 8, dtype=torch.float64)
        labels = torch.tensor([0]*7+[1]*3+[2]*2)
        coef, loo, target = residual_head(phi, prior, labels, ridge=3., balanced=True)
        counts = torch.bincount(labels, minlength=8)
        weights = len(labels)/(3*counts[labels].double())
        residual = (target-prior)*(counts > 0)[None, :]
        expected = []
        for i in range(12):
            keep = torch.arange(12) != i
            p, w = phi[keep], weights[keep]
            fitted = torch.linalg.solve(p.T@(w[:, None]*p)+3*torch.eye(6, dtype=torch.float64),
                                        p.T@(w[:, None]*residual[keep]))
            expected.append(prior[i]+phi[i]@fitted)
        torch.testing.assert_close(loo, torch.stack(expected), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(coef[:, 3:], torch.zeros_like(coef[:, 3:]), atol=0, rtol=0)
        torch.testing.assert_close(loo[:, 3:], prior[:, 3:], atol=0, rtol=0)

    def test_analytic_loo_matches_explicit_leave_one_out(self):
        torch.manual_seed(17)
        phi = torch.randn(11, 6, dtype=torch.float64)
        prior = torch.randn(11, 8, dtype=torch.float64)
        labels = torch.arange(11) % 8
        _, loo, target = residual_head(phi, prior, labels, ridge=3.)
        expected = []
        for i in range(11):
            keep = torch.arange(11) != i
            p = phi[keep]
            coef = torch.linalg.solve(p.T@p+3*torch.eye(6, dtype=torch.float64),
                                     p.T@(target[keep]-prior[keep]))
            expected.append(prior[i]+phi[i]@coef)
        torch.testing.assert_close(loo, torch.stack(expected), atol=1e-10, rtol=1e-10)

    def test_identity_and_target_gradient(self):
        torch.manual_seed(9)
        adapter = ResponseAdapter(np.ones(6, np.float32))
        x = torch.rand(11, 6, 2, 2)
        torch.testing.assert_close(adapter(x), x, atol=0, rtol=0)
        phi = adapter(x).square().mean((2, 3))
        _, loo, target = residual_head(phi, torch.zeros(11, 8), torch.arange(11) % 8)
        (loo-target).square().mean().backward()
        self.assertGreater(float(adapter.log_gain.grad.abs().sum()), 0.)
        self.assertGreater(float(adapter.shift.grad.abs().sum()), 0.)

    def test_support_selection_ignores_class_labels_and_purges_vessels(self):
        meta = pd.DataFrame(dict(mmsi=np.repeat(np.arange(100), 3), class_id=np.arange(300) % 8))
        support, query = support_design(meta, 0)
        self.assertEqual(len(support), 64)
        self.assertEqual(meta.iloc[support].mmsi.nunique(), 64)
        self.assertFalse(set(meta.iloc[support].mmsi) & set(meta.iloc[query].mmsi))
        meta.class_id = 0
        s2, q2 = support_design(meta, 0)
        np.testing.assert_array_equal(support, s2)
        np.testing.assert_array_equal(query, q2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
