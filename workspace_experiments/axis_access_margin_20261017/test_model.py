"""Check mask recovery, exact baseline and task-aligned margin gradients."""
import unittest
import numpy as np
import torch
from model import predict, decision_margin_loss, prior


def fixture():
    rng = np.random.default_rng(9)
    z = torch.tensor(rng.normal(size=(7, 5)), dtype=torch.float64)
    ns = 4
    task = dict(ns=ns, ss=z[:ns]@z[:ns].T, qs=z[ns:]@z[:ns].T,
                root=torch.ones(ns, dtype=torch.float64),
                prior_s=torch.zeros(ns, 3, dtype=torch.float64),
                prior_q=torch.zeros(3, 3, dtype=torch.float64),
                desired=2*torch.eye(3, dtype=torch.float64)[[0,1,2,0]]-1,
                base=torch.full((3, 3), 1/3, dtype=torch.float64),
                old=torch.full((3, 3), 1/3, dtype=torch.float64))
    return z, task


class LocalIntegrationTests(unittest.TestCase):
    def test_exact_legacy_path(self):
        raw, task = fixture()
        valid = torch.tensor([True, True, False, True, True, False, True])
        expected, h0 = prior.old.predict(raw, valid, task, 1.)
        actual, h1, _ = predict(raw, valid, task)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(h0, h1, atol=0, rtol=0)

    def test_axis_invalid_recovers_and_has_gradient(self):
        raw, task = fixture()
        common = (raw*.3).clone().requires_grad_(True)
        axis_valid = torch.zeros(7, dtype=torch.bool)
        p, h, valid = predict(raw, axis_valid, task, common, torch.ones(7, dtype=torch.bool))
        self.assertTrue(bool(valid.all()))
        self.assertGreater(float((p-task['old']).abs().max()), 1e-5)
        loss = -p[0,0].log()
        loss.backward()
        self.assertGreater(float(common.grad[4].abs().sum()), 1e-8)
        eigenvalues = torch.linalg.eigvalsh(h.detach()@h.detach().T)
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-10)

    def test_no_information_still_falls_back(self):
        raw, task = fixture()
        invalid = torch.zeros(7, dtype=torch.bool)
        p, _, valid = predict(raw, invalid, task, raw, invalid)
        torch.testing.assert_close(p, task['old'], atol=0, rtol=0)
        self.assertFalse(bool(valid.any()))

    def test_combined_kernel_matches_primal_ridge(self):
        from scipy.special import softmax
        raw, task = fixture()
        task['root'] = torch.tensor([1., 1.4, 1., 1.4], dtype=torch.float64)
        av = torch.tensor([True, False, True, True, False, True, True])
        cv = torch.ones(7, dtype=torch.bool)
        common = raw.flip(1)*.6
        p, _, _ = predict(raw, av, task, common, cv)
        blocks = []
        for values, mask in [(raw.numpy(), av.numpy()), (common.numpy(), cv.numpy())]:
            centered = (values-values[:4][mask[:4]].mean(0))*mask[:,None]
            blocks.append(centered/np.sqrt(1+(centered**2).sum(1,keepdims=True))/np.sqrt(2))
        design = np.concatenate([raw.numpy()]+blocks,axis=1)
        weights = task['root'].numpy()**2
        s, q = design[:4], design[4:]
        coef = np.linalg.solve(s.T@(weights[:,None]*s)+8*np.eye(s.shape[1]),
                               s.T@(weights[:,None]*task['desired'].numpy()))
        expected = .5*(task['base'].numpy()+softmax(q@coef,axis=1))
        np.testing.assert_allclose(p.numpy(),expected,atol=1e-12,rtol=0)

    def test_margin_competitor_and_class_balance(self):
        p = torch.tensor([[.2,.5,.3],[.6,.3,.1]], dtype=torch.float64, requires_grad=True)
        truth = torch.tensor([0,1])
        expected = (-torch.log(torch.tensor(.2/.7)) - torch.log(torch.tensor(.3/.9)))/2
        loss = decision_margin_loss(p, truth)
        self.assertAlmostEqual(float(loss), float(expected), places=6)
        loss.backward()
        self.assertLess(float(p.grad[0,0]), 0)
        self.assertGreater(float(p.grad[0,1]), 0)
        self.assertEqual(float(p.grad[0,2]), 0)
        repeated = torch.cat([p.detach()[:1]]*3+[p.detach()[1:]])
        torch.testing.assert_close(decision_margin_loss(repeated, torch.tensor([0,0,0,1])), loss.detach())


if __name__ == '__main__':
    unittest.main()
