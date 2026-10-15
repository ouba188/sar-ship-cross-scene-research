"""Detect scale, shared-prior, absent-class and detached-gradient mistakes."""

import unittest
import numpy as np
import torch
from frequency_standalone_core import standalone_scores, auxiliary_objective
from frequency_reliability_core import coefficients


class StandaloneTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.z = torch.randn(13, 256, dtype=torch.float64, requires_grad=True)
        self.c = torch.randn(13, 2, dtype=torch.float64)*.1
        self.sy = torch.tensor([0, 0, 1, 1, 2, 2])
        self.y = torch.tensor([0, 1, 2, 0, 1, 2, 7])
        self.ref = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)

    def test_matches_numpy_deployed_band_kernel(self):
        score = standalone_scores(self.z, self.c, self.sy).detach().numpy()
        z = self.z.detach().numpy().reshape(-1, 4, 64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        c = self.c.numpy()
        cond = np.exp(-((c[:, None]-c[None, :6])**2).sum(-1)/.5)
        for b in range(4):
            k = .25*(z[:, b]@z[:6, b].T)*cond
            expected = k@coefficients(k[:6], self.sy.numpy(), np.zeros((6, 8)))
            np.testing.assert_allclose(score[:, b], expected, atol=1e-12)

    def test_standalone_ignores_reference(self):
        a = auxiliary_objective(self.z, self.c, self.sy, self.ref, self.y, 'standalone')
        b = auxiliary_objective(self.z, self.c, self.sy, self.ref*100, self.y, 'standalone')
        torch.testing.assert_close(a, b)

    def test_every_band_receives_gradient_and_teacher_detached(self):
        for arm in ('standalone', 'pair'):
            loss = auxiliary_objective(self.z, self.c, self.sy, self.ref, self.y, arm)
            gz, gr = torch.autograd.grad(loss, (self.z, self.ref), allow_unused=True)
            self.assertIsNone(gr)
            self.assertTrue(torch.isfinite(gz).all())
            self.assertTrue((gz.reshape(-1, 4, 64).square().sum((0, 2)) > 0).all())

    def test_absent_support_classes_not_negative_competitors(self):
        ref = self.ref.detach().clone()
        ref[:, 3:] = 1e6
        a = auxiliary_objective(self.z, self.c, self.sy, self.ref, self.y, 'pair')
        b = auxiliary_objective(self.z, self.c, self.sy, ref, self.y, 'pair')
        torch.testing.assert_close(a, b)

    def test_no_supported_query_is_zero_differentiable(self):
        loss = auxiliary_objective(self.z, self.c, self.sy, self.ref, torch.full((7,), 7), 'standalone')
        self.assertEqual(float(loss.detach()), 0)
        gradient, = torch.autograd.grad(loss, self.z)
        self.assertTrue((gradient == 0).all())

    def test_target_probe_handles_noncontiguous_support_rows(self):
        from frequency_standalone_target_probe import predict
        z = self.z.detach().numpy().reshape(-1, 4, 64)
        z = z/np.linalg.norm(z, axis=-1, keepdims=True)
        support = np.array([4, 1, 7, 0, 2, 8])
        order = np.r_[support, np.setdiff1d(np.arange(13), support)]
        actual = predict(z, self.c.numpy(), support, self.sy.numpy())
        expected = standalone_scores(self.z[order], self.c[order], self.sy).argmax(-1).detach().numpy()
        np.testing.assert_array_equal(actual[order], expected)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
