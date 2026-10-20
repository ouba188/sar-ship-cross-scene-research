import unittest
import numpy as np
import torch
from run import sampled, describe, Residual


class SamplingTests(unittest.TestCase):
    def test_expected_pixels_and_axial_exchange(self):
        y, x = torch.meshgrid(torch.arange(9.), torch.arange(9.), indexing='ij')
        f = (x + 10*y)[None, None]
        masks = torch.zeros(1, 3, 9, 9); masks[:, 0] = 1
        angles = torch.tensor([[0., np.pi/2]])
        r = torch.tensor([[1., 1.]])
        n, m = sampled(f, masks, angles, r)
        torch.testing.assert_close(n[0, 0, 0, 4, 4, 0], torch.tensor([44., 45., 43., 54., 34.]))
        n2, m2 = sampled(f, masks, angles + np.pi, r)
        torch.testing.assert_close(describe(n, m), describe(n2, m2), atol=2e-5, rtol=2e-5)

    def test_foreign_region_excluded_before_interpolation(self):
        f = torch.ones(1, 1, 9, 9); masks = torch.zeros(1, 3, 9, 9)
        masks[:, 0, 3:6, 3:6] = 1; masks[:, 2] = 1-masks[:, 0]
        angle = torch.tensor([[.3, .8]]); r = torch.tensor([[1.7, .8]])
        n, m = sampled(f, masks, angle, r)
        changed = f + 100*masks[:, 2:3]
        nn, mm = sampled(changed, masks, angle, r)
        torch.testing.assert_close(describe(n, m)[:, 0], describe(nn, mm)[:, 0])

    def test_merge_precedes_nonlinearity(self):
        # Both unsigned endpoint pairings are averaged after nonlinear readout.
        n = torch.ones(1, 3, 1, 1, 1, 2, 5)
        m = torch.ones(1, 3, 1, 1, 2, 5)
        n[..., 0, 1] = 3; n[..., 0, 2] = 1
        n[..., 1, 1] = 1; n[..., 1, 2] = 3
        late = describe(n, m); early = describe(n, m, merge_refs=True)
        self.assertEqual(float(late[0, 0, 0, 3, 0]), 2.)
        self.assertEqual(float(early[0, 0, 0, 3, 0]), 1.)

    def test_each_reference_sign_is_independently_irrelevant(self):
        torch.manual_seed(4)
        f = torch.rand(2, 4, 11, 11)
        masks = torch.zeros(2, 3, 11, 11)
        masks[:, 0, 3:8, 3:8] = 1; masks[:, 1, :3, :3] = 1
        masks[:, 2] = 1-masks[:, 0]-masks[:, 1]
        angles = torch.tensor([[.3, -.7], [.9, .2]])
        radii = torch.tensor([[1.7, .8], [2., 1.1]])
        n, m = sampled(f, masks, angles, radii)
        for which in (0, 1):
            alt = angles.clone(); alt[:, which] += np.pi
            nn, mm = sampled(f, masks, alt, radii)
            for refs in (False, True):
                for roles in (False, True):
                    torch.testing.assert_close(describe(n,m,refs,roles), describe(nn,mm,refs,roles), atol=2e-5, rtol=2e-5)

    def test_zero_support_finite_and_exact_fallback(self):
        f = torch.randn(2, 4, 9, 9); masks = torch.zeros(2, 3, 9, 9)
        n, m = sampled(f, masks, torch.zeros(2, 2), torch.ones(2, 2))
        self.assertEqual(int(torch.count_nonzero(describe(n, m))), 0)
        net = Residual(); base = torch.randn(3, 8)
        self.assertTrue(torch.equal(base+net(torch.randn(3, 60, 64)), base))


if __name__ == '__main__': unittest.main()
