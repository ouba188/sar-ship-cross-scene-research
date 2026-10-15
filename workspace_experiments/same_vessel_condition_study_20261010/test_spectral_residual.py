import unittest
import numpy as np
import torch
from spectral_residual import masks, transform, reconstruct, branch, SpectralResidual


class SpectralTests(unittest.TestCase):
    def test_dct_inverse_and_spatial_axes(self):
        x = np.random.default_rng(1).normal(size=(3, 4, 8, 8))
        np.testing.assert_allclose(reconstruct(transform(x)), x, atol=1e-12)
        for b in range(3):
            for c in range(4):
                np.testing.assert_allclose(transform(x)[b, c], transform(x[b, c]), atol=1e-12)

    def test_partition_and_middle_dc(self):
        m = masks()
        np.testing.assert_allclose(m.sum(0), 1, atol=1e-7)
        self.assertTrue(bool((m >= 0).all()))
        self.assertEqual(float(m[1, 0, 0]), 0.)
        self.assertEqual(float(m[2, 0, 0]), 0.)

    def test_constant_has_no_middle_response(self):
        x = np.ones((2, 3, 8, 8), np.float32)
        np.testing.assert_allclose(branch(x, masks()[1], np.ones(3)), 0, atol=1e-7)

    def test_source_rms_matching(self):
        x = np.random.default_rng(2).normal(size=(7, 3, 8, 8)).astype(np.float32)
        raw = branch(x, masks()[1], np.ones(3))
        scale = np.sqrt(np.mean(x*x, axis=(0, 2, 3))/np.mean(raw*raw, axis=(0, 2, 3)))
        out = branch(x, masks()[1], scale)
        np.testing.assert_allclose(np.mean(out*out, axis=(0, 2, 3)), np.mean(x*x, axis=(0, 2, 3)), rtol=1e-6)

    def test_middle_preserves_spatial_channel_means(self):
        x = np.random.default_rng(5).uniform(size=(7, 3, 8, 8)).astype(np.float32)
        middle = branch(x, masks()[1], np.ones(3))
        changed = x+middle*np.array([.2, -.1, .15])[None, :, None, None]
        np.testing.assert_allclose(changed.mean((2, 3)), x.mean((2, 3)), atol=1e-7)
        self.assertGreater(float(np.linalg.norm(changed-x)), .01)

    def test_identity_gradient_and_bounds(self):
        model = SpectralResidual(3)
        h = torch.ones(2, 3, 8, 8)
        r = torch.ones_like(h)*2
        torch.testing.assert_close(model(h, r), h, atol=0, rtol=0)
        model(h, r).square().mean().backward()
        self.assertTrue(bool((model.gate.grad != 0).all()))
        with torch.no_grad():
            model.gate.fill_(1.)
        model.bound()
        torch.testing.assert_close(model.gate, torch.ones(3)*.25, atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
