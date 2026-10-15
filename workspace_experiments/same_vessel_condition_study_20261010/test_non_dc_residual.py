import unittest
import numpy as np
from non_dc_residual import remove_dc, NonDCBranch
from spectral_residual import transform, reconstruct


class NonDCTests(unittest.TestCase):
    def test_matches_dct_without_dc(self):
        h = np.random.default_rng(3).normal(size=(4, 3, 8, 8)).astype(np.float32)
        f = transform(h)
        f[:, :, 0, 0] = 0
        np.testing.assert_allclose(remove_dc(h), reconstruct(f), atol=8e-7)

    def test_scaled_branch_preserves_mean_and_matches_rms(self):
        h = np.random.default_rng(4).uniform(size=(9, 3, 8, 8)).astype(np.float32)
        raw = remove_dc(h)
        scale = np.sqrt(np.mean(h*h, axis=(0, 2, 3))/np.mean(raw*raw, axis=(0, 2, 3)))
        branch = NonDCBranch(h, scale)[np.arange(9)]
        np.testing.assert_allclose(np.mean(branch*branch, axis=(0, 2, 3)), np.mean(h*h, axis=(0, 2, 3)), rtol=1e-6)
        changed = h+branch*np.array([.2, -.1, .15])[None, :, None, None]
        np.testing.assert_allclose(changed.mean((2, 3)), h.mean((2, 3)), atol=1e-7)


if __name__ == '__main__':
    unittest.main()
