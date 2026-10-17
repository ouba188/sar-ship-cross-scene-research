"""Small regressions for the agreed universal feature contract."""
import unittest

import numpy as np
import torch

from universal_features import native_grid, pool, regions


class UniversalFeatureTests(unittest.TestCase):
    def test_normalized_regions_and_complete_coverage(self):
        for side in (13, 25, 29, 37, 53, 85):
            core, weights, positions, coverage = regions(dict(native_h=side, native_w=side), .2, .6)
            self.assertTrue(core.any())
            np.testing.assert_allclose(weights.sum((1, 2)), 1., atol=2e-7)
            np.testing.assert_allclose(coverage, 1., atol=1e-14)
            np.testing.assert_array_equal(positions[:, 3], 1.)
            np.testing.assert_array_equal(positions[:, 10:12], 0.)
            np.testing.assert_allclose(positions[:, 6:8], np.log1p(.32 * side) / 5, atol=1e-7)

    def test_no_ship_geometry_dependence(self):
        d = dict(native_h=25, native_w=25, cx=12., cy=12., length=8., width=3.,
                 ship_axis=.1, axis_valid=True)
        first = regions(d, .3, .5)
        changed = regions({**d, 'cx':0., 'cy':24., 'length':100., 'width':90.,
                           'ship_axis':2.1, 'axis_valid':False}, .3, .5)
        for a, b in zip(first, changed):
            np.testing.assert_array_equal(a, b)
        _, _, missing, _ = regions(d, np.nan, np.nan, False, False)
        self.assertTrue(np.isfinite(missing).all())
        np.testing.assert_array_equal(missing[:, 8:], 0.)

    def test_native_center_and_actual_conv1_mapping(self):
        y, x = native_grid(25, 37)
        input_x = torch.arange(112, dtype=torch.float32).reshape(1, 1, 1, 112).expand(1, 1, 112, 112)
        kernel = torch.zeros(1, 1, 7, 7)
        kernel[0, 0, 3, 3] = 1
        centers = torch.nn.functional.conv2d(input_x, kernel, stride=2, padding=3)[0, 0].numpy()
        np.testing.assert_allclose(x, (centers + .5) * 37 / 112 - .5, atol=3e-6)
        core, *_ = regions(dict(native_h=25, native_w=37, cx=0., cy=0.))
        np.testing.assert_array_equal(core, (x - 18.) ** 2 + (y - 12.) ** 2 <= 4. ** 2)

    def test_finite_pooling_including_25px(self):
        for side in (13, 25, 53):
            core, weights, _, _ = regions(dict(native_h=side, native_w=side))
            feature = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1).expand(1, 64, 56, 56)
            body, tokens = pool(feature, torch.from_numpy(core[None]), torch.from_numpy(weights[None]))
            torch.testing.assert_close(body, torch.arange(64, dtype=torch.float32).repeat(2)[None])
            torch.testing.assert_close(tokens, torch.arange(64, dtype=torch.float32).expand(1, 12, 64))
            self.assertTrue(torch.isfinite(body).all() and torch.isfinite(tokens).all())


if __name__ == '__main__':
    unittest.main()
