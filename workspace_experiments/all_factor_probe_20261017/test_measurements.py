import unittest
from types import SimpleNamespace
import numpy as np
from measure_images import spatial_cells, read_cells, weighted_extent, image_shape, native


class NativeMeasurements(unittest.TestCase):
    def test_uniform_scene_coverage_does_not_change_response(self):
        d = SimpleNamespace(cx=2.2, cy=20.2, length=18., width=6., ship_axis=.2)
        g = spatial_cells((41, 41), d)
        v = read_cells(np.ones((41, 41))*9, g, 9)
        np.testing.assert_allclose(v[v[:, 2] > 0, 0], np.log(2))
        self.assertTrue((v[:, 1] < 1).any())

    def test_unsigned_axis_preserves_tokens(self):
        image = np.random.default_rng(2).uniform(size=(51, 51))
        for cx, cy, theta in [(25.3, 23.6, .31), (25., 25., 0.)]:
            d = SimpleNamespace(cx=cx, cy=cy, length=15., width=6., ship_axis=theta)
            a = read_cells(image, spatial_cells(image.shape, d), .5)
            d.ship_axis += np.pi
            b = read_cells(image, spatial_cells(image.shape, d), .5)
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_axis_response_and_nonphysical_numeric_scale(self):
        d = SimpleNamespace(cx=20., cy=20., length=24., width=8., ship_axis=0.)
        m, _ = native.regions((41, 41), d.cx, d.cy, d.length, d.width, d.ship_axis)
        a = np.zeros((41, 41)); a[20, 11:30] = 1
        s = image_shape(a, m, d)
        self.assertGreater(s['range_extent90_px'], 10)
        self.assertEqual(s['azimuth_extent90_px'], 0)
        s2 = image_shape(a*17, m, d)
        self.assertAlmostEqual(s['body_top5_mass'], s2['body_top5_mass'])


if __name__ == '__main__':
    unittest.main()
