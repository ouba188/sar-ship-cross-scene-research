"""Focused checks: native-axis convention, coverage, polarization identities."""
import unittest
import numpy as np
from analyze_images import regions, summarize_image


class GeometryTests(unittest.TestCase):
    def test_axis_flip_and_disjoint_regions(self):
        a, _ = regions((81, 81), 40.3, 40.2, 20, 6, .23)
        b, _ = regions((81, 81), 40.3, 40.2, 20, 6, .23+np.pi)
        for name in a:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertLessEqual(np.stack(list(a.values())).sum(0).max(), 1)

    def test_image_boundary_is_missing_not_zero(self):
        masks, cov = regions((31, 31), 1, 15, 20, 6, 0)
        self.assertLess(cov['body'], 1)
        stats = summarize_image(np.ones((31, 31))*7, masks, cov, {}, 'sigma0')
        self.assertTrue(all(r['mean'] == 7 for r in stats if r['finite_pixels']))

    def test_physical_array_rotation_transforms_regions(self):
        a, _ = regions((71, 71), 35, 35, 20, 6, 0)
        b, _ = regions((71, 71), 35, 35, 20, 6, np.pi/2)
        for name in a:
            np.testing.assert_array_equal(np.rot90(a[name]), b[name])


if __name__ == '__main__':
    unittest.main()
