import unittest
from types import SimpleNamespace
import numpy as np
from transforms_regions import region_boxes, box_mask, apply


class RegionTests(unittest.TestCase):
    def test_coordinate_translation_and_equal_area_sham(self):
        row = SimpleNamespace(head_x=96, head_y=198, tail_x=104, tail_y=202,
                              center_x=100, center_y=200)
        box, sham, valid, sham_valid, _, points = region_boxes(row, (41, 41), .2)
        np.testing.assert_array_equal(points, [[16, 18], [24, 22]])
        self.assertEqual(box, (13, 15, 28, 26))
        self.assertTrue(valid and sham_valid)
        a, b = box_mask((41, 41), box), box_mask((41, 41), sham)
        self.assertEqual(a.sum(), b.sum())
        self.assertFalse(np.any(a & b))

    def test_flat_removes_inside_and_preserves_outside(self):
        pair = np.full((2, 9, 9), 7, dtype=np.uint8)
        pair[:, 3:6, 3:6] = 200
        mask = np.zeros((9, 9), bool)
        mask[3:6, 3:6] = True
        result = apply(pair, mask, "flat", 1)
        np.testing.assert_array_equal(result, np.full_like(pair, 7))
        np.testing.assert_array_equal(pair[:, 3:6, 3:6], np.full((2, 3, 3), 200))

    def test_shuffle_preserves_polarization_tuples_not_locations(self):
        pair = np.arange(2*9*9, dtype=np.uint8).reshape(2, 9, 9)
        mask = np.zeros((9, 9), bool)
        mask[2:7, 2:7] = True
        result = apply(pair, mask, "shuffle", 123)
        self.assertFalse(np.array_equal(pair, result))
        np.testing.assert_array_equal(result[:, ~mask], pair[:, ~mask])
        self.assertEqual(sorted(map(tuple, pair[:, mask].T)), sorted(map(tuple, result[:, mask].T)))

    def test_complementary_display_changes_reconstruct_global(self):
        pair = np.arange(2*9*9, dtype=np.uint8).reshape(2, 9, 9)
        mask = np.zeros((9, 9), bool)
        mask[3:6, 3:6] = True
        inside = apply(pair, mask, "display", 1)
        outside = apply(pair, ~mask, "display", 1)
        full = apply(pair, np.ones_like(mask), "display", 1)
        combined = inside.astype(int)+outside.astype(int)-pair.astype(int)
        np.testing.assert_array_equal(combined, full)


if __name__ == "__main__":
    unittest.main()
