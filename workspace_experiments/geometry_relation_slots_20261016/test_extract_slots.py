"""Focused CPU tests; never import the legacy runner or initialize CUDA."""

import unittest

import numpy as np
from scipy.ndimage import binary_dilation

from extract_slots import angular_weights, regional_slots, ship_angle


class SlotTests(unittest.TestCase):
    def setUp(self):
        self.box = np.zeros((56, 56), dtype=bool)
        self.box[22:31, 24:31] = True

    def test_soft_assignment_partition_and_slot_count(self):
        weights = angular_weights((56, 56), (27, 26), .37)
        self.assertEqual(weights.shape, (4, 56, 56))
        self.assertTrue((weights > 0).all())
        np.testing.assert_allclose(weights.sum(axis=0), 1, atol=1e-14)

    def test_regions_and_fractional_counts_match_original_masks(self):
        masks, areas, edge, center = regional_slots(self.box, .31)
        d2 = binary_dilation(self.box, iterations=2)
        d8 = binary_dilation(self.box, iterations=8)
        d16 = binary_dilation(self.box, iterations=16)
        expected = np.stack([self.box, d2 & ~self.box, d8 & ~d2, d16 & ~d8])
        self.assertEqual(masks.shape, (4, 4, 56, 56))
        self.assertEqual(areas.shape, (4, 4))
        np.testing.assert_allclose(areas.sum(axis=1), expected.sum(axis=(1, 2)), rtol=1e-6)
        np.testing.assert_allclose((masks * areas[:, :, None, None]).sum(axis=1),
                                   expected, atol=2e-7)
        np.testing.assert_array_equal(center, [27, 26])
        self.assertFalse(edge)

    def test_theta_plus_pi_invariance(self):
        first = regional_slots(self.box, .67)
        second = regional_slots(self.box, .67 + np.pi)
        np.testing.assert_allclose(first[0], second[0], atol=1e-8)
        np.testing.assert_allclose(first[1], second[1], atol=1e-6)

    def test_plus_45_degrees_cycles_slots_negative_one(self):
        first = regional_slots(self.box, .23)
        second = regional_slots(self.box, .23 + np.pi / 4)
        np.testing.assert_allclose(second[0], np.roll(first[0], -1, axis=1), atol=1e-8)
        np.testing.assert_allclose(second[1], np.roll(first[1], -1, axis=1), atol=1e-6)
        self.assertGreater(float(np.max(np.abs(second[0] - np.roll(first[0], 1, axis=1)))), 1e-4)

    def test_zero_axis_and_missing_axis_equal_image_slots(self):
        image = regional_slots(self.box)
        for axis, valid in [(0, 1), (np.nan, 0), (.7, 0), (np.nan, 1)]:
            ship = regional_slots(self.box, ship_angle(axis, valid))
            np.testing.assert_array_equal(image[0], ship[0])
            np.testing.assert_array_equal(image[1], ship[1])

    def test_nonempty_area_and_weighted_mean(self):
        masks, areas, _, _ = regional_slots(self.box, .4)
        self.assertTrue((areas > 0).all())
        np.testing.assert_allclose(masks.sum(axis=(2, 3)), 1, atol=2e-7)
        features = np.full((56, 56, 64), 7.0, dtype=np.float32)
        tokens = np.einsum("rshw,hwc->rsc", masks, features)
        self.assertEqual(tokens.shape, (4, 4, 64))
        np.testing.assert_allclose(tokens, 7, atol=1e-5)

    def test_central_pixel_phi_is_zero(self):
        for theta in (0., .4):
            actual = angular_weights((3, 3), (1, 1), theta)[:, 1, 1]
            # At phi=0 the four logits are 4cos(2theta), -4sin(2theta),
            # -4cos(2theta), 4sin(2theta), respectively.
            logits = 4 * np.array([np.cos(2 * theta), -np.sin(2 * theta),
                                   -np.cos(2 * theta), np.sin(2 * theta)])
            expected = np.exp(logits) / np.exp(logits).sum()
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_angular_direction_has_expected_slot_order(self):
        weights = angular_weights((5, 5), (2, 2))
        for y, x, slot in [(2, 4, 0), (4, 4, 1), (4, 2, 2), (4, 0, 3), (2, 0, 0)]:
            self.assertEqual(int(weights[:, y, x].argmax()), slot)

    def test_clipped_region_and_empty_region_are_finite(self):
        corner = np.zeros((56, 56), dtype=bool)
        corner[:3, :3] = True
        masks, areas, edge, _ = regional_slots(corner)
        self.assertTrue(edge)
        self.assertTrue((areas > 0).all())
        full, full_areas, _, _ = regional_slots(np.ones((56, 56), dtype=bool))
        self.assertTrue(np.isfinite(full).all())
        np.testing.assert_array_equal(full[1:], 0)
        np.testing.assert_array_equal(full_areas[1:], 0)
        with self.assertRaises(ValueError):
            regional_slots(np.zeros((56, 56), dtype=bool))


if __name__ == "__main__":
    unittest.main()
