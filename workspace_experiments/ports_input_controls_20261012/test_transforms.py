"""Focused checks for identity, crop coordinates and shared-polarization mapping."""

import unittest
import numpy as np
from PIL import Image
from transforms_fixed import transform_pair, crop_box


class InputTransforms(unittest.TestCase):
    def test_identity_preserves_native_ssl_and_legacy_r50_pixels(self):
        vv = Image.fromarray(np.arange(80*80, dtype=np.uint8).reshape(80, 80))
        vh = Image.fromarray(np.arange(96*96*3, dtype=np.uint8).reshape(96, 96, 3))
        pair, rgb, _ = transform_pair(vv, vh, "original")
        expected = np.stack([np.asarray(im.convert("L").resize((128, 128), Image.Resampling.BILINEAR)) for im in (vv, vh)])
        np.testing.assert_array_equal(pair, expected)
        np.testing.assert_array_equal(rgb, vh)

    def test_crop_known_native_window_not_resized_window(self):
        a = np.zeros((80, 80), np.uint8)
        a[10:70, 10:70] = 137
        pair, rgb, _ = transform_pair(Image.fromarray(a), Image.fromarray(a), "crop75")
        self.assertEqual(crop_box((80, 80)), (10, 10, 70, 70))
        self.assertEqual(rgb.size, (60, 60))
        np.testing.assert_array_equal(pair, np.full((2, 128, 128), 137, np.uint8))

    def test_shared_affine_not_independent_pol_normalization(self):
        vv = Image.new("L", (128, 128), 100)
        vh = Image.new("L", (128, 128), 50)
        pair, _, info = transform_pair(vv, vh, "display")
        self.assertEqual((info["q01"], info["q99"]), (50, 100))
        np.testing.assert_array_equal(pair[0], np.full((128, 128), 255, np.uint8))
        np.testing.assert_array_equal(pair[1], np.zeros((128, 128), np.uint8))
        self.assertEqual(np.asarray(vv)[0, 0], 100)

    def test_flat_pair_finite_and_unchanged(self):
        im = Image.new("L", (64, 64), 73)
        pair, _, info = transform_pair(im, im, "crop75_display")
        self.assertTrue(info["stretch_flat"])
        np.testing.assert_array_equal(pair, np.full((2, 128, 128), 73, np.uint8))


if __name__ == "__main__":
    unittest.main()
