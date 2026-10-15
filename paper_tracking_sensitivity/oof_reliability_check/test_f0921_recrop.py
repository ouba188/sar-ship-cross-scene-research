"""Relevant crop failures: truncation, border shifts, and radiometric replay."""

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from prepare_f0921_recrop import crop_side, crop_window, resize_pair, stretch, match_feature


class RecropTest(unittest.TestCase):
    def test_reordered_fid_requires_identity_and_ambiguous_match_falls_back(self):
        def feature(fid, lon, lat):
            return SimpleNamespace(GetFID=lambda: fid,
                GetField=lambda name: dict(lon=lon, lat=lat)[name])
        row = SimpleNamespace(mmsi=563110100, row_in_shp=7, lon=-70.97492, lat=42.33832)
        correct = feature(6, -70.964913, 42.338208)
        chosen, status = match_feature(row, [], {row.mmsi: [correct]})
        self.assertIs(chosen, correct)
        self.assertEqual(status, 'remapped_unique_mmsi')
        chosen, status = match_feature(row, [], {row.mmsi: [correct, feature(9, -70., 42.)]})
        self.assertIsNone(chosen)
        self.assertEqual(status, 'ambiguous_mmsi')
        chosen, status = match_feature(row, [], {})
        self.assertIsNone(chosen)
        self.assertEqual(status, 'missing_mmsi')

    def test_source_sizing_examples_and_soft_cap(self):
        self.assertEqual(crop_side(29.9), 64)
        self.assertEqual(crop_side(40), 72)
        self.assertEqual(crop_side(60), 112)
        self.assertEqual(crop_side(90), 128)
        self.assertEqual(crop_side(160), 160)

    def test_window_edges_and_small_tile(self):
        self.assertEqual(crop_window(10, 95, 64, 100, 100), (0, 36, 64, 64))
        self.assertEqual(crop_window(20, 30, 64, 40, 50), (0, 0, 40, 50))

    def test_old_percentile_recipe_is_preserved(self):
        raw = np.arange(4096, dtype=np.float32).reshape(64, 64)
        lo, hi = np.percentile(raw, 2), np.percentile(raw, 98)
        expected = np.clip((raw-lo)/(hi-lo)*255., 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(stretch(raw), expected)
        np.testing.assert_array_equal(stretch(np.zeros((8, 8))), np.zeros((8, 8), dtype=np.uint8))

    def test_ssl_rgb_resize_equals_repeated_gray_for_actual_square_sizes(self):
        rng = np.random.default_rng(0)
        for side in (64, 72, 112, 224):
            band = rng.integers(0, 256, (side, side), dtype=np.uint8)
            rgb = np.asarray(Image.fromarray(band).convert('RGB').resize(
                (128, 128), Image.Resampling.BILINEAR))
            gray = resize_pair([band, band])[1]
            np.testing.assert_array_equal(rgb, np.repeat(gray[:, :, None], 3, axis=2))


if __name__ == '__main__':
    unittest.main()
