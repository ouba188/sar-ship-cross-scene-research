import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from sar_m123.input import crop_pad, prepare


class InputTests(unittest.TestCase):
    def test_crop_and_pad_keep_native_values_and_target(self):
        image = np.arange(180*200, dtype=np.uint16).reshape(180, 200)
        masks = np.zeros((3, 180, 200), np.float32)
        masks[0, 70:80, 100:130] = 1
        masks[2] = 1-masks[0]
        pixels, regions, offset = crop_pad(image, masks)
        sx, sy = offset['crop_x'], offset['crop_y']
        self.assertEqual(regions[0].sum(), 300)
        np.testing.assert_array_equal(pixels, image[sy:sy+160, sx:sx+160])
        small = image[:65, :64]
        m = np.zeros((3, 65, 64), np.float32)
        m[0, 20:30, 20:30] = 1
        m[2] = 1-m[0]
        out, _, place = crop_pad(small, m)
        x, y = place['pad_left'], place['pad_top']
        np.testing.assert_array_equal(out[y:y+65, x:x+64], small)
        self.assertEqual(place['input_scale'], 1.)

    def test_target_too_large_is_not_silently_resized(self):
        image = np.zeros((180, 180), np.uint8)
        masks = np.zeros((3, 180, 180), np.float32)
        masks[0, 1:170, 1:4] = 1
        with self.assertRaises(ValueError):
            crop_pad(image, masks)

    def test_available_image_without_target_differs_from_missing_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            Image.fromarray(np.full((64, 64), 100, np.uint8)).save(root/'chip.png')
            detection = dict(corners=[[20, 20], [40, 20], [40, 30], [20, 30]], confidence=.8)
            record = dict(vv=dict(image='chip.png', detections=[detection], target_index=0),
                          vh=dict(image='chip.png', detections=[], target_index=None))
            p = prepare(record, root)
            self.assertTrue(p[0]['valid_target'])
            self.assertFalse(p[1]['valid_target'])
            self.assertEqual(p[1]['masks'][2].sum(), 4096)
            record['vh']['alignment_valid'] = False
            self.assertIsNone(prepare(record, root)[1])
            record['vv'] = None
            with self.assertRaises(ValueError):
                prepare(record, root)


if __name__ == '__main__':
    unittest.main()
