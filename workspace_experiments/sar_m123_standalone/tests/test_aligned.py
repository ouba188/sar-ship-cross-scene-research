"""Four-class contract and fresh-chip smoke tests; no training replay."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from sar_m123.aligned import AlignedClassifier, AlignedM3, FEATURES, local_features, raw_features, summary_features

ASSETS = Path(__file__).resolve().parents[1]/'assets_aligned/oss2'


class RepresentationTests(unittest.TestCase):
    def test_region_and_polarization_order(self):
        layer = np.zeros((1, 2, 20, 256), np.float32)
        layer[0, 0, :8] = 1
        layer[0, 0, 8:16] = 2
        layer[0, 0, 16:] = 3
        layer[0, 1] = layer[0, 0]+3
        mass = np.ones((1, 2, 20), np.float32)
        value = raw_features(dict(layer1=layer, mass=mass), [[1, 1]])
        self.assertEqual(value.shape, (1, 1538))
        np.testing.assert_array_equal(value[0, :1536].reshape(6, 256)[:, 0], np.arange(1, 7))
        mass[0, 1] = 0
        value = raw_features(dict(layer1=layer, mass=mass), [[1, 0]])
        np.testing.assert_array_equal(value[0, 768:1536], 0)
        np.testing.assert_array_equal(value[0, -2:], [1, 0])

    def test_summary_metadata_layout(self):
        summary = dict(summary1=np.arange(72).reshape(1, 2, 3, 4, 3),
                       geometry=np.zeros((1, 2, 4)), quality=np.zeros((1, 2, 3, 4)))
        summary['geometry'][0, 0] = [15, 7, .9, 1]
        summary['quality'][0, 1, 2] = [3, .5, 7, 1]
        value = summary_features(summary)
        self.assertEqual(value.shape, (1, 104))
        np.testing.assert_array_equal(value[0, :72], np.arange(72))
        np.testing.assert_allclose(value[0, 72:76], [np.log(16), np.log(8), .9, 1])
        np.testing.assert_allclose(value[0, -4:], [np.log(4), .5, np.log(8), 1])

    def test_four_class_competitors(self):
        base = np.array([[4., 1., 2., 3.]], np.float32)
        local = np.zeros((1, 4, 2, 3, 2), np.float32)
        context = local_features(local, base)
        self.assertEqual(context.shape, (1, 4, 15))
        np.testing.assert_array_equal(context[0, :, -1], [1, -3, -2, -1])


@unittest.skipUnless((ASSETS/'manifest.json').exists(), 'Export local aligned assets for inference tests')
class ExportedInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AlignedClassifier(ASSETS, 'cpu')

    def test_same_ship_exclusion(self):
        model = self.model.m3
        query = {k: model.bank[k][:1] for k in FEATURES}
        rows = model.retrieve(query, [model.bank['mmsi'][0]])
        self.assertEqual(rows.shape, (1, 4, 2))
        self.assertNotIn(model.bank['mmsi'][0], model.bank['mmsi'][rows].ravel())
        np.testing.assert_array_equal(rows//8, np.broadcast_to(np.arange(4)[None, :, None], rows.shape))

    def test_native_chip_single_polarization_and_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            pixels = np.random.default_rng(17).integers(0, 256, (47, 61), dtype=np.uint8)
            Image.fromarray(pixels).save(Path(folder)/'query.png')
            record = dict(id='new-query', vv=dict(image='query.png', target_index=0,
                detections=[dict(corners=[[18, 18], [42, 18], [42, 28], [18, 28]], confidence=.9)]), vh=None)
            feature = self.model.extract(record, folder)
            self.assertEqual(feature['raw'].shape, (1538,))
            self.assertEqual(feature['aux'].shape, (104,))
            self.assertEqual(set(feature['local']), set(FEATURES))
            np.testing.assert_array_equal(feature['valid'], [True, False])
            np.testing.assert_array_equal(feature['local']['mass'][1], 0)
            batch = dict(raw=np.stack([feature['raw']]*2), aux=np.stack([feature['aux']]*2),
                         local={k: np.stack([v]*2) for k, v in feature['local'].items()})
            for seed in (17, 29, 43):
                single, info = self.model.score(feature, seed)
                scores, _ = self.model.score(batch, seed)
                self.assertEqual(single.shape, (4,))
                self.assertTrue(np.isfinite(single).all())
                np.testing.assert_allclose(scores, np.stack([single]*2), atol=2e-5, rtol=2e-5)
                self.assertEqual(info['matching']['polar_weights'][1], 0)
            result = self.model.predict(record, folder, seeds=(43,), trace=True)
            json.dumps(result, allow_nan=False)
            self.assertEqual(len(result['predictions'][0]['class_scores']), 4)
            self.assertEqual(result['class_order'], ['bulk', 'general', 'chemical', 'container'])
            with self.assertRaisesRegex(ValueError, 'Exported seeds'):
                self.model.score(feature, 99)


if __name__ == '__main__':
    unittest.main()
