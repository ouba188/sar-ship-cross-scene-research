"""Exact prepared samples and content/config invalidation for the opt-in cache."""
from copy import deepcopy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from sar_port.two_stage import cache
from sar_port.two_stage.data import InputConfig, prepare_record
from sar_port.two_stage.geometry import GeometryConfig


class PreparedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory / 'cache'
        self.pixels = (np.arange(64 * 64).reshape(64, 64) % 256).astype(np.uint8)
        Image.fromarray(self.pixels).save(self.directory / 'image.bmp')
        valid = np.full((64, 64), 255, np.uint8)
        Image.fromarray(valid).save(self.directory / 'valid.bmp')
        q = np.zeros((64, 64), np.uint8)
        q[4:12, 4:12] = 255
        Image.fromarray(q).save(self.directory / 'q.bmp')
        view = dict(image='image.bmp', valid_mask='valid.bmp', background_mask='q.bmp',
                    background_validated=True, alignment_valid=True,
                    ship_axis=dict(center=[31.5, 31.5], length_px=16., width_px=8., angle=0.))
        self.record = dict(id='synthetic-cache-test', vv=view, vh=deepcopy(view),
                           polarizations_coregistered=True)

    def tearDown(self):
        self.temp.cleanup()

    def assert_exact(self, expected, actual):
        self.assertIs(type(actual), type(expected))
        if isinstance(expected, torch.Tensor):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.device, actual.device)
            self.assertTrue(torch.equal(expected, actual))
        elif is_dataclass(expected):
            for field in fields(expected):
                self.assert_exact(getattr(expected, field.name), getattr(actual, field.name))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.assert_exact(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.assert_exact(left, right)
        else:
            self.assertEqual(expected, actual)

    def test_disabled_is_original_path_without_cache_io(self):
        with patch.dict(os.environ, {'SAR_PORT_PREPARE_CACHE': ''}):
            sentinel = object()
            with patch.object(cache.data, 'prepare_record', return_value=sentinel) as prepare:
                self.assertIs(cache.prepare_record_cached(self.record, self.directory), sentinel)
                prepare.assert_called_once()
            self.assertIsNone(cache.save_prepared(self.record, self.directory, None, None, sentinel))
        self.assertFalse(self.root.exists())

    def test_real_roundtrip_all_tensors_geometry_and_metadata_and_hit_bypasses_prepare(self):
        expected = prepare_record(self.record, self.directory)
        with patch.dict(os.environ, {'SAR_PORT_PREPARE_CACHE': str(self.root)}):
            path = cache.save_prepared(self.record, self.directory, None, None, expected)
            raw = torch.load(path, weights_only=True, map_location='cpu')
            self.assertIsInstance(raw['prepared'], dict)
            with patch.object(cache.data, 'prepare_record', side_effect=AssertionError('cache missed')):
                actual = cache.prepare_record_cached(self.record, self.directory)
            self.assert_exact(expected, actual)
            changed = deepcopy(self.record)
            changed['vv']['ship_axis']['angle'] = .1
            with patch.object(cache.data, 'prepare_record', return_value=expected) as prepare:
                cache.prepare_record_cached(changed, self.directory)
                prepare.assert_called_once()
            self.assertEqual(len(list(self.root.glob('*.pt'))), 2)

    def test_every_source_content_and_configuration_change_invalidates(self):
        baseline = cache.cache_key(self.record, self.directory)
        for field in ('image', 'valid_mask', 'background_mask'):
            path = self.directory / self.record['vv'][field]
            original = path.read_bytes()
            stat = path.stat()
            array = np.array(Image.open(path))
            array[0, 0] = 255 - array[0, 0]
            Image.fromarray(array).save(path)
            self.assertEqual(path.stat().st_size, len(original))
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            self.assertNotEqual(baseline, cache.cache_key(self.record, self.directory), field)
            path.write_bytes(original)
            self.assertEqual(baseline, cache.cache_key(self.record, self.directory))
        changed = deepcopy(self.record)
        changed['vv']['ship_axis']['center'][0] += .25
        self.assertNotEqual(baseline, cache.cache_key(changed, self.directory))
        self.assertNotEqual(baseline, cache.cache_key(self.record, self.directory,
                                                    InputConfig(mean=(.1, 0.))))
        self.assertNotEqual(baseline, cache.cache_key(self.record, self.directory,
                                                    geometry_config=GeometryConfig(min_centers=3)))
        copies = []
        for source in cache._IMPLEMENTATION_FILES:
            target = self.directory / source.name
            target.write_bytes(source.read_bytes())
            copies.append(target)
        with patch.object(cache, '_IMPLEMENTATION_FILES', tuple(copies)):
            copied_key = cache.cache_key(self.record, self.directory)
            self.assertEqual(copied_key, baseline)
            for source in copies:
                original = source.read_bytes()
                source.write_bytes(original + b'\n# changed\n')
                self.assertNotEqual(baseline, cache.cache_key(self.record, self.directory), source.name)
                source.write_bytes(original)

    def test_absent_inactive_file_does_not_add_qualification_requirements(self):
        record = deepcopy(self.record)
        record['vh']['alignment_valid'] = False
        record['vh']['image'] = 'absent-inactive-image.png'
        record['vv']['background_validated'] = False
        record['vv']['background_mask'] = 'absent-unvalidated-q.png'
        expected = prepare_record(record, self.directory)
        with patch.dict(os.environ, {'SAR_PORT_PREPARE_CACHE': str(self.root)}):
            actual = cache.prepare_record_cached(record, self.directory)
            self.assert_exact(expected, actual)


if __name__ == '__main__':
    unittest.main()
