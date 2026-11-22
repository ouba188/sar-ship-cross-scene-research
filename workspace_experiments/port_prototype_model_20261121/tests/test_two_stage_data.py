"""Native pixels, geometry qualification, explicit trusted Q, and paired donors."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from sar_port.two_stage.data import (InputConfig, prepare_record, collate,
                                      background_targets, intervene_background)
from sar_port.two_stage.geometry import build_geometry, coverage, rectangle


class NativeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def save(self, array):
        self.counter += 1
        path = self.directory / f"image_{self.counter}.png"
        Image.fromarray(array.astype(np.uint8)).save(path)
        return path.name

    def record(self, shape=(64, 64), length=16, width=8, center=None, dual=True):
        h, w = shape
        pixels = ((np.arange(h * w).reshape(h, w) * 7) % 256).astype(np.uint8)
        center = center or [(w - 1) / 2, (h - 1) / 2]
        spec = dict(image=self.save(pixels), alignment_valid=True,
                    ship_axis=dict(center=center, length_px=length, width_px=width, angle=0.))
        record = dict(vv=spec, vh=deepcopy(spec) if dual else None,
                      polarizations_coregistered=True)
        return record, pixels

    def assert_native_pixels(self, sample, raw):
        side = sample.geometry.canvas_size
        yy, xx = np.indices((side, side))
        sx = xx - sample.placement['shift_x']
        sy = yy - sample.placement['shift_y']
        inside = (sx >= 0) & (sx < raw.shape[1]) & (sy >= 0) & (sy < raw.shape[0])
        expected = np.zeros((side, side), np.float32)
        expected[inside] = raw[sy[inside], sx[inside]].astype(np.float32) / 255.
        np.testing.assert_array_equal(sample.images[0, 0].numpy(), expected)
        np.testing.assert_array_equal(sample.valid[0, 0].numpy(), inside.astype(np.float32))

    def test_crop_and_pad_preserve_native_pixels_without_resampling(self):
        for shape, center in (((40, 48), None), ((100, 140), [120., 49.5])):
            record, raw = self.record(shape, length=12, width=6, center=center)
            sample = prepare_record(record, self.directory)
            self.assertEqual(sample.geometry.canvas_size, 64)
            self.assert_native_pixels(sample, raw)
        self.assertEqual(sample.placement['shift_x'], -71)  # Rightmost protected edge x=134 remains inside the 64-pixel crop.

    def test_buckets_prioritize_complete_protection_then_target_only(self):
        for length, size, eligible in ((20, 64, True), (60, 96, True),
                                       (90, 128, True), (126, 128, False)):
            record, raw = self.record((160, 160), length=length)
            sample = prepare_record(record, self.directory)
            self.assertEqual(sample.geometry.canvas_size, size)
            self.assertEqual(sample.pair_eligible, eligible)
            self.assert_native_pixels(sample, raw)
        record, _ = self.record((160, 160), length=140)
        with self.assertRaisesRegex(ValueError, 'maximum 128'):
            prepare_record(record, self.directory)

    def test_incomplete_target_or_internal_nodata_is_rejected(self):
        record, _ = self.record(center=[3., 31.5], length=16)
        with self.assertRaisesRegex(ValueError, 'truncated or nodata'):
            prepare_record(record, self.directory)
        record, _ = self.record()
        valid = np.full((64, 64), 255, np.uint8)
        valid[31, 31] = 0
        path = self.save(valid)
        record['vv']['valid_mask'] = path
        record['vh']['valid_mask'] = path
        with self.assertRaisesRegex(ValueError, 'truncated or nodata'):
            prepare_record(record, self.directory)

    def test_missing_polarization_is_masked_not_copied(self):
        record, _ = self.record(dual=False)
        sample = prepare_record(record, self.directory)
        for data in (sample.images, sample.valid, sample.background):
            self.assertEqual(data[1].count_nonzero().item(), 0)
        for level in sample.geometry.levels:
            self.assertFalse(level.slot_valid[1].any())
            self.assertFalse(level.match_valid[:, 1].any())
        record['vv'].pop('alignment_valid')
        with self.assertRaisesRegex(ValueError, 'alignment_valid'):
            prepare_record(record, self.directory)

    def test_q_requires_validation_and_excludes_protection(self):
        record, _ = self.record()
        q = self.save(np.full((64, 64), 255, np.uint8))
        for p in ('vv', 'vh'):
            record[p]['background_mask'] = q
        unvalidated = prepare_record(record, self.directory)
        self.assertEqual(unvalidated.background.count_nonzero().item(), 0)
        for p in ('vv', 'vh'):
            record[p]['background_validated'] = True
        sample = prepare_record(record, self.directory)
        protect = coverage(rectangle([31.5, 31.5, 16, 8, 0]).buffer(8), 64, 1) > 0
        self.assertTrue(sample.pair_eligible)
        self.assertGreater(sample.background.sum().item(), 0)
        self.assertEqual(sample.background[:, 0, protect].count_nonzero().item(), 0)
        batch = collate([sample, sample])
        altered, eligible = intervene_background(batch, torch.Generator().manual_seed(1))
        self.assertTrue(eligible.all())
        torch.testing.assert_close(altered[:, :, 0, protect], batch['images'][:, :, 0, protect])
        torch.testing.assert_close(altered[~batch['background'].bool()],
                                   batch['images'][~batch['background'].bool()])
        targets, reliable = background_targets(altered, batch['background'])
        self.assertTrue(reliable.all())
        self.assertEqual(targets.shape, (2, 2, 2))
        no_q = collate([unvalidated, unvalidated])
        unchanged, eligible = intervene_background(no_q)
        self.assertFalse(eligible.any())
        torch.testing.assert_close(unchanged, no_q['images'])

    def test_batched_inputs_do_not_silently_upsize_small_canvas(self):
        short, _ = self.record((160, 160), length=20)
        long, _ = self.record((160, 160), length=60)
        samples = [prepare_record(x, self.directory) for x in (short, long)]
        with self.assertRaisesRegex(ValueError, 'same-canvas'):
            collate(samples)


class GeometryAndInterventionTests(unittest.TestCase):
    def test_one_two_and_multiple_atoms_cut_mass_conservation(self):
        valid = np.ones((2, 64, 64), bool)
        for length, width, expected in ((10, 4, 1), (16, 8, 2), (48, 8, None)):
            geometry = build_geometry([31.5, 31.5, length, width, 0], valid)
            if expected is not None:
                self.assertEqual(geometry.atom_count, expected)
            else:
                self.assertGreater(geometry.atom_count, 2)
            self.assertLessEqual(len(geometry.area), 4)
            self.assertAlmostEqual(geometry.area.sum().item(), length * width, places=4)
            self.assertAlmostEqual(geometry.mass.sum().item(), 1, places=6)
            self.assertAlmostEqual(geometry.intervals[:, 1].sum().item(), 1, places=6)
            torch.testing.assert_close(geometry.union_regions.sum(0), geometry.union_slots[0])
            for level in geometry.levels:
                torch.testing.assert_close(level.regions.sum(0), level.slots[:, 0])
            if expected == 1:
                self.assertFalse(geometry.levels[0].slot_valid[:, 1:].any())
                self.assertTrue(geometry.levels[0].match_valid.all())
            if expected == 2:
                self.assertTrue(geometry.levels[0].slot_valid[:, 1:].all())
                self.assertFalse(geometry.levels[2].slot_valid[:, 1:].any())

    def test_dual_polarization_union_is_exact_not_max_or_sum(self):
        valid = np.zeros((2, 64, 64), bool)
        valid[0, :, ::2] = True
        valid[1, :, 1::2] = True
        geometry = build_geometry([31.5, 31.5, 16, 8, 0], valid)
        per_pol = geometry.levels[0].slots[:, 0]
        union = geometry.union_slots[0]
        self.assertAlmostEqual(union.sum().item() * 16, 128)
        self.assertAlmostEqual(geometry.area.sum().item(), 128)
        self.assertGreater(union.sum().item(), per_pol.max(0).values.sum().item())
        full = build_geometry([31.5, 31.5, 16, 8, 0], np.ones_like(valid))
        self.assertAlmostEqual(full.union_slots[0].sum().item() * 16, 128)
        self.assertLess(full.union_slots[0].sum().item(), full.levels[0].slots[:, 0].sum().item())

    def test_paired_donor_sampling_and_legal_candidate_filter(self):
        # Candidate 1 has Q in each pol but no common native donor coordinate.
        # Candidate 2 has a valid paired Q and must always remain available.
        images = torch.zeros(3, 2, 1, 4, 4)
        images[1] = -100
        images[2, 0, 0] = torch.arange(16).reshape(4, 4) + 10
        images[2, 1] = images[2, 0] + 100
        q = torch.ones_like(images)
        q[1, 0, :, :, 2:] = 0
        q[1, 1, :, :, :2] = 0
        q[0, :, :, 0] = 0
        batch = dict(images=images, valid=torch.ones_like(images), background=q,
                     pair_eligible=torch.tensor([True, False, False]))
        for seed in range(5):
            altered, eligible = intervene_background(batch, torch.Generator().manual_seed(seed))
            self.assertTrue(eligible[0])
            mask = q[0, 0].bool()
            torch.testing.assert_close(altered[0, 1][mask] - altered[0, 0][mask],
                                       torch.full_like(altered[0, 0][mask], 100))
            self.assertTrue((altered[0, 0][mask] >= 10).all())
            torch.testing.assert_close(altered[0, :, :, 0], images[0, :, :, 0])
            self.assertFalse(eligible[1:].any())
        batch['background'].zero_()
        unchanged, eligible = intervene_background(batch)
        self.assertFalse(eligible.any())
        torch.testing.assert_close(unchanged, images)


if __name__ == '__main__':
    unittest.main()
