"""Tests for protocol mistakes that would change the selected training method."""
import unittest
from pathlib import Path
import tempfile

import numpy as np
import torch
from PIL import Image

from sar_m123.current.data import assert_source, pair_mask
from sar_m123.current.interventions import select_teacher, variants
from sar_m123.current.training import adaptation_loss, margin_floor


class CurrentProtocolTests(unittest.TestCase):
    def test_exceeding_teacher_is_not_penalized(self):
        y = torch.tensor([0])
        t = torch.tensor([[0., 2., 1.]], requires_grad=True)
        s = torch.tensor([[3., 1., 1.]], requires_grad=True)
        loss = margin_floor(s, t, y, torch.ones(1), torch.ones(1))
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertIsNone(t.grad)

    def test_floor_normalizes_over_whole_batch(self):
        teacher = torch.tensor([[1., 0.], [1., 0.]])
        student = torch.tensor([[0., 1.], [0., 1.]], requires_grad=True)
        loss = margin_floor(student, teacher, torch.tensor([0, 0]),
                            torch.tensor([1., 3.]), torch.tensor([True, False]))
        self.assertEqual(float(loss), 1.)
        loss.backward()
        self.assertEqual(float(student.grad[1].abs().sum()), 0.)
        self.assertLess(float(student.grad[0, 0]), 0.)

    def test_unselected_still_receives_classification_gradient(self):
        score = torch.tensor([[0., 2.], [2., 0.]], requires_grad=True)
        teacher = score.detach().clone()
        loss, _ = adaptation_loss(score, torch.tensor([0, 1]), teacher,
                                  torch.ones(2), torch.zeros(2), .3683)
        loss.backward()
        self.assertTrue((score.grad.abs().sum(1) > 0).all())

    def test_matched_mean_control_is_required(self):
        meta = [dict(kind=k, pol='vv', region='body', band='fine')
                for k in ('original', 'structure', 'mean_control')]
        scores = np.array([[0., 2.], [1., 1.], [1.1, .9]])
        self.assertFalse(select_teacher(scores, meta, 0)['useful'])
        scores[2] = [.1, 1.9]
        self.assertTrue(select_teacher(scores, meta, 0)['useful'])
        self.assertFalse(select_teacher(scores[:2], meta[:2], 0)['useful'])

    def test_selection_still_proposes_fine_and_middle_interventions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chip.png'
            pixels = np.full((64, 64), 40, np.uint8)
            pixels[27:37, 17:47] = 180
            pixels[30:34, 21:43] = 240
            Image.fromarray(pixels).save(path)
            spec = dict(image=str(path), target_index=0,
                detections=[dict(corners=[[17, 27], [47, 27], [47, 37], [17, 37]], confidence=.9)],
                ship_axis=dict(center=[32, 32], length_px=30, width_px=10, angle=0.))
            _, metadata = variants(dict(vv=spec, vh=None))
        bands = {m['band'] for m in metadata if m['kind'] == 'structure'}
        self.assertEqual(bands, {'fine', 'middle'})
        self.assertTrue(any(m['kind'] == 'mean_control' for m in metadata))

    def test_query_labels_cannot_authorize_source_selection(self):
        with self.assertRaises(ValueError):
            assert_source([dict(id='x', label=1, split='development')])

    def test_reference_identity_exclusion(self):
        mask = pair_mask([dict(id='q', mmsi='1'), dict(id='new')],
                         [dict(id='r', mmsi='1'), dict(id='new')], 'cpu')
        self.assertEqual(mask.tolist(), [[False, True], [True, False]])


if __name__ == '__main__':
    unittest.main()
