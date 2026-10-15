"""Focused checks of scene conditioning and removal of direct semantic gradients."""

import unittest
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from relative_factor_separation import (
    FactorNet, group_center, confusion_loss, matched_scene_triples,
)


class SeparationTests(unittest.TestCase):
    def test_scene_offsets_cancel_without_losing_gradients(self):
        values = torch.tensor([[1., 2.], [3., 0.], [4., 1.], [6., 5.]], requires_grad=True)
        groups = torch.tensor([0, 0, 1, 1])
        offsets = torch.tensor([[30., -8.], [-11., 9.]])
        centered = group_center(values, groups)
        torch.testing.assert_close(centered, group_center(values+offsets[groups], groups))
        centered.square().sum().backward()
        self.assertGreater(float(values.grad.abs().sum()), 0.)

    def test_confusion_uniform_on_present_classes_only(self):
        mask = torch.tensor([[True, False, True, False, False, False, False, False]])
        uniform = torch.zeros((1, 8), requires_grad=True)
        self.assertAlmostEqual(float(confusion_loss(uniform, mask, torch.ones(1))), 0., places=6)
        biased = torch.tensor([[2., 500., -2., 0., 0., 0., 0., 0.]], requires_grad=True)
        loss = confusion_loss(biased, mask, torch.ones(1))
        loss.backward()
        self.assertGreater(float(loss), 0.)
        self.assertEqual(float(biased.grad[0, 1]), 0.)

    def test_stable_only_classification_has_no_condition_gradient(self):
        torch.manual_seed(42)
        model = FactorNet()
        stable, _, _, corrected, _ = model(torch.randn(12, 512))
        labels = torch.arange(12) % 8
        F.cross_entropy(model.classifier(stable), labels).backward(retain_graph=True)
        self.assertTrue(all(p.grad is None for p in model.condition.parameters()))
        model.zero_grad()
        F.cross_entropy(model.classifier(corrected), labels).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.condition.parameters()), 0.)

    def test_scene_comparison_matches_negative_class_and_port(self):
        meta = pd.DataFrame(dict(class_id=[0, 1, 1, 0, 2, 1], product=['a', 'a', 'b', 'b', 'c', 'c'],
                                 port=['p']*4+['q']*2, mmsi=np.arange(6)))
        triples = matched_scene_triples(meta)
        self.assertGreater(len(triples), 0)
        for a, p, n in triples:
            self.assertEqual(meta.loc[a, 'product'], meta.loc[p, 'product'])
            self.assertNotEqual(meta.loc[a, 'product'], meta.loc[n, 'product'])
            self.assertEqual(meta.loc[p, 'class_id'], meta.loc[n, 'class_id'])
            self.assertEqual(meta.loc[a, 'port'], meta.loc[n, 'port'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
