"""Tests for signed candidate semantics, action mapping and missing polarizations."""

import unittest

import numpy as np  # Import before torch in the existing Windows scientific environment.
import torch

from sar_port.priors import PriorEffectStudent, attention_from_response
from sar_port.supervision import candidate_effects, distillation_losses, effects_from_legacy


class SupervisionTests(unittest.TestCase):
    def test_all_candidate_signs_and_competitor_switch(self):
        clean = torch.tensor([[4., 3., 1.]], requires_grad=True)
        changed = torch.tensor([2., 0., 3.]).reshape(1, 1, 1, 1, 3).expand(1, 2, 3, 2, 3)
        # Original margins [1,-1,-3]; changed [-1,-3,1]. c0's rival switches.
        result = candidate_effects(clean, changed)
        torch.testing.assert_close(result[0, 0, :, 0, 0], torch.tensor([2., 2., -4.]))
        self.assertFalse(result.requires_grad)

    def test_named_legacy_regions_and_operation_order(self):
        clean = torch.tensor([[4., 3., 1.]])
        scores = torch.tensor([[[2., 0., 3.]], [[1., 2., 3.]]])
        result = effects_from_legacy(clean, scores,
                                    [(1, "side", "middle"), (0, 1, "fine")],
                                    eligible=torch.tensor([[True], [False]]))
        self.assertEqual(tuple(result["effects"].shape), (1, 2, 3, 3, 2))
        torch.testing.assert_close(result["effects"][0, 1, :, 2, 1], torch.tensor([2., 2., -4.]))
        torch.testing.assert_close(result["effects"][0, 0, :, 1, 0], torch.tensor([3., 0., -4.]))
        self.assertEqual(result["effect_mask"].sum().item(), 1)
        self.assertEqual(result["effect_mask"][0, 1, 2, 1].item(), 1)

    def test_missing_view_cannot_change_available_prediction(self):
        torch.manual_seed(7)
        model = PriorEffectStudent(channels=6, classes=3, hidden=8)
        features = torch.randn(2, 2, 6, 4, 4)
        valid = torch.ones(2, 2, 1, 4, 4)
        available = torch.tensor([[1., 0.], [0., 0.]])
        regions = torch.ones(2, 2, 3, 4, 4)
        first = model(features, valid, regions, available)
        features[:, 1] += 1000
        second = model(features, valid, regions, available)
        for name in first:
            torch.testing.assert_close(first[name], second[name])
            self.assertTrue(torch.isfinite(first[name]).all())
            self.assertEqual(first[name][:, 1].abs().max().item(), 0)
            self.assertEqual(first[name][1].abs().max().item(), 0)
        torch.testing.assert_close(first["attention"][0, 0].sum((-1, -2)), torch.ones(3))

    def test_empty_supervision_is_zero_and_targets_are_detached(self):
        logits = torch.randn(1, 2, 3, 2, 2, requires_grad=True)
        effects = torch.randn(1, 2, 3, 3, 2, requires_grad=True)
        target_attention = torch.rand_like(logits, requires_grad=True)
        target_effects = torch.randn_like(effects, requires_grad=True)
        pred = {"attention": attention_from_response(logits, torch.ones(1, 2, 1, 2, 2)),
                "effects": effects}
        losses = distillation_losses(pred, target_attention, target_effects,
                                     torch.zeros(1, 2, 3), torch.zeros(1, 2, 3, 2))
        sum(losses.values()).backward()
        self.assertEqual(sum(loss.item() for loss in losses.values()), 0)
        self.assertEqual(logits.grad.abs().max().item(), 0)
        self.assertEqual(effects.grad.abs().max().item(), 0)
        self.assertIsNone(target_attention.grad)
        self.assertIsNone(target_effects.grad)

    def test_all_candidate_heads_receive_supervised_gradients(self):
        torch.manual_seed(13)
        model = PriorEffectStudent(channels=6, classes=4, hidden=8)
        features = torch.randn(2, 2, 6, 4, 4, requires_grad=True)
        valid = torch.ones(2, 2, 1, 4, 4)
        pred = model(features, valid, torch.ones(2, 2, 3, 4, 4), torch.ones(2, 2))
        target_attention = torch.zeros_like(pred["attention"])
        target_attention[..., 0, 0] = 1
        targets = torch.randn_like(pred["effects"], requires_grad=True)
        losses = distillation_losses(pred, target_attention, targets,
                                     torch.ones(2, 2, 4), torch.ones(2, 2, 3, 2))
        sum(losses.values()).backward()
        for p in range(2):
            attention_grad = model.attention_heads[p].weight.grad.flatten(1).abs().sum(1)
            effect_grad = model.effect_heads[p].weight.grad.reshape(4, 2, -1).abs().sum((1, 2))
            self.assertTrue((attention_grad > 0).all())
            self.assertTrue((effect_grad > 0).all())
        self.assertIsNone(targets.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
