"""Detect wrong utility targets and accidental auxiliary training in the control."""

import unittest

import torch
from torch.nn import functional as F

from neural_complement import pilot_loss


class PilotLossTest(unittest.TestCase):
    def test_control_is_exact_weighted_cross_entropy(self):
        torch.manual_seed(7)
        scores = torch.randn(20, 8, requires_grad=True)
        utility = torch.randn(20, 3, requires_grad=True)
        labels = torch.arange(20) % 8
        weights = torch.arange(1, 9).float()
        loss, parts = pilot_loss(scores, utility, labels, scores.detach(), torch.ones(20, dtype=torch.bool),
                                 weights, candidate=False)
        torch.testing.assert_close(loss, F.cross_entropy(scores, labels, weight=weights))
        loss.backward()
        self.assertIsNone(utility.grad)
        self.assertEqual(parts['selected_count'], 0)

    def test_recovery_harm_neutral_targets_and_eligible_only(self):
        labels = torch.tensor([0, 0, 0, 0])
        expert = torch.tensor([[3., 0.], [0., 3.], [3., 0.], [0., 3.]], requires_grad=True)
        base = torch.tensor([[0., 3.], [3., 0.], [3., 0.], [0., 3.]])
        utility = torch.tensor([[-9., -9., 9.], [9., -9., -9.], [-9., 9., -9.], [9., -9., -9.]], requires_grad=True)
        eligible = torch.tensor([True, True, True, False])
        loss, parts = pilot_loss(expert, utility, labels, base, eligible, torch.ones(2), candidate=True)
        self.assertLess(float(parts['utility']), 1e-5)
        self.assertEqual(parts['selected_count'], 1)
        loss.backward()
        self.assertTrue(torch.isfinite(expert.grad).all())
        self.assertTrue(torch.equal(utility.grad[3], torch.zeros(3)))

    def test_no_source_episode_gives_control_objective(self):
        logits = torch.randn(10, 8)
        loss, parts = pilot_loss(logits, torch.randn(10, 3), torch.zeros(10, dtype=torch.long),
            torch.zeros(10, 8), torch.zeros(10, dtype=torch.bool), torch.ones(8), candidate=True)
        torch.testing.assert_close(loss, F.cross_entropy(logits, torch.zeros(10, dtype=torch.long)))
        self.assertEqual(parts['selected_count'], 0)


if __name__ == '__main__':
    unittest.main()
