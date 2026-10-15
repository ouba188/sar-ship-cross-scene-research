"""Check unique-anchor semantics, gradients, and no forced disagreement."""

import unittest
import torch
from frequency_preserve_core import unique_anchors, retention_loss


class PreservationTest(unittest.TestCase):
    def setUp(self):
        self.labels = torch.zeros(4, dtype=torch.long)
        self.scores = torch.zeros(4, 4, 3, requires_grad=True)
        with torch.no_grad():
            self.scores[:, :, 1] = 1
            self.scores[0, 2, 0] = 2
            self.scores[1, 0:2, 0] = 2
            self.scores[2, 1, 0] = 2
        self.main = torch.tensor([[0., 1., 0.], [0., 1., 0.], [2., 1., 0.], [0., 1., 0.]], requires_grad=True)

    def test_selects_actual_unique_rescue_only(self):
        mask, margin = unique_anchors(self.scores, self.main, self.labels)
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(mask[0, 2])
        self.assertAlmostEqual(margin[0, 2].item(), .1)
        self.assertFalse(margin.requires_grad)

    def test_penalty_restores_correct_class_only_in_anchor_branch(self):
        mask, margin = unique_anchors(self.scores, self.main, self.labels)
        current = self.scores.detach().clone()
        current[0, 2, 0] = .5
        current.requires_grad_()
        loss = retention_loss(current, self.labels, mask, margin)
        g, = torch.autograd.grad(loss, current)
        self.assertGreater(float(loss.detach()), 0)
        self.assertLess(g[0, 2, 0].item(), 0)
        self.assertGreater(g[0, 2, 1].item(), 0)
        self.assertEqual(g[~mask].abs().sum().item(), 0)

    def test_other_branches_may_become_correct_without_penalty(self):
        mask, margin = unique_anchors(self.scores, self.main, self.labels)
        current = self.scores.detach().clone()
        current[0, :, 0] = 3
        self.assertEqual(retention_loss(current, self.labels, mask, margin).item(), 0)

    def test_empty_anchors_have_zero_finite_gradient(self):
        mask = torch.zeros((4, 4), dtype=torch.bool)
        loss = retention_loss(self.scores, self.labels, mask, torch.ones(4, 4))
        g, = torch.autograd.grad(loss, self.scores)
        self.assertEqual(g.abs().sum().item(), 0)

    def test_teacher_is_detached(self):
        mask, margin = unique_anchors(self.scores, self.main, self.labels)
        current = torch.zeros(4, 4, 3, requires_grad=True)
        loss = retention_loss(current, self.labels, mask, margin)
        gs, gm = torch.autograd.grad(loss, (self.scores, self.main), allow_unused=True)
        self.assertIsNone(gs)
        self.assertIsNone(gm)


if __name__ == '__main__':
    unittest.main()
