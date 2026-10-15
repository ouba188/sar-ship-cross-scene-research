import unittest
import torch
from target_loss_repair import support_loss, supervised_source_loss
from target_response_adapter import residual_head


class LossRepairTests(unittest.TestCase):
    def test_mask_removes_direct_missing_class_gradient_without_rescaling_others(self):
        scores = torch.randn(6, 8, requires_grad=True)
        y = torch.tensor([0, 0, 0, 1, 1, 2])
        counts = torch.bincount(y, minlength=8)
        present = counts > 0
        weights = len(y)/(present.sum()*counts[y].float())
        target = 2*torch.eye(8)[y]-1
        original = torch.autograd.grad(support_loss(scores, target, weights, present, False), scores)[0]
        repaired = torch.autograd.grad(support_loss(scores, target, weights, present, True), scores)[0]
        self.assertGreater(float(original[:, ~present].abs().sum()), 0.)
        torch.testing.assert_close(repaired[:, ~present], torch.zeros_like(repaired[:, ~present]), atol=0, rtol=0)
        torch.testing.assert_close(repaired[:, present], original[:, present], atol=0, rtol=0)

    def test_complete_coverage_is_identical(self):
        score = torch.randn(8, 8, requires_grad=True)
        target = 2*torch.eye(8)-1
        values = [support_loss(score, target, torch.ones(8), torch.ones(8, dtype=torch.bool), flag)
                  for flag in (False, True)]
        torch.testing.assert_close(*values, atol=0, rtol=0)

    def test_missing_prior_direct_gradient_through_residual_solver(self):
        torch.manual_seed(3)
        phi = torch.randn(6, 5)
        prior = torch.randn(6, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        _, loo, target = residual_head(phi, prior, labels, balanced=True)
        present = torch.arange(8) < 3
        loss = support_loss(loo, target, torch.ones(6), present, True)
        grad = torch.autograd.grad(loss, prior)[0]
        torch.testing.assert_close(grad[:, 3:], torch.zeros_like(grad[:, 3:]), atol=0, rtol=0)
        self.assertGreater(float(grad[:, :3].abs().sum()), 0.)

    def test_source_supervision_contains_positive_signal_for_every_class(self):
        scores = torch.zeros(32, 8, requires_grad=True)
        labels = torch.arange(8).repeat_interleave(4)
        grad = torch.autograd.grad(supervised_source_loss(scores, labels), scores)[0]
        for c in range(8):
            self.assertTrue(bool((grad[labels == c, c] < 0).all()))
            self.assertTrue(bool((grad[labels != c, c] > 0).all()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
