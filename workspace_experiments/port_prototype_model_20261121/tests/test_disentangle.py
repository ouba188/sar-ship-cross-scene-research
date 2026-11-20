"""Gated reconstruction, actual donor influence and consistency masking."""

import unittest

import numpy as np
import torch

from sar_port.disentangle import PolarDisentangler, donor_permutation, weighted_consistency


class DisentangleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.features = torch.randn(3, 2, 6, 5, 5, requires_grad=True)
        self.valid = torch.ones(3, 2, 1, 10, 10)
        self.available = torch.tensor([[1., 1.], [1., 1.], [1., 0.]])
        self.module = PolarDisentangler(channels=6, hidden=4, initial_gate=.8)

    def test_complement_conservation_missing_view_and_all_open(self):
        content, remainder, gate = self.module.split(self.features, self.valid, self.available)
        visible = self.features * self.available[:, :, None, None, None]
        torch.testing.assert_close(content + remainder, visible)
        self.assertEqual(content[2, 1].abs().max().item(), 0)
        self.assertEqual(remainder[2, 1].abs().max().item(), 0)
        self.assertEqual(gate[2, 1].abs().max().item(), 0)
        torch.testing.assert_close(self.module(self.features, self.valid, self.available, mode="full"), visible)
        with torch.no_grad():
            for network in self.module.gates:
                network[-1].weight.zero_()
                network[-1].bias.fill_(40)
        torch.testing.assert_close(self.module(self.features, self.valid, self.available), visible)

    def test_swap_changes_features_and_trains_gates_and_fusion(self):
        donor = torch.tensor([1, 0, 2])
        clean = self.module(self.features, self.valid, self.available)
        changed = self.module(self.features, self.valid, self.available, "swap", donor)
        self.assertGreater((changed[:2] - clean[:2]).abs().sum().item(), 0)
        torch.testing.assert_close(changed[2], clean[2])
        changed.square().mean().backward()
        for network, fusion in zip(self.module.gates, self.module.fusions):
            self.assertTrue(torch.isfinite(network[-1].weight.grad).all())
            self.assertGreater(network[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(fusion.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.features.grad.abs().sum().item(), 0)

    def test_swap_preserves_outside_overlap_and_availability_mismatch(self):
        self.valid[1, :, :, :6] = 0
        donor = torch.tensor([1, 0, 0])
        clean = self.module(self.features, self.valid, self.available)
        changed = self.module(self.features, self.valid, self.available, "swap", donor)
        torch.testing.assert_close(changed[0, :, :, :3], clean[0, :, :, :3])
        # The third receiver has no VH; incompatible externally supplied donor is ignored.
        torch.testing.assert_close(changed[2], clean[2])

    def test_donors_exclude_shared_identity_group_and_missing_views(self):
        rows = [dict(vessel_id="a", group_id="x"), dict(vessel_id="b", group_id="y"),
                dict(vessel_id="a", group_id="y"), dict(vessel_id="c", group_id="z")]
        available = torch.tensor([[1, 1], [1, 1], [1, 1], [1, 0]])
        donor, eligible = donor_permutation(rows, available, seed=7)
        self.assertEqual(donor.tolist(), [1, 0, 2, 3])
        self.assertEqual(eligible.tolist(), [True, True, False, False])
        again = donor_permutation(rows, available, seed=7)
        torch.testing.assert_close(donor, again[0])

    def test_no_legal_donor_has_zero_loss_and_clean_is_stop_gradient(self):
        rows = [dict(vessel_id="a", group_id="x"), dict(vessel_id="a", group_id="y")]
        donor, eligible = donor_permutation(rows, torch.ones(2, 2), seed=9)
        self.assertEqual(donor.tolist(), [0, 1])
        changed = torch.randn(2, 3, requires_grad=True)
        clean = torch.randn(2, 3, requires_grad=True)
        loss = weighted_consistency(changed, clean, eligible, temperature=2.)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(changed.grad.abs().max().item(), 0)
        self.assertIsNone(clean.grad)
        changed.grad = None
        weighted_consistency(changed, clean, torch.ones(2, dtype=torch.bool)).backward()
        self.assertGreater(changed.grad.abs().sum().item(), 0)
        self.assertIsNone(clean.grad)


if __name__ == "__main__":
    unittest.main()
