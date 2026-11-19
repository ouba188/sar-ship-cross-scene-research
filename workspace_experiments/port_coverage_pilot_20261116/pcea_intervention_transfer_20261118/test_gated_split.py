import unittest
import torch
from gated_split import PolarGatedSplit, weighted_consistency


class GatedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.net = PolarGatedSplit(4, 3)
        self.x = torch.rand(3, 2, 4, 5, 5)
        self.valid = torch.ones(3, 2, 1, 5, 5)
        self.available = torch.ones(3, 2)

    def test_full_width_continuous_and_sample_conditioned(self):
        c, e, gate = self.net(self.x, self.valid, self.available)
        torch.testing.assert_close(c+e, self.x)
        self.assertTrue(((gate > 0) & (gate < 1)).all())
        self.assertGreater(float((gate[0]-gate[1]).abs().max()), 0.)
        self.assertEqual(c.shape, self.x.shape)

    def test_invalid_padding_does_not_drive_gate(self):
        self.valid[:, :, :, :2] = 0
        changed = self.x.clone()
        changed[:, :, :, :2] = 1000
        a = self.net(self.x, self.valid, self.available)[2]
        b = self.net(changed, self.valid, self.available)[2]
        torch.testing.assert_close(a, b)

    def test_recombination_has_donor_effect_and_identity_initialization(self):
        c, e, _ = self.net(self.x, self.valid, self.available)
        donor = torch.tensor([1, 2, 0])
        mixed = self.net.recombine(c, e, donor, self.valid, self.available)
        torch.testing.assert_close(mixed, c+e[donor])
        self.assertGreater(float((mixed-c).norm()), 0.)
        mixed.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.net.parameters()))

    def test_unavailable_polarization_and_singleton_not_injected(self):
        self.available[:, 1] = 0
        c, e, _ = self.net(self.x, self.valid, self.available)
        mixed = self.net.recombine(c, e, torch.tensor([1, 0, 2]), self.valid, self.available)
        torch.testing.assert_close(mixed[:, 1], c[:, 1])
        torch.testing.assert_close(mixed[2], c[2])

    def test_target_and_confidence_detached_but_changed_trainable(self):
        clean = torch.tensor([[3., 0.], [1., -1.]], requires_grad=True)
        changed = torch.tensor([[0., 2.], [2., 0.]], requires_grad=True)
        loss, _ = weighted_consistency(changed, clean, torch.tensor([True, False]))
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertGreater(float(changed.grad[0].norm()), 0.)
        torch.testing.assert_close(changed.grad[1], torch.zeros(2))

    def test_uniform_prediction_not_forced_and_matched_prediction_zero(self):
        clean = torch.zeros(3, 4)
        loss, stats = weighted_consistency(torch.randn(3, 4), clean, torch.ones(3, dtype=torch.bool))
        self.assertAlmostEqual(float(loss), 0., places=6)
        self.assertAlmostEqual(stats['reliability'], 0., places=6)
        clean = torch.randn(3, 4)
        loss, _ = weighted_consistency(clean, clean, torch.ones(3, dtype=torch.bool))
        self.assertAlmostEqual(float(loss), 0., places=6)


if __name__ == '__main__':
    unittest.main()
