import unittest

import torch

from sar_m123.backbone_regions import RegionFeedback


class RegionFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.block = RegionFeedback(8, hidden=4)
        self.x = torch.randn(2, 8, 6, 6)
        self.masks = torch.zeros(2, 3, 6, 6)
        self.masks[:, 0, 1:3, 1:3] = 1
        self.masks[:, 1, 3:5, 1:3] = 1
        self.masks[:, 2, 1:3, 3:5] = 1
        self.valid = torch.ones(2, 1, 6, 6)

    def test_initial_identity_and_output_gradient(self):
        y = self.block(self.x, self.masks, self.valid)
        torch.testing.assert_close(y, self.x, rtol=0, atol=0)
        y.square().mean().backward()
        self.assertGreater(self.block.output.weight.grad.abs().sum().item(), 0)

    def test_empty_regions_and_padding_unchanged(self):
        torch.nn.init.normal_(self.block.output.weight)
        self.valid[:, :, :, 4:] = 0
        y = self.block(self.x, self.masks, self.valid)
        torch.testing.assert_close(y[:, :, :, 4:], self.x[:, :, :, 4:], rtol=0, atol=0)
        empty = self.block(self.x, self.masks * 0, self.valid)
        torch.testing.assert_close(empty, self.x, rtol=0, atol=0)

    def test_rotation_covariance_and_nonuniform_feedback(self):
        torch.nn.init.normal_(self.block.output.weight)
        y, d = self.block(self.x, self.masks, self.valid, return_details=True)
        yr = self.block(self.x.rot90(1, (-2, -1)), self.masks.rot90(1, (-2, -1)),
                        self.valid.rot90(1, (-2, -1)))
        torch.testing.assert_close(yr, y.rot90(1, (-2, -1)))
        self.assertGreater(d['weights'][0, 0, 1:3, 1:3].std().item(), 0)

    def test_feedback_can_train_through_frozen_next_layer(self):
        downstream = torch.nn.Conv2d(8, 3, 3, padding=1).requires_grad_(False)
        loss = downstream(self.block(self.x, self.masks, self.valid)).square().mean()
        loss.backward()
        self.assertGreater(self.block.output.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(downstream.weight.grad)

    def test_temperature_softens_gate_without_changing_parameters(self):
        soft = RegionFeedback(8, hidden=4, temperature=4.)
        soft.load_state_dict(self.block.state_dict())
        _, a = self.block(self.x, self.masks, self.valid, return_details=True)
        _, b = soft(self.x, self.masks, self.valid, return_details=True)
        active = self.masks > 0
        self.assertTrue(((b['weights'][active]-.5).abs() <= (a['weights'][active]-.5).abs()+1e-7).all())
        self.assertEqual(sum(p.numel() for p in soft.parameters()), sum(p.numel() for p in self.block.parameters()))

    def test_cosine_gate_ignores_positive_projection_rescaling(self):
        block = RegionFeedback(8, hidden=4, gate_mode='cosine')
        _, a = block(self.x, self.masks, self.valid, return_details=True)
        with torch.no_grad():
            block.query.weight.mul_(100)
            block.key.weight.mul_(30)
        _, b = block(self.x, self.masks, self.valid, return_details=True)
        torch.testing.assert_close(a['gates'], b['gates'])
        self.assertGreaterEqual(float(b['gates'].min()), .2689)
        self.assertLessEqual(float(b['gates'].max()), .7311)

    def test_fixed_gate_still_trains_value_feedback(self):
        block = RegionFeedback(8, hidden=4, gate_mode='fixed')
        torch.nn.init.normal_(block.output.weight)
        out, d = block(self.x, self.masks, self.valid, return_details=True)
        torch.testing.assert_close(d['gates'], torch.full_like(d['gates'], .5))
        out.square().mean().backward()
        self.assertIsNone(block.query.weight.grad)
        self.assertGreater(float(block.value.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
