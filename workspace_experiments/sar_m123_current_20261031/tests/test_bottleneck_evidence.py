"""Checks for a learnable, spatially bounded M2 writeback."""
import unittest
import torch
from sar_m123.bottleneck_evidence import RegionPoolWriteback


class PoolWritebackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        torch.set_num_threads(1)

    def fixture(self):
        x = torch.randn(2, 8, 4, 4)
        valid = torch.ones(2, 1, 4, 4)
        valid[1] = 0
        masks = torch.zeros(2, 3, 4, 4)
        masks[0, 0, 0] = .5
        masks[0, 1, 1] = 1
        masks[0, 2, 2] = 1
        energy = masks[:, :, None]*torch.rand(2, 3, 3, 4, 4)
        return x, masks, valid, energy

    def test_identity_then_gradient_opening(self):
        op = RegionPoolWriteback(8, hidden=4)
        x, masks, valid, energy = self.fixture()
        out, _ = op(x, masks, valid, energy)
        self.assertTrue(torch.equal(out, x))
        target = torch.randn_like(out)
        opt = torch.optim.SGD(op.parameters(), lr=.1)
        (out-target).square().mean().backward()
        self.assertGreater(op.transforms[0][-1].weight.grad.abs().sum().item(), 0)
        opt.step()
        opt.zero_grad()
        out, d = op(x, masks, valid, energy)
        (out-target).square().mean().backward()
        self.assertGreater(op.transforms[0][0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.equal(out[1], x[1]))
        self.assertTrue(torch.equal(out[0, :, 3], x[0, :, 3]))
        self.assertTrue(torch.equal(d['write_weights'][0, 0, 0], masks[0, 0, 0]))

    def test_distinct_region_values_do_not_broadcast_globally(self):
        op = RegionPoolWriteback(8, hidden=4)
        x, masks, valid, energy = self.fixture()
        with torch.no_grad():
            op.transforms[1][-1].weight.fill_(1)
        out, d = op(x, masks, valid, energy)
        self.assertTrue(torch.equal(out[:, :, 0], x[:, :, 0]))
        self.assertTrue(torch.equal(out[:, :, 2:], x[:, :, 2:]))
        self.assertGreater(d['update'][0, :, 1].abs().sum().item(), 0)
        op.enabled = False
        disabled, _ = op(x, masks, valid, energy)
        self.assertTrue(torch.equal(disabled, x))


if __name__ == '__main__':
    unittest.main()
