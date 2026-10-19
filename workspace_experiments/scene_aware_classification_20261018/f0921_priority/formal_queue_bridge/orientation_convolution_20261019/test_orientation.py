import math
import sys
import unittest
from pathlib import Path

import torch

from orientation_model import OrientationReader, centered

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'convolution_prepool_20261019'))
from conv_model import ConvReader


class OrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(19)
        self.maps = torch.randn(3, 2, 256, 32, 32)
        self.masks = torch.rand(3, 3, 32, 32).softmax(1)
        self.position = torch.randn(3, 128, 8)
        self.geometry = torch.tensor([[0.37, -0.62, 1, 1, 1.7, 0.8],
                                      [1.13, 0.45, 1, 1, 2.2, 1.3],
                                      [0.21, 0.83, 0, 0, 1.2, 0.7]])
        self.available = torch.tensor([True, True, False])

    def reader(self, arm):
        reader = OrientationReader(arm)
        with torch.no_grad():
            reader.spatial_weight.normal_(0, 0.3)
        return reader

    def run_reader(self, reader, geometry=None):
        return reader(self.maps, self.masks, self.position,
                      self.geometry if geometry is None else geometry, self.available)

    def test_parameter_equality_and_zero_initialization(self):
        counts = [sum(p.numel() for p in OrientationReader(a).parameters()) for a in 'ABCD']
        self.assertEqual(len(set(counts)), 1)
        reader = OrientationReader('A')
        self.assertEqual(tuple(reader.spatial_weight.shape), (2, 3, 32))
        self.assertEqual(int(torch.count_nonzero(reader.spatial_weight)), 0)
        self.assertEqual(counts[0], sum(p.numel() for p in ConvReader('point_first').parameters()) + 192)

    def test_a_ignores_angles_and_validity(self):
        reader = self.reader('A')
        changed = self.geometry.clone()
        changed[:, :2] += 0.91
        changed[:, 2:4] = 1 - changed[:, 2:4]
        with torch.no_grad():
            for a, b in zip(self.run_reader(reader), self.run_reader(reader, changed)):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_b_ship_axis_pi_invariance_with_active_weights(self):
        reader = self.reader('B')
        changed = self.geometry.clone()
        changed[:, 0] += math.pi
        with torch.no_grad():
            for a, b in zip(self.run_reader(reader), self.run_reader(reader, changed)):
                torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_c_missing_radar_falls_back_to_b_and_d_matches_c(self):
        b, c, d = self.reader('B'), self.reader('C'), self.reader('D')
        c.load_state_dict(b.state_dict())
        d.load_state_dict(b.state_dict())
        missing = self.geometry.clone()
        missing[:, 3] = 0
        missing[:, 1] = float('nan')
        with torch.no_grad():
            for first, second in zip(self.run_reader(b, missing), self.run_reader(c, missing)):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            for first, second in zip(self.run_reader(c), self.run_reader(d)):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_invalid_ship_falls_back_to_a(self):
        a, b = self.reader('A'), self.reader('B')
        b.load_state_dict(a.state_dict())
        missing = self.geometry.clone()
        missing[:, 2] = 0
        missing[:, 0] = float('nan')
        with torch.no_grad():
            for first, second in zip(self.run_reader(a, missing), self.run_reader(b, missing)):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_mask_before_sampling_blocks_foreign_features(self):
        reader = self.reader('C')
        projected = torch.randn(1, 2, 32, 7, 7)
        masks = torch.zeros(1, 3, 7, 7)
        masks[:, 0, :, :3] = 1
        masks[:, 1, :, 3:] = 1
        changed = projected.clone()
        changed[..., 3:] += 100
        with torch.no_grad():
            base = reader.local_response(projected, masks, self.geometry[:1])
            other = reader.local_response(changed, masks, self.geometry[:1])
        torch.testing.assert_close(base[:, :, 0, :, :, :3], other[:, :, 0, :, :, :3], rtol=0, atol=0)
        self.assertGreater(float((base[:, :, 1] - other[:, :, 1]).abs().max()), 1)
        self.assertEqual(int(torch.count_nonzero(base[:, :, 2])), 0)

    def test_signed_weights_use_unsigned_valid_mass(self):
        reader = OrientationReader('B')
        with torch.no_grad():
            reader.spatial_weight[0, 0].fill_(1)
            reader.spatial_weight[0, 1].fill_(2)
            reader.spatial_weight[0, 2].fill_(-1)
        projected = torch.full((1, 2, 32, 7, 7), 2.0)
        masks = torch.zeros(1, 3, 7, 7)
        masks[:, 0] = 0.25
        geometry = torch.tensor([[0., 0., 1., 1., 1., 1.]])
        actual = reader.local_response(projected, masks, geometry)
        # Residual 2 + 2*(1+2+2-1-1)/5; soft ownership cancels once.
        torch.testing.assert_close(actual[:, :, 0, :, 3, 3], torch.full((1, 2, 32), 3.2))

    def test_zero_init_matches_old_point_reader_with_soft_masks(self):
        reader = OrientationReader('C')
        old = ConvReader('point_first')
        old.load_state_dict({k: v for k, v in reader.state_dict().items() if k != 'spatial_weight'})
        merged = torch.stack((self.masks[:, 0], self.masks[:, 1] + self.masks[:, 2]), 1)
        with torch.no_grad():
            actual = self.run_reader(reader)
            expected = old(self.maps, merged, self.position, self.geometry, self.available)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)

    def test_shapes_empty_regions_and_centered_compatibility(self):
        reader = self.reader('C')
        self.masks[0, 1:] = 0
        self.masks[1] = 0
        with torch.no_grad():
            raw, weights = self.run_reader(reader)
            embedding = centered(raw, self.available, 2)
        self.assertEqual(tuple(raw.shape), (3, 2, 64))
        self.assertEqual(tuple(weights.shape), (3, 2, 2, 64))
        self.assertEqual(tuple(embedding.shape), (3, 128))
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertEqual(int(torch.count_nonzero(raw[1:])), 0)
        self.assertEqual(int(torch.count_nonzero(raw[0, :, 32:])), 0)
        self.assertEqual(int(torch.count_nonzero(weights[0, :, 1])), 0)
        self.assertEqual(int(torch.count_nonzero(weights[1])), 0)
        torch.testing.assert_close(weights[0, :, 0].sum(-1), torch.ones(2))

    def test_zero_init_finite_gradients_and_new_weights_learn(self):
        reader = OrientationReader('C')
        self.maps.requires_grad_()
        raw, _ = self.run_reader(reader)
        loss = (raw * torch.randn_like(raw)).sum()
        loss.backward()
        for name, parameter in reader.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        gradient = reader.spatial_weight.grad
        self.assertTrue((gradient.abs().sum(-1) > 0).all())
        self.assertTrue(torch.isfinite(self.maps.grad).all())
        before = raw.detach().clone()
        torch.optim.SGD(reader.parameters(), lr=0.01).step()
        after, _ = self.run_reader(reader)
        self.assertGreater(float((before - after.detach()).abs().max()), 1e-6)

    def test_active_sampling_responds_to_reference_change_and_bounds_radii(self):
        reader = self.reader('C')
        projected = torch.randn(1, 2, 32, 9, 9)
        masks = torch.ones(1, 3, 9, 9)
        changed = self.geometry[:1].clone()
        changed[:, 1] += 0.7
        with torch.no_grad():
            a = reader.local_response(projected, masks, self.geometry[:1])
            b = reader.local_response(projected, masks, changed)
            self.assertGreater(float((a - b).abs().max()), 1e-4)
            changed[:, 4:6] = 40
            large = reader.local_response(projected, masks, changed)
            changed[:, 4:6] = 4
            bounded = reader.local_response(projected, masks, changed)
        torch.testing.assert_close(large, bounded, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward_smoke(self):
        if torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled():
            self.skipTest('CUDA grid_sample backward needs explicitly permitted nondeterminism')
        torch.cuda.reset_peak_memory_stats()
        reader = OrientationReader('C').cuda()
        raw, weights = reader(self.maps[:1].expand(24, -1, -1, -1, -1).cuda(),
                              self.masks[:1].expand(24, -1, -1, -1).cuda(),
                              self.position[:1].expand(24, -1, -1).cuda(),
                              self.geometry[:1].expand(24, -1).cuda(),
                              self.available[:1].expand(24).cuda())
        raw.square().sum().backward()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(raw).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(reader.spatial_weight.grad).all())
        self.assertGreater(float(reader.spatial_weight.grad.abs().sum()), 0)
        print(f'CUDA N=24 peak allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB')


if __name__ == '__main__':
    unittest.main(verbosity=2)
