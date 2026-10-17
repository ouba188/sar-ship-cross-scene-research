"""Focused reader contracts; no dataset, pipeline or target training imports."""

import math
import unittest

import torch

from geometry_reader import GeometryReader


class GeometryReaderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(183)
        torch.set_num_threads(1)
        self.tokens = torch.randn(3, 4, 16, 64, dtype=torch.float64)
        self.positions = torch.rand(3, 4, 16, 2, dtype=torch.float64) * 2 - 1
        self.areas = torch.rand(3, 4, 16, dtype=torch.float64) + 0.1
        self.geometry = torch.tensor([[0.3, 1.1, 0.6, 1, 1, 1, 0.8],
                                      [1.2, -0.4, 0.8, 1, 1, 1, 1.0],
                                      [-0.7, 2.0, 0.5, 1, 1, 1, 0.6]], dtype=torch.float64)

    def run_reader(self, model, **changes):
        args = dict(tokens=self.tokens, positions=self.positions, areas=self.areas, geometry=self.geometry)
        args.update(changes)
        return model(**args)

    def test_pi_ambiguity_and_radar_full_turn(self):
        for mode in "BC":
            model = GeometryReader(mode).double()
            original = self.run_reader(model)
            for column, shift in ((0, math.pi), (1, 2 * math.pi)):
                geometry = self.geometry.clone()
                geometry[:, column] += shift
                torch.testing.assert_close(original, self.run_reader(model, geometry=geometry), atol=1e-12, rtol=1e-12)

    def test_missing_angles_and_confidence(self):
        for mode in "ABC":
            model = GeometryReader(mode).double()
            for missing in ((0,), (1,), (2,), (0, 1, 2)):
                geometry = self.geometry.clone()
                for col in missing:
                    geometry[:, col + 3] = 0
                    geometry[:, col] = 0
                reference = self.run_reader(model, geometry=geometry)
                for col in missing:
                    geometry[:, col] = float("nan")
                if 0 in missing:
                    geometry[:, 6] = float("nan")
                torch.testing.assert_close(reference, self.run_reader(model, geometry=geometry), atol=0, rtol=0)
            geometry = self.geometry.clone()
            geometry[:, 6] = 0
            reference = self.run_reader(model, geometry=geometry)
            geometry[:, 0] += 0.6
            torch.testing.assert_close(reference, self.run_reader(model, geometry=geometry), atol=1e-12, rtol=1e-12)

    def test_unsigned_radar_pi_and_coincident_position_gradients(self):
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 2e-6)):
            for mode in "BC":
                model = GeometryReader(mode).to(dtype=dtype)
                tokens = self.tokens.to(dtype).clone().requires_grad_()
                positions = self.positions.to(dtype).clone()
                positions[:, 1:, 0] = positions[:, :1, 0]
                positions.requires_grad_()
                geometry = self.geometry.to(dtype).clone().requires_grad_()
                areas = self.areas.to(dtype)
                original = model(tokens, positions, areas, geometry)
                reversed_axis = geometry.detach().clone()
                reversed_axis[:, 1] += math.pi
                actual = model(tokens, positions, areas, reversed_axis)
                torch.testing.assert_close(original, actual, atol=tolerance, rtol=tolerance)
                (original * torch.randn_like(original)).sum().backward()
                for gradient in (tokens.grad, positions.grad, geometry.grad):
                    self.assertTrue(torch.isfinite(gradient).all())
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_unit_norm_empty_regions_and_finite_backward(self):
        for mode in "ABC":
            model = GeometryReader(mode).double()
            tokens = self.tokens.clone().requires_grad_()
            areas = self.areas.clone()
            areas[0, 2:] = 0
            areas[1, 1:] = 0
            areas[2] = 0
            output = self.run_reader(model, tokens=tokens, areas=areas)
            self.assertEqual(tuple(output.shape), (3, 64))
            torch.testing.assert_close(output.norm(dim=-1), torch.ones(3, dtype=output.dtype))
            self.assertEqual(output[2, 0].item(), 1)
            (output * torch.randn_like(output)).sum().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(model.image_relation[0].weight.grad.abs().sum().item(), 0)
            if mode == "C":
                self.assertGreater(model.geometry_relation[0].weight.grad.abs().sum().item(), 0)

    def test_masked_content_cannot_leak_and_absent_body(self):
        areas = self.areas.clone()
        areas[:, :, ::2] = 0
        areas[0, 0] = 0
        areas[1, 2] = 0
        tokens, positions = self.tokens.clone(), self.positions.clone()
        tokens[areas == 0] = float("nan")
        positions[areas == 0] = float("nan")
        for mode in "ABC":
            model = GeometryReader(mode).double()
            reference = self.run_reader(model, areas=areas)
            actual = self.run_reader(model, tokens=tokens, positions=positions, areas=areas)
            torch.testing.assert_close(reference, actual, atol=0, rtol=0)
            torch.testing.assert_close(actual.norm(dim=-1), torch.ones(3, dtype=actual.dtype))

    def test_region_area_scaling_does_not_change_region_importance(self):
        factors = torch.tensor([7, 0.2, 11, 3], dtype=torch.float64)[None, :, None]
        for mode in "ABC":
            model = GeometryReader(mode).double()
            torch.testing.assert_close(self.run_reader(model), self.run_reader(model, areas=self.areas * factors), atol=1e-12, rtol=1e-12)

    def test_exact_area_weighted_attention_and_equal_regions(self):
        model = GeometryReader("A").double()
        q = torch.zeros(1, 16, 64, dtype=torch.float64)
        k = torch.zeros(1, 3, 16, 64, dtype=torch.float64)
        v = torch.zeros_like(k)
        v[0, 0, 0] = 2
        v[0, 0, 1] = 6
        v[0, 1, 0] = 9
        area = torch.zeros(1, 4, 16, dtype=torch.float64)
        area[0, 0, :2] = torch.tensor([1, 4])
        area[0, 1, :2] = torch.tensor([1, 3])
        area[0, 2, 0] = 200
        rel = torch.zeros(1, 3, 16, 16, 64, dtype=torch.float64)
        result = model._attend(q, k, v, rel, area)
        # First region is (2+3*6)/4=5, second 9: equal-region mean 7.
        torch.testing.assert_close(result, torch.full_like(result, 7))

    def test_relative_position_and_content_response(self):
        for mode in "ABC":
            model = GeometryReader(mode).double()
            reference = self.run_reader(model)
            torch.testing.assert_close(reference, self.run_reader(model, positions=self.positions + 2.3), atol=1e-12, rtol=1e-12)
            positions = self.positions.clone()
            positions[:, 1:, :, 0] *= -1
            self.assertGreater((reference - self.run_reader(model, positions=positions)).abs().max().item(), 1e-6)
        model = GeometryReader("C").double()
        shifted = self.geometry.clone()
        shifted[:, 0] += 0.7
        shifted[:, 1] += 0.4
        change = self.run_reader(model, geometry=shifted) - self.run_reader(model)
        tokens = self.tokens.clone()
        tokens[:, 0] = torch.randn_like(tokens[:, 0])
        other = self.run_reader(model, tokens=tokens, geometry=shifted) - self.run_reader(model, tokens=tokens)
        self.assertGreater(change.abs().max().item(), 1e-6)
        self.assertGreater((change - other).abs().max().item(), 1e-6)
        model = GeometryReader("A").double()
        torch.testing.assert_close(self.run_reader(model), self.run_reader(model, geometry=shifted), atol=0, rtol=0)

    def test_body_queries_use_actual_area(self):
        model = GeometryReader("A").double()
        q = torch.zeros(1, 16, 64, dtype=torch.float64)
        q[:, 1] = 1
        k = torch.zeros(1, 3, 16, 64, dtype=torch.float64)
        k[:, 0, 1] = math.log(3) / 4
        v = torch.zeros_like(k)
        v[:, 0, 1] = 8
        areas = torch.zeros(1, 4, 16, dtype=torch.float64)
        areas[:, 0, :2] = torch.tensor([1, 3])
        areas[:, 1, :2] = 1
        relation = torch.zeros(1, 3, 16, 16, 64, dtype=torch.float64)
        # Query zero gives 4, query one gives 6; actual-area mean is 5.5.
        actual = model._attend(q, k, v, relation, areas)
        torch.testing.assert_close(actual, torch.full_like(actual, 5.5))

    def test_float32_missing_geometry_backward(self):
        for mode in "ABC":
            model = GeometryReader(mode)
            tokens = self.tokens.float().requires_grad_()
            geometry = self.geometry.float()
            geometry[:, 3:6] = 0
            geometry[:, :3] = float("nan")
            geometry[:, 6] = float("nan")
            geometry.requires_grad_()
            areas = self.areas.float()
            areas[0, 0] = 0
            areas[1, 1:] = 0
            areas[2, 2] = 0
            output = model(tokens, self.positions.float(), areas, geometry)
            torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))
            (output * torch.randn_like(output)).sum().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            for parameter in model.parameters():
                self.assertTrue(torch.isfinite(parameter.grad).all())
            if mode != "A":
                self.assertTrue(torch.isfinite(geometry.grad).all())
                self.assertEqual(geometry.grad[:, :3].abs().sum().item(), 0)

    def test_parameter_counts(self):
        counts = {mode: sum(p.numel() for p in GeometryReader(mode).parameters()) for mode in "ABC"}
        self.assertLess(max(counts.values()) / min(counts.values()), 1.02)
        self.assertLess(max(counts.values()), 40000)


if __name__ == "__main__":
    unittest.main()
