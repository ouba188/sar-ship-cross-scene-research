"""First-stage shapes, readout semantics, freezing, quotas, and gradient owners."""
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sar_port.two_stage.contracts import LevelGeometry, ShipGeometry, weighted_pool
from sar_port.two_stage.stage1 import Stage1, Stage1Config, JointGate, SpatialReadout
from sar_port.two_stage.stage1_losses import (
    classification_loss, scene_regression_loss, content_invariance_loss,
    discriminator_loss, leakage_loss,
)


def geometry(size):
    levels = []
    for stride in (4, 8, 16):
        h = size // stride
        valid = torch.ones(2, 1, h, h)
        slots = torch.ones(2, 3, h, h)
        slots[:, 1, :, h // 2:] = 0
        slots[:, 2, :, :h // 2] = 0
        levels.append(LevelGeometry(valid, slots, torch.ones(2, 3, dtype=torch.bool),
                                    slots[:, :1].permute(1, 0, 2, 3),
                                    torch.ones(1, 2, dtype=torch.bool)))
    return ShipGeometry(tuple(levels), torch.ones(1), torch.tensor([[0., 1.]]),
                        torch.ones(1, 2, 3), True, 2, size)


def synthetic_output(batch=2):
    content = tuple(torch.randn(batch, 2, 256, h, h, requires_grad=True) for h in (4, 3, 2))
    residual = tuple(torch.randn_like(x, requires_grad=True) for x in content)
    valid = tuple(torch.ones(batch, 2, 1, x.shape[-2], x.shape[-1]) for x in content)
    return dict(content=content, residual=residual, valid=valid)


class Stage1EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(417)
        cls.model = Stage1(Stage1Config(classes=3))

    def setUp(self):
        self.model.zero_grad(set_to_none=True)

    def test_independent_polarizations_and_native_shapes(self):
        self.model.eval()
        left, right = self.model.backbones
        self.assertEqual(len(left), 6)
        self.assertEqual([len(left[k]) for k in (1, 3, 5)], [3, 3, 9])
        self.assertNotEqual(left[0][0].weight.data_ptr(), right[0][0].weight.data_ptr())
        torch.testing.assert_close(left[0][0].weight, right[0][0].weight)
        self.assertNotEqual(self.model.spatial_readouts[0].project.weight.data_ptr(),
                            self.model.spatial_readouts[1].project.weight.data_ptr())
        for size in (64, 96, 128):
            x = torch.randn(1, 2, 1, size, size)
            valid = torch.ones_like(x)
            valid[:, 1] = 0
            with torch.no_grad():
                out = self.model(x, valid)
            for i, stride in enumerate((4, 8, 16)):
                self.assertEqual(out["content"][i].shape, (1, 2, 256, size // stride, size // stride))
                self.assertEqual(out["spatial"][i].shape, (1, 2, 64, size // stride, size // stride))
                for name in ("content", "spatial", "residual", "gates"):
                    self.assertEqual(out[name][i][:, 1].count_nonzero().item(), 0)

    def test_disabled_fusion_removes_parameters_and_keeps_local_paths(self):
        model = Stage1(Stage1Config(3, fusion_enabled=False))
        self.assertFalse(hasattr(model, "fusion_injection_logits"))
        self.assertFalse(hasattr(model, "fusion_mixture_logits"))
        self.assertFalse(any("fusion_" in name for name, _ in model.named_parameters()))
        self.assertFalse(any("fusion_" in name for name in model.state_dict()))
        x = torch.randn(1, 2, 1, 64, 64)
        out = model(x, torch.ones_like(x))
        for level, side in enumerate((16, 8, 4)):
            self.assertEqual(out["content"][level].shape, (1, 2, 256, side, side))
            self.assertEqual(out["spatial"][level].shape, (1, 2, 64, side, side))
        # Fine output must have no dependency on either coarse projection.
        grads = torch.autograd.grad(out["spatial"][0].square().mean(),
                                    [p.weight for p in model.projections[0]],
                                    retain_graph=True, allow_unused=True)
        self.assertGreater(grads[0].abs().sum().item(), 0)
        self.assertIsNone(grads[1])
        self.assertIsNone(grads[2])
        geom = [geometry(64)]
        loss = classification_loss(model.class_logits(out, geom), torch.tensor([1]),
                                   model.class_valid(geom))
        loss.backward()
        for pol in range(2):
            for project in model.projections[pol]:
                self.assertGreater(project.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.backbones[pol][0][0].weight.grad.abs().sum().item(), 0)
        for gate, readout in zip(model.gates, model.spatial_readouts):
            self.assertGreater(gate.local[-1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(readout.project.weight.grad.abs().sum().item(), 0)

    def test_hard_quota_ties_support_and_straight_through(self):
        gate = JointGate(Stage1Config(3, gate_rho_min=.5, gate_rho_max=.75))
        with torch.no_grad():
            for p in gate.parameters():
                p.zero_()
        x = torch.ones(2, 256, 2, 2, requires_grad=True)
        valid = torch.ones(2, 1, 2, 2)
        valid[0, :, 0, 0] = 0
        valid[1] = 0
        mask = gate(x, valid, hard=True)
        self.assertEqual(mask[0].sum().item(), 576)  # .75 * (3 cells * 256)
        self.assertEqual(mask[0, :, 0, 0].count_nonzero().item(), 0)
        self.assertEqual(mask[1].count_nonzero().item(), 0)
        torch.testing.assert_close(mask, gate(x, valid, hard=True), rtol=0, atol=0)
        mask.sum().backward()
        self.assertGreater(gate.local[-1].bias.grad.abs().sum().item(), 0)
        with torch.no_grad():
            gate.local[-1].bias.fill_(-30)
        low = gate(x, valid, hard=True)
        self.assertEqual(low[0].sum().item(), 384)
        self.assertTrue(((low == 0) | (low == 1)).all())

    def test_spatial_readout_preserves_information_lost_before_pool(self):
        readout = SpatialReadout()
        with torch.no_grad():
            for p in readout.parameters():
                p.zero_()
            readout.project.weight[0, 0, 0, 0] = 1
        x = torch.zeros(2, 256, 1, 2)
        x[0, 0, 0] = torch.tensor([-1., 1.])
        valid = torch.ones(2, 1, 1, 2)
        raw = weighted_pool(x, valid[:, 0])
        torch.testing.assert_close(raw[0], raw[1])
        encoded = weighted_pool(readout(x, valid), valid[:, 0])
        self.assertGreater((encoded[0] - encoded[1]).norm().item(), .3)

    def test_class_readout_axis_symmetry_and_gradient_path(self):
        self.model.train()
        x = torch.randn(1, 2, 1, 64, 64)
        out = self.model(x, torch.ones_like(x))
        geom = geometry(64)
        logits = self.model.class_logits(out, [geom])
        swapped = deepcopy(geom)
        for level in swapped.levels:
            level.slots = level.slots[:, [0, 2, 1]]
            level.slot_valid = level.slot_valid[:, [0, 2, 1]]
        torch.testing.assert_close(logits, self.model.class_logits(out, [swapped]))
        loss = classification_loss(logits, torch.tensor([1]), self.model.class_valid([geom]))
        loss.backward()
        for pol in range(2):
            self.assertGreater(self.model.backbones[pol][0][0].weight.grad.abs().sum().item(), 0)
        for g in self.model.spatial_readouts:
            self.assertGreater(g.project.weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.fusion_injection_logits.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in self.model.discriminator.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.scene_heads.parameters()))

    def test_class_loss_skips_invalid_scales_and_normalizes_views(self):
        logits = torch.tensor([[[2., 0., 0.], [0., 100., 0.], [0., 100., 0.]]])
        flags = torch.tensor([[True, False, False]])
        expected = F.cross_entropy(logits[:, 0], torch.tensor([0]))
        torch.testing.assert_close(classification_loss(logits, torch.tensor([0]), flags), expected)
        two = logits[:, None].repeat(1, 2, 1, 1)
        torch.testing.assert_close(classification_loss(two, torch.tensor([0]), flags[:, None].repeat(1, 2, 1)), expected)
        empty = logits.clone().requires_grad_()
        zero = classification_loss(empty, torch.tensor([0]), torch.zeros_like(flags))
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertEqual(empty.grad.count_nonzero().item(), 0)

    def test_scene_and_invariance_gradient_owners(self):
        output = synthetic_output()
        loss = scene_regression_loss(self.model, output, output["valid"],
                                     torch.randn(2, 2, 2), torch.ones(2, 2, dtype=torch.bool))
        loss.backward()
        self.assertTrue(all(x.grad is not None and x.grad.abs().sum() > 0 for x in output["residual"]))
        self.assertTrue(all(x.grad is None for x in output["content"]))
        self.assertTrue(all(p.grad is None for p in self.model.discriminator.parameters()))
        self.assertGreater(self.model.scene_heads[0][0].weight.grad.abs().sum().item(), 0)
        original, changed = synthetic_output(), synthetic_output()
        loss = content_invariance_loss(original, changed, original["valid"], torch.tensor([True, False]))
        loss.backward()
        for source in (original, changed):
            for x in source["content"]:
                self.assertGreater(x.grad[0].abs().sum().item(), 0)
                self.assertEqual(x.grad[1].abs().sum().item(), 0)
            self.assertTrue(all(x.grad is None for x in source["residual"]))
        zero = scene_regression_loss(self.model, synthetic_output(), original["valid"],
                                     torch.full((2, 2, 2), float("nan")), torch.zeros(2, 2, dtype=torch.bool))
        self.assertEqual(zero.item(), 0)

    def test_alternating_discriminator_has_exact_gradient_boundary(self):
        output = synthetic_output()
        labels = torch.tensor([0, 1])
        legal = torch.tensor([True, False])
        discriminator_loss(self.model, output, labels, legal).backward()
        self.assertTrue(all(x.grad is None for x in output["residual"]))
        self.assertGreater(self.model.discriminator.head[0].weight.grad.abs().sum().item(), 0)
        self.model.zero_grad(set_to_none=True)
        loss = leakage_loss(self.model, output, torch.ones(3) / 3, legal)
        loss.backward()
        self.assertTrue(all(p.grad is None for p in self.model.discriminator.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.discriminator.parameters()))
        for x in output["residual"]:
            self.assertGreater(x.grad[0].abs().sum().item(), 0)
            self.assertEqual(x.grad[1].abs().sum().item(), 0)

    def test_freeze_encode_and_save_reload_include_G(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "stage1.pt"
            torch.save(self.model.state_dict(), path)
            restored = Stage1(Stage1Config(3))
            restored.load_state_dict(torch.load(path, weights_only=True))
        x = torch.randn(1, 2, 1, 64, 64)
        geom = [geometry(64)]
        with self.assertRaises(RuntimeError):
            restored.encode(x, torch.ones_like(x), geom)
        restored.freeze_for_h().train()
        self.assertFalse(restored.training)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        encoded = restored.encode(x, torch.ones_like(x), geom)[0]
        self.model.eval()
        with torch.no_grad():
            expected = self.model(x, torch.ones_like(x), hard_gate=True)
        for i in range(3):
            torch.testing.assert_close(encoded.content[i], expected["content"][i][0])
            torch.testing.assert_close(encoded.spatial[i], expected["spatial"][i][0])
            self.assertFalse(encoded.spatial[i].requires_grad)
            self.assertFalse(encoded.content[i].requires_grad)


if __name__ == "__main__":
    unittest.main()
