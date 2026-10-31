"""Focused CPU checks for spatial M1/M2 and unchanged M3 checkpoint state."""
from pathlib import Path
import tempfile
import unittest

import torch

from sar_m123.backbone_attention import ClassResponseFeedback
from sar_m123.bottleneck_evidence import BottleneckEvidenceResNet50
from sar_m123.current.model import CLASSES, load_model, save_model
from sar_m123.current.spatial_m12 import (
    SpatialClassResponseFeedback, SpatialEvidencePool, SpatialM12ResNet50,
    SpatialM123ResNet50, from_legacy, without_m3_frequency,
)
from sar_m123.pool_replacement import grid_occupancy, regional_mean
from sar_m123.selective_readout import EvidencePool


class SpatialM12Tests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nonzero_m1_matches_class_response_and_ignores_frequency(self):
        baseline = ClassResponseFeedback(8, classes=3, hidden=5)
        with torch.no_grad():
            baseline.output.weight.normal_()
            baseline.gamma.fill_(.7)
        spatial = SpatialClassResponseFeedback(8, classes=3, hidden=5)
        spatial.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(set(dict(spatial.named_parameters())),
                         set(dict(baseline.named_parameters())))
        self.assertFalse(hasattr(spatial, 'channel_selectors'))
        x = torch.randn(3, 8, 3, 4)
        valid = torch.ones(3, 1, 3, 4)
        valid[0, :, :, 0] = .25
        valid[0, :, :, -1] = 0
        valid[1] = 0
        expected, old = baseline(x, valid, return_details=True)
        self.assertGreater(old['update'].abs().sum().item(), 0)
        self.assert_exact(expected[1], x[1])
        field = torch.rand(3, 3, 3, 4)
        for energy in (None, field, field[:, [2, 0, 1]] * 100,
                       torch.full_like(field, float('nan'))):
            with self.subTest(energy='none' if energy is None else 'provided'):
                actual, details = spatial(x, valid, energy, True)
                self.assert_exact(actual, expected)
                for key, value in old.items():
                    self.assert_exact(details[key], value)
                self.assert_exact(spatial(x, valid, energy_field=energy), expected)

    def test_m2_regional_mean_partial_valid_missing_rows_and_gradient(self):
        x = torch.randn(3, 4, 2, 2, requires_grad=True)
        valid = torch.ones(3, 1, 4, 4)
        valid[0] = 0
        valid[0, :, :2, :2] = 1
        valid[1] = 0
        masks = torch.rand(3, 3, 2, 2)
        masks[:, 0] = 1
        masks[2, 2] = 0
        pool = SpatialEvidencePool(4)
        self.assertEqual(list(pool.named_parameters()), [])
        reference_x = x.detach().clone().requires_grad_()
        expected, mass = regional_mean(reference_x, masks, valid)
        occupancy = grid_occupancy(valid, x.shape[-2:]).clamp(0, 1)
        self.assertTrue(((occupancy > 0) & (occupancy < 1)).any())
        energy = torch.rand(3, 3, 3, 2, 2, requires_grad=True)
        result = pool(x, masks, valid, energy, True)
        self.assert_exact(result['pooled'], expected)
        self.assert_exact(result['mean'], expected)
        self.assert_exact(result['mass'], mass)
        self.assert_exact(result['valid'], occupancy)
        self.assert_exact(result['pooled'][1], torch.zeros_like(expected[1]))
        self.assert_exact(result['pooled'][2, 2], torch.zeros_like(expected[2, 2]))
        for changed in (None, energy.detach() * 100,
                        energy.detach()[:, :, [2, 0, 1]],
                        torch.full_like(energy, float('nan'))):
            for detailed in (False, True):
                with self.subTest(details=detailed, absent=changed is None):
                    again = pool(x, masks, valid, changed, detailed)
                    for key in ('pooled', 'mean', 'mass'):
                        self.assert_exact(again[key], result[key])
                    if detailed:
                        self.assert_exact(again['valid'], occupancy)
        target = torch.randn_like(expected)
        (result['pooled'] * target).sum().backward()
        (expected * target).sum().backward()
        self.assert_exact(x.grad, reference_x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assert_exact(x.grad[1], torch.zeros_like(x.grad[1]))
        self.assertIsNone(energy.grad)

    def assert_spatial_modules(self, model):
        for block in (*model.internal_m1, *model.m1.values()):
            self.assertIsInstance(block, SpatialClassResponseFeedback)
            self.assertFalse(hasattr(block, 'channel_selectors'))
        pools = [writer.pool for writer in model.internal_m2]
        pools.extend(model.m2)
        pools.extend(model.deep_pool)
        for pool in pools:
            self.assertIsInstance(pool, SpatialEvidencePool)
            self.assertEqual(list(pool.named_parameters()), [])
        for operators in model.m3:
            for operator in operators:
                self.assertIsInstance(operator.reader2, EvidencePool)
                self.assertIsInstance(operator.reader3, EvidencePool)
                self.assertTrue(operator.frequency)

    def test_from_legacy_retains_all_remaining_state_and_m3(self):
        for position, policy in (('pre3x3', 'local'), ('post3x3', 'frozen'),
                                 ('stage', 'local')):
            with self.subTest(position=position, policy=policy):
                legacy = BottleneckEvidenceResNet50(
                    None, position=position, policy=policy, classes=3)
                # Open zero projections and distinguish copied weights from defaults.
                with torch.no_grad():
                    for i, parameter in enumerate(legacy.parameters()):
                        parameter.add_(.01 * (1 + i % 11))
                    legacy.backbone.model.bn1.running_mean.fill_(.25)
                old = legacy.state_dict()
                removed = set()
                for name, module in legacy.named_modules():
                    if isinstance(module, EvidencePool) and not name.startswith('m3.'):
                        removed.update(f'{name}.{key}' for key in module.state_dict())
                removed.update(key for key in old if '.channel_selectors.' in key)
                self.assertTrue(removed)
                spatial = from_legacy(legacy)
                self.assertIsInstance(spatial, SpatialM12ResNet50)
                self.assertEqual((spatial.position, spatial.policy), (position, policy))
                self.assert_spatial_modules(spatial)
                new = spatial.state_dict()
                self.assertEqual(set(new), set(old) - removed)
                for name, value in new.items():
                    with self.subTest(state=name):
                        self.assert_exact(value, old[name])
                old_parameters = dict(legacy.named_parameters())
                new_parameters = dict(spatial.named_parameters())
                self.assertEqual(set(new_parameters), set(old_parameters) - removed)
                old_m3 = legacy.m3.state_dict()
                new_m3 = spatial.m3.state_dict()
                self.assertTrue(old_m3)
                self.assertEqual(set(new_m3), set(old_m3))
                for name, value in old_m3.items():
                    with self.subTest(m3=name):
                        self.assert_exact(new_m3[name], value)
                del legacy, spatial, old, new, old_parameters, new_parameters

    def test_new_architecture_cpu_save_load(self):
        model = SpatialM12ResNet50()
        self.assertEqual((model.position, model.policy), ('pre3x3', 'local'))
        self.assert_spatial_modules(model)
        with torch.no_grad():
            for i, parameter in enumerate(model.parameters()):
                parameter.add_(.01 * (1 + i % 7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spatial.pt'
            save_model(model, path, purpose='spatial_m12_test')
            blob = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(blob['architecture'], 'native160_pre3x3_spatial_m12_m3')
            self.assertEqual(blob['classes'], CLASSES)
            self.assertEqual(blob['metadata'], {'purpose': 'spatial_m12_test'})
            self.assertTrue(all(value.device.type == 'cpu'
                                for value in blob['state'].values()))
            del blob
            restored = load_model(path, device='cpu')
        self.assertIsInstance(restored, SpatialM12ResNet50)
        self.assert_spatial_modules(restored)
        self.assertFalse(restored.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in restored.parameters()))
        original, loaded = model.state_dict(), restored.state_dict()
        self.assertEqual(set(loaded), set(original))
        for name, value in original.items():
            with self.subTest(state=name):
                self.assertEqual(loaded[name].device.type, 'cpu')
                self.assert_exact(loaded[name], value)

    def test_spatial_m3_transfer_has_distinct_checkpoint_and_keeps_weights(self):
        source = SpatialM12ResNet50()
        changed = without_m3_frequency(source)
        self.assertIsInstance(changed, SpatialM123ResNet50)
        self.assertTrue(all(not operator.frequency for group in changed.m3 for operator in group))
        self.assertTrue(all(operator.frequency for group in source.m3 for operator in group))
        for name, value in source.state_dict().items():
            self.assert_exact(changed.state_dict()[name], value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spatial_m3.pt'
            save_model(changed, path)
            restored = load_model(path, device='cpu')
        self.assertIsInstance(restored, SpatialM123ResNet50)
        self.assertTrue(all(not operator.frequency for group in restored.m3 for operator in group))
        for name, value in changed.state_dict().items():
            self.assert_exact(restored.state_dict()[name], value)


if __name__ == '__main__':
    unittest.main()
