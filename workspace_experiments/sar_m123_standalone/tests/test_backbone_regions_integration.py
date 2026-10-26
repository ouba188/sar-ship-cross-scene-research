"""CPU integration checks with the local pretrained R50 and native-size inputs."""
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn

from sar_m123.backbone_regions import RegionResNet50
from sar_m123.regions import soft_regions


WEIGHTS = Path(__file__).resolve().parents[1] / 'assets/backbone/resnet50.pth'


class BackboneRegionsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        if not WEIGHTS.is_file():
            raise unittest.SkipTest(f'Local pretrained R50 asset required: {WEIGHTS}')

    def setUp(self):
        self.rng = torch.random.fork_rng(devices=[])
        self.rng.__enter__()
        self.addCleanup(self.rng.__exit__, None, None, None)
        torch.manual_seed(17)
        valid = torch.zeros(2, 2, 1, 64, 64)
        valid[..., 8:56, 4:60] = 1
        images = torch.randint(0, 256, valid.shape).float() / 255 * valid
        masks = torch.from_numpy(np.stack([
            soft_regions(np.array([32., 32., 24., 10., angle, .9]), shape=(64, 64))
            for angle in (.3, .8)
        ])).float().unsqueeze(0).repeat(2, 1, 1, 1, 1) * valid
        available = torch.ones(2, 2, dtype=torch.bool)
        available[1, 1] = False
        valid[1, 1] = 0
        images[1, 1] = 0
        masks[1, 1] = 0
        self.inputs = images, masks, valid, available

    def model(self, stages=(2,), mode='regional'):
        return RegionResNet50(WEIGHTS, classes=8, stages=stages, mode=mode, hidden=4)

    def test_zero_init_full_forward_equals_no_adapter_with_same_head(self):
        adapted = self.model(stages=(1, 3)).eval()
        baseline = self.model(mode='none').eval()
        baseline.head.load_state_dict(adapted.head.state_dict())
        for block in adapted.blocks.values():
            self.assertEqual(torch.count_nonzero(block.output.weight).item(), 0)
        with torch.no_grad():
            actual = adapted(*self.inputs)
            expected = baseline(*self.inputs)
        self.assertEqual(actual.shape, (2, 8))
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_stage2_full_forward_equals_cached_prefix_forward(self):
        model = self.model().eval()
        for block in model.blocks.values():
            nn.init.normal_(block.output.weight, std=.02)
        images, masks, valid, available = self.inputs
        with torch.no_grad():
            expected = model(*self.inputs)
            prefix = model.prefix(images, stop=2)
        self.assertFalse(prefix.requires_grad)
        self.assertEqual(prefix.shape, (2, 2, 512, 8, 8))
        actual = model.forward_from(prefix, masks, valid, available, start=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        for block in model.blocks.values():
            self.assertGreater(block.output.weight.grad.abs().sum().item(), 0)

    def test_stage1_nonzero_feedback_changes_logits_and_matches_cached_forward(self):
        images, masks, valid, available = self.inputs
        for gate_mode in ('dot', 'fixed', 'cosine'):
            with self.subTest(gate_mode=gate_mode):
                model = RegionResNet50(
                    WEIGHTS, classes=8, stages=(1,), hidden=4,
                    gate_mode=gate_mode).eval()
                with torch.no_grad():
                    baseline = model(*self.inputs)
                    for block in model.blocks.values():
                        nn.init.normal_(block.output.weight, std=.02)
                    expected = model(*self.inputs)
                    prefix = model.prefix(images, stop=1)
                self.assertFalse(prefix.requires_grad)
                self.assertEqual(prefix.shape, (2, 2, 256, 16, 16))
                self.assertGreater((expected - baseline).abs().max().item(), 1e-6)
                actual = model.forward_from(prefix, masks, valid, available, start=1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().mean().backward()
                for block in model.blocks.values():
                    self.assertIsNotNone(block.output.weight.grad)
                    self.assertGreater(block.output.weight.grad.abs().sum().item(), 0)

    def test_stage2_cached_entry_rejects_skipping_stage1_adapter(self):
        images, masks, valid, available = self.inputs
        baseline = self.model(mode='none').eval()
        with torch.no_grad():
            prefix = baseline.prefix(images, stop=2)
        model = self.model(stages=(1,)).eval()
        with self.assertRaisesRegex(ValueError, 'skip an earlier trainable adapter'):
            model.forward_from(prefix, masks, valid, available, start=2)

    def test_repeated_stages_receive_gradients_and_backbone_stays_frozen(self):
        model = self.model(stages=(1, 3))
        self.assertEqual(set(model.blocks), {'0_1', '1_1', '0_3', '1_3'})
        before = {name: value.clone() for name, value in model.backbone.named_buffers()}
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=.01)
        for step in range(2):
            model.train()
            self.assertTrue(model.head.training)
            self.assertTrue(all(block.training for block in model.blocks.values()))
            self.assertTrue(all(not bn.training for bn in model.backbone.modules()
                                if isinstance(bn, nn.BatchNorm2d)))
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(*self.inputs), torch.tensor([0, 5]))
            loss.backward()
            for block_name, block in model.blocks.items():
                # Zero output initialization initially blocks gradients to earlier projections.
                parameters = block.named_parameters() if step else [('output.weight', block.output.weight)]
                for name, parameter in parameters:
                    with self.subTest(step=step, parameter=f'{block_name}.{name}'):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())
                        self.assertGreater(parameter.grad.abs().sum().item(), 0)
            self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(not p.requires_grad and p.grad is None
                                for p in model.backbone.parameters()))
            optimizer.step()
            for name, value in model.backbone.named_buffers():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_merged_and_regional_have_equal_parameter_counts(self):
        regional = self.model(stages=(1, 3), mode='regional')
        merged = self.model(stages=(1, 3), mode='merged')
        for trainable_only in (False, True):
            with self.subTest(trainable_only=trainable_only):
                counts = [sum(p.numel() for p in model.parameters()
                              if not trainable_only or p.requires_grad)
                          for model in (regional, merged)]
                self.assertGreater(counts[0], 0)
                self.assertEqual(*counts)
        self.assertEqual(
            {name: p.shape for name, p in regional.blocks.named_parameters()},
            {name: p.shape for name, p in merged.blocks.named_parameters()})


if __name__ == '__main__':
    unittest.main()
