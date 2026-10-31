"""Native-size CPU integration checks for layer2 pre-add spatial M2."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from sar_m123 import tokens
from sar_m123.bottleneck_evidence import RegionPoolWriteback
from sar_m123.current.data import FREQUENCY_KEYS
from sar_m123.current.model import (
    CLASSES, LIVE, forward_batch, load_model, reference_bank, save_model,
    training_policy,
)
from sar_m123.current.spatial_layer2 import SpatialLayer2M123ResNet50, from_spatial
from sar_m123.current.spatial_m12 import SpatialEvidencePool, SpatialM123ResNet50
from sar_m123.pool_replacement import grid_occupancy


class SpatialLayer2M123Tests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.query = self.sample(1)
        self.reference = self.sample(2)
        self.pairs = torch.tensor([[True, True]])

    @staticmethod
    def sample(batch):
        # The native tiling follows test_joint_readout; no dataset or weights.
        valid = torch.ones(batch, 2, 1, 160, 160)
        valid[..., 136:] = 0
        native = torch.zeros(batch, 2, 20, 160, 160)
        for index in range(20):
            row, col = divmod(index, 5)
            native[:, :, index, row*40:(row+1)*40, col*32:(col+1)*32] = 1
        native *= valid
        masks = torch.stack([native[:, :, group.tolist()].sum(2)
                             for group in (tokens.REGIONS[i] for i in (0, 2, 1))], 2)
        result = dict(images=torch.rand(batch, 2, 1, 160, 160), masks=masks,
                      valid=valid, available=torch.ones(batch, 2, dtype=torch.bool))
        for size, slots, energy, field in (
                (20, 'local_slot_masks', 'local_frequency', 'field2'),
                (10, 'slot_masks', 'frequency', 'field3')):
            result[slots] = grid_occupancy(native.flatten(0, 1), (size, size)).reshape(
                batch, 2, 20, size, size)
            result[field] = torch.rand(batch, 2, 3, size, size)
            result[energy] = result[slots][:, :, :, None] * result[field][:, :, None]
        return result

    def assert_exact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def assert_nonzero(self, value):
        self.assertIsNotNone(value)
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertGreater(value.abs().sum().item(), 0)

    @staticmethod
    def open_writers(writers):
        with torch.no_grad():
            for writer in writers:
                for transform in writer.transforms:
                    nn.init.normal_(transform[-1].weight, std=.03)

    @classmethod
    def open_existing(cls, model):
        cls.open_writers(model.internal_m2)
        with torch.no_grad():
            for block in model.internal_m1:
                nn.init.normal_(block.output.weight, std=.03)
            for output in model.deep_output:
                nn.init.normal_(output.weight, std=.01)
            for group in model.m3:
                for block in group:
                    nn.init.normal_(block.output.weight, std=.03)
                    nn.init.normal_(block.selector[-1].weight, std=.1)

    @staticmethod
    def deep_head_only(model):
        # Prevent the shallow regional readout from satisfying a deep-path test.
        with torch.no_grad():
            for pol in range(2):
                offset = pol * (2048 + 3*512)
                model.head.weight[:, offset+2048:offset+2048+3*512].zero_()

    def cached(self, model, query=None, reference=None):
        query = self.query if query is None else query
        reference = self.reference if reference is None else reference
        with torch.no_grad():
            prefix = model.prefix(query['images'], stop=model.cache_start)
            bank = {**reference, 'prefix': model.prefix(reference['images'], stop=model.cache_start)}
        return prefix, bank

    def call(self, model, prefix, bank, query=None, **kwargs):
        query = self.query if query is None else query
        options = dict(start=model.cache_start, reference=bank, pair_valid=self.pairs,
                       return_details=True)
        options.update(kwargs)
        return model.forward_from(prefix, **{k: v for k, v in query.items() if k != 'images'},
                                  **options)

    def call_with_paths(self, model, prefix, bank):
        captured = []
        original = model.stage3_paths

        def capture(*args, **kwargs):
            paths = original(*args, **kwargs)
            captured.append(paths)
            return paths

        with patch.object(model, 'stage3_paths', side_effect=capture):
            scores, details = self.call(model, prefix, bank)
        self.assertEqual(len(captured), 1)
        self.assertEqual(len(captured[0]), 7)
        return scores, details, captured[0]

    def native_hook_path(self, model, prefix, values):
        """Let real torchvision bottlenecks perform residual addition and ReLU."""
        count = len(prefix)
        effective_valid = values['valid'] * values['available'][..., None, None, None]

        def write_after_bn3(module, args, output):
            views = output.reshape(count, 2, 512, 20, 20)
            changed = []
            for pol in range(2):
                regions, bands = model.region_integrals(
                    values['local_slot_masks'][:, pol], values['local_frequency'][:, pol])
                changed.append(model.layer2_m2[pol](
                    views[:, pol], regions, effective_valid[:, pol], bands)[0])
            return torch.stack(changed, 1).flatten(0, 1)

        handle = model.backbone.model.layer2[2].bn3.register_forward_hook(write_after_bn3)
        try:
            x2 = model.backbone.model.layer2(prefix.flatten(0, 1))
        finally:
            handle.remove()
        x3 = model.backbone.model.layer3(x2)
        return x2.reshape(count, 2, 512, 20, 20), x3.reshape(count, 2, 1024, 10, 10)

    def test_nonzero_writeback_matches_bn3_hook_and_reaches_deep_scores(self):
        model = SpatialLayer2M123ResNet50().eval()
        self.open_writers(model.layer2_m2)
        self.deep_head_only(model)
        prefix, bank = self.cached(model)
        downstream = (model.backbone.model.layer2[3].conv1,
                      model.backbone.model.layer3[0].conv1,
                      model.backbone.model.layer3[4].conv2)

        def observed():
            inputs = [[] for _ in downstream]
            with ExitStack() as stack:
                for module, captured in zip(downstream, inputs):
                    handle = module.register_forward_pre_hook(
                        lambda mod, args, captured=captured: captured.append(args[0].detach().clone()))
                    stack.callback(handle.remove)
                result = self.call_with_paths(model, prefix, bank)
            return result, inputs

        with torch.no_grad():
            (score, details, paths), active_inputs = observed()
            expected2, expected3 = self.native_hook_path(model, prefix, self.query)
            _, expected_reference3 = self.native_hook_path(model, bank['prefix'], bank)
            for actual, expected in ((paths[0], expected2), (paths[1], expected3),
                                     (paths[2], expected_reference3)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
            for writer in model.layer2_m2:
                writer.enabled = False
            (baseline, _, plain), plain_inputs = observed()
        self.assert_nonzero(score - baseline)
        for active, unchanged in zip(active_inputs, plain_inputs):
            self.assertEqual(len(active), len(unchanged))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(active, unchanged)))
        for index in (0, 1, 2):
            self.assert_nonzero(paths[index] - plain[index])
        for item in details['operators']['layer2_block2']:
            self.assert_nonzero(item['update'])

    def test_final_ce_reaches_zero_initialized_m2_on_query_and_reference(self):
        model = SpatialLayer2M123ResNet50().eval()
        training_policy(model, 'fit')
        self.deep_head_only(model)
        with torch.no_grad():
            for group in model.m3:
                for operator in group:
                    nn.init.normal_(operator.output.weight, std=.03)
        prefix, bank = self.cached(model)
        captured = [[], []]

        def keep_gradient(pol):
            def capture(module, args, result):
                result[0].retain_grad()
                captured[pol].append(result[0])
            return capture

        with ExitStack() as stack:
            for pol, writer in enumerate(model.layer2_m2):
                for transform in writer.transforms:
                    self.assertEqual(transform[-1].weight.count_nonzero().item(), 0)
                handle = writer.register_forward_hook(keep_gradient(pol))
                stack.callback(handle.remove)
            score, _, paths = self.call_with_paths(model, prefix, bank)
            F.cross_entropy(score, torch.tensor([3])).backward()
        for pol, writer in enumerate(model.layer2_m2):
            self.assertEqual([len(x) for x in captured[pol]], [1, 2])
            for output in captured[pol]:
                self.assert_nonzero(output.grad)
            for transform in writer.transforms:
                self.assert_nonzero(transform[-1].weight.grad)
        for anchor in paths[3:]:
            self.assertFalse(anchor.requires_grad)
        self.assertTrue(all(p.grad is None for p in model.backbone.model.layer2.parameters()))
        self.assertTrue(all(p.grad is None for p in model.anchor_last.parameters()))

    def test_m3_uses_unchanged_raw2_raw3_after_m2_update(self):
        model = SpatialLayer2M123ResNet50().eval()
        self.open_existing(model)
        prefix, bank = self.cached(model)
        with torch.no_grad():
            _, before, old = self.call_with_paths(model, prefix, bank)
            self.open_writers(model.layer2_m2)
            with ExitStack() as stack:
                matches = [stack.enter_context(patch.object(op, 'forward', wraps=op.forward))
                           for op in model.m3[0]]
                _, after, new = self.call_with_paths(model, prefix, bank)
            for pol, matching in enumerate(matches):
                args = matching.call_args.args
                for actual, expected in ((args[1], new[3][:, pol]),
                                         (args[2], new[5][:, pol]),
                                         (args[8]['anchor'], new[4][:, pol]),
                                         (args[8]['local'], new[6][:, pol])):
                    self.assert_exact(actual, expected)
                    self.assertFalse(actual.requires_grad)
            for values, raw2, raw3 in ((self.query, new[5], new[3]),
                                      (self.reference, new[6], new[4])):
                native2 = model.backbone.model.layer2(
                    model.prefix(values['images'], stop=1).flatten(0, 1))
                self.assert_exact(raw2, native2.reshape_as(raw2))
                native3 = native2
                for block in list(model.backbone.model.layer3.children())[:-1]:
                    native3 = block(native3)
                self.assert_exact(raw3, model.anchor_last(native3).reshape_as(raw3))
        for first, second in zip(old[3:], new[3:]):
            self.assert_exact(first, second)
        for index in (0, 1, 2):
            self.assert_nonzero(old[index] - new[index])
        for first, second in zip(before['relations'], after['relations']):
            self.assert_exact(first[0], second[0])
            self.assert_exact(first[1], second[1])

    def test_zero_writeback_migration_retains_state_and_predictions(self):
        source = SpatialM123ResNet50().eval()
        self.open_existing(source)
        converted = from_spatial(source).eval()
        self.assertIsInstance(converted, SpatialLayer2M123ResNet50)
        self.assertEqual((source.cache_start, converted.cache_start), (2, 1))
        old_state, new_state = source.state_dict(), converted.state_dict()
        self.assertEqual(set(new_state) - set(old_state),
                         {key for key in new_state if key.startswith('layer2_m2.')})
        for name, value in old_state.items():
            self.assert_exact(new_state[name], value)
        self.assertIsInstance(converted.layer2_m2, nn.ModuleList)
        self.assertEqual(len(converted.layer2_m2), 2)
        for writer in converted.layer2_m2:
            self.assertIsInstance(writer, RegionPoolWriteback)
            self.assertIsInstance(writer.pool, SpatialEvidencePool)
            self.assertEqual(writer.pool.channels, 512)
            self.assertEqual(list(writer.pool.parameters()), [])
            for transform in writer.transforms:
                self.assertEqual(transform[-1].weight.count_nonzero().item(), 0)
        for missing_vh in (False, True):
            with self.subTest(missing_vh=missing_vh), torch.no_grad():
                if missing_vh:
                    for values in (self.query, self.reference):
                        for value in values.values():
                            value[0, 1] = 0
                old_prefix, old_bank = self.cached(source)
                new_prefix, new_bank = self.cached(converted)
                before, _ = self.call(source, old_prefix, old_bank)
                after, _ = self.call(converted, new_prefix, new_bank)
                torch.testing.assert_close(after, before, rtol=0, atol=1e-5)

    def test_cache_contract_full_forward_and_current_wrappers(self):
        for model_type, start, shape in ((SpatialM123ResNet50, 2, (1, 2, 512, 20, 20)),
                                         (SpatialLayer2M123ResNet50, 1, (1, 2, 256, 40, 40))):
            with self.subTest(model=model_type.__name__), torch.no_grad():
                model = model_type().eval()
                if start == 1:
                    self.open_writers(model.layer2_m2)
                items = [{k: v[i].numpy() for k, v in self.reference.items()} for i in range(2)]
                with patch('sar_m123.current.model.prepare_record', side_effect=items), \
                        patch.object(model, 'prefix', wraps=model.prefix) as prefix_call:
                    bank = reference_bank(model, [{'id': 'r0'}, {'id': 'r1'}], 'cpu')
                    self.assertEqual(prefix_call.call_args.kwargs['stop'], start)
                self.assertFalse(bank['prefix'].requires_grad)
                with patch.object(model, 'prefix', wraps=model.prefix) as prefix_call:
                    wrapped = forward_batch(model, self.query, bank, self.pairs)
                    self.assertEqual(prefix_call.call_args.kwargs['stop'], start)
                with patch.object(model, 'prefix', wraps=model.prefix) as prefix_call:
                    direct = model(**self.query, reference=self.reference, pair_valid=self.pairs)
                    self.assertEqual([call.kwargs['stop'] for call in prefix_call.call_args_list],
                                     [start, start])
                prefix, bank = self.cached(model)
                self.assertEqual(tuple(prefix.shape), shape)
                cached, _ = self.call(model, prefix, bank)
                self.assert_exact(direct, cached)
                self.assert_exact(wrapped, cached)
                with self.assertRaises(ValueError):
                    self.call(model, prefix, bank, start=3-start)
                if start == 1:
                    with self.assertRaises(ValueError):
                        model.prefix(self.query['images'], stop=2)
                    old_cache = model.backbone.model.layer2(prefix.flatten(0, 1)).reshape(1, 2, 512, 20, 20)
                    with self.assertRaises(ValueError):
                        self.call(model, old_cache, bank, start=1)

    def test_nonzero_model_ignores_explicit_query_and_reference_frequency(self):
        model = SpatialLayer2M123ResNet50().eval()
        self.open_existing(model)
        self.open_writers(model.layer2_m2)
        prefix, bank = self.cached(model)
        changed_query, changed_bank = dict(self.query), dict(bank)
        for key in FREQUENCY_KEYS:
            changed_query[key] = self.query[key].flip(-3) * 23 + 7
            changed_bank[key] = bank[key].flip(-3) * .01 + 19
        with torch.no_grad():
            before, details = self.call(model, prefix, bank)
            after, other = self.call(model, prefix, changed_bank, query=changed_query)
        self.assert_exact(before, after)
        for key in ('layer2_block2', 'reference_layer2_block2', 'm3_0'):
            for original, changed in zip(details['operators'][key], other['operators'][key]):
                self.assert_nonzero(original['update'])
                self.assert_exact(original['update'], changed['update'])
        for first, second in zip(details['relations'], other['relations']):
            self.assert_exact(first[0], second[0])
        self.assertTrue(all(not op.frequency for group in model.m3 for op in group))

    def test_missing_vh_and_empty_regions_receive_no_writeback(self):
        model = SpatialLayer2M123ResNet50().eval()
        self.open_existing(model)
        self.open_writers(model.layer2_m2)
        for values in (self.query, self.reference):
            for value in values.values():
                value[0, 1] = 0
            # Empty the end-region slots while retaining body and side evidence.
            for key in ('slot_masks', 'local_slot_masks', 'frequency', 'local_frequency'):
                values[key][:, :, tokens.REGIONS[2].tolist()] = 0
            values['masks'][:, :, 1] = 0
        prefix, bank = self.cached(model)
        with torch.no_grad():
            score, details = self.call(model, prefix, bank)
        self.assertTrue(torch.isfinite(score).all().item())
        for key, values in (('layer2_block2', self.query),
                            ('reference_layer2_block2', self.reference)):
            for pol, detail in enumerate(details['operators'][key]):
                occupancy = grid_occupancy(values['valid'][:, pol], (20, 20))
                invalid = (occupancy == 0).expand_as(detail['update'])
                self.assertEqual(detail['update'][invalid].count_nonzero().item(), 0)
                self.assertEqual(detail['mass'][:, 1].count_nonzero().item(), 0)
                self.assertEqual(detail['write_weights'][:, 1].count_nonzero().item(), 0)
            self.assert_nonzero(details['operators'][key][0]['update'])
            self.assertEqual(details['operators'][key][1]['update'][0].count_nonzero().item(), 0)
        self.assertEqual(details['operators']['m3_0'][1]['update'].count_nonzero().item(), 0)
        altered_query = {**self.query, 'images': self.query['images'].clone()}
        altered_reference = {**self.reference, 'images': self.reference['images'].clone()}
        altered_query['images'][0, 1].fill_(9)
        altered_reference['images'][0, 1].fill_(13)
        prefix, bank = self.cached(model, altered_query, altered_reference)
        with torch.no_grad():
            altered, _ = self.call(model, prefix, bank, query=altered_query)
        self.assert_exact(altered, score)

    def test_save_load_new_architecture_preserves_nonzero_predictions(self):
        model = SpatialLayer2M123ResNet50().eval()
        self.open_existing(model)
        self.open_writers(model.layer2_m2)
        # load_model freezes parameters; compare the same inference policy.
        model.requires_grad_(False)
        prefix, bank = self.cached(model)
        with torch.no_grad():
            expected, _ = self.call(model, prefix, bank)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spatial_layer2.pt'
            save_model(model, path, purpose='layer2_cpu_integration')
            blob = torch.load(path, map_location='cpu', weights_only=True)
            self.assertEqual(blob['architecture'], 'native160_layer2block2_preadd_spatial_m123')
            self.assertEqual(blob['classes'], CLASSES)
            self.assertEqual(blob['metadata'], {'purpose': 'layer2_cpu_integration'})
            restored = load_model(path, device='cpu')
        self.assertIsInstance(restored, SpatialLayer2M123ResNet50)
        self.assertEqual(restored.cache_start, 1)
        self.assertFalse(restored.training)
        self.assertTrue(all(not p.requires_grad and p.device.type == 'cpu' for p in restored.parameters()))
        original, loaded = model.state_dict(), restored.state_dict()
        self.assertEqual(set(original), set(loaded))
        for name in original:
            self.assert_exact(loaded[name], original[name])
        prefix, bank = self.cached(restored)
        with torch.no_grad():
            actual, _ = self.call(restored, prefix, bank)
        self.assert_exact(actual, expected)

    def test_fit_opens_new_m2_but_adapt_only_opens_last_three_convolutions(self):
        model = SpatialLayer2M123ResNet50()
        training_policy(model, 'fit')
        self.assertTrue(all(p.requires_grad for p in model.layer2_m2.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.model.layer2.parameters()))
        self.assertFalse(model.training)
        training_policy(model, 'adapt')
        self.assertEqual({name for name, p in model.named_parameters() if p.requires_grad}, set(LIVE))
        self.assertTrue(all(not p.requires_grad for p in model.layer2_m2.parameters()))
        self.assertTrue(all(not module.training for module in model.modules()
                            if isinstance(module, nn.BatchNorm2d)))


if __name__ == '__main__':
    unittest.main()
