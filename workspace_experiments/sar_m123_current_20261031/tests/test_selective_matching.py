"""Small-grid behavioral checks for frozen anchors and selective transport.

No backbone weights or dataset are needed. The relation supervision interface
is details['reference_logits']; invalid pairs must be excluded from its loss.
"""
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from sar_m123.pool_replacement import grid_occupancy
from sar_m123.selective_matching import SelectiveReferenceMatch


class SelectiveReferenceMatchTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(17)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.query = self.sample(2)
        self.reference = self.sample(3)

    @staticmethod
    def sample(batch):
        slots = torch.eye(20).reshape(1, 20, 4, 5).repeat(batch, 1, 1, 1)
        local_slots = slots.repeat_interleave(2, -2).repeat_interleave(2, -1)
        return dict(features=torch.randn(batch, 8, 4, 5),
                    anchor=torch.randn(batch, 1024, 4, 5),
                    local=torch.randn(batch, 512, 8, 10),
                    slots=slots, local_slots=local_slots,
                    valid=torch.ones(batch, 1, 8, 10),
                    energy=slots[:, :, None] * torch.rand(batch, 20, 3, 4, 5),
                    local_energy=local_slots[:, :, None] * torch.rand(batch, 20, 3, 8, 10))

    def call(self, block, query=None, reference=None, pairs=None, details=True):
        query = self.query if query is None else query
        reference = self.reference if reference is None else reference
        if pairs is None:
            pairs = torch.ones(len(query['features']), len(reference['features']), dtype=torch.bool)
        return block(query['features'], query['anchor'], query['local'], query['slots'],
                     query['local_slots'], query['valid'], query['energy'],
                     query['local_energy'], reference, pairs, return_details=details)

    @staticmethod
    def open_output(block):
        nn.init.normal_(block.output.weight, std=.05)

    def assert_zero(self, value):
        self.assertTrue(torch.isfinite(value).all().item())
        self.assertEqual(value.count_nonzero().item(), 0)

    def assert_gradient(self, value):
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(value.grad).all().item())
        self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_initial_identity_shapes_and_first_output_gradient(self):
        block = SelectiveReferenceMatch(channels=8)
        out, d = self.call(block)
        torch.testing.assert_close(out, self.query['features'], rtol=0, atol=0)
        self.assert_zero(d['update'])
        self.assertEqual(d['reference_logits'].shape, (2, 3))
        self.assertEqual(d['reference_valid'].shape, (2, 3))
        self.assertEqual(d['reference_weights'].shape, (2, 3, 20))
        for index, count in enumerate((8, 8, 4)):
            self.assertEqual(d['transport'][index].shape, (2, 3, count, count))
            self.assertEqual(d['branch_similarity'][index].shape, (2, 3, count, count, 5))
            self.assertEqual(d['branch_weights'][index].shape, (2, 3, count, count, 5))
        for value in (d['reference_logits'], *d['transport'], *d['similarity']):
            self.assertTrue(torch.isfinite(value).all().item())
        (out * torch.randn_like(out)).sum().backward()
        self.assert_gradient(block.output.weight)
        self.assert_zero(block.project.weight.grad)
        plain, plain_details = self.call(block, details=False)
        torch.testing.assert_close(plain, out, rtol=0, atol=0)
        torch.testing.assert_close(plain_details['reference_logits'], d['reference_logits'])

    def controlled_bands(self):
        query, reference = self.sample(1), self.sample(2)
        for sample in (query, reference):
            sample['anchor'].zero_()
            sample['anchor'][:, 0] = 1
            sample['local'].zero_()
            sample['local'][:, 0, :, ::2] = 1
            sample['local'][:, 1, :, 1::2] = 1
            field = torch.zeros(len(sample['features']), 3, 8, 10)
            field[:, 0, :, ::2] = 1
            field[:, 1, :, 1::2] = 1
            field[:, 2] = 1
            sample['local_energy'] = sample['local_slots'][:, :, None] * field[:, None]
        reference['local_energy'][1] = reference['local_energy'][1][:, [1, 0, 2]]
        return query, reference

    def test_energy_position_distinguishes_equal_means_and_equal_band_totals(self):
        query, reference = self.controlled_bands()
        torch.testing.assert_close(reference['local'][0], reference['local'][1], rtol=0, atol=0)
        torch.testing.assert_close(reference['local_energy'][0].sum((-2, -1)),
                                   reference['local_energy'][1].sum((-2, -1)), rtol=0, atol=0)
        torch.testing.assert_close(reference['local_energy'][0].sum(1),
                                   reference['local_energy'][1].sum(1), rtol=0, atol=0)
        block = SelectiveReferenceMatch(channels=8, dynamic=False)
        _, d = self.call(block, query, reference)
        # Both means match. Orthogonal channels at swapped band positions have
        # cosine zero, while matching positions have cosine one.
        for evidence in d['branch_similarity']:
            torch.testing.assert_close(evidence[..., :2], torch.ones_like(evidence[..., :2]), atol=1e-6, rtol=0)
            torch.testing.assert_close(evidence[:, 0, ..., 2:], torch.ones_like(evidence[:, 0, ..., 2:]), atol=1e-6, rtol=0)
            self.assert_zero(evidence[:, 1, ..., 2:4])
        torch.testing.assert_close(d['reference_logits'], torch.tensor([[10., 25/3]]), atol=1e-5, rtol=0)
        swapped = {**query, 'local_energy': query['local_energy'][:, :, [1, 0, 2]]}
        _, reverse = self.call(block, swapped, reference)
        torch.testing.assert_close(reverse['reference_logits'], d['reference_logits'].flip(1), atol=1e-5, rtol=0)
        scaled = {**reference, 'local_energy': reference['local_energy'] * 100}
        _, rescaled = self.call(block, query, scaled)
        torch.testing.assert_close(rescaled['reference_logits'], d['reference_logits'], atol=1e-5, rtol=0)

    def test_same_mmsi_pair_mask_excludes_outputs_and_reference_gradients(self):
        block = SelectiveReferenceMatch(channels=8)
        self.open_output(block)
        query_mmsi = torch.tensor([111, 222])
        reference_mmsi = torch.tensor([111, 333, 222])
        pairs = query_mmsi[:, None] != reference_mmsi[None]
        pairs[:, 2] = False
        self.reference['features'].requires_grad_()
        out, d = self.call(block, pairs=pairs)
        self.assertGreater(d['update'].abs().sum().item(), 0)
        self.assert_zero(d['reference_weights'][~pairs])
        self.assertFalse(d['reference_valid'][~pairs].any().item())
        for transport in d['transport']:
            self.assert_zero(transport[~pairs])
        (d['update'][0] * torch.randn_like(d['update'][0])).sum().backward()
        self.assert_zero(self.reference['features'].grad[[0, 2]])
        self.assert_gradient(self.reference['features'])
        changed = {key: value.detach().clone() for key, value in self.reference.items()}
        for key in ('features', 'anchor', 'local', 'energy', 'local_energy'):
            changed[key][[0, 2]] = torch.randn_like(changed[key][[0, 2]]) * 20
        altered, after = self.call(block, reference=changed, pairs=pairs)
        torch.testing.assert_close(altered[0], out[0], rtol=0, atol=0)
        torch.testing.assert_close(after['reference_logits'][0], d['reference_logits'][0], rtol=0, atol=0)

    def test_missing_reference_polarization_and_all_invalid_zero_update(self):
        block = SelectiveReferenceMatch(channels=8)
        self.open_output(block)
        self.reference['valid'][1].zero_()
        _, partial = self.call(block)
        self.assert_zero(partial['reference_weights'][:, 1])
        self.assertFalse(partial['reference_valid'][:, 1].any().item())
        for missing in ('pairs', 'query_valid', 'reference_valid', 'query_slots', 'reference_slots'):
            with self.subTest(missing=missing):
                query = {key: value.clone() for key, value in self.query.items()}
                reference = {key: value.clone() for key, value in self.reference.items()}
                pairs = torch.ones(2, 3, dtype=torch.bool)
                if missing == 'pairs':
                    pairs.zero_()
                elif missing == 'query_valid':
                    query['valid'].zero_()
                elif missing == 'reference_valid':
                    reference['valid'].zero_()
                elif missing == 'query_slots':
                    query['slots'].zero_()
                    query['local_slots'].zero_()
                else:
                    reference['slots'].zero_()
                    reference['local_slots'].zero_()
                out, d = self.call(block, query, reference, pairs)
                torch.testing.assert_close(out, query['features'], rtol=0, atol=0)
                self.assertFalse(d['reference_valid'].any().item())
                for value in (d['update'], d['reference_weights'], *d['transport']):
                    self.assert_zero(value)
                self.assertTrue(torch.isfinite(d['reference_logits']).all().item())
                block.zero_grad(set_to_none=True)
                (out.square().mean() + d['reference_logits'].sum()).backward()
                for parameter in block.parameters():
                    if parameter.requires_grad:
                        self.assert_zero(parameter.grad)

    def test_dynamic_selector_changes_with_evidence_static_weights_do_not(self):
        dynamic = SelectiveReferenceMatch(channels=8, dynamic=True)
        static = SelectiveReferenceMatch(channels=8, dynamic=False)
        static.load_state_dict(dynamic.state_dict())
        _, initial = self.call(dynamic)
        _, matched = self.call(static)
        torch.testing.assert_close(initial['reference_logits'], matched['reference_logits'], rtol=0, atol=0)
        nn.init.normal_(dynamic.selector[-1].weight, std=.3)
        static.load_state_dict(dynamic.state_dict())
        changed = {**self.reference, 'anchor': -self.reference['anchor'], 'local': -self.reference['local']}
        for block in (dynamic, static):
            _, before = self.call(block)
            _, after = self.call(block, reference=changed)
            difference = sum((a - b).abs().sum().item()
                             for a, b in zip(before['branch_weights'], after['branch_weights']))
            if block.dynamic:
                self.assertGreater(difference, 1e-4)
            else:
                self.assertEqual(difference, 0)
                self.assertTrue(all(not p.requires_grad for p in block.selector.parameters()))

    def test_relation_loss_trains_selector_without_anchor_or_value_gradient(self):
        block = SelectiveReferenceMatch(channels=8)
        leaves = [sample[key].requires_grad_() for sample in (self.query, self.reference)
                  for key in ('features', 'anchor', 'local', 'energy', 'local_energy')]
        optimizer = torch.optim.SGD(block.parameters(), lr=.2)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            _, d = self.call(block)
            F.cross_entropy(d['reference_logits'], torch.tensor([0, 2])).backward()
            self.assert_gradient(block.selector[-1].weight)
            self.assert_gradient(block.selector[-1].bias)
            self.assert_gradient(block.branch_bias)
            if step == 0:
                self.assert_zero(block.selector[0].weight.grad)
            else:
                for parameter in block.selector.parameters():
                    self.assert_gradient(parameter)
            for leaf in leaves:
                self.assertIsNone(leaf.grad)
            self.assertIsNone(block.project.weight.grad)
            self.assertIsNone(block.output.weight.grad)
            self.assertTrue(all(p.grad is None for p in block.reader2.parameters()))
            self.assertTrue(all(p.grad is None for p in block.reader3.parameters()))
            optimizer.step()

    def test_after_one_output_step_reference_values_receive_gradient(self):
        block = SelectiveReferenceMatch(channels=8)
        optimizer = torch.optim.SGD(block.parameters(), lr=.2)
        self.reference['features'].requires_grad_()
        for sample in (self.query, self.reference):
            sample['anchor'].requires_grad_()
            sample['local'].requires_grad_()
        probe = torch.randn_like(self.query['features'])
        _, initial = self.call(block)
        (initial['update'] * probe).sum().backward()
        self.assert_gradient(block.output.weight)
        self.assert_zero(self.reference['features'].grad)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.reference['features'].grad = None
        _, opened = self.call(block)
        (opened['update'] * probe).sum().backward()
        for parameter in (block.output.weight, block.project.weight, block.gain,
                          self.reference['features'], block.selector[-1].weight):
            self.assert_gradient(parameter)
        for sample in (self.query, self.reference):
            self.assertIsNone(sample['anchor'].grad)
            self.assertIsNone(sample['local'].grad)

    def test_reference_permutation_preserves_output_and_pair_alignment(self):
        block = SelectiveReferenceMatch(channels=8)
        self.open_output(block)
        pairs = torch.tensor([[True, False, True], [False, True, True]])
        out, d = self.call(block, pairs=pairs)
        perm = torch.tensor([2, 0, 1])
        reference = {key: value[perm] for key, value in self.reference.items()}
        altered, after = self.call(block, reference=reference, pairs=pairs[:, perm])
        torch.testing.assert_close(altered, out)
        for key in ('reference_logits', 'reference_valid', 'reference_weights'):
            torch.testing.assert_close(after[key], d[key][:, perm])

    def test_repeated_blocks_keep_anchors_fixed_and_both_outputs_learn(self):
        first, second = SelectiveReferenceMatch(channels=8), SelectiveReferenceMatch(channels=8)
        self.assertNotEqual(first.output.weight.data_ptr(), second.output.weight.data_ptr())
        optimizer = torch.optim.SGD(list(first.parameters()) + list(second.parameters()), lr=.1)
        middle, _ = self.call(first)
        out, _ = self.call(second, query={**self.query, 'features': middle})
        torch.testing.assert_close(out, self.query['features'], rtol=0, atol=0)
        (out * torch.randn_like(out)).sum().backward()
        self.assert_gradient(first.output.weight)
        self.assert_gradient(second.output.weight)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        middle, d1 = self.call(first)
        out, d2 = self.call(second, query={**self.query, 'features': middle})
        torch.testing.assert_close(out, self.query['features'] + d1['update'] + d2['update'])
        for d in (d1, d2):
            self.assertGreater(d['update'].abs().sum().item(), 0)
        (out * torch.randn_like(out)).sum().backward()
        self.assert_gradient(first.project.weight)
        self.assert_gradient(second.project.weight)

    @staticmethod
    def partial_slot_padding(sample):
        padded = {key: value.clone() for key, value in sample.items()}
        # Each deep slot spans two columns; its second column may be invalid.
        for key in ('features', 'anchor', 'slots', 'energy'):
            padded[key] = padded[key].repeat_interleave(2, -1)
        padded['valid'][..., -1] = 0
        return padded

    def test_query_invalid_cell_cannot_receive_slot_writeback(self):
        block = SelectiveReferenceMatch(channels=8)
        self.open_output(block)
        query = self.partial_slot_padding(self.query)
        out, d = self.call(block, query=query)
        invalid = (grid_occupancy(query['valid'], out.shape[-2:]) == 0).expand_as(out)
        self.assertGreater(d['update'][~invalid].abs().sum().item(), 0)
        self.assert_zero(d['update'][invalid])
        torch.testing.assert_close(out[invalid], query['features'][invalid], rtol=0, atol=0)

    def test_reference_invalid_cell_cannot_change_values_or_get_gradient(self):
        block = SelectiveReferenceMatch(channels=8)
        self.open_output(block)
        reference = self.partial_slot_padding(self.reference)
        reference['features'].requires_grad_()
        invalid = (grid_occupancy(reference['valid'], reference['features'].shape[-2:]) == 0).expand_as(reference['features'])
        out, d = self.call(block, reference=reference)
        changed = reference['features'].detach().clone()
        changed[invalid] = torch.randn_like(changed[invalid]) * 100
        altered, _ = self.call(block, reference={**reference, 'features': changed})
        torch.testing.assert_close(altered, out, rtol=0, atol=0)
        d['update'].square().mean().backward()
        self.assert_zero(reference['features'].grad[invalid])
        self.assert_gradient(reference['features'])

    def test_frequency_disabled_removes_band_input_from_dynamic_selector(self):
        query, reference = self.controlled_bands()
        block = SelectiveReferenceMatch(channels=8, frequency=False, dynamic=True)
        with torch.no_grad():
            # A trained selector can route band0 evidence into the deep weight.
            block.selector[0].weight.zero_()
            block.selector[0].bias.zero_()
            block.selector[0].weight[0, 2] = 1
            block.selector[-1].weight.zero_()
            block.selector[-1].bias.zero_()
            block.selector[-1].weight[0, 0] = 3
        reference['anchor'][:, 0] = 0
        reference['anchor'][:, 1] = 1
        _, before = self.call(block, query, reference)
        swapped = {**query, 'local_energy': query['local_energy'][:, :, [1, 0, 2]]}
        _, after = self.call(block, swapped, reference)
        for weights in before['branch_weights']:
            self.assert_zero(weights[..., 2:])
        torch.testing.assert_close(after['reference_logits'], before['reference_logits'], rtol=0, atol=0)

    def test_frequency_disabled_keeps_nonzero_spatial_writeback_invariant(self):
        block = SelectiveReferenceMatch(channels=8, frequency=False, dynamic=True)
        self.open_output(block)
        original, evidence = self.call(block)
        query = {**self.query, 'energy': self.query['energy'] * 11,
                 'local_energy': self.query['local_energy'][:, :, [2, 0, 1]] * 7}
        reference = {**self.reference, 'energy': self.reference['energy'] * 13,
                     'local_energy': self.reference['local_energy'][:, :, [1, 2, 0]] * 5}
        changed, other = self.call(block, query, reference)
        self.assertGreater(evidence['update'].abs().sum().item(), 0)
        torch.testing.assert_close(changed, original, rtol=0, atol=0)
        torch.testing.assert_close(other['reference_logits'], evidence['reference_logits'], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
