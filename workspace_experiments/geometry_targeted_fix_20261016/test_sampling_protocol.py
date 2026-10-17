"""CPU-only checks of actual sampling interfaces and query-label independence."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import run_fix as fix


def mock_data():
    n = 60
    valid = torch.ones(n, dtype=torch.bool)
    valid[59] = False
    specs = [([0, 1, 2], list(range(10, 59)), 'A', 0),
             ([2, 3, 4], list(reversed(range(10, 59))), 'A', 1),
             ([4, 5, 6], list(range(10, 59)), 'B', 0),
             ([7, 8, 9], [59], 'A', 2),
             ([0, 59], list(range(10, 59)), 'A', 3)]
    cache = []
    for s, q, support_scene, draw in specs:
        cache.append(dict(s=torch.tensor(s), q=torch.tensor(q),
                          t=dict(port='port', acquisition='query_scene', support_acquisition=support_scene,
                                 draw=draw, origin='cross', budget=32)))
    meta = pd.DataFrame(dict(mmsi=[str(i) for i in range(n)], acquisition=['scene']*n,
                             class_id=np.arange(n)%8))
    return dict(cache={'fit': cache}, tensors={'valid': valid}, meta=meta)


def step_trace(data, groups, seed, arm, steps=200):
    rng = np.random.default_rng(20261016 + seed)
    trace, paired_steps, paired_flags = [], 0, []
    for step in range(steps):
        sampled = fix.sample_step(data, groups, rng, arm.endswith(('_pair', '_rank')))
        paired_flags.append(len(sampled) == 2)
        paired_steps += int(len(sampled) == 2)
        if len(sampled) == 1:
            sampled += fix.sample_step(data, groups, rng, False)
        trace.append([(ti, pos.tolist()) for ti, pos in sampled])
    return trace, paired_steps, paired_flags


class SamplingProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_inventory_filters_and_returns_original_row_ids(self):
        data = mock_data()
        eligible, groups, rows, inventory = fix.base.fit_inventory(data)
        self.assertEqual(eligible, [0, 1, 2])
        self.assertEqual(groups, {'port': {'query_scene': [0, 1, 2]}})
        self.assertEqual(rows, list(range(7)) + list(range(10, 59)))
        self.assertEqual(inventory['query_samples'], 49)
        self.assertEqual(inventory['support_samples'], 7)

    def test_pair_matches_query_identity_not_position(self):
        data = mock_data()
        groups = {'port': {'query_scene': [0, 1]}}
        rng = np.random.default_rng(123)
        for _ in range(100):
            sampled = fix.sample_step(data, groups, rng, True)
            self.assertEqual(len(sampled), 2)
            (a, ap), (b, bp) = sampled
            left, right = data['cache']['fit'][a], data['cache']['fit'][b]
            self.assertNotEqual(left['t']['draw'], right['t']['draw'])
            self.assertEqual(left['t']['support_acquisition'], right['t']['support_acquisition'])
            torch.testing.assert_close(left['q'][ap], right['q'][bp], rtol=0, atol=0)
            self.assertEqual(len(ap), 32)
            self.assertTrue(data['tensors']['valid'][left['q'][ap]].all())

    def test_no_partner_or_order_only_support_returns_one_task(self):
        for case in ('different_acquisition', 'order_only'):
            with self.subTest(case=case):
                data = mock_data()
                indices = [0, 2]
                if case == 'order_only':
                    # Same scene, different draw, same support identities in reverse order.
                    data['cache']['fit'][1]['s'] = data['cache']['fit'][0]['s'].flip(0)
                    indices = [0, 1]
                groups = {'port': {'query_scene': indices}}
                for seed in range(3):
                    self.assertEqual(len(fix.sample_step(data, groups, np.random.default_rng(seed), True)), 1)

    def test_three_paired_arms_same_seed_same_actual_samples(self):
        data = mock_data()
        _, groups, _, _ = fix.base.fit_inventory(data)
        for seed in range(3):
            traces = [step_trace(data, groups, seed, arm) for arm in
                      ('geometry_pair', 'geometry_rank', 'conditional_rank')]
            self.assertEqual(traces[0], traces[1])
            self.assertEqual(traces[1], traces[2])

    def test_target_labels_do_not_change_baseline_or_increment_predictions(self):
        torch.manual_seed(47)
        n = 12
        s, q = torch.arange(4), torch.arange(4, n)
        d = dict(base=torch.randn(n, 12, dtype=torch.float64),
                 expert=torch.randn(n, 257, dtype=torch.float64)*.1,
                 prior_b=torch.randn(n, 8, dtype=torch.float64),
                 prior_e=torch.randn(n, 8, dtype=torch.float64),
                 relation=torch.randn(n, 8, dtype=torch.float64), weights=torch.ones(8, dtype=torch.float64),
                 ship=torch.nn.functional.normalize(torch.randn(n, 64, dtype=torch.float64), dim=-1),
                 labels=torch.arange(n)%8, valid=torch.ones(n, dtype=torch.bool),
                 geometry=torch.tensor([[.3, .4, .5, 1., 1., 1., .8]]).repeat(n, 1))
        for prefix in ('ship', 'img'):
            d[prefix+'_tokens'] = torch.randn(n, 4, 4, 64)
            d[prefix+'_areas'] = torch.ones(n, 4, 4)
        context = torch.nn.functional.normalize(torch.randn(n, 64, dtype=torch.float64), dim=-1)
        task = dict(s=s.tolist(), q=q.tolist())
        data = dict(tensors=d, tasks={'target': [task]}, port='mock')

        class FrozenReader(torch.nn.Module):
            def __init__(self, mode):
                super().__init__()

        with patch.object(fix.base, 'DEVICE', 'cpu'), patch.object(fix.base.old, 'DEVICE', 'cpu'), \
             patch.object(fix.base, 'GeometryReader', FrozenReader), \
             patch.object(fix.base.torch, 'load', return_value={'state': {}, 'best_step': 0}), \
             patch.object(fix.base.old, 'contexts', return_value=context):
            model = fix.OptionalGeometryReader('geometry').eval()
            fix.base.prepare_baseline(data, 0)
            baseline = data['cache']['target'][0]['old_prob'].clone()
            result = fix.predict(model, data, data['cache']['target'][0])
            d['labels'][q] = (d['labels'][q]+3)%8
            fix.base.prepare_baseline(data, 0)
            mutated = fix.predict(model, data, data['cache']['target'][0])
            torch.testing.assert_close(data['cache']['target'][0]['old_prob'], baseline, rtol=0, atol=0)
            for actual, expected in zip(mutated, result):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            d['labels'][s] = (d['labels'][s]+1)%8
            fix.base.prepare_baseline(data, 0)
            changed_support = fix.predict(model, data, data['cache']['target'][0])[0]
            self.assertGreater(float((changed_support-result[0]).abs().max()), 1e-6)

    def test_real_fixed_source_tasks_pairing_and_sampling(self):
        root = Path(__file__).resolve().parent
        pools = json.loads((root.parent / 'prepool_local_probe_20261015/body_context_expansion/'
                           'ring_relaxed/query_coverage/frozen.json').read_text(encoding='utf-8'))['pools']
        geometry = pd.read_csv(root/'recovered_geometry.csv', low_memory=False).set_index('global_row')
        for pool in pools:
            meta = pd.read_csv(Path(pool['source_folder'])/'metadata.csv')
            tasks = [t for t in pool['splits']['fit'] if t['budget'] == 32]
            valid = torch.as_tensor(geometry.loc[meta.global_row, 'ship_valid'].to_numpy() > 0)
            data = dict(meta=meta, tensors={'valid': valid}, cache={'fit': [
                dict(t=t, s=torch.tensor(t['s']), q=torch.tensor(t['q'])) for t in tasks]})
            eligible, groups, _, inventory = fix.base.fit_inventory(data)
            if not eligible:
                continue
            same_set_pairs = 0
            candidate_pairs = 0
            for inds in [ids for scenes in groups.values() for ids in scenes.values()]:
                for ai, a in enumerate(inds):
                    for b in inds[ai+1:]:
                        x, y = data['cache']['fit'][a], data['cache']['fit'][b]
                        if x['t']['support_acquisition'] != y['t']['support_acquisition']:
                            continue
                        intersection = set(x['q'][valid[x['q']]].tolist()) & set(y['q'][valid[y['q']]].tolist())
                        if not intersection or torch.equal(x['s'], y['s']):
                            continue
                        same_set_pairs += int(set(x['s'].tolist()) == set(y['s'].tolist()))
                        candidate_pairs += 1
            paired_counts = []
            fallback_counts = []
            order_only_steps = []
            for seed in range(3):
                traces = [step_trace(data, groups, seed, arm) for arm in
                          ('geometry_pair', 'geometry_rank', 'conditional_rank')]
                self.assertEqual(traces[0], traces[1])
                self.assertEqual(traces[1], traces[2])
                paired_counts.append(traces[0][1])
                fallback_counts.append(len(traces[0][0])-traces[0][1])
                repeats = 0
                for sampled, is_pair in zip(traces[0][0], traces[0][2]):
                    if not is_pair:
                        continue
                    (a, ap), (b, bp) = sampled
                    x, y = data['cache']['fit'][a], data['cache']['fit'][b]
                    self.assertNotEqual(a, b)
                    self.assertEqual(x['t']['support_acquisition'], y['t']['support_acquisition'])
                    self.assertNotEqual(x['t']['draw'], y['t']['draw'])
                    torch.testing.assert_close(x['q'][ap], y['q'][bp], rtol=0, atol=0)
                    repeats += int(set(x['s'].tolist()) == set(y['s'].tolist()))
                order_only_steps.append(repeats)
            print('ACTUAL_SOURCE', pool['pool'], 'eligible', inventory['tasks'],
                  'candidate_pairs', candidate_pairs, 'order_only_candidates', same_set_pairs,
                  'paired_steps_of_200', paired_counts, 'fallback_independent_steps', fallback_counts,
                  'order_only_paired_steps', order_only_steps, flush=True)
            self.assertEqual(order_only_steps, [0, 0, 0], 'Actual paired steps must use different support sets')


if __name__ == '__main__':
    unittest.main()
