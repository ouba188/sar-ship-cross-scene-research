"""Bounded checks for partial arms/seeds, paired evaluation, and native pixels."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pairwise_analysis_tested', HERE.parent / 'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnalysisChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix='_synthetic_')
        self.folder = Path(self.temp.name)
        self.group = [dict(id=str(i), label=y, mmsi=str(i), pass_group=str(i))
                      for i, y in enumerate([0, 0, 1, 1])]
        self.old = np.array([[3., 0, 0]] * 4)
        self.new = np.array([[3., 0, 0], [0., 3, 0], [0., 3, 0], [0., 0, 3]])
        self.d = dict(dataset='synthetic', group=self.group, classes=['a', 'b', 'absent'],
                      seeds=[17, 29], requested=[17, 29, 43], pending=[], input_paths=[],
                      manifest='synthetic', receipts={}, y=np.array([0, 0, 1, 1]), scores={})
        for seed, arms in ((17, a.ARMS), (29, ('ce', 'hybrid'))):
            self.d['scores']['saved_baseline', seed] = self.old
            for arm in arms:
                self.d['scores'][arm, seed] = self.old if arm == 'ce' else self.new

    def tearDown(self):
        self.temp.cleanup()

    def test_partial_arms_seeds_and_four_states(self):
        with patch.object(a, 'OUT', self.folder):
            result = a.summarize(self.d)
        rank = next(r for r in result['paired_means'] if r['arm'] == 'rank' and r['reference'] == 'ce')
        hybrid = next(r for r in result['paired_means'] if r['arm'] == 'hybrid' and r['reference'] == 'ce')
        self.assertEqual(rank['seeds'], [17])
        self.assertEqual(hybrid['seeds'], [17, 29])
        self.assertIsNone(rank['sd_OA'])
        self.assertFalse(result['three_seed_complete'])
        self.assertEqual([rank[s + '_events'] for s in a.STATES], [1, 1, 1, 1])
        self.assertEqual((hybrid['rescued_events'], hybrid['rescued_queries']), (2, 1))

    def test_id_alignment_and_incomplete_cohort(self):
        path = self.folder / 'fixture.npz'
        for ix in ([3, 1, 0, 2], [0, 1, 2]):
            np.savez(path, ids=np.array([self.group[i]['id'] for i in ix]),
                     y=self.d['y'][ix], score=self.new[ix], baseline=self.old[ix])
            pack, absent = a.load_pack(path, self.group, self.d['classes'])
            if len(ix) == 4:
                np.testing.assert_array_equal(pack['score'], self.new)
            else:
                self.assertIsNone(pack)
                self.assertEqual(absent['missing_ids'], ['3'])

    def test_source_predictions_not_used(self):
        folder = self.folder / 'private' / 'predictions'
        folder.mkdir(parents=True)
        np.savez(folder / 'ce_source_s17.npz', score=self.new)
        with patch.object(a, 'HERE', self.folder):
            d = a.load_dataset('private', [17])
        self.assertEqual(d['scores'], {})
        self.assertEqual(len(d['pending']), 3)

    def test_shared_shift_and_softmax_invisible_translation(self):
        r = a.shared_shift(self.old, self.old + [1., -1., 0.], self.d['y'])
        self.assertAlmostEqual(r['common_energy_fraction'], 1)
        r = a.shared_shift(self.old, self.old + 5., self.d['y'])
        self.assertIsNone(r['common_energy_fraction'])
        self.assertEqual(r['centered_delta_energy'], 0)

    def test_guard_uses_matching_ce_guard_and_keeps_raw_scores(self):
        p = self.folder / 'synthetic' / 'predictions'
        r = self.folder / 'synthetic' / 'alignment'
        p.mkdir(parents=True)
        r.mkdir(parents=True)
        shift = np.array([1., -1., 0.])
        for arm in ('ce', 'rank'):
            np.savez(p / f'{arm}_guard_s17.npz', score=self.d['scores'][arm, 17]-shift,
                     baseline=self.old, ids=np.array([g['id'] for g in self.group]), y=self.d['y'])
            a.dump(r / f'{arm}_guard_s17.json', dict(target_used_for_shift=False,
                   evaluation_used=False, source_shift=shift.tolist(), source_ids=['synthetic_source']))
        with patch.object(a, 'HERE', self.folder), patch.object(a, 'OUT', self.folder):
            status = a.guard_analysis(self.d)
        summary = a.read(Path(status['output']) / 'summary.json')
        pair = next(x for x in summary['paired_means'] if x['arm'] == 'rank_guard' and x['reference'] == 'ce_guard')
        self.assertEqual(pair['seeds'], [17])
        self.assertEqual(status['available_reports'], 2)
        self.assertTrue(all('source_ids' not in x['receipt'] for x in summary['reports']))
        self.assertNotIn(('rank_guard', 17), self.d['scores'])
        target_shift = (self.new - self.old).mean(axis=0)
        np.savez(p / 'rank_guard_s17.npz', score=self.new-target_shift,
                 baseline=self.old, ids=np.array([g['id'] for g in self.group]), y=self.d['y'])
        with patch.object(a, 'HERE', self.folder), patch.object(a, 'OUT', self.folder):
            with self.assertRaisesRegex(ValueError, 'reported source offset'):
                a.guard_analysis(self.d)
        a.dump(r / 'ce_guard_s17.json', dict(target_used_for_shift=True,
               evaluation_used=False, source_shift=shift.tolist()))
        with patch.object(a, 'HERE', self.folder), patch.object(a, 'OUT', self.folder):
            with self.assertRaisesRegex(ValueError, 'source-only alignment'):
                a.guard_analysis(self.d)

    def test_no_model_runtime_imported(self):
        self.assertNotIn('torch', sys.modules)

    def test_pair_exclusion_missing_classes_and_common_bias_invariance(self):
        self.group[2]['mmsi'] = self.group[0]['mmsi']
        self.group[3]['pass_group'] = self.group[1]['pass_group']
        self.d['scores']['rank', 17] = self.old + [4., -3., 1.]
        r = a.query_pair_metrics(self.d)
        self.assertEqual([p['pairs'] for p in r['coverage']], [2, 0, 0])
        old = next(p for p in r['endpoints'] if p['arm'] == 'saved_baseline' and p['seed'] == 17)
        new = next(p for p in r['endpoints'] if p['arm'] == 'rank')
        self.assertAlmostEqual(old['pair_logistic'], np.log(2))
        self.assertEqual(new['pair_logistic'], old['pair_logistic'])
        self.assertEqual(new['pair_rank_accuracy'], 0)
        self.assertTrue(all(p['pair_logistic'] is None for p in r['class_pair_metrics'] if p['missing']))

    def test_native_rendering_original_pixel_scale(self):
        helper = a.replay.coverage.module('pairwise_panel_check', a.ROOT / 'spatial_m12_refit_20261029/analysis.py')
        native = a.replay.coverage.module('pairwise_native_check', a.replay.WORK / 'sar_m123_standalone/sar_m123/input.py')
        d = a.load_dataset('private', [])
        q = d['group'][0]
        r = next(r for r in d['refs'] if helper.eligible(q, r) and r.get('vv'))
        for record in (q, r):
            view = native.prepare_view(record['vv'])
            tile, receipt = helper.native_tile(record, 'vv', native, 'CHECK')
            w, h = receipt['raw_size']
            np.testing.assert_array_equal(np.asarray(tile)[63:63+h, 8:8+w, 0], view['raw'])
            self.assertEqual(receipt['pixel_scale'], 1)
            tile.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
