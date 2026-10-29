"""Focused CPU checks; all temporary inputs/outputs stay inside analysis/."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('replay_analysis_tested', HERE.parent / 'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnalysisChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=HERE, prefix='_synthetic_')
        self.folder = Path(self.temp.name)
        self.group = [dict(id=str(i), label=y) for i, y in enumerate([0, 0, 1, 1])]
        self.old = np.array([[3., 0, 0], [3., 0, 0], [3., 0, 0], [3., 0, 0]])
        self.new = np.array([[3., 0, 0], [0., 3, 0], [0., 3, 0], [0., 0, 3]])
        self.d = dict(dataset='synthetic', group=self.group, classes=['a', 'b', 'absent'],
                      seeds=[17, 29, 43], requested=[17, 29, 43], pending=[], input_paths=[],
                      manifest='synthetic', receipts={}, y=np.array([0, 0, 1, 1]), scores={})
        for seed in self.d['seeds']:
            for arm, z in [('saved_baseline', self.old), ('adapt_only', self.old), ('source_mix', self.new)]:
                self.d['scores'][arm, seed] = z

    def tearDown(self):
        self.temp.cleanup()

    def save_pack(self, order):
        path = self.folder / 'fixture.npz'
        np.savez(path, ids=np.array([self.group[i]['id'] for i in order]),
                 y=self.d['y'][order], score=self.new[order], baseline=self.old[order])
        return path

    def test_id_reordering(self):
        pack, missing = a.load_pack(self.save_pack([3, 1, 0, 2]), self.group, self.d['classes'])
        self.assertIsNone(missing)
        np.testing.assert_array_equal(pack['score'], self.new)

    def test_missing_samples_not_silent_subset(self):
        pack, missing = a.load_pack(self.save_pack([0, 1, 2]), self.group, self.d['classes'])
        self.assertIsNone(pack)
        self.assertEqual(missing['missing_ids'], ['3'])

    def test_four_states_absent_class_and_ww_change(self):
        _, paired, classes, _ = a.paired_tables(self.d)
        row = next(r for r in paired if r['arm'] == 'source_mix' and r['reference'] == 'adapt_only')
        self.assertEqual([row[k] for k in a.STATES], [1, 1, 1, 1])
        self.assertEqual(row['top1_changes'], 3)
        self.assertEqual(row['wrong_to_wrong_changes'], 1)
        self.assertEqual(row['delta_OA'], 0)
        self.assertEqual(row['delta_BA'], 0)
        self.assertTrue(all(r['old_recall'] is None for r in classes if r['label'] == 2))
        selected, missing = a.select_cases(self.d, 'source_mix', 'adapt_only')
        self.assertEqual({r['state'] for r in selected}, set(a.STATES))
        self.assertEqual(missing, [])

    def test_three_seed_events_not_independent_queries(self):
        with patch.object(a, 'OUT', self.folder):
            result = a.summarize(self.d)
        self.assertTrue(result['paired_three_orders_complete'])
        row = result['paired_means'][-1]
        self.assertEqual((row['rescued_events'], row['rescued_queries']), (3, 1))
        self.assertEqual(row['sd_OA'], 0)
        self.d['seeds'] = [17]
        self.d['receipts'] = {}
        with patch.object(a, 'OUT', self.folder):
            result = a.summarize(self.d)
        self.assertFalse(result['paired_three_orders_complete'])
        self.assertIsNone(result['paired_means'][-1]['sd_OA'])

    def test_no_files_no_fallback(self):
        with patch.object(a, 'HERE', self.folder):
            d = a.load_dataset('private', [17, 29, 43])
        self.assertEqual(d['seeds'], [])
        self.assertEqual(d['scores'], {})
        self.assertEqual(sum(len(r['missing_files']) for r in d['pending']), 12)
        with patch.object(a, 'OUT', self.folder):
            result = a.summarize(d)
        self.assertIsNone(result['paired_means'][-1]['delta_OA'])
        self.assertEqual(result['analyzed_query_count'], 0)

    def test_real_native_pixels_unchanged(self):
        helper = a.coverage.module('replay_native_check', a.ROOT / 'spatial_m12_refit_20261029/analysis.py')
        native = a.coverage.module('replay_input_check', a.WORK / 'sar_m123_standalone/sar_m123/input.py')
        d = a.load_dataset('private', [])
        q = d['group'][0]
        r = next(r for r in d['refs'] if helper.eligible(q, r) and r.get('vv'))
        for record, role in [(q, 'QUERY'), (r, 'REF')]:
            view = native.prepare_view(record['vv'])
            tile, receipt = helper.native_tile(record, 'vv', native, role)
            w, h = receipt['raw_size']
            raw_rendered = np.asarray(tile)[63:63+h, 8:8+w, 0]
            np.testing.assert_array_equal(raw_rendered, view['raw'])
            self.assertEqual(receipt['pixel_scale'], 1)
            tile.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
