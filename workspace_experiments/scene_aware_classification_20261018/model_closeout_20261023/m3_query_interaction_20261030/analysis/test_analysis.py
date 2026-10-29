"""Synthetic-only adaptation checks; never load experiment predictions."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('interaction_analysis_tested', HERE.parent / 'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AdaptationChecks(unittest.TestCase):
    def test_schema_alignment_and_partial_cohort(self):
        group = [dict(id=str(i), label=i % 2) for i in range(4)]
        z = np.arange(8).reshape(4, 2)
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path = Path(tmp) / 'synthetic.npz'
            for order in ([3, 1, 0, 2], [0, 1, 2]):
                np.savez(path, ids=np.array([group[i]['id'] for i in order]),
                         y=np.array([group[i]['label'] for i in order]), score=z[order], baseline=z[order])
                pack, pending = a.load_pack(path, group, ['a', 'b'])
                if len(order) == 4:
                    np.testing.assert_array_equal(pack['score'], z)
                else:
                    self.assertIsNone(pack)
                    self.assertEqual(pending['missing_ids'], ['3'])

    def test_partial_arms_and_four_states(self):
        old = np.array([[3., 0, 0]] * 4)
        new = np.array([[3., 0, 0], [0., 3, 0], [0., 3, 0], [0., 0, 3]])
        group = [dict(id=str(i), label=y) for i, y in enumerate([0, 0, 1, 1])]
        d = dict(dataset='synthetic', group=group, classes=['a', 'b', 'absent'],
                 y=np.array([r['label'] for r in group]), requested=[17, 29, 43], seeds=[17, 29],
                 pending=[], input_paths=[], scores={})
        for seed in d['seeds']:
            d['scores']['saved_baseline', seed] = old
            d['scores']['ce', seed] = old
            d['scores']['interaction', seed] = new
        d['scores']['query_only', 17] = old
        with tempfile.TemporaryDirectory(dir=HERE) as tmp, patch.object(a, 'OUT', Path(tmp)):
            summary = a.summarize(d)
        row = next(r for r in summary['paired_means'] if (r['arm'], r['reference']) == ('interaction', 'ce'))
        self.assertEqual([row[s + '_events'] for s in a.STATES], [2, 2, 2, 2])
        self.assertEqual(row['rescued_queries'], 1)
        self.assertEqual(row['delta_OA'], 0)
        row = next(r for r in summary['paired_means'] if (r['arm'], r['reference']) == ('query_only', 'ce'))
        self.assertEqual(row['seeds'], [17])
        self.assertIsNone(row['sd_OA'])
        self.assertFalse(summary['three_seed_complete'])
        self.assertEqual(summary['available_by_arm']['ref_only'], [])
        endpoint = next(r for r in summary['endpoint_means'] if r['arm'] == 'query_only')
        self.assertEqual(endpoint['seeds'], [17])
        self.assertIsNone(endpoint['sd_OA'])

    def test_imports_are_cpu_only(self):
        self.assertNotIn('torch', sys.modules)

    def test_shift_absolute_energies(self):
        old = np.zeros((2, 2))
        d = dict(dataset='synthetic', y=np.array([0, 1]), seeds=[17, 29],
                 scores={('saved_baseline', 17): old,
                         ('interaction', 17): np.array([[2., -2.], [0., 0.]]),
                         ('saved_baseline', 29): old,
                         ('interaction', 29): np.array([[4., -4.], [0., 0.]])})
        with tempfile.TemporaryDirectory(dir=HERE) as tmp, patch.object(a, 'OUT', Path(tmp)):
            (Path(tmp) / 'synthetic').mkdir()
            status = a.shared_shift_summary(d)
            report = a.read(Path(status['output']))
            row = report['rows'][0]
        self.assertEqual(row['centered_delta_energy'], 8)
        self.assertEqual(row['common_energy'], 4)
        self.assertEqual(row['residual_energy'], 4)
        self.assertEqual(row['residual_energy_fraction'], .5)
        mean = next(r for r in report['means'] if r['arm'] == 'interaction')
        self.assertEqual(mean['seeds'], [17, 29])
        self.assertEqual(mean['centered_delta_energy'], 20)
        self.assertEqual(mean['residual_energy'], 10)
        self.assertAlmostEqual(mean['sd_centered_delta_energy'], np.sqrt(288))
        self.assertFalse(report['three_seed_complete'])

    def test_control_pair_direction_receipts_and_off_restoration(self):
        old = np.array([[3., 0.], [3., 0.]])
        normal = np.array([[3., 0.], [0., 3.]])
        y = np.array([0, 1])
        d = dict(dataset='synthetic', group=[dict(id=str(i), label=int(v)) for i, v in enumerate(y)],
                 classes=['a', 'b'], y=y, seeds=[17], requested=[17],
                 scores={('saved_baseline', 17): old, ('interaction', 17): normal},
                 receipts={('interaction', 17): dict(controls={})})
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            pred = root / 'synthetic' / 'predictions'
            pred.mkdir(parents=True)
            def save_control(control, scores):
                order = [1, 0]
                np.savez(pred / f'interaction_{control}_s17.npz', ids=np.array(['1', '0']), y=y[order],
                         score=scores[order], baseline=old[order])
                d['receipts']['interaction', 17]['controls'][control] = dict(
                    metrics={k.lower(): v for k, v in a.metric(scores, y).items()},
                    max_logit_difference=float(np.abs(scores-normal).max()),
                    changed=int((scores.argmax(1) != normal.argmax(1)).sum()))
            save_control('off', old)
            save_control('uniform', old)
            with patch.object(a, 'HERE', root), patch.object(a, 'OUT', root / 'output'):
                status = a.dependence_summary(d)
                summary = a.read(Path(status['output']) / 'summary.json')
                self.assertEqual(status['available_controls'], 2)
                row = next(r for r in summary['paired_means'] if
                           (r['arm'], r['reference']) == ('interaction_uniform', 'interaction'))
                self.assertEqual(row['delta_OA'], -50)
                self.assertEqual(row['harmed_events'], 1)
                self.assertEqual(row['rescued_events'], 0)
                self.assertIsNone(row['sd_OA'])
                self.assertTrue(summary['diagnostics'][0]['off_restoration_passed'])
                self.assertNotIn(('interaction_off', 17), d['scores'])
                d['receipts']['interaction', 17]['controls']['uniform']['changed'] = 0
                with self.assertRaisesRegex(ValueError, 'changed count mismatch'):
                    a.dependence_summary(d)
                save_control('uniform', old)
                save_control('off', old + .01)
                with self.assertRaisesRegex(ValueError, 'off does not restore'):
                    a.dependence_summary(d)


if __name__ == '__main__':
    unittest.main(verbosity=2)
