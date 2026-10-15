"""Focused integration checks for support-only adaptation and handoff budgets."""

import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_opensarship2_transfer import freeze_draw, evaluate


class TransferPolicyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.n = 200
        self.b = rng.normal(size=(self.n, 8))
        self.e = rng.normal(size=(self.n, 8))
        self.pb = np.column_stack((rng.normal(size=(self.n, 4)), np.ones(self.n)))
        self.pe = np.column_stack((rng.normal(size=(self.n, 6)), np.ones(self.n)))
        self.support = np.arange(64)
        self.y = np.arange(64) % 8
        self.gate = dict(center=np.zeros(4), scale=np.ones(4), lr=np.array([0., 0., 0., 0., 1.]))

    def run_policy(self, y=None):
        return freeze_draw(self.b, self.e, self.pb, self.pe, self.gate,
                           self.support, self.y if y is None else y, np.arange(self.n))

    def test_cap_excludes_support_and_controls_exact_changes(self):
        result = self.run_policy()
        self.assertEqual(int(result['caps']), 6)
        self.assertEqual(int(result['selected'].sum()), 6)
        self.assertFalse(result['selected'][self.support].any())
        np.testing.assert_array_equal(result['predictions'][2] != result['predictions'][1], result['selected'])
        np.testing.assert_array_equal(result['predictions'][0], self.b.argmax(1))

    def test_absent_support_classes_cannot_receive_handoff(self):
        result = self.run_policy(np.arange(64) % 2)
        chosen = result['expert_scores'].argmax(1)[result['selected']]
        self.assertTrue(np.isin(chosen, [0, 1]).all())

    def test_rejects_query_labels_in_support_api(self):
        with self.assertRaisesRegex(ValueError, '64'):
            self.run_policy(np.arange(self.n) % 8)

    def test_readout_missing_target_class_is_null_not_zero(self):
        result = self.run_policy()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            pd.DataFrame({'class_id': np.arange(self.n) % 7}).to_csv(out/'target_manifest.csv', index=False)
            np.savez_compressed(out/'policies.npz', **{k: np.stack([v]*5) for k, v in result.items()})
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(out, target_name='seven-class transfer test', encoder_scope='single-image adapter test')
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['target'], 'seven-class transfer test')
            self.assertEqual(summary['encoder_scope'], 'single-image adapter test')
            for arm in summary['means']:
                self.assertIsNone(arm['class_recall'][7])
                self.assertTrue(all(x is not None for x in arm['class_recall'][:7]))


if __name__ == '__main__':
    unittest.main()
