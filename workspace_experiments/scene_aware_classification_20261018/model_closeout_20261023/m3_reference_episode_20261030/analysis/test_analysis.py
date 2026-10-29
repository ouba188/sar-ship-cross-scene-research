"""Small synthetic checks for ID alignment and prediction-versus-metric equality."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('episode_analysis_tested', HERE.parent / 'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnalysisChecks(unittest.TestCase):
    def test_id_alignment_and_partial(self):
        group = [dict(id=str(i), label=i % 2) for i in range(4)]
        z = np.arange(8).reshape(4, 2)
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            path = Path(tmp) / 'synthetic.npz'
            for order in ([3, 1, 0, 2], [0, 1, 2]):
                np.savez(path, ids=np.array([group[i]['id'] for i in order]),
                         y=np.array([group[i]['label'] for i in order]), score=z[order], baseline=z[order])
                pack, missing = a.load_pack(path, group, ['a', 'b'])
                if len(order) == 4:
                    np.testing.assert_array_equal(pack['score'], z)
                else:
                    self.assertIsNone(pack)
                    self.assertEqual(missing['missing_ids'], ['3'])

    def test_states_and_same_accuracy_different_predictions(self):
        old = np.array([[3., 0, 0]] * 4)
        new = np.array([[3., 0, 0], [0., 3, 0], [0., 3, 0], [0., 0, 3]])
        group = [dict(id=str(i), label=y) for i, y in enumerate([0, 0, 1, 1])]
        d = dict(dataset='synthetic', group=group, classes=['a', 'b', 'absent'],
                 y=np.array([r['label'] for r in group]), requested=[17], seeds=[17],
                 pending=[], input_paths=[], scores={('saved_baseline', 17): old,
                 ('ce', 17): old, ('fixed_alt', 17): old, ('episodic', 17): new})
        with tempfile.TemporaryDirectory(dir=HERE) as tmp, patch.object(a, 'OUT', Path(tmp)):
            result = a.summarize(d)
        row = next(r for r in result['paired_means'] if r['reference'] == 'fixed_alt')
        self.assertEqual([row[s] for s in a.STATES], [1, 1, 1, 1])
        self.assertEqual(row['delta_OA'], 0)
        self.assertIsNone(row['sd_OA'])
        identity = result['prediction_identity'][0]
        self.assertEqual(identity['changed'], 3)
        self.assertFalse(identity['logits_identical'])

    def test_cpu_imports(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
