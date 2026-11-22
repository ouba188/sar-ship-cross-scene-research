"""Small NumPy/AST checks; no Torch, checkpoint, SAR image or test split access."""
import ast
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import analyze_head_reuse as analysis


LAB = Path(__file__).resolve().parent.parent


class HeadReuseTests(unittest.TestCase):
    def test_valid_scale_formula_is_identical_to_training_cli(self):
        def formula(path, name):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            values = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'logits' for t in node.targets)
                      and isinstance(node.value, ast.BinOp)]
            self.assertEqual(len(values), 1)
            return ast.dump(values[0])
        self.assertEqual(formula(LAB/'oss2_three_arm_experiment_20261122/branch_diagnostics.py', 'extract'),
                         formula(LAB/'port_prototype_model_20261121/sar_port/two_stage/cli.py', 'stage1_metrics'))

    def test_identity_alignment_and_label_rejection(self):
        ids = np.asarray([f'fixture_{i}' for i in range(945)])
        targets = np.arange(945) % 4
        logits = np.zeros((945, 4), dtype=np.float32)
        logits[np.arange(945), targets] = 1
        classes = ['0', '1', '2', '3']
        natural = dict(ids=ids, split=np.asarray(['train']*718+['validation']*227),
                       label=targets.astype(str), class_order=np.asarray(classes), stage1_logits=logits,
                       metadata={'class_order': classes})
        query_ids, y = list(ids[718:][::-1]), targets[718:][::-1]
        actual = analysis.align_heads(natural, query_ids, y, classes, np)
        np.testing.assert_array_equal(actual, logits[718:][::-1])
        natural['label'][718] = '1'
        with self.assertRaisesRegex(ValueError, 'labels differ'):
            analysis.align_heads(natural, query_ids, y, classes, np)

    def test_gamma0_and_fixed_evidence_rescue(self):
        classes = ['0', '1', '2', '3']
        y = np.arange(227) % 4
        original = np.zeros((227, 4), dtype=np.float32)
        original[np.arange(227), y] = 10
        replacement = original.copy()
        original[0] = (0, 2, -5, -5)
        evidence = np.zeros_like(original)
        evidence[0, 0] = 1
        first = analysis.scan(original, evidence, y, classes, replacement.argmax(1), np)
        second = analysis.scan(replacement, evidence, y, classes, replacement.argmax(1), np)
        self.assertEqual(first[0]['rescued_vs_own_gamma0'], 0)
        self.assertEqual(first[1]['rescued_vs_own_gamma0'], 0)
        self.assertEqual(first[2]['rescued_vs_own_gamma0'], 1)
        self.assertEqual(second[0]['accuracy'], 1)
        self.assertTrue(all(row['harmed_vs_own_gamma0'] == 0 for row in second))

    def test_checkpoint_metric_replay_requires_exact_confusion(self):
        path = LAB/'oss2_three_arm_experiment_20261122/branch_diagnostics.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'verify_head_replay')
        namespace = {'math': math}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<replay-function-only>', 'exec'), namespace)
        value = dict(count=2, confusion_matrix=[[1, 0], [0, 1]], accuracy=1., balanced_accuracy=1., macro_f1=1.)
        self.assertTrue(namespace['verify_head_replay'](value, value)['passed'])
        different = dict(value, confusion_matrix=[[0, 1], [1, 0]])
        with self.assertRaisesRegex(RuntimeError, 'confusion_matrix'):
            namespace['verify_head_replay'](value, different)

    def test_no_torch_loaded(self):
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
