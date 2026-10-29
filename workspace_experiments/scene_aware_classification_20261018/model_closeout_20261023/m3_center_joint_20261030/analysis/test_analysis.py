"""Synthetic checks only; no experiment predictions or final-test manifests."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('joint_analysis_test', Path(__file__).resolve().parents[1]/'analysis.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class AnalysisTests(unittest.TestCase):
    def test_equal_accuracy_does_not_hide_swaps(self):
        group = [dict(id=str(i),label=0) for i in range(4)]
        old = np.array([[3.,0,0],[0,3,0],[0,3,0],[3,0,0]])
        new = np.array([[0.,3,0],[3,0,0],[0,0,3],[3,0,0]])
        d = dict(group=group,y=np.zeros(4,dtype=int),classes=('a','b','c'),seeds=[17],
                 scores={('saved_baseline',17):old,('raw_joint',17):old,('center_joint',17):new},
                 receipts={(arm,17):dict(before={k.lower():0 for k in a.METRICS}) for arm in a.ARMS})
        _, paired, _, _ = a.paired_tables(d)
        row = paired[-1]
        self.assertEqual(row['delta_OA'],0)
        self.assertEqual((row['rescued'],row['harmed'],row['top1_changes'],row['wrong_to_wrong_changes']),(1,1,3,1))
        selected, absent = a.select_cases(d,'raw_joint')
        self.assertEqual({e['state'] for e in selected},set(a.STATES))
        self.assertEqual(absent,[])

    def test_inverse_weights_and_normalized_exposure(self):
        records = [dict(id=str(i),label=int(i>=12),mmsi=str(i//2),pass_group='p') for i in range(16)]
        teachers = {17:[dict(original=[1,0],useful=False) for _ in records]}
        rows = a.source_composition(records,('major','minor'),[17],teachers)
        self.assertAlmostEqual(rows[0]['weight_sum'],8,places=5)
        self.assertAlmostEqual(rows[1]['weight_sum'],8,places=5)
        self.assertEqual(rows[0]['source_mmsi'],6)
        self.assertEqual(rows[1]['source_mmsi'],2)
        self.assertAlmostEqual(rows[0]['s17_ce_normalized_mass_6epochs'],3,places=5)
        self.assertAlmostEqual(rows[1]['s17_ce_normalized_mass_6epochs'],3,places=5)
        self.assertEqual(rows[1]['s17_kl_eligible_mass_6epochs'],0)
        self.assertNotIn('torch',sys.modules)


if __name__=='__main__':
    unittest.main()
