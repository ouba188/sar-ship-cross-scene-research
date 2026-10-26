"""Checks for the actual source-label and staged-optimization boundaries."""
import unittest
import numpy as np
import pandas as pd
import torch

from sar_m123.train_aligned import validate_roles, source_scaler, scale_input
from sar_m123.train_ops import exact_j, Gate


class TrainingTests(unittest.TestCase):
    def example(self):
        rows = []
        for role in ('fit_train', 'fit_validation', 'utility_train', 'selection', 'evaluation'):
            for c in range(4):
                for _ in range(8 if role == 'fit_train' else 1):
                    i = len(rows)
                    rows.append(dict(obs_id=i, mmsi=str(i), pass_group=role, role=role, class_id=c))
        table = pd.DataFrame(rows)
        bank = np.arange(32).reshape(4, 8)
        refs = np.broadcast_to(bank[:, :2], (len(table), 4, 2)).copy()
        for i in range(32):
            c = i//8
            refs[i, c] = bank[c][bank[c] != i][:2]
        return table, dict(obs_id=table.obs_id.to_numpy(), bank=bank, refs=refs)

    def test_evaluation_labels_not_required(self):
        table, data = self.example()
        table.loc[table.role.eq('evaluation'), 'class_id'] = -1
        groups = validate_roles(table, data)
        self.assertEqual(len(groups['evaluation']), 4)

    def test_same_ship_retrieval_rejected(self):
        table, data = self.example()
        data['refs'][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, 'own vessel'):
            validate_roles(table, data)

    def test_pass_overlap_rejected(self):
        table, data = self.example()
        table.loc[table.role.eq('selection'), 'pass_group'] = 'utility_train'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_roles(table, data)

    def test_source_constant_channel_not_activated_by_queries(self):
        x = np.array([[1, 2], [1, 4], [999, 3]], np.float32)
        sc = source_scaler(x, [0, 1])
        z = scale_input(x, sc)
        self.assertEqual(z[2, 0], 0)
        self.assertEqual(z[2, 1], 0)

    def test_gate_missing_polar_not_renormalized(self):
        gate = Gate(185, .3)
        x = torch.zeros(3, 4, 185)
        av = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])
        np.testing.assert_allclose(gate(x, av).detach(), [[.3, .7], [.3, 0], [0, .7]], atol=1e-7)

    def test_exact_j_is_oa_ba_mean(self):
        y = np.array([0, 0, 0, 1])
        score = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        self.assertAlmostEqual(float(exact_j(y, score)), (.75+(2/3+1)/2)/2)


if __name__ == '__main__':
    unittest.main()
