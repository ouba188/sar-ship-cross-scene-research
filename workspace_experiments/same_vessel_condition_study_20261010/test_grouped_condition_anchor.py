"""Check real failure modes: target-content leakage and missing-label gradients."""

import unittest
import numpy as np
import pandas as pd
import torch
from grouped_condition_anchor import swap_inputs, anchor_loss, metadata_targets


class GroupedAnchorTests(unittest.TestCase):
    def test_swap_excludes_target_content(self):
        stable = torch.tensor([[10.], [20.], [30.], [40.]], requires_grad=True)
        condition = torch.tensor([[1.], [2.], [3.], [4.]], requires_grad=True)
        result = swap_inputs(stable, condition)
        torch.testing.assert_close(result, torch.tensor([[20., 1.], [10., 2.], [40., 3.], [30., 4.]]))
        result[0].sum().backward()
        self.assertEqual(float(stable.grad[0]), 0.)
        self.assertEqual(float(stable.grad[1]), 1.)
        self.assertEqual(float(condition.grad[0]), 1.)
        self.assertEqual(float(condition.grad[1]), 0.)

    def test_all_missing_is_zero_with_zero_gradient(self):
        pred = torch.randn(4, 3, requires_grad=True)
        loss = anchor_loss(pred, torch.zeros(4), torch.zeros(4, dtype=torch.bool),
                           torch.full((4,), -1), torch.ones(2))
        loss.backward()
        self.assertEqual(float(loss), 0.)
        self.assertEqual(float(pred.grad.abs().sum()), 0.)

    def test_invalid_angle_does_not_contribute(self):
        pred = torch.zeros(3, 3, requires_grad=True)
        loss = anchor_loss(pred, torch.tensor([1., 999., 999.]), torch.tensor([True, False, False]),
                           torch.tensor([-1, 0, -1]), torch.ones(2))
        loss.backward()
        self.assertNotEqual(float(pred.grad[0, 0]), 0.)
        self.assertEqual(float(pred.grad[1, 0]), 0.)
        self.assertEqual(float(pred.grad[2].abs().sum()), 0.)

    def test_metadata_statistics_use_source_fit_only(self):
        meta = pd.DataFrame(dict(incidence_deg=[30., 40., 1000., np.nan],
                                 pass_name=['Ascending', 'Descending', 'Ascending', None]))
        angle, valid, orbit, weights, stats = metadata_targets(meta, np.array([0, 1]))
        self.assertEqual(stats['angle_center'], 35.)
        self.assertEqual(stats['angle_scale'], 5.)
        self.assertEqual(int(orbit[-1]), -1)
        self.assertFalse(bool(valid[-1]))
        torch.testing.assert_close(weights, torch.ones(2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
