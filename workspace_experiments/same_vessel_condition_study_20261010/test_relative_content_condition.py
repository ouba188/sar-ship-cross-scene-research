"""Acquisition weighting, factor targets, and actual operator capability."""

import unittest
import numpy as np
import pandas as pd
import torch
from relative_content_condition import vessel_targets, auxiliary_objective, BoundedOperator


class RelativeContentTests(unittest.TestCase):
    def test_group_mean_equal_acquisition_not_chip_weight(self):
        x = np.array([[0.], [0.], [9.], [50.]], np.float32)
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2], product=['a', 'a', 'b', 'c']))
        shared, other, valid = vessel_targets(x, meta)
        np.testing.assert_array_equal(valid, [True, True, True, False])
        np.testing.assert_allclose(shared[:3, 0], [4.5, 4.5, 4.5])
        np.testing.assert_allclose(other[:3, 0], [9., 9., 0.])

    def test_group_loss_blocks_compensating_wrong_parts(self):
        s = torch.tensor([[8.]], requires_grad=True)
        c = torch.tensor([[-2.]], requires_grad=True)
        x, shared = torch.tensor([[6.]]), torch.tensor([[4.]])
        self.assertEqual(float(auxiliary_objective(s, c, x, shared, 'self_aux')), 0.)
        loss = auxiliary_objective(s, c, x, shared, 'group_aux')
        self.assertEqual(float(loss), 32.)
        loss.backward()
        self.assertGreater(float(s.grad), 0.)
        self.assertLess(float(c.grad), 0.)

    def test_same_parameter_count_and_zero_initial_identity(self):
        h, c = torch.randn(3, 8), torch.randn(3, 32)
        for op in ('scale', 'residual'):
            model = BoundedOperator(torch.eye(8), torch.ones(8), op)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 264)
            torch.testing.assert_close(model(h, c), h, rtol=0, atol=0)

    def test_additive_can_change_zero_projection_and_is_bounded(self):
        h, c = torch.zeros(1, 8), torch.zeros(1, 32)
        models = [BoundedOperator(torch.eye(8), 2*torch.ones(8), op) for op in ('scale', 'residual')]
        for m in models:
            with torch.no_grad():
                m.map.bias.fill_(10.)
        torch.testing.assert_close(models[0](h, c), h)
        out = models[1](h, c)
        self.assertTrue(bool((out > .19).all()))
        self.assertTrue(bool((out.abs() <= .2).all()))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
