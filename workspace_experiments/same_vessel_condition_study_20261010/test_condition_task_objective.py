import unittest
import numpy as np
import pandas as pd
import torch
from torch import nn
from condition_task_objective import ConditionResidual, identity_cv


class ObjectiveChecks(unittest.TestCase):
    def make_model(self):
        torch.manual_seed(3)
        return ConditionResidual(nn.Linear(5, 32), np.zeros(32), np.ones(32))

    def test_zero_residual_preserves_baseline_and_reversal(self):
        model = self.make_model()
        za, zb, sa, sb = torch.randn(4, 32), torch.randn(4, 32), torch.randn(4, 10), torch.randn(4, 10)
        torch.testing.assert_close(model.pair_residual(za, zb, sa, sb), torch.zeros(4, 2))
        nn.init.normal_(model.signed_head.weight)
        nn.init.normal_(model.magnitude_head.weight)
        p, r = model.pair_residual(za, zb, sa, sb), model.pair_residual(zb, za, sb, sa)
        torch.testing.assert_close(p[:, 0], -r[:, 0])
        torch.testing.assert_close(p[:, 1], r[:, 1])

    def test_only_task_arm_sends_error_gradient_to_encoder(self):
        model = self.make_model()
        nn.init.normal_(model.signed_head.weight)
        state = torch.randn(6, 10)
        for detach in (True, False):
            model.zero_grad(set_to_none=True)
            z = model.condition(torch.randn(6, 5))
            pred = model.pair_residual(z[:3], z[3:], state[:3], state[3:], detach=detach)
            pred.square().sum().backward()
            grad = model.condition.weight.grad
            if detach:
                self.assertIsNone(grad)
            else:
                self.assertGreater(float(grad.abs().sum()), 0.)

    def test_cv_assignment_is_by_ship(self):
        meta = pd.DataFrame(dict(mmsi=np.repeat(np.arange(18), 2), class_id=np.repeat(np.arange(18) % 3, 2)))
        pairs = pd.DataFrame(dict(mmsi=np.repeat(np.arange(18), 4)))
        partition, mapping = identity_cv(meta, pairs)
        self.assertEqual(set(partition), {0, 1, 2})
        self.assertTrue(all(len(set(partition[pairs.mmsi == ship])) == 1 for ship in mapping))


if __name__ == '__main__':
    unittest.main()
