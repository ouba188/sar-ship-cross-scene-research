import unittest
import numpy as np
import pandas as pd
import torch
from conditional_evidence_gate import EvidenceGate, source_partitions


class GateChecks(unittest.TestCase):
    def test_identity_zero_evidence_and_bound(self):
        torch.manual_seed(17)
        basis = torch.linalg.qr(torch.randn(512, 8)).Q
        model = EvidenceGate(basis)
        h, c = torch.randn(40, 512), torch.randn(40, 32)
        torch.testing.assert_close(model(h, c), h, rtol=0, atol=0)
        with torch.no_grad():
            model.map.weight.normal_(std=5.)
            model.map.bias.normal_()
        torch.testing.assert_close(model(torch.zeros_like(h), c), torch.zeros_like(h), rtol=0, atol=0)
        ratio = torch.linalg.norm(model(h, c)-h, dim=1)/torch.linalg.norm(h, dim=1)
        self.assertLessEqual(float(ratio.max().detach()), .10001)
        # A component orthogonal to the classifier directions is unchanged.
        residual = h-(h@basis)@basis.T
        torch.testing.assert_close(model(residual, c), residual, rtol=1e-5, atol=1e-6)

    def test_validation_identity_and_observation_isolation(self):
        meta = pd.DataFrame([dict(mmsi=ship, class_id=ship % 8, product=f'p{j}')
            for ship in range(80) for j in range(3)])
        roles = source_partitions(meta, np.arange(len(meta)))
        train, identity, time = [roles[k] for k in ('fit', 'val_new_ship', 'val_new_time')]
        self.assertFalse(set(meta.iloc[train].mmsi)&set(meta.iloc[identity].mmsi))
        self.assertFalse(set(train)&set(time))
        self.assertEqual(set(meta.iloc[train].mmsi), set(meta.iloc[time].mmsi))


if __name__ == '__main__':
    unittest.main()
