import unittest
import numpy as np
import pandas as pd
import torch
from relational_factors import FactorNet, relation_pools, sample_triplets, design
from target_response_adapter import residual_head


class RelationTests(unittest.TestCase):
    def test_pair_semantics(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 2, 3, 3, 4, 4], class_id=[0, 0, 0, 0, 1, 1, 1, 1],
                                 product=['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']))
        pairs, pools = relation_pools(meta)
        for a, b in pairs:
            self.assertEqual(meta.iloc[a].mmsi, meta.iloc[b].mmsi)
            self.assertNotEqual(meta.iloc[a]['product'], meta.iloc[b]['product'])
        for kind in ('semantic', 'scene'):
            for a, p, n in sample_triplets(pools[kind], np.random.default_rng(1)):
                aa, pp, nn = [meta.iloc[i] for i in (a, p, n)]
                self.assertNotEqual(aa.mmsi, pp.mmsi)
                self.assertNotEqual(aa.mmsi, nn.mmsi)
                self.assertEqual(aa.class_id == pp.class_id, kind == 'semantic')
                self.assertEqual(aa['product'] == pp['product'], kind == 'scene')
                self.assertEqual(aa.class_id == nn.class_id, kind == 'scene')
                self.assertNotEqual(aa['product'], nn['product'])

    def test_correction_bound_and_both_factor_gradients(self):
        model = FactorNet()
        s, d, c, r, rec = model(torch.randn(5, 512))
        self.assertEqual(tuple(d.shape), (5, 32))
        self.assertLessEqual(float(c.abs().max().detach()), .1)
        torch.testing.assert_close(r, s+c, atol=0, rtol=0)
        (r.square().mean()+rec.square().mean()).backward()
        self.assertGreater(float(model.stable[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.condition[0].weight.grad.norm()), 0)

    def test_generic_readout_and_absent_class(self):
        x = torch.randn(8, 128)
        state = [torch.zeros(128), torch.ones(128), torch.randn(128, 8), torch.randn(8)]
        ph, prior = design(x, state)
        beta, _, _ = residual_head(ph, prior, torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]), balanced=True)
        self.assertEqual(tuple(ph.shape), (8, 129))
        torch.testing.assert_close(beta[:, 2:], torch.zeros_like(beta[:, 2:]), atol=0, rtol=0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
