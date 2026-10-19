import json
import unittest
import numpy as np
import pandas as pd
import torch
from prepare import HERE, BRIDGE, load
from model import Reader, centered, residual_probability, substitute_geometry


class Tests(unittest.TestCase):
    def test_ridge_matches_existing_and_gradients(self):
        core = load('existing_core','E:/Hermes/paper_tracking/sensitivity/oof_reliability_check/unified_head_core.py')
        rng = np.random.default_rng(19)
        z = rng.normal(size=(20,9)); k = z@z.T
        p = rng.normal(size=(20,7)); e = rng.normal(size=(20,7))
        labels = np.array([0,0,1,2,2,2,4,6])
        tk = torch.tensor(k,requires_grad=True)
        prob,alpha,score = residual_probability(tk,torch.tensor(p),torch.tensor(e),torch.tensor(labels),8)
        expected = core.coefficients(k[:8,:8],p[:8],labels,8.)
        np.testing.assert_allclose(alpha.detach(),expected,atol=1e-12)
        np.testing.assert_allclose(prob.detach(),core.late((p+k[:,:8]@expected)[8:],e[8:]),atol=1e-12)
        prob[:,0].sum().backward()
        self.assertTrue(torch.isfinite(tk.grad).all())
        self.assertGreater(float(tk.grad[:8,:8].abs().sum()),0)

    def test_arms_capacity_and_late_attention(self):
        torch.manual_seed(3)
        a = Reader('A'); b = Reader('B'); c = Reader('C'); d = Reader('D')
        for m in [b,c,d]: m.load_state_dict(a.state_dict())
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in [a,b,c,d]}),1)
        content,pos,mass = torch.randn(3,2,128,256),torch.randn(3,128,8),torch.ones(3,128)
        geo,other = torch.randn(3,4),torch.randn(3,4)
        valid = torch.tensor([True,True,False])
        for m in [a,b,c,d]:
            raw,w = m(content,pos,mass,geo,valid)
            alternate,wa = m(content,pos,mass,other,valid)
            self.assertTrue(torch.equal(raw[2],torch.zeros_like(raw[2])))
            if m.arm=='A': torch.testing.assert_close(raw,alternate,atol=0,rtol=0)
            if m.arm=='B': torch.testing.assert_close(w,wa,atol=0,rtol=0)
            if m.arm=='C': self.assertGreater(float((w-wa).detach().abs().max()),1e-6)
            raw[:2].square().sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        z = centered(raw,valid,2)
        self.assertTrue(torch.equal(z[2],torch.zeros_like(z[2])))

    def test_tasks_fixed_queries_nested_and_isolated(self):
        tasks = json.loads((HERE/'tasks.json').read_text())['tasks']
        meta = pd.read_csv(HERE/'cohort.csv')
        for t in tasks:
            for s in t['supports']:
                self.assertEqual(len(s),t['budget'])
                for x,y in [(s,t['query']),(s,t['source']),(t['query'],t['source'])]:
                    self.assertFalse(set(meta.iloc[x].mmsi)&set(meta.iloc[y].mmsi))
                if t['line']=='cross_pass':
                    self.assertFalse(set(meta.iloc[s].pass_key)&set(meta.iloc[t['query']].pass_key))
            pair = next(o for o in tasks if o['target_pass']==t['target_pass'] and o['line']==t['line'] and o['budget']!=t['budget'])
            self.assertEqual(t['query'],pair['query'])
            self.assertEqual(t['source'],pair['source'])
            for s,r in zip(t['supports'],pair['supports']): self.assertEqual(s[:16],r[:16])

    def test_mismatch_preserves_tuples_unknown_and_donor_scope(self):
        g = np.arange(40,dtype=np.float32).reshape(10,4)
        valid = np.array([True]*9+[False])
        altered,receivers,donors = substitute_geometry(g,valid,np.array([0,2,4]),np.random.default_rng(8))
        np.testing.assert_array_equal(altered[9],g[9])
        np.testing.assert_array_equal(altered[receivers],g[donors])
        self.assertTrue(set(donors)<=set([0,2,4]))
        self.assertTrue(np.all(receivers!=donors))


if __name__=='__main__': unittest.main()
