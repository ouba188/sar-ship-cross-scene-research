import importlib.util
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[2]/'sar_m123_standalone'
sys.path.insert(0,str(PACKAGE))
from sar_m123.selective_matching import SelectiveReferenceMatch
from task_anchor import TaskAnchorMatch
spec=importlib.util.spec_from_file_location('anchor_test_helper',PACKAGE/'tests/test_selective_matching.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class AnchorChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        self.query=base.SelectiveReferenceMatchTests.sample(2)
        self.reference=base.SelectiveReferenceMatchTests.sample(3)
        for s in (self.query,self.reference):
            s['features']=torch.randn_like(s['anchor'])

    def call(self,block,q=None,r=None,pairs=None):
        return base.SelectiveReferenceMatchTests.call(self,block,
            self.query if q is None else q,self.reference if r is None else r,pairs)

    def test_matches_explicit_anchor_substitution_only(self):
        old=SelectiveReferenceMatch().eval()
        torch.nn.init.normal_(old.output.weight,std=.1)
        new=TaskAnchorMatch().eval();new.load_state_dict(old.state_dict())
        q=dict(self.query,anchor=self.query['features'])
        r=dict(self.reference,anchor=self.reference['features'])
        expected,ed=self.call(old,q,r)
        actual,ad=self.call(new)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(ad['reference_weights'],ed['reference_weights'],rtol=0,atol=0)
        self.assertFalse(torch.equal(self.reference['anchor'],self.reference['features']))
        _,raw=self.call(old)
        self.assertGreater(float(abs(raw['reference_weights']-ad['reference_weights']).max()),0)

    def test_comparison_features_detached_but_selector_learns(self):
        block=TaskAnchorMatch().eval()
        q=dict(self.query,features=self.query['features'].clone().requires_grad_(True))
        r=dict(self.reference,features=self.reference['features'].clone().requires_grad_(True))
        _,d=self.call(block,q,r)
        term=d['reference_logits'][:,0].sum()
        term.backward()
        self.assertIsNone(q['features'].grad)
        self.assertIsNone(r['features'].grad)
        self.assertGreater(float(block.branch_bias.grad.norm()),0)

    def test_no_reference_no_update(self):
        block=TaskAnchorMatch().eval()
        torch.nn.init.normal_(block.output.weight,std=.1)
        z,d=self.call(block,pairs=torch.zeros(2,3,dtype=torch.bool))
        torch.testing.assert_close(z,self.query['features'],atol=0,rtol=0)
        self.assertEqual(int(d['update'].count_nonzero()),0)


if __name__=='__main__':unittest.main(verbosity=2)
