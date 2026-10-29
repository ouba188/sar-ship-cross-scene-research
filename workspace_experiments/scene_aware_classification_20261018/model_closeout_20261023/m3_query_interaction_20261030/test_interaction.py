import importlib.util
from pathlib import Path
import sys
import unittest
import torch

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[2]/'sar_m123_standalone'
sys.path.insert(0,str(PACKAGE))
from interaction import QueryInteractionMatch
from sar_m123.selective_matching import SelectiveReferenceMatch

spec=importlib.util.spec_from_file_location('base_matching_tests',PACKAGE/'tests/test_selective_matching.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


class InteractionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(1)
        self.query=base.SelectiveReferenceMatchTests.sample(2)
        self.ref=base.SelectiveReferenceMatchTests.sample(3)

    def call(self, block, q=None, r=None, pairs=None):
        return base.SelectiveReferenceMatchTests.call(self,block,
            self.query if q is None else q,self.ref if r is None else r,pairs)

    def test_zero_start_preserves_old_nonzero_writer_and_gets_gradient(self):
        old=SelectiveReferenceMatch(channels=8).eval().requires_grad_(False)
        torch.nn.init.normal_(old.output.weight,std=.1)
        expected,_=self.call(old)
        for mode in ('query_only','ref_only','interaction'):
            new=QueryInteractionMatch(mode,channels=8).eval()
            new.load_state_dict(old.state_dict(),strict=False)
            new.requires_grad_(False);new.extra_output.requires_grad_(True)
            actual,d=self.call(new)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            (actual*torch.randn_like(actual)).sum().backward()
            self.assertGreater(float(new.extra_output.weight.grad.norm()),0)
            self.assertIsNone(new.project.weight.grad)

    def test_query_and_reference_content_dependencies(self):
        q=dict(self.query,features=self.query['features'].roll(1,1))
        r=dict(self.ref,features=self.ref['features'].roll(1,1))
        for mode in ('query_only','ref_only','interaction'):
            block=QueryInteractionMatch(mode,channels=8).eval().requires_grad_(False)
            torch.nn.init.normal_(block.extra_output.weight,std=.1)
            _,original=self.call(block)
            _,qchange=self.call(block,q=q)
            _,rchange=self.call(block,r=r)
            dq=(qchange['extra_update']-original['extra_update']).abs().max().item()
            dr=(rchange['extra_update']-original['extra_update']).abs().max().item()
            self.assertEqual(dq==0,mode=='ref_only')
            self.assertEqual(dr==0,mode=='query_only')

    def test_no_legal_reference_adds_nothing(self):
        pairs=torch.zeros(2,3,dtype=torch.bool)
        for mode in ('query_only','ref_only','interaction'):
            block=QueryInteractionMatch(mode,channels=8).eval().requires_grad_(False)
            torch.nn.init.normal_(block.extra_output.weight,std=.1)
            for control in ('normal','uniform'):
                block.extra_control=control
                _,d=self.call(block,pairs=pairs)
                self.assertTrue(torch.isfinite(d['extra_update']).all())
                self.assertEqual(int(d['extra_update'].count_nonzero()),0)

    def test_reference_order_equivariance(self):
        block=QueryInteractionMatch(channels=8).eval().requires_grad_(False)
        torch.nn.init.normal_(block.extra_output.weight,std=.1)
        r={k:v.flip(0) for k,v in self.ref.items()}
        for control in ('normal','uniform'):
            block.extra_control=control
            a,_=self.call(block);b,_=self.call(block,r=r)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)


if __name__=='__main__':unittest.main(verbosity=2)
