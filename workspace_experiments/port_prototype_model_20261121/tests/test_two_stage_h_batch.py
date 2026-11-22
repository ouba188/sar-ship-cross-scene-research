"""Independent legacy-pair equivalence for grouped matching, including gradients."""
from copy import deepcopy
import unittest
import numpy as np
import torch
from sar_port.two_stage.h import HConfig, HModel, partial_transport, partial_transport_batched
from test_two_stage_h import example


def compare_tree(case, expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    elif isinstance(expected, dict):
        case.assertEqual(list(expected), list(actual))
        for key in expected: compare_tree(case, expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        case.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual): compare_tree(case, before, after)
    else:
        case.assertEqual(actual, expected)


class BatchedMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_transport_forward_and_score_dustbin_gradients(self):
        torch.manual_seed(26)
        for n, m in ((1,1), (1,4), (2,3), (4,2)):
            scores = torch.randn(5,n,m, requires_grad=True)
            valid = torch.rand(5,n,m) > .3; valid[0] = False
            a = torch.rand(5,n).softmax(-1); b = torch.rand(5,m).softmax(-1)
            dust = torch.tensor(.04, requires_grad=True)
            old = torch.stack([partial_transport(scores[i], valid[i], a[i], b[i], dust, iterations=60)
                               for i in range(5)])
            new = partial_transport_batched(scores, valid, a, b, dust, iterations=60)
            weight = torch.randn_like(old)
            old_grad = torch.autograd.grad((old*weight).sum(), (scores,dust), retain_graph=True)
            new_grad = torch.autograd.grad((new*weight).sum(), (scores,dust))
            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
            for before, after in zip(old_grad,new_grad):
                torch.testing.assert_close(after,before,rtol=3e-5,atol=3e-6)

    def test_full_forward_and_all_parameter_gradients_for_ragged_pairs(self):
        for adjacent in (True, False):
            torch.manual_seed(81)
            old = HModel(HConfig(3, descriptor_dim=12, sinkhorn_iterations=60,
                                 adjacent_scales=adjacent, batched_matching=False)).set_stage('match')
            # Nonzero scorers exercise both projection-dependent weights and learned mixtures.
            with torch.no_grad():
                old.edge_score[-1].weight.normal_(0,.15)
                old.prototype_score[-1].weight.normal_(0,.15)
                old.attention.score[-1].weight.normal_(0,.15)
            new = deepcopy(old); new.config.batched_matching = True; new.config.pair_batch_size = 2
            queries = [example(1,4,gradients=True),example(2,5,missing_vh=True,gradients=True),
                       example(4,6,gradients=True),example(2,11,gradients=True)]
            refs = [example(1,7,gradients=True),example(2,8,gradients=True),
                    example(4,9,missing_vh=True,gradients=True),example(3,10,gradients=True),
                    example(2,12,gradients=True)]
            # A reference with no usable local evidence still has ordinary whole-ship features.
            for level in refs[-1].geometry.levels: level.match_valid.zero_()
            queries[-1].geometry.axis_valid = False
            newqueries, newrefs = deepcopy(queries), deepcopy(refs)
            labels = torch.tensor([0,0,1,2,1]); ports = ['a','a','b','c','a']
            legal = torch.ones(4,5,dtype=torch.bool); legal[1,0] = False
            before = old(queries,refs,labels,ports,legal)
            after = new(newqueries,newrefs,labels,ports,legal)
            compare_tree(self,before,after)
            targets = torch.tensor([0,1,2,1])
            old.loss(before,targets)['total'].backward()
            new.loss(after,targets)['total'].backward()
            for (name,p), (newname,q) in zip(old.named_parameters(),new.named_parameters()):
                self.assertEqual(name,newname)
                self.assertEqual(p.grad is None,q.grad is None,name)
                if p.grad is not None:
                    torch.testing.assert_close(q.grad,p.grad,rtol=2e-4,atol=5e-6,msg=name)
            for before_ship, after_ship in zip(queries+refs,newqueries+newrefs):
                for x,y in zip(before_ship.spatial,after_ship.spatial):
                    self.assertEqual(x.grad is None,y.grad is None)
                    if x.grad is not None: torch.testing.assert_close(y.grad,x.grad,rtol=2e-4,atol=5e-6)

    def test_orientation_identification_reversal_and_exact_tie_prefer_plus(self):
        torch.manual_seed(31)
        model = HModel(HConfig(2,descriptor_dim=8,sinkhorn_iterations=60)).set_stage('match')
        query, ref = example(3,7), example(3,9)
        # Compact separated intervals distinguish the two orientation costs.
        for ship in (query,ref): ship.geometry.intervals[:,1] = .1
        q = torch.zeros(3,2,3,8); r = torch.zeros_like(q)
        for i in range(3):
            q[i,:,:,i] = 1; r[2-i,:,:,i] = 1
        valid = torch.ones(3,2,3,dtype=torch.bool)
        qdesc,rdesc = (q,valid),(r,valid)
        before = model.match_pair(query,ref,qdesc,rdesc)
        actual = model._match_all_batched([query],[ref],[qdesc],[rdesc],torch.ones(1,1,dtype=torch.bool))[0][0]
        compare_tree(self,before,actual)
        self.assertEqual(actual['direction'],-1)
        self.assertEqual(actual['direction_solves'],2)
        # A single legal central edge gives exactly equal qualities for differing costs.
        center = torch.zeros_like(valid); center[1] = True
        qdesc,rdesc = (q,center),(r,center)
        before = model.match_pair(query,ref,qdesc,rdesc)
        actual = model._match_all_batched([query],[ref],[qdesc],[rdesc],torch.ones(1,1,dtype=torch.bool))[0][0]
        compare_tree(self,before,actual)
        self.assertEqual(actual['direction'],1)
        self.assertTrue(actual['direction_identifiable'])
        # No edge also ties at zero evidence without inventing support.
        zero = torch.zeros_like(valid)
        actual = model._match_all_batched([query],[ref],[(q,zero)],[(r,zero)],
                                          torch.ones(1,1,dtype=torch.bool))[0][0]
        self.assertEqual(actual['direction'],1)
        self.assertFalse(actual['valid'].any())
        self.assertEqual(float(actual['evidence'].sum().detach()),0.)

    def test_chunking_and_initialization_do_not_change_parameter_layout_or_rng(self):
        torch.manual_seed(19)
        old = HModel(HConfig(2,descriptor_dim=8,batched_matching=False))
        old_next = torch.rand(3)
        torch.manual_seed(19)
        new = HModel(HConfig(2,descriptor_dim=8,batched_matching=True,pair_batch_size=1))
        new_next = torch.rand(3)
        compare_tree(self,old.state_dict(),new.state_dict())
        torch.testing.assert_close(old_next,new_next,rtol=0,atol=0)


if __name__ == '__main__': unittest.main()
