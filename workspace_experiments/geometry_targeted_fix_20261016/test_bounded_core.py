import unittest
import numpy as np
import torch
from bounded_core import bounded_features,bounded_predict,OptionalGeometryReader,relation_loss


class BoundedTests(unittest.TestCase):
    def test_gram_bounded_and_tiny_not_amplified(self):
        torch.manual_seed(1)
        for factor in (1e-7,1.,1e5):
            x=torch.randn(9,12,dtype=torch.float64)*factor
            h=bounded_features(x,torch.ones(9,dtype=torch.bool),4)
            self.assertLessEqual(float((.75*h@h.T).abs().max()),.75+1e-12)
            if factor<1:
                self.assertLess(float(h.norm()),1e-4)

    def test_constant_zero_and_query_batch_independent(self):
        v=torch.ones(9,dtype=torch.bool)
        self.assertEqual(float(bounded_features(torch.ones(9,12),v,4).abs().max()),0.)
        x=torch.randn(9,12)
        torch.testing.assert_close(bounded_features(x,v,4)[:6],bounded_features(x[:6],v[:6],4))

    def test_masked_conditions_not_fabricated_zero_angle(self):
        g=torch.tensor([[.5,float('nan'),float('nan'),1,0,0,.8]])
        a=OptionalGeometryReader.angular(g)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(float(a[:,2:].abs().max()),0.)

    def test_source_relation_gradient_and_missing_class_skip(self):
        torch.manual_seed(3)
        x=torch.randn(8,12,dtype=torch.float64,requires_grad=True)
        v=torch.ones(8,dtype=torch.bool)
        h=bounded_features(x,v,4)
        loss,n=relation_loss(h,torch.tensor([0,0,1,1]),torch.tensor([0,1,2,2]),v,torch.ones(4,8,dtype=torch.float64)/8)
        self.assertEqual(n,2)
        loss.backward()
        self.assertGreater(float(x.grad[:4].norm()),0.)
        self.assertGreater(float(x.grad[4:6].norm()),0.)
        self.assertEqual(float(x.grad[6:].norm()),0.)

    def test_full_solver_numpy_replay_gradient_and_fallback(self):
        from run_fix import _coefficients
        torch.manual_seed(7)
        ns,nq=8,3
        f=torch.randn(ns+nq,6,dtype=torch.float64)
        sy=torch.tensor([0,0,0,1,1,2,3,3])
        item=dict(s=torch.arange(ns),ss=f[:ns]@f[:ns].T,qs=f[ns:]@f[:ns].T,
                  sy=sy,prior_s=torch.randn(ns,8,dtype=torch.float64),
                  prior_q=torch.randn(nq,8,dtype=torch.float64),
                  base_prob=torch.randn(nq,8,dtype=torch.float64).softmax(-1),
                  old_prob=torch.randn(nq,8,dtype=torch.float64).softmax(-1))
        raw=torch.randn(ns+nq,12,dtype=torch.float64,requires_grad=True)
        valid=torch.ones(ns+nq,dtype=torch.bool); valid[-1]=False
        p,_,h=bounded_predict(item,torch.arange(nq),raw,valid,_coefficients)
        a=raw.detach().numpy(); a=a-a[:ns].mean(0); a[~valid.numpy()]=0
        a=a/np.sqrt(1+(a*a).sum(1,keepdims=True))
        ss=item['ss'].numpy()+.75*a[:ns]@a[:ns].T
        qs=item['qs'].numpy()+.75*a[ns:]@a[:ns].T
        labels=sy.numpy(); counts=np.bincount(labels,minlength=8)
        w=np.sqrt(ns/(4*counts[labels]))
        residual=2*np.eye(8)[labels]-1-item['prior_s'].numpy()
        coef=w[:,None]*np.linalg.solve(w[:,None]*ss*w[None,:]+8*np.eye(ns),w[:,None]*residual)
        scores=item['prior_q'].numpy()+qs@coef
        exp=np.exp(scores-scores.max(1,keepdims=True))
        expected=.5*(item['base_prob'].numpy()+exp/exp.sum(1,keepdims=True))
        expected[-1]=item['old_prob'][-1].numpy()
        np.testing.assert_allclose(p.detach().numpy(),expected,atol=1e-12)
        (-p[0,1].log()).backward()
        self.assertGreater(float(raw.grad.norm()),0.)
        valid[:ns]=False; valid[0]=True
        fallback,_,_=bounded_predict(item,torch.arange(nq),raw,valid,_coefficients)
        torch.testing.assert_close(fallback,item['old_prob'],rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
