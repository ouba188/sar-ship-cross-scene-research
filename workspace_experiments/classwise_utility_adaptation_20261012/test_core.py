import unittest
import numpy as np
from scipy.special import softmax
from core import loss_grad, fit_gate, support_oof


class CoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.p = rng.normal(size=(24,4))
        self.d = rng.normal(size=(24,4))
        self.b = softmax(rng.normal(size=(24,4)),axis=1)
        self.y = np.repeat(np.arange(4),[9,7,5,3])

    def test_gradient_matches_finite_difference(self):
        g = np.array([.7,1.2,.5,1.1])
        _, grad = loss_grad(g,self.p,self.d,self.b,self.y,.01)
        delta = np.eye(4)*1e-6
        numerical = np.array([(loss_grad(g+h,self.p,self.d,self.b,self.y,.01)[0]-
                              loss_grad(g-h,self.p,self.d,self.b,self.y,.01)[0])/2e-6 for h in delta])
        np.testing.assert_allclose(grad,numerical,atol=1e-9,rtol=1e-6)

    def test_identity_risk_and_joint_improvement(self):
        truth = np.eye(4)[self.y]
        full = .5*(self.b+softmax(self.p+self.d,axis=1))
        expected = np.mean([np.square(full[self.y==c]-truth[self.y==c]).sum(1).mean() for c in range(4)])
        initial = loss_grad(np.ones(4),self.p,self.d,self.b,self.y,.01)[0]
        self.assertAlmostEqual(initial,expected)
        for mode in ('joint','global'):
            g,_ = fit_gate(self.p,self.d,self.b,self.y,.01,mode)
            self.assertTrue(((g>=0)&(g<=2)).all())
            self.assertLessEqual(loss_grad(g,self.p,self.d,self.b,self.y,.01)[0],initial+1e-10)

    def test_class_permutation(self):
        perm = np.array([2,0,3,1])
        mapped = np.argsort(perm)[self.y]
        g,_ = fit_gate(self.p,self.d,self.b,self.y,.01,'joint')
        h,_ = fit_gate(self.p[:,perm],self.d[:,perm],self.b[:,perm],mapped,.01,'joint')
        np.testing.assert_allclose(h,g[perm],atol=1e-7)

    def test_no_residual_no_gate_change(self):
        for mode in ('joint','global','self'):
            g,_ = fit_gate(self.p,np.zeros_like(self.d),self.b,self.y,.01,mode)
            np.testing.assert_allclose(g,1.)

    def test_oof_holds_out_whole_groups(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(24,5))
        k = x@x.T
        groups = np.repeat(np.arange(12),2)
        def direct(kernel,prior,y,strength):
            # Independent unweighted solver is sufficient to test data flow.
            return np.linalg.solve(kernel+strength*np.eye(len(y)),2*np.eye(4)[y]-1-prior)
        result,yy,folds,_ = support_oof(k,self.p,k,self.p,self.y,groups,direct)
        self.assertEqual(result[0].shape,(72,4))
        for part in folds:
            self.assertFalse(set(groups[part['train']]) & set(groups[part['validation']]))
        np.testing.assert_array_equal(yy,np.tile(self.y,3))


if __name__=='__main__':
    unittest.main()
