import unittest
import numpy as np
import torch
from objective import make_interventions, select_targets, auxiliary


class ObjectiveTests(unittest.TestCase):
    def test_selection_uses_new_rival_and_level_control(self):
        y=torch.tensor([0,0,0])
        base=torch.tensor([[0.,1.,-2.]]*3)
        # First: genuine gain. Second: original rival suppressed, new rival worse.
        # Third: an intensity-only change explains the apparent improvement.
        changed=torch.tensor([[[.4,.5,-2.],[.4,-1.,2.],[.4,.5,-2.]]])
        level=torch.tensor([[[0.,1.,-2.],[0.,1.,-2.],[.8,.5,-2.]]])
        t=select_targets(base,changed,level,y,torch.ones(1,3,dtype=torch.bool))
        self.assertEqual(t['use'].tolist(),[True,False,False])
        self.assertLessEqual(float(t['relation_gain'].max()),.100001)

    def test_locality_and_missing_polarization(self):
        rng=np.random.default_rng(7)
        x=np.zeros((1,2,1,20,20),np.float32)
        x[:,:,0,3:17,3:17]=rng.uniform(.1,.9,(1,2,14,14))
        valid=(x>0).astype(np.float32)
        masks=np.zeros((1,2,3,20,20),bool)
        masks[:,:,0,7:13,7:13]=True
        arr,level,ok,meta=make_interventions(x,valid,masks,np.array([[1,0]]))
        self.assertTrue(ok[:2].all())
        self.assertFalse(ok[6:].any())
        np.testing.assert_array_equal(arr[0,:,1],x[:,1])
        delta=arr[0,0,0,0]-x[0,0,0]
        self.assertEqual(float(abs(delta[~masks[0,0,0]]).max()),0.)
        self.assertLess(meta[0][0]['mean_control_error'],1e-5)

    def test_detached_targets_and_inactive_loss(self):
        y=torch.tensor([0])
        base=torch.tensor([[0.,1.,-2.]],requires_grad=True)
        changed=torch.tensor([[[.4,.5,-2.]]],requires_grad=True)
        target=select_targets(base,changed,base.detach()[None],y,torch.ones(1,1,dtype=torch.bool))
        on=base.detach().clone().requires_grad_()
        off=base.detach().clone().requires_grad_()
        loss,_=auxiliary(on,off,y,torch.ones(1),target,torch.ones(1,dtype=torch.bool))
        loss.backward()
        self.assertLess(float(on.grad[0,0]),0)
        self.assertIsNone(off.grad)
        self.assertIsNone(base.grad)
        self.assertIsNone(changed.grad)
        loss,_=auxiliary(on,off,y,torch.ones(1),target,torch.zeros(1,dtype=torch.bool))
        self.assertEqual(float(loss),0.)

    def test_confident_correct_is_not_a_gain_target(self):
        y=torch.tensor([0])
        base=torch.tensor([[2.,0.,-2.]])
        target=select_targets(base,torch.tensor([[[3.,0.,-2.]]]),base[None],y,
                              torch.ones(1,1,dtype=torch.bool))
        self.assertTrue(bool(target['use'][0]))
        loss,stats=auxiliary(base,base,y,torch.ones(1),target,torch.ones(1,dtype=torch.bool))
        self.assertEqual(stats['selected'],0)
        self.assertEqual(float(loss),0.)


if __name__ == '__main__':
    unittest.main()
