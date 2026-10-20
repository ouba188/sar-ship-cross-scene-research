import unittest
import numpy as np
import pandas as pd
import torch
from run import relation_moments, standard_block, matrices


class Relations(unittest.TestCase):
    def example(self):
        x=torch.zeros(1,2,4,2,3,3,2)
        m=torch.zeros(1,2,4,2,3,3,2)
        x[...,0,:,0]=torch.tensor([1.,0.,0.])
        x[...,1:,:,0]=torch.tensor([1.,0.,0.])
        m[...,0,:,0]=1; m[...,1:,:,1]=1
        return x,m

    def test_joint_differs_but_marginals_match(self):
        x,m=self.example(); a=relation_moments(x,m)
        xx=x.clone(); xx[...,1:,:,:]=xx[...,1:,:,:].flip(-2)
        b=relation_moments(xx,m)
        torch.testing.assert_close(a[0],b[0]); torch.testing.assert_close(a[1],b[1])
        self.assertGreater(float((a[2]-b[2]).abs().max()),.3)
        self.assertAlmostEqual(float(a[2][0,0,0,0]),1/3,places=6)

    def test_common_reversal_and_side_exchange(self):
        x,m=self.example(); a=relation_moments(x,m)
        mixed=relation_moments(x,m.half())
        b=relation_moments(x.flip(-2),m.flip(-2))
        c=relation_moments(x.index_select(-3,torch.tensor([0,2,1])),m.index_select(-3,torch.tensor([0,2,1])))
        for j in range(4):
            torch.testing.assert_close(a[j],mixed[j])
            torch.testing.assert_close(a[j],b[j]); torch.testing.assert_close(a[j],c[j])

    def test_no_pair_and_mask_weight(self):
        x,m=self.example(); z=relation_moments(x,m*0)
        for v in z:
            self.assertEqual(float(v.abs().sum()) if v.dtype!=torch.bool else int(v.sum()),0)
        m[...,0,:,0]=torch.tensor([.5,1.,0.])
        joint=relation_moments(x,m)[2]
        self.assertAlmostEqual(float(joint[0,0,0,0]),1/3,places=6)

    def test_source_only_standard_and_missing(self):
        x=np.arange(6*2*2*2,dtype=np.float32).reshape(6,2,2,2)
        present=np.ones((6,2,2),bool); present[4]=False
        a=standard_block(x,present,np.arange(3)); changed=x.copy(); changed[5]*=100
        b=standard_block(changed,present,np.arange(3))
        np.testing.assert_allclose(a[:3],b[:3]); np.testing.assert_array_equal(a[4],0)

    def test_missing_condition_not_created_by_centering(self):
        rng=np.random.default_rng(4)
        data={k:rng.normal(size=(6,2,2,2)).astype(np.float32) for k in ('marginal','factorized','joint')}
        data['present']=np.ones((6,2,2),bool)
        data['condition']=rng.normal(size=(6,2,3)).astype(np.float32)
        data['condition_valid']=np.ones((6,2),bool); data['condition_valid'][5,1]=False
        f=pd.DataFrame({'port':['A']*6,'platform':['S1A']*6,'quarter':[3]*6})
        z,_=matrices(data,f,(np.arange(3),np.array([3]),np.array([4,5])),0)
        block=z['joint_condition'].reshape(6,2,2,8)
        np.testing.assert_array_equal(block[5,1,:,2:],0)


if __name__=='__main__':
    unittest.main()
