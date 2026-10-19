import unittest
import numpy as np
import torch
from conv_model import ConvReader,region_pool,scramble_cells
from run_probe import matched_drop


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(2)
        self.maps=torch.randn(3,2,256,32,32)
        f=torch.rand(3,1,32,32)
        self.masks=torch.cat([f,1-f],1)
        self.position=torch.randn(3,128,8)
        self.geo=torch.zeros(3,4)
        self.valid=torch.tensor([True,True,False])
        self.permutation=torch.randperm(16)

    def test_local_permutation_preserves_both_region_means(self):
        changed,mask=scramble_cells(self.maps,self.masks,self.permutation)
        a,m=region_pool(self.maps,self.masks)
        b,n=region_pool(changed,mask)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(m,n,atol=1e-7,rtol=1e-5)
        for mode in ['pool_first','point_first']:
            reader=ConvReader(mode)
            a,_=reader(self.maps,self.masks,self.position,self.geo,self.valid)
            b,_=reader(changed,mask,self.position,self.geo,self.valid)
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    def test_zero_spatial_start_and_active_spatial_gradient(self):
        point=ConvReader('point_first');conv=ConvReader('conv_first')
        conv.load_state_dict(point.state_dict(),strict=False)
        a,_=point(self.maps,self.masks,self.position,self.geo,self.valid)
        b,_=conv(self.maps,self.masks,self.position,self.geo,self.valid)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        b[:2].square().sum().backward()
        self.assertGreater(float(conv.spatial.weight.grad.abs().sum()),0)
        self.assertTrue(torch.equal(b[2],torch.zeros_like(b[2])))

    def test_matched_area_drop(self):
        w=np.zeros((1,2,2,64));m=np.zeros((1,2,64))
        m[:,:,:3]=[.4,.41,.8]
        w[..., :3]=[.5,.1,.4]
        high,low=matched_drop(w,m,np.array([True]))
        self.assertEqual(int(high.sum()),4)
        self.assertTrue(high[:,:,:,0].all())
        self.assertTrue(low[:,:,:,1].all())
        self.assertFalse((high&low).any())


if __name__=='__main__':unittest.main()
