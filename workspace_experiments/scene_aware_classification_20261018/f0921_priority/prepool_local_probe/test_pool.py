import unittest
import torch
from extract_local import spatial_grid,ownership_masks,pool


class PoolTests(unittest.TestCase):
    def test_all_pixels_partition_once(self):
        regions=torch.zeros((2,4,128,128));regions[:,2]=1
        regions[:,2,40:70,50:80]=0;regions[:,0,40:70,50:80]=1
        masks=ownership_masks(regions)
        torch.testing.assert_close(masks.sum(1),torch.ones((2,128,128)))
        torch.testing.assert_close(spatial_grid(3,4).sum(0),torch.ones((128,128)))

    def test_prepool_recovers_distinct_evidence(self):
        fmap=torch.tensor([[[[1.,3.],[7.,9.]]]])
        masks=torch.tensor([[[[1.,0.],[0.,0.]],[[0.,1.],[1.,1.]],[[0.,0.],[0.,0.]]]])
        tokens,mass=pool(fmap,masks)
        self.assertAlmostEqual(tokens[0,0,0].item(),1)
        self.assertAlmostEqual(tokens[0,1,0].item(),19/3,places=5)
        self.assertEqual(tokens[0,2,0].item(),0)
        self.assertAlmostEqual((tokens[0,:,0]*mass[0]).sum().item()/mass.sum().item(),5)


if __name__=='__main__':unittest.main()
