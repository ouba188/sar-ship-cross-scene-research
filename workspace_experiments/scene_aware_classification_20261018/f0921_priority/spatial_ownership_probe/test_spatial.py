import unittest
import numpy as np
from run import cell_fractions,normalize_grid,lift,inner,build_kernels,cosine


class SpatialTests(unittest.TestCase):
    def test_cell_order_and_map_independence(self):
        masks=np.zeros((1,4,128,128),np.uint8)
        masks[0,0,:64,:64]=1;masks[0,1,:64,64:]=1
        masks[0,2,64:,:64]=1;masks[0,3,64:,64:]=1
        p=cell_fractions(masks)
        np.testing.assert_array_equal(p[0],[[1,0,0],[0,1,0],[0,0,1],[0,0,1]])
        masks[:,[2,3]]=masks[:,[3,2]]
        np.testing.assert_array_equal(p,cell_fractions(masks))

    def test_lift_preserves_norm_and_empty_channels_are_zero(self):
        z=np.random.default_rng(3).normal(size=(5,2,4,7))
        p=np.tile([.25,0,.75],(5,4,1))
        lifted=lift(z,p).reshape(5,2,4,3,7)
        np.testing.assert_allclose(np.square(lifted).sum((2,3,4)),np.square(z).sum((2,3)))
        np.testing.assert_array_equal(lifted[:,:,:,1],0)

    def test_shared_ownership_recovers_original_kernel(self):
        z=np.random.default_rng(4).normal(size=(5,2,4,7))
        p=np.tile([.2,.3,.5],(5,4,1))
        np.testing.assert_allclose(inner(z),inner(lift(z,p)),atol=1e-12)

    def test_disjoint_ownership_removes_only_that_cell_product(self):
        z=np.ones((2,2,4,1))/2
        p=np.tile([1.,0,0],(2,4,1));p[1,0]=[0,1,0]
        self.assertAlmostEqual(inner(lift(z,p))[0,1],.75)
        self.assertAlmostEqual(inner(lift(z,p))[0,0],1)

    def test_no_query_fit_and_grid_replay(self):
        rng=np.random.default_rng(6)
        f=dict(grid=rng.normal(size=(8,2,16)),global_=rng.normal(size=(8,2,24)))
        f['global']=f.pop('global_');p=np.tile([.2,.3,.5],(8,4,1));s=np.arange(4)
        k,z=build_kernels(f,p,s)
        np.testing.assert_allclose(inner(z),cosine(f['grid'],s),atol=1e-12)
        f['grid'][4:]*=20;f['global'][4:]*=30;p[4:]=[1,0,0]
        after,_=build_kernels(f,p,s)
        for name in k:np.testing.assert_allclose(k[name][:4,:4],after[name][:4,:4],atol=1e-12)


if __name__=='__main__':unittest.main()
