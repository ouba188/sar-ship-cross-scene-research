import unittest
import numpy as np
from run import token_descriptor,kernel


class TokenTests(unittest.TestCase):
    def test_absence_and_unseen_support_tokens_stay_zero(self):
        x=np.arange(4*2*3*2,dtype=float).reshape(4,2,3,2)
        valid=np.ones((4,3),bool);valid[:2,1]=False;valid[2,2]=False
        z,count=token_descriptor(x,valid,[0,1])
        np.testing.assert_array_equal(z[:,:,1],0)
        np.testing.assert_array_equal(z[2,:,2],0)
        self.assertEqual(count[1],0)

    def test_missing_values_cannot_pollute_center(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(4,2,3,5))
        valid=np.ones((4,3),bool);valid[0,1]=False
        before,_=token_descriptor(x,valid,[0,1])
        x[0,:,1]=1e8
        after,_=token_descriptor(x,valid,[0,1])
        np.testing.assert_allclose(before,after)

    def test_queries_cannot_change_support_kernel(self):
        x=np.random.default_rng(6).normal(size=(8,2,12,5));valid=np.ones((8,12),bool)
        z,_=token_descriptor(x,valid,[0,1,2,3]);before=kernel(z)
        x[4:]*=100;valid[4:,2:]=False
        z,_=token_descriptor(x,valid,[0,1,2,3]);after=kernel(z)
        np.testing.assert_allclose(before[:4,:4],after[:4,:4])
        np.testing.assert_allclose(np.diag(before),1,atol=1e-12)

    def test_known_token_direction_and_equal_norm(self):
        x=np.array([[[[1.,0],[0,2]]],[[[3.,0],[0,6]]]])
        z,_=token_descriptor(x,np.ones((2,2),bool),[0,1])
        np.testing.assert_allclose(kernel(z),[[1,-1],[-1,1]],atol=1e-12)


if __name__=='__main__':unittest.main()
