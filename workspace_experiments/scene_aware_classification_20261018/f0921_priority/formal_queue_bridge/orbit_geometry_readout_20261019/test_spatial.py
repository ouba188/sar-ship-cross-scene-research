import unittest
import numpy as np
from spatial_readout import joint_windows


class SpatialTests(unittest.TestCase):
    def test_partition_unsigned_ship_and_signed_radar(self):
        y,x=np.mgrid[-7:8,-9:10]
        args=(x,y,.3,6.,2.,.6,.8,.7)
        w=joint_windows(*args)
        np.testing.assert_allclose(w.sum(0),1.,atol=1e-14)
        self.assertGreaterEqual(w.min(),0.)
        np.testing.assert_allclose(w,joint_windows(x,y,.3+np.pi,6,2,.6,.8,.7),atol=1e-14)
        reversed_radar=joint_windows(x,y,.3,6,2,.6,.8+np.pi,.7)
        np.testing.assert_allclose(w[[1,0,3,2]],reversed_radar,atol=1e-14)

    def test_joint_coordinate_rotation(self):
        y,x=np.mgrid[-7:8,-9:10]
        a=.41
        expected=joint_windows(x,y,.3,6,2,.6,.8,.7)
        actual=joint_windows(x*np.cos(a)-y*np.sin(a),x*np.sin(a)+y*np.cos(a),
                             .3+a,6,2,.6,.8+a,.7)
        np.testing.assert_allclose(expected,actual,atol=1e-14)

    def test_semantics_at_selected_positions(self):
        w=joint_windows(np.array([6.,-6.,0]),np.array([0.,0.,6.]),0,6,2,1,0,np.pi/2)
        self.assertGreater(w[:2,0].sum(),.5)
        self.assertLess(w[:2,2].sum(),.5)
        self.assertGreater(w[[0,2],0].sum(),.99)
        self.assertLess(w[[0,2],1].sum(),.01)


if __name__=='__main__':
    unittest.main()
