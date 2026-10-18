import unittest
import numpy as np
from readout import angular_windows, rectangle_axis


class GeometryTests(unittest.TestCase):
    def test_axis_is_unsigned_and_corner_order_independent(self):
        box = np.array([[-4,-1],[4,-1],[4,1],[-4,1]],float)
        angle = .63
        rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for corners in [box@rot.T, np.roll(box@rot.T,1,axis=0), (box@rot.T)[::-1]]:
            _, actual, reliability = rectangle_axis(corners)
            self.assertAlmostEqual(np.cos(2*(actual-angle)),1.)
            self.assertAlmostEqual(reliability,.6)

    def test_partition_and_no_axis(self):
        y,x=np.mgrid[-5:6,-7:8]
        for reliability in [0., .6, 1.]:
            masks=angular_windows(x,y,.72,reliability)
            np.testing.assert_allclose(masks.sum(0),1.)
            self.assertGreaterEqual(masks.min(),0)
        np.testing.assert_allclose(angular_windows(x,y,2.,0),.25)

    def test_half_turn_and_joint_coordinate_rotation(self):
        y,x=np.mgrid[-5:6,-7:8]
        original=angular_windows(x,y,.31,.7)
        np.testing.assert_allclose(original,angular_windows(x,y,.31+np.pi,.7),atol=1e-14)
        alpha=.58
        xr=np.cos(alpha)*x-np.sin(alpha)*y
        yr=np.sin(alpha)*x+np.cos(alpha)*y
        np.testing.assert_allclose(original,angular_windows(xr,yr,.31+alpha,.7),atol=1e-14)


if __name__=='__main__':
    unittest.main()
