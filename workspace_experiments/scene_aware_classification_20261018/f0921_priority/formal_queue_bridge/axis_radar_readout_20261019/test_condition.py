import unittest
import numpy as np
import pandas as pd
from run_condition import geometry_kernels


class ConditionTests(unittest.TestCase):
    def test_actual_pair_geometry_and_missingness(self):
        axes=pd.DataFrame(dict(axis_reliability=[.5]*4,ship_axis_rad=[0,0,np.pi/2,0]))
        geo=pd.DataFrame(dict(geometry_valid=[True,True,True,False],
                             radar_angle_image_rad=[0,0,0,np.nan],incidence_deg=[30,40,30,np.nan]))
        availability,angle,joint,valid=geometry_kernels(axes,geo)
        self.assertAlmostEqual(joint[0,1],np.exp(-.5))
        self.assertAlmostEqual(joint[0,2],np.exp(-1))
        self.assertEqual(joint[0,3],0)
        self.assertEqual(joint[3,3],1)
        np.testing.assert_allclose(np.diag(joint),1)
        self.assertGreaterEqual(np.linalg.eigvalsh(joint).min(),-1e-12)
        axes.ship_axis_rad+=np.pi
        np.testing.assert_allclose(geometry_kernels(axes,geo)[2],joint,atol=1e-14)


if __name__=='__main__':
    unittest.main()
