import unittest
import numpy as np
from run import polygon_inside, edge_distance, weighted_span, paired_table
import pandas as pd


class MeasurementsTest(unittest.TestCase):
    def test_geometry_translation_rotation_and_edge_distance(self):
        box=np.array([[-2,-1],[2,-1],[2,1],[-2,1]],float)
        points=np.array([[0,0],[3,0],[3,2]],float)
        np.testing.assert_array_equal(polygon_inside(points,box),[True,False,False])
        np.testing.assert_allclose(edge_distance(points,box),[1,1,np.sqrt(2)])
        rotation=np.array([[0,-1],[1,0]])
        np.testing.assert_array_equal(polygon_inside(points@rotation+10,box@rotation+10),[True,False,False])
        np.testing.assert_allclose(edge_distance(points@rotation+10,box@rotation+10),[1,1,np.sqrt(2)])

    def test_weighted_span_units_and_endpoint_sign(self):
        x=np.arange(101,dtype=float);w=np.ones(101)
        self.assertAlmostEqual(weighted_span(x,w),90.9)
        self.assertAlmostEqual(weighted_span(x*10,w),909.)
        self.assertAlmostEqual(weighted_span(-x,w),weighted_span(x,w))

    def test_incomplete_window_is_not_paired_as_zero(self):
        pairs=pd.DataFrame([dict(obs_id_a=1,obs_id_b=2,days_apart=1)])
        rows=[]
        for oid in (1,2):
            for pol in ('vv','vh'):
                rows.append(dict(obs_id=oid,pol=pol,incidence=35,sin2phi=0,cos2phi=1,platform='S1A',pass_name='Ascending',
                    other_boxes=0,length_m=100,width_m=20,aspect=5,r3_valid=oid==1,r5_valid=False,
                    r3_length_m=80 if oid==1 else 90))
        result=paired_table(pairs,pd.DataFrame(rows))
        self.assertTrue(result.dlog_r3_length_m.isna().all())
        self.assertTrue(result.dlog_length_m.eq(0).all())


if __name__=='__main__':
    unittest.main()
