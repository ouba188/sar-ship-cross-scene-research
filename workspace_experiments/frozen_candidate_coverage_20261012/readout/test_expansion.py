import unittest
import numpy as np
from run_expansion import measures, ship_groups
import pandas as pd


class ExpansionTests(unittest.TestCase):
    def test_present_class_balancing(self):
        y = np.array([0,0,0,1])
        p = np.full((4,8), .01)
        p[np.arange(4),[0,0,1,1]] = .93
        result = measures(y,p)
        self.assertAlmostEqual(result['oa'],75.)
        self.assertAlmostEqual(result['ba'],100*5/6)
        self.assertTrue(np.isfinite(list(result.values())).all())

    def test_groups_keep_unknown_rows_separate(self):
        frame = pd.DataFrame(dict(id=['a','b','c','d'],mmsi=['123456789','123456789','','000000000']))
        groups,known = ship_groups(frame,'id','mmsi')
        self.assertEqual(groups[0],groups[1])
        self.assertNotEqual(groups[2],groups[3])
        np.testing.assert_array_equal(known,[True,True,False,False])


if __name__=='__main__':
    unittest.main()
