"""Compare accelerated cell selection with exhaustive GEOS integration."""
import unittest
import numpy as np
from shapely.geometry import Polygon, box
from sar_port.two_stage.geometry import coverage, rectangle, grid


class ExactCoverageTests(unittest.TestCase):
    def test_exhaustive_integration(self):
        for side in (64,96,128):
            shapes=[Polygon(),box(-.5,-.5,7.5,7.5),
                    rectangle([31.2,28.9,37.1,6.3,.73]),
                    rectangle([63.1,62.9,25.,3.,-.11]).difference(box(60,60,65,65))]
            for stride in (4,8,16):
                for poly in shapes:
                    expected=np.array([poly.intersection(cell).area/stride**2
                                       for cell in grid(side,stride)[1]],dtype=np.float32)
                    np.testing.assert_array_equal(coverage(poly,side,stride).ravel(),expected)


if __name__=='__main__':unittest.main()
