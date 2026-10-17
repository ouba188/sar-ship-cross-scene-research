"""Focused geometry checks for the two-arm, no-relabeling region study."""
import math
import unittest
import numpy as np
import torch
from ultralytics.engine.results import OBB

from redetect_regions import project
from extract_regions import masks_for


class RegionChecks(unittest.TestCase):
    def test_fixed_box_matches_detector_corners_after_padding_and_clip(self):
        for cx,cy,w,h,theta in ((32.,32.,6.,20.,.1),(32.,32.,6.,20.,1.4),
                                (32.,32.,20.,6.,1.4),(4.,5.,10.,24.,.4)):
            with self.subTest(cx=cx,w=w,theta=theta):
                raw=OBB(torch.tensor([[cx,cy,w,h,theta,.9,0.]]),(64,64))
                points=raw.xyxyxyxy[0].numpy()
                half=np.maximum(8.,1.25*np.ptp(points,axis=0))/2
                expected=np.r_[np.floor([cx,cy]-half),np.ceil([cx,cy]+half)].clip(0,64)
                np.testing.assert_array_equal(project(cx,cy,w,h,theta,64,64,False),expected)

    def test_unchanged_adapter_and_affected_orientation(self):
        np.testing.assert_array_equal(project(32,32,20,6,.2,64,64,True),
                                      project(32,32,20,6,.2,64,64,False))
        wrong=project(32,32,6,20,math.radians(80),64,64,True)
        fixed=project(32,32,6,20,math.radians(80),64,64,False)
        self.assertLess(wrong[2]-wrong[0],wrong[3]-wrong[1])
        self.assertGreater(fixed[2]-fixed[0],fixed[3]-fixed[1])

    def test_pol_union_uses_native_width_and_height(self):
        records={(7,'vv'):dict(native_w=64,native_h=128,fixed_box=[16,32,32,64]),
                 (7,'vh'):dict(native_w=64,native_h=128,fixed_box=[32,64,48,96])}
        expected=np.zeros((56,56),bool)
        expected[14:28,14:28]=True
        expected[28:42,28:42]=True
        np.testing.assert_array_equal(masks_for([7],records,'fixed')[0],expected)


if __name__=='__main__':
    unittest.main()
