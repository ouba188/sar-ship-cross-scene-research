import unittest
import numpy as np
from extract import mask_from_boxes


class MaskTests(unittest.TestCase):
    def test_target_excludes_overlap_from_other(self):
        boxes=[dict(pol=0,corners=[[2,2],[6,2],[6,6],[2,6]]),dict(pol=1,corners=[[5,5],[8,5],[8,8],[5,8]])]
        masks,info=mask_from_boxes(boxes,[3,3],10)
        self.assertEqual(info['target_polarizations'],1)
        self.assertEqual(masks[0,5,5],1);self.assertEqual(masks[1,5,5],0)
        np.testing.assert_array_equal(masks.sum(0),np.ones((10,10)))

    def test_ambiguous_anchor_not_arbitrarily_assigned(self):
        box=dict(pol=0,corners=[[2,2],[6,2],[6,6],[2,6]])
        masks,info=mask_from_boxes([box,dict(box)],[3,3],10)
        self.assertEqual(info['target_polarizations'],0);self.assertEqual(info['ambiguous_polarizations'],1)
        self.assertEqual(masks[0].sum(),0)


if __name__=='__main__':unittest.main()
