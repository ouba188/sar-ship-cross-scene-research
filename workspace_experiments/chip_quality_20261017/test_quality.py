"""Focused checks for dimension conversion and non-circular target matching."""
import unittest
import numpy as np
from analyze_quality import analyze_image


def box(cx, cy, length, beam, angle=0):
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-length/2,-beam/2],[length/2,-beam/2],[length/2,beam/2],[-length/2,beam/2]]) @ rotation.T + [cx,cy]
    return dict(xywhr=[cx,cy,length,beam,angle],corners=corners.tolist(),confidence=.8,length_px=length,
                beam_px=beam,shape_usable=True,center_inside=True,touches_edge=False,visible_fraction=1.)


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(port="test",class_name="test",mmsi="123",product="test",target_chip_center_x_vv="32",target_chip_center_y_vv="32")
        self.record = dict(global_row=0,instance_id="test",pol="vv",native_w=64,native_h=64,image="",detections=[box(32,32,20,4)])
        self.meta = dict(pixel_x_m_vv="10",pixel_y_m_vv="20",source_transform_json_vv="[100,10,0,200,0,-20]",ais_length_m="200",ais_width_m="80")

    def test_anisotropic_rotated_ground_length(self):
        self.record["detections"] = [box(32,32,20,4,np.pi/2)]
        out = analyze_image(self.record,self.row,self.meta)
        self.assertAlmostEqual(out["target_length_m"],400)
        self.assertAlmostEqual(out["target_width_m"],40)

    def test_ais_size_never_selects_other_ship(self):
        self.record["detections"].append(box(55,55,8,4))
        self.meta.update(ais_length_m="80",ais_width_m="80")
        out = analyze_image(self.record,self.row,self.meta)
        self.assertEqual(out["target_index"],0)
        self.assertEqual(out["n_25"],2)
        self.assertFalse(out["primary_size_eligible"])

    def test_unknown_scale_not_imputed(self):
        self.meta["pixel_x_m_vv"] = ""
        out = analyze_image(self.record,self.row,self.meta)
        self.assertFalse(out["size_comparable"])
        self.assertTrue(np.isnan(out["target_length_m"]))

    def test_ambiguous_boxes_excluded_from_primary(self):
        self.record["detections"].append(box(33,32,12,6))
        out = analyze_image(self.record,self.row,self.meta)
        self.assertEqual(out["target_match_status"],"inside_multiple_boxes_ambiguous")
        self.assertFalse(out["primary_size_eligible"])


if __name__ == "__main__":
    unittest.main()
