import unittest
import sys
import tempfile
from pathlib import Path
import numpy as np
from osgeo import gdal, osr
from land_mask import WaterMap, interior_distance, region_stats
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'object_bridge'))
from tile_policy import select_tiles


class LandMaskTests(unittest.TestCase):
    def test_detector_policy_is_optional_and_complete(self):
        tiles=[dict(tile_id='water'),dict(tile_id='land')]
        self.assertEqual(select_tiles(tiles),(tiles,[]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'policy.csv'
            path.write_text('tile_id,skip_land_interior\nwater,False\nland,True\n',encoding='utf-8')
            self.assertEqual(select_tiles(tiles,path),([tiles[0]],[tiles[1]]))
            with self.assertRaises(KeyError):
                select_tiles([dict(tile_id='unknown_tile')],path)

    def test_coastal_box_is_kept_despite_land_center(self):
        classes=np.full((80,80),50,dtype=np.uint8)
        classes[:,:35]=80
        region=np.zeros_like(classes,dtype=bool); region[30:45,30:55]=True
        result=region_stats(classes,interior_distance(classes,10),region)
        self.assertGreater(result['water_fraction'],0)
        self.assertNotEqual(result['action'],'exclude_land_interior')

    def test_deep_land_excluded(self):
        classes=np.full((80,80),50,dtype=np.uint8)
        region=np.zeros_like(classes,dtype=bool); region[35:45,35:45]=True
        self.assertEqual(region_stats(classes,interior_distance(classes,10),region)['action'],'exclude_land_interior')

    def test_missing_and_wetland_retained(self):
        for value in [0,90,95]:
            classes=np.full((20,20),value,dtype=np.uint8)
            result=region_stats(classes,interior_distance(classes,10),np.ones_like(classes,dtype=bool))
            self.assertEqual(result['uncertain_fraction'],1)
            self.assertNotEqual(result['action'],'exclude_land_interior')

    def test_warp_grid_alignment(self):
        srs=osr.SpatialReference();srs.ImportFromEPSG(32615)
        ds=gdal.GetDriverByName('MEM').Create('',30,20,1,gdal.GDT_Byte)
        ds.SetProjection(srs.ExportToWkt());ds.SetGeoTransform((400000,10,0,3300000,0,-10))
        a=np.full((20,30),50,dtype=np.uint8);a[:,12:]=80;ds.GetRasterBand(1).WriteArray(a)
        mapper=WaterMap.__new__(WaterMap);mapper.vrt=ds
        actual,_=mapper.window(ds,(7,3,15,10))
        np.testing.assert_array_equal(actual,a[3:13,7:22])


if __name__=='__main__':
    unittest.main()
