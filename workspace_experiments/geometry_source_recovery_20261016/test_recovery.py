"""Focused source-identity, coordinate and preserved-field regression checks."""
import json
import unittest

import numpy as np
import pandas as pd
from pyproj import Geod

from assemble_recovered import HERE, ROOT, coordinate, prior
from recover_source_bridge import choose


class SourceTests(unittest.TestCase):
    def test_exact_path_wins_over_same_name_elsewhere(self):
        header=dict(path='F:/port_a/product_VV.tif',epsg=32631,width=100,height=100,
                    transform=[500000,10,0,100000,0,-10])
        other=dict(header,path='E:/port_a/product_VV.tif',transform=[600000,10,0,100000,0,-10])
        result,status=choose([other,header],header['path'],'port_a')
        self.assertEqual(status,'exact_path_header')
        self.assertEqual(result['transform'],header['transform'])

    def test_ambiguous_fallback_is_not_accepted(self):
        a=dict(path='E:/port_a/product_VV.tif',epsg=32631,width=100,height=100,
               transform=[500000,10,0,100000,0,-10])
        b=dict(a,path='D:/port_a/product_VV.tif',transform=[600000,10,0,100000,0,-10])
        value,status=choose([a,b],'F:/port_a/product_VV.tif','port_a')
        self.assertIsNone(value)
        self.assertEqual(status,'conflicting_headers')
        self.assertIsNone(choose([a],'F:/port_a/product_VH.tif','port_a')[0])

    def test_known_utm_origins_and_pixel_direction(self):
        for epsg,northing in [(32631,0),(32731,10000000)]:
            geo=dict(epsg=epsg,transform=[500000,10,0,northing,0,-10])
            np.testing.assert_allclose(coordinate(geo,0,0),[3,0],atol=1e-10)
            lon,lat=coordinate(geo,10,10)
            self.assertGreater(lon,3)
            self.assertLess(lat,0)


class OutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=pd.read_csv(ROOT/'geometry_targeted_fix_20261016/recovered_geometry.csv',low_memory=False)
        cls.after=pd.read_csv(HERE/'recovered_geometry.csv',low_memory=False)
        cls.needed=np.flatnonzero(cls.before.ship_recovery_status.notna())

    def test_old_values_and_weak_axis_decisions_preserved(self):
        prior.verify_preservation(self.before,self.after,self.needed)
        np.testing.assert_array_equal(self.before.ship_valid,self.after.ship_valid)
        weak=self.before.ship_recovery_status.str.startswith('previously_rejected',na=False)
        np.testing.assert_array_equal(self.before.loc[weak,'ship_recovery_status'],
                                      self.after.loc[weak,'ship_recovery_status'])
        centers=pd.read_csv(HERE/'crop_center_recovery.csv')
        self.assertEqual(centers.global_row.nunique(),1)
        self.assertEqual(set(centers.polarization),{'vv','vh'})

    def test_restored_coordinates_against_historical_exact_table(self):
        links=pd.read_csv(HERE/'source_links.csv')
        ids=links.loc[(links.source_link=='exact_original_crop_manifest') &
                     (links.status=='recovered_header') & (links.polarization=='vv'),'global_row']
        previous=pd.read_csv(prior.old.INCIDENCE)
        joined=self.after.iloc[ids].merge(previous,on='instance_id',validate='one_to_one')
        self.assertEqual(len(joined),len(ids))
        self.assertTrue(joined.coordinate_kind.isin(prior.KINDS).all())
        distance=Geod(ellps='WGS84').inv(joined.recovery_longitude.to_numpy(),joined.recovery_latitude.to_numpy(),
                                        joined.longitude.to_numpy(),joined.latitude.to_numpy())[2]
        self.assertTrue(np.isfinite(distance).all())
        self.assertLess(float(np.max(distance)),.001)
        (HERE/'coordinate_check.json').write_text(json.dumps(dict(compared_samples=len(ids),
            coordinate_kinds=joined.coordinate_kind.value_counts().to_dict(),max_distance_m=float(max(distance)),
            interpretation='agreement with separately stored historical exact coordinates, not independent field surveying'),indent=2),encoding='utf-8')

    def test_coverage_change_matches_actual_eligibility(self):
        table=pd.read_csv(HERE/'task_coverage.csv')
        keys=['pool','split','origin','criterion']
        old=table[table.version=='old'].drop(columns='version').set_index(keys).sort_index()
        new=table[table.version=='recovered'].drop(columns='version').set_index(keys).sort_index()
        pd.testing.assert_frame_equal(old.xs('ship_required',level='criterion'),
                                      new.xs('ship_required',level='criterion'))
        delta=(new-old)
        self.assertTrue((delta>=0).all().all())
        delta[delta.any(axis=1)].to_csv(HERE/'task_coverage_delta.csv')
        missing_before=~prior.masks(self.before)['triple']
        valid_after=prior.masks(self.after)['triple']
        improved=self.after.loc[missing_before & valid_after]
        self.assertTrue((improved.ship_valid==1).all())
        summary=dict(new_complete_geometry_samples=len(improved),
                     ship_required_task_coverage_unchanged=True,changed_task_groups=int(delta.any(axis=1).sum()))
        (HERE/'eligibility_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    unittest.main(verbosity=2)
