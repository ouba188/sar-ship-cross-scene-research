"""Checks for exact joins, duplicate exposure, raw semantics and segment units."""
import copy
import math
import unittest

import pandas as pd

import probe_motion as p


class MotionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohort = pd.read_csv(p.OUT/'cohort_attributes.csv', dtype={'mmsi': str})
        cls.row = cohort[cohort.motion_match_status.eq('exact_raw_record')].iloc[0]
        cls.records, _ = p.load_records(p.Path(cls.row.source_txt), {cls.row.mmsi})
        cls.record = next(r for r in cls.records if r['epoch'] == cls.row.ais_epoch)

    def test_exact_keys_and_raw_columns(self):
        r = self.row
        result = p.exact_join(r, self.records)
        self.assertEqual(result['motion_match_status'], 'exact_raw_record')
        self.assertEqual(result['draught_raw'], float(self.record['raw_fields'][16]))
        self.assertEqual(self.record['accuracy_raw'], self.record['raw_fields'][5])
        wrong = r.copy()
        wrong['mmsi'] = '000000000'
        self.assertEqual(p.exact_join(wrong,self.records)['motion_match_status'], 'exact_time_not_found')
        wrong = r.copy()
        wrong['ais_epoch'] += .25
        self.assertEqual(p.exact_join(wrong,self.records)['motion_match_status'], 'exact_time_not_found')

    def test_ambiguity_is_not_arbitrarily_resolved(self):
        changed = copy.deepcopy(self.record)
        changed['raw_motion'] = ('123', '456', '100')
        result = p.exact_join(self.row, [self.record, changed])
        self.assertEqual(result['motion_match_status'], 'ambiguous_exact_records')

    def test_duplicates_do_not_add_messages(self):
        duplicate = dict(self.record, source_line=123456)
        result = p.dedup_records([self.record, duplicate])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['duplicate_rows'], 2)

    def test_decode_and_circular_units(self):
        self.assertEqual(p.decode_motion('0','0','0'), (0.,0.,0.))
        self.assertTrue(all(math.isnan(x) for x in p.decode_motion('1023','3600','511')))
        self.assertEqual(p.angle_delta(359.,1.), 2.)

    def test_geodesic_speed_and_zero_displacement(self):
        row = self.row.copy()
        t = row.sar_midpoint_epoch
        a = copy.deepcopy(self.record)
        b = copy.deepcopy(self.record)
        for rec, epoch in [(a,t-15),(b,t+15)]:
            rec.update(epoch=epoch, lon=0., lat=0., sog_knots=1., cog_deg=90., true_heading_deg=90., raw_motion=('10','900','90'))
            rec['raw_fields'][0] = str(epoch)
            rec['raw_fields'][1] = row.mmsi
        b['lon'], b['lat'], _ = p.GEOD.fwd(0.,0.,90.,1852/120)
        summary, _, segments = p.summarize_track(row,[a,b])
        self.assertEqual(summary['observed_span_s'],30.)
        self.assertAlmostEqual(segments[0]['displacement_speed_knots'],1.,places=7)
        self.assertAlmostEqual(segments[0]['abs_bearing_cog_delta_deg'],0.,places=7)
        b['lon'], b['lat'] = 0.,0.
        _, _, segments = p.summarize_track(row,[a,b])
        self.assertFalse(segments[0]['moving_cog_eligible'])
        self.assertTrue(math.isnan(segments[0]['abs_bearing_cog_delta_deg']))
        self.assertTrue(math.isnan(segments[0]['displacement_bearing_deg']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
