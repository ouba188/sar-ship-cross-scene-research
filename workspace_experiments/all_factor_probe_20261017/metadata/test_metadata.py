"""Focused unit, sign, timestamp and independently parsed source checks."""

from datetime import datetime, timezone
from pathlib import Path
import io
import math
import sys
import unittest
import xml.etree.ElementTree as XML

import numpy as np
import pandas as pd
from scipy.interpolate import CubicHermiteSpline

import recover_metadata as r


class FactorTests(unittest.TestCase):
    def test_ais_units_and_missing_codes(self):
        x = r.ais_fields(dict(Sog="100", Cog="900.00", True_Head="270", Nav_Status="1", Draught="115"), {})
        self.assertAlmostEqual(x["ais_sog_knots"], 10)
        self.assertAlmostEqual(x["ais_sog_mps"], 5.144444444444445)
        self.assertEqual(x["ais_cog_deg"], 90)
        self.assertEqual(x["ais_true_heading_deg"], 270)
        self.assertEqual(x["ais_draught_m"], 11.5)
        bad = r.ais_fields(dict(Sog="1023", Cog="3600", True_Head="511", Nav_Status="15", Draught="0"), {})
        for field in ("sog", "cog", "true_heading", "nav_status", "draught"):
            self.assertFalse(bad[f"ais_{field}_valid"])
        for field in ("ais_sog_mps", "ais_cog_deg", "ais_true_heading_deg", "ais_nav_status", "ais_draught_m"):
            self.assertTrue(math.isnan(bad[field]))

    def test_censoring_zero_and_out_of_range(self):
        x = r.ais_fields(dict(Sog="1022", Draught="255"), {})
        self.assertEqual(x["ais_sog_knots"], 102.2)
        self.assertEqual(x["ais_draught_m"], 25.5)
        self.assertTrue(x["ais_sog_censored"])
        self.assertTrue(x["ais_draught_censored"])
        self.assertFalse(x["ais_sog_point_valid"])
        self.assertFalse(x["ais_draught_point_valid"])
        self.assertTrue(r.ais_fields(dict(Sog="0", Cog="0", True_Head="0"), {})["ais_sog_point_valid"])
        for value in ("-1", "256", "600", "NA", "1.5"):
            self.assertFalse(r.ais_fields(dict(Draught=value), {})["ais_draught_valid"])

    def test_timestamp_sign_utc_and_rollover(self):
        x = r.ais_fields(dict(Time="1577836800"), dict(Time_UTC="2020/1/1.00:00:30"))
        self.assertEqual(x["ais_minus_sar_seconds"], -30)
        self.assertEqual(x["ais_time_utc"], "2020-01-01T00:00:00Z")
        y = r.ais_fields(dict(Time="1577836800"), dict(Time_UTC="2019/12/31.23:59:30"))
        self.assertEqual(y["ais_minus_sar_seconds"], 30)
        self.assertTrue(y["ais_within_60s"])
        self.assertFalse(r.ais_fields(dict(Time="0"), dict(Time_UTC="2020/1/1.00:00:30"))["ais_timegap_valid"])

    def test_known_enu_los_incidence_and_flight(self):
        # At lon=lat=0: ECEF +y is east, +z north, +x up.
        a = 6378137.0
        geo = r.look_geometry(np.array([a+700000., 700000., 0.]), np.array([0., 0., 7500.]), 0, 0)
        self.assertAlmostEqual(geo["orbit_incidence_deg"], 45)
        self.assertAlmostEqual(geo["orbit_los_bearing_deg"], 90)
        self.assertAlmostEqual(geo["orbit_flight_bearing_deg"], 0)
        self.assertAlmostEqual(geo["orbit_los_east_unit"], 1/math.sqrt(2))
        self.assertAlmostEqual(geo["orbit_los_horizontal_east_unit"], 1)
        self.assertAlmostEqual(geo["orbit_slant_range_m"], 700000*math.sqrt(2))

    def test_ground_cardinal_direction_units_and_sign(self):
        geo = r.look_geometry(np.array([7078137., 700000., 0.]), np.array([0., 0., 7500.]), 0, 0)
        for course, east, north in ((0, 0, 10), (90, 10, 0), (180, 0, -10), (270, -10, 0)):
            with self.subTest(course=course):
                m = r.ground_motion(10, course, geo)
                self.assertAlmostEqual(m["ground_east_mps"], east)
                self.assertAlmostEqual(m["ground_north_mps"], north)
                self.assertAlmostEqual(m["ground_los_horizontal_signed_mps"], east)
                self.assertAlmostEqual(m["ground_range_away_signed_mps"], -east)
                self.assertAlmostEqual(m["ground_los_3d_signed_mps"], east/math.sqrt(2))
                self.assertAlmostEqual(m["ground_alongflight_signed_mps"], north)
                self.assertAlmostEqual(m["ground_los_horizontal_abs_mps"], abs(east))

    def test_orbit_interpolation_against_circular_orbit_no_extrapolation(self):
        t = np.array([0., 10.])
        radius, w = 7000000., 0.001
        p = np.column_stack((radius*np.cos(w*t), radius*np.sin(w*t), t*0))
        v = np.column_stack((-radius*w*np.sin(w*t), radius*w*np.cos(w*t), t*0))
        product = dict(vectors=np.column_stack((t+1000, p, v)), interpolator=CubicHermiteSpline(t, p, v, extrapolate=False))
        midpoint, velocity = r.orbit_at(product, 1005)
        expected = np.array([radius*math.cos(.005), radius*math.sin(.005), 0])
        np.testing.assert_allclose(midpoint, expected, rtol=0, atol=.001)
        np.testing.assert_allclose(velocity, [-radius*w*math.sin(.005), radius*w*math.cos(.005), 0], rtol=0, atol=.001)
        self.assertIsNone(r.orbit_at(product, 999))
        self.assertIsNone(r.orbit_at(product, 1011))

    def test_exact_full_cohort_alignment_and_projection_masks(self):
        for dataset, source, count in (("oss1", r.GEOMETRY, 1349), ("oss2", r.OSS2_MANIFEST, 3745)):
            f = pd.read_csv(r.HERE / f"{dataset}_factor_metadata.csv", dtype={"sample_id": str})
            original = pd.read_csv(source, dtype={"sample_id": str})
            self.assertEqual(len(f), count)
            self.assertEqual(f.sample_id.tolist(), original.sample_id.tolist())
            self.assertEqual(f.global_row.tolist(), list(range(count)))
            self.assertTrue(f.identity_match_valid.all())
            self.assertTrue(f.loc[~f.ground_projection_valid, "ground_los_horizontal_signed_mps"].isna().all())
            self.assertTrue(f.loc[f.ground_projection_valid, "sar_time_within_product_span"].all())
            self.assertTrue((f.loc[f.ground_projection_valid, "ground_los_3d_abs_mps"] <= f.loc[f.ground_projection_valid, "ais_sog_mps"]+1e-10).all())
            self.assertFalse(f.full_chip_georeference_valid.any())
        self.assertEqual(f.product_type.value_counts().to_dict(), {"GRDH": 2746, "SLC": 999})

    def test_independent_xml_samples(self):
        checks = []
        # Strict stdlib streaming validates each selected ship independently;
        # a malformed final document closing tag does not invalidate that ship.
        for dataset in ("oss1", "oss2"):
            f = pd.read_csv(r.HERE / f"{dataset}_factor_metadata.csv", dtype={"mmsi": str})
            for kind, group in f.groupby("product_type", sort=False):
                checked = 0
                for row in group.groupby("scene_id", sort=False).head(1).itertuples():
                    node = None
                    try:
                        with Path(row.ship_xml_path).open("rb") as handle:
                            for _, candidate in XML.iterparse(handle, events=("end",)):
                                if candidate.tag == "ship" and candidate.findtext("SARShipInformation/Center_x") == str(row.center_x) and candidate.findtext("SARShipInformation/Center_y") == str(row.center_y):
                                    node = candidate
                                    break
                    except XML.ParseError:
                        continue
                    self.assertIsNotNone(node)
                    self.assertEqual(node.findtext("AISShipInformation/MMSI"), row.mmsi)
                    self.assertEqual(float(node.findtext("AISShipInformation/Sog"))/10, row.ais_sog_knots)
                    raw_cog = float(node.findtext("AISShipInformation/Cog"))
                    if raw_cog < 3600:
                        self.assertEqual(raw_cog/10, row.ais_cog_deg)
                    sar_time = datetime.strptime(node.findtext("SARShipInformation/Time_UTC"), "%Y/%m/%d.%H:%M:%S").replace(tzinfo=timezone.utc)
                    gap = int(node.findtext("AISShipInformation/Time")) - sar_time.timestamp()
                    self.assertEqual(gap, row.ais_minus_sar_seconds)
                    self.assertEqual(float(node.findtext("SARShipInformation/Manual_Length")), row.sar_manual_length_m)
                    if row.ais_length_valid:
                        self.assertEqual(float(node.findtext("AISShipInformation/Length")), row.ais_length_m)
                    checks.append(dict(dataset=dataset, product_type=kind, sample_id=row.sample_id, xml=row.ship_xml_path,
                                       mmsi=row.mmsi, ais_minus_sar_seconds=gap, sar_manual_length_m=row.sar_manual_length_m,
                                       ais_length_m=row.ais_length_m, strict_independent_ship_parse_pass=True))
                    checked += 1
                    if checked == 3:
                        break
                self.assertEqual(checked, 3, f"Not enough strict samples in {dataset}/{kind}")
        pd.DataFrame(checks).to_csv(r.HERE / "independent_xml_checks.csv", index=False)

    def test_known_source_first_row_no_length_heading_substitution(self):
        row = pd.read_csv(r.HERE / "oss1_factor_metadata.csv").iloc[0]
        self.assertEqual(row.ais_sog_raw, 1)
        self.assertEqual(row.ais_sog_knots, .1)
        self.assertEqual(row.ais_cog_deg, 162)
        self.assertEqual(row.ais_true_heading_deg, 45)
        self.assertEqual(row.ais_length_m, 190)
        self.assertEqual(row.sar_manual_length_m, 245)
        self.assertAlmostEqual(row.ais_draught_m, 11.1)
        self.assertNotEqual(row.sar_chip_direction_deg, row.ais_true_heading_deg)


if __name__ == "__main__":
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FactorTests))
    text = stream.getvalue()
    (r.HERE / "test_results.txt").write_text(text, encoding="utf-8")
    print(text)
    sys.exit(0 if result.wasSuccessful() else 1)
