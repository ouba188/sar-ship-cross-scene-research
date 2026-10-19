"""Analytic convention/coverage tests, plus frozen CSV contract verification."""
import unittest

import numpy as np
import pandas as pd
from pyproj import Transformer

from recover_orbit import (BRIDGE, HERE, NUMERIC_COLUMNS, Unavailable, axis_difference,
                           enu_to_map_jacobian, image_direction, interpolate_grid,
                           local_basis, look_geometry, orbit_at, recover_row)


class OrbitTests(unittest.TestCase):
    def test_enu_equator_and_east_longitude(self):
        np.testing.assert_allclose(local_basis(0, 0) @ [3, 4, 5], [4, 5, 3])
        np.testing.assert_allclose(local_basis(90, 0) @ [3, 4, 5], [-3, 5, 4], atol=1e-12)

    def test_target_to_sensor_sign_and_incidence(self):
        # At Greenwich/equator ECEF +x=up, +y=east, +z=north.
        los, flight, incidence = look_geometry([6378137+700000, 700000, 0], [0, 0, 7500], 0, 0)
        np.testing.assert_allclose(los, [2**-.5, 0, 2**-.5], atol=1e-12)
        np.testing.assert_allclose(flight, [0, 7500, 0])
        self.assertAlmostEqual(incidence, 45.)
        self.assertGreater(los[0], 0)

    def test_image_sign_rotation_and_unsigned_axis(self):
        affine = np.diag([10., -10.])
        for horizontal, expected in [([1, 0], 0), ([0, 1], 1.5*np.pi),
                                     ([-1, 0], np.pi), ([0, -1], .5*np.pi)]:
            _, _, angle = image_direction(horizontal, np.eye(2), affine)
            self.assertAlmostEqual(angle, expected)
        rotation = np.array([[0., -1.], [1., 0.]])
        _, _, angle = image_direction([1, 0], rotation, affine)
        self.assertAlmostEqual(angle, 1.5*np.pi)
        self.assertAlmostEqual(axis_difference(.01, np.pi-.01), np.degrees(.02))
        self.assertAlmostEqual(axis_difference(0, np.pi), 0)

    def test_utm_true_north_is_not_grid_north(self):
        transform = Transformer.from_crs(4326, 32611, always_xy=True)
        central = enu_to_map_jacobian(-117, 40, transform)
        np.testing.assert_allclose(central, np.eye(2)*.9996, atol=2e-8)
        east_of_central = enu_to_map_jacobian(-114, 40, transform)
        self.assertLess(east_of_central[0, 1], 0)
        self.assertGreater(east_of_central[1, 0], 0)

    def test_per_position_time_interpolation(self):
        xy = np.array([[0., 0.], [10., 0.], [0., 10.]])
        origin = 1738547400.
        t, values = interpolate_grid(xy, origin+np.array([0., 20., 40.]),
                                     np.array([2., 3.]), np.array([[0.], [100.], [200.]]))
        self.assertAlmostEqual(t-origin, 16.)
        self.assertAlmostEqual(values[0], 80.)
        with self.assertRaisesRegex(Unavailable, 'outside_geolocation_grid'):
            interpolate_grid(xy, origin+np.array([0., 20., 40.]), np.array([9., 9.]), xy)

    def test_hermite_analytic_cubic_and_coverage(self):
        origin = 1738547400.
        vectors = np.array([[origin, 0, 0, 5, 0, 2, 0],
                            [origin+10, 1000, 20, 5, 300, 2, 0]], dtype=float)
        position, velocity = orbit_at(vectors, origin+4, origin+1, origin+9)
        np.testing.assert_allclose(position, [64, 8, 5], atol=1e-11)
        np.testing.assert_allclose(velocity, [48, 2, 0], atol=1e-11)
        with self.assertRaisesRegex(Unavailable, 'outside_actual_acquisition_time'):
            orbit_at(vectors, origin, origin+1, origin+9)
        with self.assertRaisesRegex(Unavailable, 'outside_orbit_time_coverage'):
            orbit_at(vectors, origin+11, origin, origin+12)

    def test_unknown_never_synthesizes_direction(self):
        row = pd.Series(dict(row=17, clean_row=22, observation_key='test', resolved_product='test',
                             geometry_valid=False, pixel_product_unique=False,
                             native_window_matches_manifest_raster=True, geometry_status='ambiguous'))
        out = recover_row(row, None, None, None)
        self.assertFalse(out['orbit_valid'])
        self.assertTrue(all(np.isnan(out[k]) for k in NUMERIC_COLUMNS))
        row.geometry_valid = row.pixel_product_unique = True
        out = recover_row(row, None, None, None)
        self.assertEqual(out['orbit_status'], 'missing_annotation_orbit')
        self.assertTrue(all(np.isnan(out[k]) for k in NUMERIC_COLUMNS))

    def test_recovered_csv_alignment_and_coverage(self):
        output = HERE / 'orbit_geometry.csv'
        if not output.exists():
            self.skipTest('Run recovery before output-contract test')
        frame = pd.read_csv(output)
        cohort = pd.read_csv(BRIDGE / 'cohort.csv')
        old = pd.read_csv(BRIDGE / 'axis_radar_readout_20261019/geometry.csv').set_index('row')
        self.assertEqual(frame.row.tolist(), cohort.row.tolist())
        self.assertEqual(frame.clean_row.tolist(), cohort.clean_row.tolist())
        self.assertTrue(frame.loc[~frame.orbit_valid, NUMERIC_COLUMNS].isna().all().all())
        valid = frame[frame.orbit_valid]
        self.assertGreater(len(valid), 0)
        self.assertTrue(old.loc[valid.row, 'geometry_valid'].all())
        self.assertTrue(old.loc[valid.row, 'pixel_product_unique'].all())
        self.assertTrue(valid.los_angle_image_rad.between(0, 2*np.pi, inclusive='left').all())
        self.assertTrue(valid.los_axis_image_rad.between(0, np.pi, inclusive='left').all())
        self.assertTrue(valid.orbit_incidence_deg.between(0, 90, inclusive='neither').all())
        self.assertTrue((valid[['time_since_start_s', 'time_until_stop_s',
                                'orbit_left_margin_s', 'orbit_right_margin_s']] >= 0).all().all())


if __name__ == '__main__':
    unittest.main(verbosity=2)
