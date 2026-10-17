"""Focused CPU checks for independent recovery and preserved measurements."""

import unittest
import numpy as np
import pandas as pd

from recover_geometry import radar_recovery, verify_preservation, old, masks


class RecoveryTest(unittest.TestCase):
    def test_independent_radar_geometry_and_units(self):
        grid = [dict(line=y, pixel=x, latitude=float(y), longitude=float(x), incidenceAngle=35.)
                for y in (0, 1) for x in (0, 1)]
        geo = dict(epsg=4326, transform=[0., .01, 0., 1., 0., -.01])
        # At a grid node: ship quality is neither supplied nor needed.
        update, status = radar_recovery(grid, 0., 0., geo, 128, 128)
        self.assertEqual(status, 'recovered')
        self.assertEqual(update['radar_valid'], 1)
        self.assertLess(float(old.axis_error(update['radar_axis_rad'], 0.)), .02)
        north = old.image_axis(0., 0., 0., geo, 128, 128)
        self.assertAlmostEqual(north, np.pi/2, places=6)

    def test_existing_range_quality_not_relaxed(self):
        grid = [dict(line=y, pixel=x, latitude=float(y), longitude=float(x), incidenceAngle=35.)
                for y in (0, 1) for x in (0, 1)]
        geo = dict(epsg=4326, transform=[0., .01, 0., 1., 0., -.01])
        update, status = radar_recovery(grid, .5, .5, geo, 128, 128)
        self.assertFalse(update)
        self.assertEqual(status, 'outside_grid')

    def test_preservation_detects_changed_valid_value(self):
        before = pd.DataFrame(dict(global_row=[0], ship_valid=[1], ship_axis_rad=[.2],
            radar_valid=[0], radar_axis_rad=[np.nan], incidence_valid=[1], incidence_rad=[.6]))
        after = before.copy()
        after.loc[0, 'radar_valid'] = 1
        after.loc[0, 'radar_axis_rad'] = .8
        verify_preservation(before, after, [0])
        after.loc[0, 'ship_axis_rad'] = .3
        with self.assertRaises(AssertionError):
            verify_preservation(before, after, [0])

    def test_optional_fields_do_not_invent_ship_axis(self):
        frame = pd.DataFrame(dict(ship_valid=[1, 0], radar_valid=[0, 1], incidence_valid=[0, 1]))
        np.testing.assert_array_equal(masks(frame)['ship_required'], [True, False])
        np.testing.assert_array_equal(masks(frame)['triple'], [False, False])


if __name__ == '__main__':
    unittest.main()
