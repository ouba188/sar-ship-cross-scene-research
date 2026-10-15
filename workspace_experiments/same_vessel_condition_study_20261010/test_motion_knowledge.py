"""Auxiliary target checks: zero speed, dynamic values and missing rows."""

import unittest
import numpy as np
import pandas as pd
from train_motion_knowledge import prepare_speed
from scene_training import sample_weights


class MotionTests(unittest.TestCase):
    def test_valid_zero_and_dynamic_speed_are_not_removed(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 3, 4], class_id=[0, 0, 0, 1, 1]))
        sog = np.array([0., 8., 2., 3., np.nan])
        valid = np.isfinite(sog)
        w = sample_weights(meta); w /= w.sum()
        a = prepare_speed(sog, valid, meta, w)
        np.testing.assert_array_equal(a['valid'], valid)
        np.testing.assert_allclose(a['log_mean'], [.125*np.log(9)+.25*np.log(3)+.5*np.log(4)])
        np.testing.assert_allclose(w*a['importance'], [.125, .125, .25, .5, 0.])
        self.assertNotEqual(float(a['real'][0, 0]), float(a['real'][1, 0]))
        self.assertEqual(float(a['control'][0, 0]), float(a['control'][1, 0]))
        self.assertTrue(np.isfinite(a['real']).all())
        self.assertEqual(float(a['real'][-1, 0]), 0.)

    def test_invalid_and_censored_exact_regression_targets_rejected(self):
        meta = pd.DataFrame(dict(mmsi=[1, 2], class_id=[0, 1]))
        for speed in (-1., 102.2, 102.3, np.nan):
            with self.assertRaises(ValueError):
                prepare_speed(np.array([0., speed]), np.ones(2, bool), meta, np.ones(2)/2)


if __name__ == '__main__':
    unittest.main()
