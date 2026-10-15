"""Tests for leave-acquisition-out, missingness and direction of knowledge use."""

import unittest
import numpy as np
import pandas as pd
from geometry_train_teacher import make_teachers


class TeacherTests(unittest.TestCase):
    def fixture(self):
        meta = pd.DataFrame(dict(mmsi=[1]*5+[2, 2], product=['a', 'a', 'b', 'c', 'd', 'e', 'f'],
            incidence_deg=[30., 30., 31., 40., np.nan, 30., 45.],
            class_id=[0]*5+[1, 1], port=['p']*7))
        scores = np.zeros((7, 8), np.float32)
        scores[:, 0] = [100., 100., 1., 10., 7., -1., 2.]
        return meta, scores

    def test_whole_focal_product_excluded_and_direction_correct(self):
        meta, scores = self.fixture()
        teachers, valid, _, edges = make_teachers(scores, meta)
        self.assertTrue(valid.all())
        self.assertTrue((edges.focal_product != edges.donor_product).all())
        # Unknown donor retains 1/3; known donor total remains 2/3.
        far = (2/3)*(1.2*1+2*10)/3.2+7/3
        near = (2/3)*(1.8*1+1*10)/2.8+7/3
        self.assertAlmostEqual(float(teachers['far'][0, 0]), far, places=5)
        self.assertAlmostEqual(float(teachers['near'][0, 0]), near, places=5)
        self.assertGreater(far, float(teachers['B'][0, 0]))
        self.assertLess(near, float(teachers['B'][0, 0]))
        altered = scores.copy(); altered[:2] = -1e5
        changed = make_teachers(altered, meta)[0]
        for arm in teachers:
            np.testing.assert_array_equal(teachers[arm][:2], changed[arm][:2])

    def test_unknown_focal_and_single_donor_are_exact_baseline(self):
        meta, scores = self.fixture()
        teachers, _, _, _ = make_teachers(scores, meta)
        for arm in teachers:
            np.testing.assert_array_equal(teachers[arm][4:], teachers['B'][4:])

    def test_shuffle_does_not_change_missingness_or_cross_ship(self):
        meta, scores = self.fixture()
        _, _, _, edges = make_teachers(scores, meta)
        real = edges[edges.arm == 'far'].reset_index(drop=True)
        sham = edges[edges.arm == 'sham'].reset_index(drop=True)
        np.testing.assert_array_equal(np.isnan(real.used_donor), np.isnan(sham.used_donor))
        for ship in (1, 2):
            a = real[real.mmsi == ship][['donor_product', 'used_donor']].drop_duplicates()
            b = sham[sham.mmsi == ship][['donor_product', 'used_donor']].drop_duplicates()
            np.testing.assert_array_equal(np.sort(a.used_donor), np.sort(b.used_donor))
        sums = edges.groupby(['mmsi', 'focal_product', 'arm']).weight.sum()
        np.testing.assert_allclose(sums, 1.)


if __name__ == '__main__':
    unittest.main()
