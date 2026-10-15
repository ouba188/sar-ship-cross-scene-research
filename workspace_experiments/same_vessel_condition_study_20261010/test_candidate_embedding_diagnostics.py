"""Focused checks against explicit pairs and known embedding geometries."""

import itertools
import unittest

import numpy as np
import pandas as pd

from candidate_embedding_diagnostics import embedding_diagnostics


def explicit_within(values, meta):
    drifts, directions, pairs = [], [], 0
    meta = meta.reset_index(drop=True)
    for _, group in meta.groupby('mmsi', sort=True):
        eligible = [(i, j) for i, j in itertools.combinations(group.index, 2)
                    if meta['product'][i] != meta['product'][j]]
        if eligible:
            pairs += len(eligible)
            drifts.append(np.mean([np.square(values[i]-values[j]).mean()
                                   for i, j in eligible]))
            directions.append(np.mean([1-np.dot(values[i], values[j]) /
                                       (np.linalg.norm(values[i])*np.linalg.norm(values[j]))
                                       for i, j in eligible]))
    return np.mean(drifts), np.mean(directions), pairs, len(drifts)


class CandidateEmbeddingDiagnosticsTests(unittest.TestCase):
    def test_explicit_cross_product_pairs_general_dimensions_and_position(self):
        meta = pd.DataFrame(dict(mmsi=[2, 1, 1, 1, 2, 1, 3, 3],
                                 product=['a', 'a', 'a', 'b', 'b', 'c', 'a', 'a'],
                                 class_id=[0]*8), index=[9, 4, 4, -1, 8, 3, 1, 2])
        for dimension in (1, 3, 32):
            with self.subTest(dimension=dimension):
                values = np.random.default_rng(13).normal(size=(8, dimension))
                expected = explicit_within(values, meta)
                row = embedding_diagnostics(values, meta)
                self.assertEqual(row['dimension'], dimension)
                self.assertAlmostEqual(row['same_vessel_drift'], expected[0])
                self.assertAlmostEqual(row['same_vessel_direction_distance'], expected[1])
                self.assertEqual(row['same_vessel_pairs'], expected[2])
                self.assertEqual(row['paired_vessels'], expected[3])
                self.assertAlmostEqual(row['total_centered_variance'],
                                       np.square(values-values.mean(axis=0)).sum(axis=1).mean())

    def test_same_scene_duplicates_excluded_and_vessels_weighted_equally(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2, 2, 3, 3],
                                 product=['a', 'a', 'b', 'a', 'b', 'c', 'c'],
                                 class_id=[0]*7))
        values = np.array([[1, 1], [5, 1], [3, 1], [1, 1], [9, 1], [-100, 1], [100, 1]])
        row = embedding_diagnostics(values, meta)
        self.assertEqual(row['paired_vessels'], 2)
        self.assertEqual(row['same_vessel_pairs'], 3)
        self.assertAlmostEqual(row['same_vessel_drift'], (4/2+64/2)/2)
        self.assertAlmostEqual(row['same_vessel_direction_distance'], explicit_within(values, meta)[1])

    def test_fixed_draw_control_reference_unequal_identity_chip_counts(self):
        ids = [np.arange(0, 3), np.arange(3, 5), np.arange(5, 12)]
        meta = pd.DataFrame(dict(mmsi=[1]*3+[2]*2+[3]*7,
                                 product=['a', 'a', 'b', 'a', 'b']+['c']*7,
                                 class_id=[0]*12))
        values = np.random.default_rng(55).normal(size=(12, 32))
        rng = np.random.default_rng(41033)
        explicit_controls = []
        for k in (0, 1):
            anchors = rng.choice(ids[k], 64)
            others = [indices for j, indices in enumerate(ids) if j != k]
            donor_ships = rng.integers(2, size=64)
            donors = [rng.choice(others[j]) for j in donor_ships]
            explicit_controls.append(np.mean([np.square(values[a]-values[b]).mean()
                                              for a, b in zip(anchors, donors)]))
        row = embedding_diagnostics(values, meta)
        self.assertAlmostEqual(row['same_class_other_distance'], np.mean(explicit_controls))
        self.assertEqual(row['control_vessels'], 2)
        self.assertEqual(row['control_pairs'], 128)
        self.assertEqual(row['same_class_other_distance'],
                         embedding_diagnostics(values, meta)['same_class_other_distance'])

    def test_control_and_matched_drift_share_covered_anchor_cohort(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 3, 3],
                                 product=['a', 'b', 'a', 'a', 'b'],
                                 class_id=[0, 0, 0, 1, 1]))
        values = np.array([[1., 0.], [-1., 0.], [0., 1.], [10., 0.], [110., 0.]])
        row = embedding_diagnostics(values, meta)
        self.assertEqual(row['paired_vessels'], 2)
        self.assertEqual(row['control_vessels'], 1)
        self.assertEqual(row['control_pairs'], 64)
        self.assertEqual(row['same_class_other_distance'], 1.)
        self.assertEqual(row['matched_same_vessel_drift'], 2.)
        self.assertEqual(row['same_vessel_drift'], (2.+5000.)/2)
        self.assertEqual(row['drift_ratio'], 2.)

    def test_scale_invariance_and_raw_quadratic_scaling(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 2, 3, 3],
                                 product=['a', 'b']*3, class_id=[0, 0, 0, 0, 1, 1]))
        values = np.random.default_rng(7).normal(size=(6, 32))
        before = embedding_diagnostics(values, meta)
        for scale in (0.001, 7., -3.):
            after = embedding_diagnostics(scale*values, meta)
            for key in ('drift_ratio', 'normalized_drift', 'effective_rank',
                        'same_vessel_direction_distance', 'class_separation'):
                self.assertAlmostEqual(before[key], after[key], msg=key)
            for key in ('same_vessel_drift', 'matched_same_vessel_drift',
                        'same_class_other_distance', 'total_centered_variance'):
                np.testing.assert_allclose(after[key], before[key]*scale**2, err_msg=key)

    def test_rank_entropy_and_exact_collapse(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 2], product=['a', 'b']*2, class_id=[0]*4))
        values = np.zeros((4, 32))
        values[:, :2] = [[2., 0.], [-2., 0.], [0., 1.], [0., -1.]]
        row = embedding_diagnostics(values, meta)
        self.assertAlmostEqual(row['effective_rank'], np.exp(-0.8*np.log(0.8)-0.2*np.log(0.2)))
        self.assertEqual(row['total_centered_variance'], 2.5)
        values[:, 1] = 0.
        self.assertEqual(embedding_diagnostics(values, meta)['effective_rank'], 1.)
        for constant in (0., 0.1):
            row = embedding_diagnostics(np.full((4, 32), constant), meta)
            self.assertEqual(row['dimension'], 32)
            self.assertEqual(row['effective_rank'], 0.)
            self.assertEqual(row['total_centered_variance'], 0.)
            self.assertEqual(row['same_vessel_drift'], 0.)
            self.assertTrue(np.isnan(row['normalized_drift']))
            self.assertTrue(np.isnan(row['drift_ratio']))
            if constant == 0:
                self.assertTrue(np.isnan(row['same_vessel_direction_distance']))
            else:
                self.assertAlmostEqual(row['same_vessel_direction_distance'], 0.)

    def test_equal_class_scatter_with_unequal_class_sizes(self):
        meta = pd.DataFrame(dict(mmsi=range(6), product=['a']*6, class_id=[0, 0, 1, 1, 1, 1]))
        values = np.array([[0.], [2.], [8.], [10.], [10.], [12.]])
        row = embedding_diagnostics(values, meta)
        self.assertAlmostEqual(row['class_separation'], 20.25/1.5)

    def test_empty_missing_pairs_missing_controls_and_invalid_shape(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1], product=['a', 'a'], class_id=[0, 0]))
        values = np.ones((2, 32))
        for x, frame in ((values, meta), (values[:0], meta.iloc[:0])):
            row = embedding_diagnostics(x, frame)
            self.assertEqual(row['dimension'], 32)
            self.assertEqual(row['effective_rank'], 0.)
            self.assertEqual(row['paired_vessels'], 0)
            self.assertEqual(row['control_vessels'], 0)
            for key in ('same_vessel_drift', 'same_vessel_direction_distance',
                        'same_class_other_distance', 'matched_same_vessel_drift',
                        'drift_ratio', 'normalized_drift'):
                self.assertTrue(np.isnan(row[key]), key)
        meta['product'] = ['a', 'b']
        row = embedding_diagnostics(values, meta)
        self.assertEqual(row['paired_vessels'], 1)
        self.assertEqual(row['same_vessel_drift'], 0.)
        self.assertTrue(np.isnan(row['matched_same_vessel_drift']))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            embedding_diagnostics(values[:1], meta)
        with self.assertRaisesRegex(ValueError, 'finite'):
            embedding_diagnostics(np.full((2, 32), np.nan), meta)


if __name__ == '__main__':
    unittest.main()
