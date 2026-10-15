"""Focused checks for collision exclusion, measured zero, and exported alignment."""
import unittest

import numpy as np
import pandas as pd

from audit_context import DATA, OUT, LAYERS, _osm_columns, derive_relations, unique_table


class ContextAuditTests(unittest.TestCase):
    def test_collision_excludes_both_records_without_averaging(self):
        frame = pd.DataFrame({'pair_id': ['duplicate', 'duplicate', 'unique', ''],
                              'value': [1, 5, 3, 7]})
        collisions = []
        accepted = unique_table(frame, ['pair_id'], 'fixture', collisions)
        self.assertEqual(accepted.pair_id.tolist(), ['unique'])
        self.assertEqual(accepted.value.tolist(), [3])
        self.assertEqual(collisions[0]['rows'], 2)

    def test_measured_zero_is_distinct_from_missing_no_centroid_fallback(self):
        raw = pd.DataFrame(np.nan, index=range(2), columns=_osm_columns())
        layer = LAYERS[0]
        raw.loc[0, [layer + suffix for suffix in
                    ('_available', '_distance_km', '_count_1km_log1p', '_count_3km_log1p')]] = [1, 10, 0, 0]
        empty = pd.DataFrame(index=range(2))
        values, known, names = derive_relations(empty, empty, raw, np.zeros(2, bool))
        column = names.index('osm_' + layer + '_local_background_contrast')
        self.assertEqual(values[0, column], 0)
        self.assertTrue(known[0, column])
        self.assertFalse(known[1].any())
        self.assertTrue(np.all(values[1] == 0))

    def test_export_keeps_full_manifest_and_no_unsupported_values(self):
        ids = pd.read_csv(DATA / 'manifest.csv', usecols=['instance_id']).instance_id.to_numpy(str)
        with np.load(OUT / 'facility_features.npz', allow_pickle=False) as pack:
            np.testing.assert_array_equal(pack['instance_id'], ids)
            np.testing.assert_array_equal(pack['row_index'], np.arange(len(ids)))
            self.assertEqual(len(ids), 38091)
            self.assertTrue(all(name.startswith('osm_') for name in pack['feature_names']))
            for value_key, mask_key in [('values', 'known'), ('raw_values', 'raw_known'),
                                        ('role_values', 'role_known')]:
                values, known = pack[value_key], pack[mask_key]
                self.assertTrue(np.isfinite(values).all())
                self.assertTrue(np.all(values[~known] == 0))
            self.assertTrue(np.all(~pack['known'] | pack['historical_known']))
            self.assertTrue(np.any(pack['known'] & (pack['values'] == 0)))
            self.assertFalse(any(name in pack.files for name in ['y', 'y_ids', 'class_id', 'class_name']))


if __name__ == '__main__':
    unittest.main()
