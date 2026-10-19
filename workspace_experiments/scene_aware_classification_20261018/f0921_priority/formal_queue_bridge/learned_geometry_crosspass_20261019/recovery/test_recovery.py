"""Focused regression checks for the relaxed gate and recovered artifact."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import recover_expanded as recovery


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohort, cls.old, cls.baseline = recovery.load_inputs()
        cls.frame = pd.read_csv(recovery.HERE / 'expanded_geometry.csv')

    def test_old_132_value_replay_and_all_269_rows(self):
        checks = recovery.verify_output(self.frame, self.old, self.baseline)
        self.assertEqual(checks['old_valid_replayed'], 132)
        self.assertEqual(len(self.frame), 269)
        self.assertEqual(list(self.frame.columns[:len(self.baseline.columns)]), list(self.baseline.columns))

    def test_new_eligible_sources_and_xml_identity(self):
        new = self.frame.geometry_eligible & ~self.baseline.orbit_valid
        self.assertEqual(int(new.sum()), 108)
        self.assertEqual(set(self.old.loc[new, 'geometry_status']), {'incidence_only'})
        self.assertTrue(self.old.loc[new, 'annotation_path'].str.endswith('.json').all())
        receipt = json.loads((recovery.HERE / 'receipt.json').read_text(encoding='utf-8'))
        metadata = {m['product']: m for m in receipt['annotation_metadata']}
        for product, group in self.frame[new].groupby('resolved_product'):
            entry = metadata[product]
            if entry['status'] != 'ready':
                self.assertFalse(group.orbit_valid.any())
                self.assertTrue(group.orbit_status.str.startswith('annotation_unavailable:').all())
                continue
            path = Path(entry['annotation_path'])
            self.assertEqual(path.suffix, '.xml')
            annotation = recovery.orbit.load_annotation(path, product, path)
            self.assertGreater(len(annotation['vectors']), 1)
            self.assertTrue(np.isfinite(annotation['points']).all())
            self.assertEqual(group.observation_key.unique().tolist(), [recovery.product_key(product)])
            for _, row in group.iterrows():
                self.assertEqual(json.loads(row.contributor_observation_keys_json), [row.observation_key])
            # A complete but wrong acquisition must not be accepted as this product.
            other = next(p for p in metadata if p != product)
            with self.assertRaisesRegex(recovery.orbit.Unavailable, 'annotation_product_identity_mismatch'):
                recovery.orbit.load_annotation(path, other, path)
        valid = self.frame.loc[new & self.frame.orbit_valid]
        required = [c for c in recovery.orbit.NUMERIC_COLUMNS if c != 'axis_difference_deg']
        self.assertTrue(np.isfinite(valid[required].to_numpy()).all())
        self.assertTrue(valid.axis_difference_deg.isna().all())
        self.assertTrue(valid.los_angle_image_rad.between(0, 2*np.pi, inclusive='left').all())
        self.assertTrue(valid.orbit_incidence_deg.between(0, 90, inclusive='neither').all())

    def test_mixed_product_and_window_mismatch_never_get_angles(self):
        rejected = ~self.frame.geometry_eligible
        self.assertEqual(int(rejected.sum()), 29)
        self.assertFalse(self.frame.loc[rejected, 'orbit_valid'].any())
        self.assertTrue(self.frame.loc[rejected, recovery.orbit.NUMERIC_COLUMNS].isna().all().all())
        row = self.old.loc[self.old.geometry_valid].iloc[0].copy()
        for column in ('pixel_product_unique', 'native_window_matches_manifest_raster'):
            changed = row.copy()
            changed[column] = False
            with patch.object(recovery.orbit, 'recover_row', side_effect=AssertionError('Must not compute')):
                out = recovery.recover_row(changed, object())
            self.assertFalse(out['orbit_valid'])
            self.assertTrue(all(np.isnan(out[k]) for k in recovery.orbit.NUMERIC_COLUMNS))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (recovery.HERE / 'test_receipt.json').write_text(json.dumps(dict(
        status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors),
        tests=[name for name in dir(RecoveryTests) if name.startswith('test_')]), indent=2), encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
