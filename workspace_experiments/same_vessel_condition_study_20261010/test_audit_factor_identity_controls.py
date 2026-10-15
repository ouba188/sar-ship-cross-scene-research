import unittest

import numpy as np
import pandas as pd

from audit_factor_identity_controls import (
    aligned_held, anchor_metrics, coverage, held_cohort, summarize,
)


def obs(ship, product, port, label=0, angle=30.):
    return dict(mmsi=ship, product=product, port=port, class_id=label,
                incidence_deg=angle, pass_name='ASCENDING', platform='S1A')


class IdentityControlTests(unittest.TestCase):
    def test_held_positions_identity_class_and_exact_scene_exclusions(self):
        meta = pd.DataFrame([
            obs(99, 'b', 'B'),                 # Excluded non-held exact donor.
            obs(1, 'a', 'A'), obs(1, 'b', 'B'),
            obs(2, 'b', 'B', angle=32.),       # Included boundary angle.
            obs(3, 'b', 'B', label=1),
            obs(4, 'b', 'wrong_port'),
            obs(5, 'c', 'B'),                 # Fallback is not allowed.
            obs(6, 'b', 'B', angle=32.01),
        ], index=[90, 90, -4, 8, 600, 12, 3, 2])
        held = np.array([7, 2, 5, 1, 6, 4, 3])
        frame, matches = held_cohort(meta, held)
        row = matches[matches.anchor == 3].iloc[0]
        self.assertEqual(tuple(frame.iloc[list(row.same_indices)].obs_id), (2,))
        self.assertEqual(tuple(frame.iloc[list(row.other_indices)].obs_id), (3,))
        self.assertEqual(row.incidence_gap, 2.)
        for row in matches.itertuples(index=False):
            anchor = frame.iloc[row.anchor]
            same, other = frame.iloc[list(row.same_indices)], frame.iloc[list(row.other_indices)]
            self.assertTrue((same.mmsi == anchor.mmsi).all())
            self.assertTrue((other.mmsi != anchor.mmsi).all())
            self.assertTrue((other.class_id == anchor.class_id).all())
            self.assertTrue((same.port != anchor.port).all())
            self.assertTrue((same['product'] != anchor['product']).all())
            self.assertTrue((other['product'] == same['product'].iloc[0]).all())
        # The only exact eligible donor was non-held: fallback must not replace it.
        _, empty = held_cohort(meta, np.array([1, 2, 4, 5, 6, 7]))
        self.assertTrue(empty.empty)

    def test_unknown_angles_reported_and_wrong_predictions_not_filtered(self):
        meta = pd.DataFrame([obs(1, 'a', 'A'), obs(1, 'b', 'B', angle=np.nan),
                             obs(2, 'b', 'B'), obs(3, 'c', 'C', label=1)])
        frame, matches = held_cohort(meta, np.arange(len(meta)))
        self.assertEqual(matches.anchor.tolist(), [0])
        self.assertEqual(matches.angle_status.tolist(), ['unknown_angle'])
        scores = np.zeros((4, 8))
        scores[0, 1] = 1.                    # Wrong anchor must remain eligible.
        scores[1, 0] = 1.                    # Same-vessel teacher corrects all rivals.
        scores[2, 1] = 2.                    # Other-vessel teacher corrects none.
        values = np.array([[0.], [1.], [2.], [99.]])
        records = anchor_metrics(values, scores, frame, matches)
        row = records[records.stratum == 'all'].iloc[0]
        self.assertEqual(row.same_net_pct, 100.)
        self.assertEqual(row.other_net_pct, 0.)
        self.assertEqual((row.same_distance, row.other_distance), (1., 4.))
        summary, _ = summarize(records)
        self.assertTrue((summary.distance_ratio == .25).all())
        cov = coverage(frame, matches)
        self.assertEqual(cov.query("stratum == 'unknown_angle' and class_id == 0").iloc[0].anchors, 1)
        self.assertEqual(cov.query("stratum == 'all' and class_id == 1").iloc[0].anchors, 0)
        self.assertEqual(cov.query("stratum == 'known_angle'").anchors.sum(), 0)
        reversed_records = anchor_metrics(values, -scores, frame, matches)
        self.assertEqual(set(reversed_records.obs_id), set(records.obs_id))
        self.assertAlmostEqual(reversed_records.iloc[0].same_harm_pct, 100/7)

    def test_export_order_uses_obs_ids_not_array_positions(self):
        archive = dict(obs_ids=np.array([9, 3, 7]), z_candidate=np.array([[90.], [30.], [70.]]))
        np.testing.assert_array_equal(aligned_held(archive, [7, 9], 'z_candidate'), [[70.], [90.]])
        with self.assertRaisesRegex(ValueError, 'missing'):
            aligned_held(archive, [1], 'z_candidate')

    def test_acquisitions_vessels_and_covered_classes_are_balanced(self):
        rows = []
        for ship, label, product, count, distance in [
                (1, 0, 'a', 8, 0.), (1, 0, 'b', 1, 2.),
                (2, 0, 'c', 1, 3.), (3, 1, 'd', 1, 8.)]:
            for _ in range(count):
                rows.append(dict(stratum='all', obs_id=len(rows), mmsi=ship,
                    class_id=label, product=product, port='p', pairs=1,
                    same_distance=distance, other_distance=2.,
                    same_rescue_pct=0., same_harm_pct=0., same_net_pct=0.,
                    other_rescue_pct=0., other_harm_pct=0., other_net_pct=0.,
                    anchor_positive_pct=0., same_positive_pct=0., other_positive_pct=0.))
        summary, classes = summarize(pd.DataFrame(rows))
        self.assertEqual(classes.loc[classes.class_id == 0, 'same_distance'].iloc[0], 2.)
        self.assertEqual(summary.loc[summary.weighting == 'equal_class', 'same_distance'].iloc[0], 5.)
        self.assertEqual(summary.loc[summary.weighting == 'equal_ship', 'same_distance'].iloc[0], 4.)
        self.assertEqual(summary.loc[summary.weighting == 'equal_class', 'distance_ratio'].iloc[0], 2.5)


if __name__ == '__main__':
    unittest.main()
