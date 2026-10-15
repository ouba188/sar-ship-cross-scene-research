"""Focused source diagnostics: alignment, populations, and centered geometry."""

import unittest
import numpy as np
import pandas as pd

from shared_evidence_diagnose import diagnose_scores, CONTROL_SAMPLES_PER_VESSEL


def logits(predictions):
    return 2*np.eye(8)[predictions]


class SharedEvidenceDiagnoseTests(unittest.TestCase):
    def test_positional_alignment_ignores_shuffled_duplicate_index_labels(self):
        meta = pd.DataFrame(dict(mmsi=[2, 1, 2, 1], product=['a', 'a', 'b', 'b'],
                                 class_id=[1, 0, 1, 0]), index=[99, 4, 99, -1])
        score, base = logits([1, 0, 0, 0]), logits([0, 0, 1, 1])
        row = diagnose_scores(score, base, score, meta, [1, 1, 1, 1])
        self.assertEqual((row['rescue'], row['harm']), (2, 1))
        self.assertEqual((row['OA'], row['BA']), (75., 75.))
        self.assertEqual(row['same_vessel_pairs'], 2)
        order = [3, 0, 1, 2]
        reordered = diagnose_scores(score[order], base[order], score[order],
                                    meta.iloc[order], np.ones(4, bool))
        for key in ('OA', 'BA', 'rescue', 'harm', 'centered_score_drift', 'teacher_mse'):
            self.assertAlmostEqual(row[key], reordered[key])
        with self.assertRaisesRegex(ValueError, 'aligned'):
            diagnose_scores(score[:-1], base, score, meta, np.ones(4, bool))

    def test_same_product_excluded_and_vessels_equal_not_pair_counts(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2, 2, 3, 3],
                                 product=['a', 'a', 'b', 'a', 'b', 'c', 'c'],
                                 class_id=[0]*7))
        score = np.zeros((7, 8))
        score[:, 0] = [0, 4, 2, 0, 8, -100, 100]
        row = diagnose_scores(score, score, score, meta, [1, 1, 1, 1, 1, 0, 0])
        self.assertEqual(row['same_vessel_pairs'], 3)
        self.assertEqual(row['paired_vessels'], 2)
        # A one-coordinate delta d has centered mean square 7*d*d/64.
        self.assertAlmostEqual(row['centered_score_drift'], (4+64)/2*7/64)

    def test_independent_row_offsets_preserve_all_diagnostics(self):
        rng = np.random.default_rng(19)
        arrays = [rng.normal(size=(6, 8)) for _ in range(3)]
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 2, 3, 3],
                                 product=['a', 'b']*3, class_id=[0]*6))
        valid = np.ones(6, bool)
        before = diagnose_scores(*arrays, meta, valid)
        shifted = [x+rng.normal(size=(6, 1))*20 for x in arrays]
        after = diagnose_scores(*shifted, meta, valid)
        for key in before:
            np.testing.assert_allclose(before[key], after[key], atol=1e-12, equal_nan=True,
                                       err_msg=key)

    def test_unequal_teacher_groups_keep_counts_and_eligible_denominators(self):
        meta = pd.DataFrame(dict(mmsi=[1]*8, product=['a', 'b']*4, class_id=[0]*8))
        base = logits([1, 1, 0, 0, 1, 0, 1, 0])
        score = logits([0, 1, 1, 0, 0, 1, 0, 1])
        teacher = logits([0, 0, 0, 0, 0, 0, 1, 0])
        teacher[-1] = np.nan
        row = diagnose_scores(score, base, teacher, meta, [1]*7+[0])
        self.assertEqual(row['teacher_correct_count'], 6)
        self.assertEqual(row['teacher_wrong_count'], 1)
        self.assertEqual(row['teacher_correct_rescue'], 2)
        self.assertEqual(row['teacher_correct_harm'], 2)
        self.assertAlmostEqual(row['teacher_correct_rescue_rate'], 2/3)
        self.assertAlmostEqual(row['teacher_correct_harm_rate'], 2/3)
        self.assertEqual(row['teacher_wrong_rescue_rate'], 1.)
        self.assertTrue(np.isnan(row['teacher_wrong_harm_rate']))
        self.assertEqual(row['harm'], 3)
        self.assertEqual(row['repeated_harm'], 2)
        self.assertAlmostEqual(row['teacher_mse'], 4/7)
        self.assertAlmostEqual(row['baseline_teacher_mse'], 3/7)
        self.assertAlmostEqual(row['correct_margin'], 0.)

    def test_fixed_control_budget_and_class_matching(self):
        meta = pd.DataFrame(dict(mmsi=[1]*100+[2]*2+[3]*2,
                                 product=['a', 'b']*52, class_id=[0]*102+[1]*2))
        score = np.zeros((104, 8))
        score[100:102, 0] = 4
        score[102:, 0] = [100, 200]
        row = diagnose_scores(score, score, score, meta, np.ones(104, bool))
        again = diagnose_scores(score, score, score, meta, np.ones(104, bool))
        self.assertEqual(row['control_pairs'], 2*CONTROL_SAMPLES_PER_VESSEL)
        self.assertEqual(row['control_vessels'], 2)
        self.assertAlmostEqual(row['same_class_other_vessel_distance'], 16*7/64)
        self.assertEqual(row['drift_ratio'], 0.)
        self.assertGreater(row['centered_score_drift'], 0.)
        self.assertEqual(row['same_class_other_vessel_distance'], again['same_class_other_vessel_distance'])

    def test_missing_repeats_empty_input_and_zero_control_distance(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1], product=['a', 'a'], class_id=[0, 0]))
        score = logits([0, 0])
        row = diagnose_scores(score, score, np.full((2, 8), np.nan), meta, [0, 0])
        self.assertEqual(row['paired_vessels'], 0)
        self.assertEqual(row['repeated_count'], 0)
        for key in ('teacher_mse', 'baseline_teacher_mse', 'centered_score_drift', 'drift_ratio'):
            self.assertTrue(np.isnan(row[key]), key)
        empty = diagnose_scores(score[:0], score[:0], score[:0], meta.iloc[:0], [])
        self.assertEqual(empty['n'], 0)
        self.assertTrue(np.isnan(empty['OA']))
        meta = pd.DataFrame(dict(mmsi=[1, 1, 2, 2], product=['a', 'b']*2, class_id=[0]*4))
        zero = np.zeros((4, 8))
        row = diagnose_scores(zero, zero, zero, meta, [1]*4)
        self.assertEqual(row['same_class_other_vessel_distance'], 0.)
        self.assertTrue(np.isnan(row['drift_ratio']))


if __name__ == '__main__':
    unittest.main()
