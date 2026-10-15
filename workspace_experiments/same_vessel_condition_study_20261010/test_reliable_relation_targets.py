import unittest

import numpy as np
import pandas as pd
import torch

from relation_teacher_matching import build_matches
from reliable_relation_targets import ARMS, AUDIT_COLUMNS, build_targets, interval_rows


def fixture():
    meta = pd.DataFrame(dict(
        class_id=[3] * 8, mmsi=[1, 1, 1, 1, 2, 2, 3, 1],
        product=['a', 'b', 'b', 'c', 'b', 'b', 'c', 'unmatched'],
        port=['p'] * 7 + ['elsewhere'], incidence_deg=[30.] * 8,
        pass_name=['ASCENDING'] * 8, platform=['S1A'] * 8))
    prior = -np.repeat(np.array([1., .25, .75, 1.5, .5, 1.5, 2., 100.])[:, None],
                       8, axis=1)
    prior[:, 3] = 0.
    prior[0] = -np.array([.2, 1., 2., 0., 1.25, 1.5, .5, 1.75])
    matches = build_matches(meta).query('anchor == 0').reset_index(drop=True)
    return prior, meta, matches


class ReliableRelationTargetsTests(unittest.TestCase):
    def test_scene_chip_means_and_audit_with_actual_matcher(self):
        prior, meta, matches = fixture()
        self.assertEqual(matches.same_indices.tolist(), [(1, 2), (3,)])
        self.assertEqual(matches.other_indices.tolist(), [(4, 5), (6,)])
        targets, audit = build_targets(prior, meta, matches)
        rivals = [0, 1, 2, 4, 5, 6, 7]
        expected = {'mean': (1.25, 1.25, 6), 'same': (.5, 1.5, 3),
                    'joint': (.5, 2., 1)}
        self.assertEqual(len(audit), len(meta))
        self.assertEqual(audit.loc[0, 'n_same_scenes'], 2)
        self.assertEqual(audit.loc[0, 'n_other_identities'], 2)
        self.assertEqual(audit.loc[0, 'class_id'], 3)
        for arm, (lower, upper, violated) in expected.items():
            with self.subTest(arm=arm):
                arrays = targets[arm]
                np.testing.assert_array_equal(arrays['lower'][0, rivals], lower)
                np.testing.assert_array_equal(arrays['upper'][0, rivals], upper)
                self.assertTrue(arrays['mask'][0, rivals].all())
                self.assertFalse(arrays['mask'][:, 3].any())
                self.assertEqual(audit.loc[0, arm + '_edges'], 7)
                self.assertEqual(audit.loc[0, arm + '_initial_violated_edges'], violated)
                np.testing.assert_array_equal(arrays['eligible'], [True] + [False] * 7)
                for name in ('lower', 'upper', 'mask'):
                    self.assertEqual(arrays[name].shape, (8, 8))
                    self.assertFalse(arrays[name][1:].any())
                    self.assertFalse(arrays[name][:, 3].any())
                    self.assertEqual(arrays[name].dtype,
                                     np.dtype(bool if name == 'mask' else np.float32))
                self.assertEqual(arrays['eligible'].dtype, np.dtype(bool))

    def test_eligibility_requires_distinct_acquisitions_and_identities(self):
        prior, meta, matches = fixture()
        one_scene = pd.concat([matches.iloc[:1]] * 3, ignore_index=True)
        one_identity = meta.copy()
        one_identity.loc[6, 'mmsi'] = 2
        for frame, pairs, counts in ((meta, one_scene, (1, 1)),
                                      (one_identity, matches, (2, 1))):
            with self.subTest(counts=counts):
                targets, audit = build_targets(prior, frame, pairs)
                self.assertEqual(tuple(audit.loc[0, ['n_same_scenes',
                                                    'n_other_identities']]), counts)
                self.assertFalse(audit.eligible.any())
                for arrays in targets.values():
                    for values in arrays.values():
                        self.assertFalse(values.any())

    def test_duplicate_rows_and_chip_order_do_not_change_targets(self):
        prior, meta, matches = fixture()
        original_prior, original_meta = prior.copy(), meta.copy(deep=True)
        original_matches = matches.copy(deep=True)
        repeated = pd.concat([matches.iloc[::-1], matches.iloc[:1]] * 2,
                             ignore_index=True)
        for column in ('same_indices', 'other_indices'):
            repeated[column] = repeated[column].map(lambda ids: tuple(reversed(ids)))
        expected, expected_audit = build_targets(prior, meta, matches)
        actual, actual_audit = build_targets(prior, meta, repeated)
        for arm in ARMS:
            for name in expected[arm]:
                np.testing.assert_array_equal(actual[arm][name], expected[arm][name])
        pd.testing.assert_frame_equal(actual_audit, expected_audit)
        np.testing.assert_array_equal(prior, original_prior)
        pd.testing.assert_frame_equal(meta, original_meta)
        pd.testing.assert_frame_equal(matches, original_matches)

    def test_reused_other_scene_deduplicated_and_sides_weighted_equally(self):
        prior, meta, matches = fixture()
        meta.loc[7, ['product', 'port']] = ['d', 'p']
        prior[7] = -1.
        prior[7, 3] = 0.
        extra = pd.DataFrame([dict(anchor=0, same_indices=(7,), other_indices=(4, 5))])
        matches = pd.concat([matches, extra], ignore_index=True)
        targets, audit = build_targets(prior, meta, matches)
        # Same scene means: .5, 1.5, 1; distinct other scene means: 1, 2.
        self.assertEqual(targets['mean']['lower'][0, 0], 1.25)
        self.assertEqual(audit.loc[0, 'n_same_scenes'], 3)
        self.assertEqual(audit.loc[0, 'n_other_identities'], 2)

    def test_positive_mean_can_hide_disagreement_and_zero_is_not_positive(self):
        prior, meta, matches = fixture()
        prior[[1, 2], 0], prior[3, 0], prior[[4, 5, 6], 0] = 1., -2., -1.
        prior[[1, 2], 1], prior[3, 1] = -.5, -1.
        prior[[4, 5], 1], prior[6, 1] = 1., -2.
        prior[[1, 2], 2] = 0.
        targets, _ = build_targets(prior, meta, matches)
        self.assertTrue(targets['mean']['mask'][0, :3].all())
        np.testing.assert_array_equal(targets['same']['mask'][0, :3],
                                      [False, True, False])
        self.assertFalse(targets['joint']['mask'][0, :3].any())
        self.assertEqual(targets['mean']['lower'][0, 0], .75)
        self.assertEqual(targets['same']['lower'][0, 0], 0.)
        for arrays in targets.values():
            self.assertTrue(arrays['eligible'][0])

    def test_eligibility_independent_of_margin_scores(self):
        prior, meta, matches = fixture()
        # Use full eight-class tied scores, then strictly negative teacher margins.
        for values in (np.zeros_like(prior), -prior):
            targets, audit = build_targets(values, meta, matches)
            self.assertTrue(audit.loc[0, 'eligible'])
            for arrays in targets.values():
                self.assertTrue(arrays['eligible'][0])
                self.assertFalse(arrays['mask'][0].any())

    def test_offset_invariance_and_interval_cap(self):
        prior, meta, matches = fixture()
        prior *= 4
        targets, audit = build_targets(prior, meta, matches)
        shifted, shifted_audit = build_targets(prior + np.arange(8)[:, None] * 8,
                                               meta, matches)
        for arm in ARMS:
            for name in targets[arm]:
                np.testing.assert_array_equal(targets[arm][name], shifted[arm][name])
            self.assertTrue((targets[arm]['upper'] <= 2).all())
            self.assertEqual(targets[arm]['lower'][0, 0], 2.)
            self.assertEqual(targets[arm]['upper'][0, 0], 2.)
        pd.testing.assert_frame_equal(audit, shifted_audit)

    def test_empty_matches_and_empty_input(self):
        prior, meta, matches = fixture()
        for values, frame in ((prior, meta), (prior[:0], meta.iloc[:0])):
            targets, audit = build_targets(values, frame, matches.iloc[:0])
            self.assertEqual(list(audit.columns), AUDIT_COLUMNS)
            self.assertEqual(len(audit), len(frame))
            for arrays in targets.values():
                for values in arrays.values():
                    self.assertFalse(values.any())
                self.assertEqual(arrays['lower'].shape, (len(frame), 8))

    def test_interval_loss_boundaries_symmetry_normalization_and_gradients(self):
        scores = torch.zeros((6, 8), dtype=torch.float64)
        scores[:, 1] = -torch.tensor([1., 1.5, 2., .5, 2.5, 20.])
        scores[3:5, 2] = -1.5
        scores.requires_grad_()
        labels = torch.zeros(6, dtype=torch.long)
        lower, upper = np.ones((6, 8)), np.full((6, 8), 2.)
        mask = np.zeros((6, 8), dtype=bool)
        mask[:5, 1] = True
        mask[3:5, 2] = True
        losses = interval_rows(scores, labels, lower, upper, mask)
        torch.testing.assert_close(losses, torch.tensor([0., 0., 0., .125, .125, 0.],
                                                        dtype=torch.float64))
        shifted = interval_rows(scores + torch.arange(6)[:, None] * 8,
                                labels, lower, upper, mask)
        torch.testing.assert_close(losses, shifted)
        losses.sum().backward()
        expected_grad = torch.zeros_like(scores)
        expected_grad[3, 0], expected_grad[3, 1] = -.5, .5
        expected_grad[4, 0], expected_grad[4, 1] = .5, -.5
        torch.testing.assert_close(scores.grad, expected_grad)


if __name__ == '__main__':
    unittest.main()
