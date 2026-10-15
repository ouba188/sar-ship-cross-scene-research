"""Concrete grouped-leakage, semantic-alignment and actual upstream-fit checks."""

import unittest
import numpy as np
import pandas as pd
import torch
from crossfit_evidence import (
    ship_folds, assert_excluded, class_contrasts, semantic_features,
    correction_episodes, fit_pipeline, assemble_training_view,
)
from explicit_candidate_projection import identity_episode


class CrossfitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_group_assignment_and_episode_isolation(self):
        meta = pd.DataFrame(dict(mmsi=np.repeat(np.arange(525), 2),
                                 class_id=np.repeat(np.arange(525)%8, 2)))
        folds = ship_folds(meta)
        self.assertTrue((meta.assign(fold=folds).groupby('mmsi').fold.nunique() == 1).all())
        for s, q in correction_episodes(meta, folds, 31, count=12):
            self.assertEqual(len(np.unique(folds[np.r_[s, q]])), 1)
            self.assertFalse(set(meta.iloc[s].mmsi)&set(meta.iloc[q].mmsi))
            self.assertEqual((len(s), len(q)), (64, 96))

    def test_new_time_identity_cannot_choose_checkpoint(self):
        meta = pd.DataFrame(dict(mmsi=[1, 2, 3, 3]))
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            assert_excluded(meta, np.array([0]), np.array([3]), np.array([2]))

    def test_size_matched_and_oof_use_same_rows_but_different_fit_exposure(self):
        fit = np.arange(10, 19); folds = np.arange(9)%3
        bundles = {}
        for k in range(3):
            bundles[f'fold{k}'] = dict(fitting=fit[folds != k],
                features={'paired': np.full((19, 7), k+1.)},
                prior=np.full((19, 8), k+1.), kernel=np.full((19, 32), k+1.))
        out = assemble_training_view(bundles, fit, folds, 'paired', 'crossfit')
        seen = assemble_training_view(bundles, fit, folds, 'paired', 'size_matched_in')
        for component in out:
            np.testing.assert_array_equal(component[:, 0], folds+1.)
        for component in seen:
            np.testing.assert_array_equal(component[:, 0], (folds+1)%3+1.)

    def test_class_basis_and_rotation_invariant_meta_features(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(120, 7)); fit = np.arange(80); y = np.arange(120)%8
        weights = np.ones(80)/80
        rotate = np.linalg.qr(rng.normal(size=(7, 7)))[0]
        a, _ = semantic_features(x, fit, y, weights)
        b, _ = semantic_features(x@rotate, fit, y, weights)
        np.testing.assert_allclose(a, b, atol=1e-10)
        basis = class_contrasts()
        np.testing.assert_allclose(basis.T@basis, np.eye(7), atol=1e-12)
        np.testing.assert_allclose(basis.T@np.ones(8), 0, atol=1e-12)

    def test_probe_and_scale_do_not_fit_query(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=(120, 7)); fit = np.arange(80); y = np.arange(120)%8
        weights = np.ones(80)/80
        _, first = semantic_features(x, fit, y, weights)
        x[80:] *= 100; y[80:] = 0
        _, second = semantic_features(x, fit, y, weights)
        np.testing.assert_array_equal(first['probe'], second['probe'])
        self.assertEqual(first['rms'], second['rms'])

    def test_actual_two_stage_fit_ignores_excluded_labels_and_features(self):
        rng = np.random.default_rng(9)
        ships = np.repeat(np.arange(170), 2)
        meta = pd.DataFrame(dict(mmsi=ships, class_id=ships%8,
                                 product=np.tile(['a', 'b'], 170), port=np.tile(['p', 'q'], 170)))
        raw = rng.normal(size=(len(meta), 40)).astype(np.float32)
        fit, val, excluded = np.arange(160), np.arange(160, 320), np.arange(320, 340)
        vm = meta.iloc[val].reset_index(drop=True)
        s, q = identity_episode(vm, rng, query_size=None)
        validation = [(val[s], val[q])]
        a = fit_pipeline(raw, meta, fit, validation, excluded, 13, steps=2, every=1)
        poisoned = raw.copy(); poisoned[excluded] += 100
        changed = meta.copy(); changed.loc[excluded, 'class_id'] = 0
        b = fit_pipeline(poisoned, changed, fit, validation, excluded, 13, steps=2, every=1)
        for key in ('raw_center', 'raw_scale', 'teacher_coef'):
            np.testing.assert_array_equal(a[key], b[key])
        for stage in ('case_state', 'prior_state'):
            for key in a[stage]['selected']:
                torch.testing.assert_close(a[stage]['selected'][key], b[stage]['selected'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
