"""Focused tests for signed contribution and whole-vessel removal."""

import unittest
import numpy as np
from scipy.special import softmax
from evidence_core import contributions, deletion_features
from extract_evidence import prepare, c, j, load, STUDY


class EvidenceTests(unittest.TestCase):
    def test_signed_group_contributions_and_same_class(self):
        # Two observations are one vessel: opposing contributions cancel in it.
        ar = np.zeros((3, 2)); ai = np.array([[0., 2.], [0., -1.], [0., 3.]])
        k = np.ones((2, 3)); r = np.array([0, 1]); i = np.array([1, 1])
        f, terms = contributions(k, k, ar, ai, r, i, np.array([1, 0, 1]),
                                  np.array(['a', 'a', 'b']))
        np.testing.assert_allclose(terms[0], [2., -1., 3.])
        np.testing.assert_allclose(f[0, [0, 2, 3, 4, 5]], [4., 1., .75, .8, 1.])
        np.testing.assert_array_equal(f[1], np.zeros(8))

    def test_deletion_summary_uses_original_roles(self):
        raw = np.array([[[.8, .2], [.4, .6]]])
        initial = np.array([[[.1, .9], [.7, .3]]])
        f = deletion_features(raw, initial, np.array([0]), np.array([1]))
        np.testing.assert_allclose(f[0], [.5, .5, .5, .4, 1., -.6])

    def test_whole_vessel_refit_against_direct_solver(self):
        maps = load(STUDY/'case_relation_headroom_20261012/Los Angeles/maps.npz')
        model = load(STUDY/'case_completion_class_control_20261012/Los Angeles/fold_0_model.npz')
        xs, ps = maps['single'][:6], maps['prior'][:6]
        sy = np.array([0, 0, 1, 2, 1, 2])
        groups = np.array(['a', 'a', 'b', 'c', 'd', 'e'])
        kb = xs@xs.T
        fit = prepare(xs, ps, sy, groups, model, kb, ps)
        keep = np.array([2, 3, 4, 5])
        np.testing.assert_array_equal(fit['remaining'][0], keep)
        xq, pq = maps['single'][10:13], maps['prior'][10:13]
        changed = j.corrected_support(xs[keep], model)
        direct = pq+(xq@changed.T)@c.coefficients(changed@changed.T, ps[keep], sy[keep], 2.)
        batched = pq+xq@fit['ew'][:, 8:16]
        np.testing.assert_allclose(softmax(batched,axis=1),softmax(direct,axis=1),atol=1e-10)
        # Re-centering after removal really matters in this supported use case.
        self.assertGreater(float(abs(changed-fit['xi'][keep]).max()), 1e-6)
        np.testing.assert_array_equal(fit['bw'][:2,:8], np.zeros((2,8)))


if __name__ == '__main__':
    unittest.main()
