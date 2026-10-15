"""Check forecast-to-support semantics and preservation of the original readout."""

import unittest
import numpy as np
import pandas as pd
from condition_support_matching import (
    ARMS, comparable_pairs, multipliers, incremental_factors, score_episode, decision_accounting, metrics,
)
from condition_evidence_response import fit_model
from incidence_transport import matched_pairs


class SupportMatchingTests(unittest.TestCase):
    def test_increment_cancels_generic_effect_and_keeps_unknown_noop(self):
        raw = dict(anchor=np.array([[.2, .9, 1.]]), condition=np.array([[.7, .3, 1.]]))
        inc = incremental_factors(raw)
        np.testing.assert_array_equal(inc['anchor'], np.ones((1, 3)))
        np.testing.assert_allclose(inc['condition'], [[1.5, .4, 1.]])
        self.assertTrue(((inc['condition'] >= 0)&(inc['condition'] <= 2)).all())

    def setUp(self):
        rng = np.random.default_rng(81)
        self.meta = pd.DataFrame(dict(mmsi=np.repeat(np.arange(20), 3),
            class_id=np.repeat(np.arange(20)%4, 3), port='p', platform='S1A',
            pass_name='Ascending', product=[f'p{i}' for i in range(60)],
            incidence_deg=np.tile([32., 38., 44.], 20), obs_id=np.arange(60),
            acq_time=np.tile(['2020-01-01', '2020-02-01', '2020-03-01'], 20)))
        self.ratio = rng.normal(size=60)
        self.context = rng.normal(size=60)
        pairs = matched_pairs(self.meta, np.arange(60))
        self.models = {a: fit_model(self.meta, np.arange(60), pairs, self.ratio,
            self.context, self.ratio, 'magnitude', a) for a in ('anchor','capacity','condition','sham')}
        self.gate = dict(feature_index=np.array(-1), threshold=np.array(0.), leaf_mix=np.array([0., 0.]))
        self.limits = dict(angle_min=31., angle_max=45., max_gap=12., floor=.05, fixed=.5)

    def test_scope_and_missing_conditions_are_exact_noop(self):
        meta = self.meta.copy()
        meta.loc[3, 'incidence_deg'] = np.nan
        meta.loc[4, 'port'] = 'outside'
        meta.loc[5, 'pass_name'] = 'Descending'
        s, q = np.array([0, 1]), np.array([3, 4, 5])
        factors, detail = multipliers(meta, s, q, self.ratio, self.context, self.models, self.gate, self.limits)
        self.assertEqual(detail['gate'].shape, (3, 2))
        self.assertFalse(detail['valid'].any())
        for value in factors.values():
            np.testing.assert_array_equal(value, np.ones((3, 2)))

    def test_matching_is_symmetric_bounded_and_gate_zero_is_anchor(self):
        s, q = np.array([0, 1]), np.array([3, 4, 5])
        a, _ = multipliers(self.meta, s, q, self.ratio, self.context, self.models, self.gate, self.limits)
        b, _ = multipliers(self.meta, q, s, self.ratio, self.context, self.models, self.gate, self.limits)
        for arm in ARMS:
            np.testing.assert_allclose(a[arm], b[arm].T, rtol=0, atol=1e-15)
            self.assertTrue(((a[arm] >= 0)&(a[arm] <= 1)).all())
        np.testing.assert_array_equal(a['gated'], a['anchor'])

    def test_larger_tolerance_protects_identical_observed_difference(self):
        low, high = dict(self.limits), dict(self.limits)
        low['fixed'], high['fixed'] = .1, 1.
        args = (self.meta, np.array([0, 1]), np.array([3, 4]), self.ratio, self.context, self.models, self.gate)
        a, _ = multipliers(*args, low)
        b, _ = multipliers(*args, high)
        self.assertTrue((b['fixed'] >= a['fixed']).all())

    def test_readout_matches_primal_ridge_and_unit_factors_do_nothing(self):
        rng = np.random.default_rng(18)
        k, p = rng.normal(size=(20, 5)), rng.normal(size=(20, 8))
        s, q, labels = np.arange(8), np.arange(8, 20), np.arange(8)
        factors = {a: np.ones((len(q), len(s))) for a in ARMS}
        base, delta, op, residual = score_episode(p, k, s, q, labels, factors)
        xs, xq = np.c_[k[s], np.ones(len(s))], np.c_[k[q], np.ones(len(q))]
        beta = np.linalg.solve(xs.T@xs+8*np.eye(6), xs.T@(2*np.eye(8)[labels]-1-p[s]))
        np.testing.assert_allclose(base, p[q]+xq@beta, rtol=0, atol=1e-14)
        for d in delta.values():
            np.testing.assert_array_equal(d, np.zeros_like(d))

    def test_nonunit_factors_with_missing_classes_match_primal_readout(self):
        rng = np.random.default_rng(72)
        k, p = rng.normal(size=(5, 2)), rng.normal(size=(5, 8))
        s, q, labels = np.arange(3), np.arange(3, 5), np.array([0, 0, 1])
        factor = np.array([[.2, 1.4, .7], [1.3, .4, 1.6]])
        base, delta, op, residual = score_episode(p, k, s, q, labels, {'condition': factor})
        xs, xq = np.c_[k[s], np.ones(len(s))], np.c_[k[q], np.ones(len(q))]
        weight = np.array([.75, .75, 1.5])
        targets = -np.ones((3, 8))
        targets[np.arange(3), labels] = 1.
        target_residual = targets-p[s]
        target_residual[:, 2:] = 0.
        lhs = xs.T@(weight[:, None]*xs)+8*np.eye(3)
        expected = p[q].copy()
        # Per-query primal solves independently check post-solve edge modulation.
        for j in range(len(q)):
            rhs = xs.T@((weight*factor[j])[:, None]*target_residual)
            expected[j] += xq[j]@np.linalg.solve(lhs, rhs)
        actual = base+delta['condition']
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(actual[:, 2:], p[q, 2:])
        legacy_residual = target_residual.copy()
        legacy_residual[:, :2] += 1.
        legacy = base+(op*(factor-1))@legacy_residual
        self.assertGreater(np.max(np.abs(legacy-expected)), .01)

    def test_exact_margin_accounting_and_query_labels_not_model_inputs(self):
        import inspect
        self.assertNotIn('query_labels', inspect.signature(score_episode).parameters)
        rng = np.random.default_rng(19)
        base = rng.normal(size=(5, 8))
        op = rng.normal(size=(5, 7))
        residual = rng.normal(size=(7, 8))
        factor = rng.uniform(0, 1, (5, 7))
        result = base+(op*(factor-1))@residual
        y = np.array([0, 1, 2, 3, 4])
        rival = base.copy(); rival[np.arange(5), y] = -np.inf
        r = rival.argmax(1)
        actual = np.mean((result[np.arange(5), y]-result[np.arange(5), r])-
                         (base[np.arange(5), y]-base[np.arange(5), r]))
        account = decision_accounting(base, result, op, residual, factor, np.ones_like(factor), y,
                                      np.ones_like(factor, dtype=bool))
        self.assertAlmostEqual(actual, account['margin_delta'])
        self.assertAlmostEqual(actual, account['harmful_removed']-account['helpful_removed'])


if __name__ == '__main__':
    unittest.main()
