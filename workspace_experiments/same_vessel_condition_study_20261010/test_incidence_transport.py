import unittest
import numpy as np
import pandas as pd
from incidence_transport import (matched_pairs, pair_weights, fit_response, predict_delta,
    canonical_delta, shuffled_angles)


def sample():
    rng = np.random.default_rng(123)
    rows, context, z = [], [], []
    for ship in range(20):
        c = rng.normal(size=7)
        for t, angle in enumerate([32., 36., 42.]):
            rows.append(dict(mmsi=ship, class_id=ship%8, product=f'p{ship}_{t}',
                port=f'port{ship%2}', platform='S1A', pass_name='Ascending',
                acq_time=f'2025-01-0{t+1}', obs_id=len(rows), incidence_deg=angle))
            context.append(c)
            z.append(np.r_[c, np.zeros(25)]+((angle-36)/5)*np.r_[c*2+1, np.zeros(25)])
    return np.array(z), np.array(context), pd.DataFrame(rows)


class TransportTests(unittest.TestCase):
    def test_pairs_and_weights(self):
        z, c, m = sample()
        p = matched_pairs(m, np.arange(len(m)))
        self.assertEqual(len(p), 60)
        np.testing.assert_allclose(pair_weights(m, p).sum(), 1)
        m.loc[1, 'pass_name'] = None
        p2 = matched_pairs(m, np.arange(len(m)))
        self.assertFalse(np.isin(1, p2))
        for a, b in p2:
            self.assertEqual(m.loc[a, 'mmsi'], m.loc[b, 'mmsi'])
            self.assertNotEqual(m.loc[a, 'product'], m.loc[b, 'product'])

    def test_known_context_response_generalizes_in_synthetic_system(self):
        z, c, m = sample(); fit = np.arange(45)
        model = fit_response(z, c, m, fit, 'angle_context')
        pairs = matched_pairs(m, np.arange(45, 60)); a, b = pairs.T
        prediction = predict_delta(c[a], m.incidence_deg.to_numpy()[a], m.incidence_deg.to_numpy()[b], model)
        truth = z[b]-z[a]
        self.assertLess(np.square(truth-prediction).sum()/np.square(truth).sum(), .1)

    def test_no_target_feature_or_label_enters_prediction_or_fit(self):
        z, c, m = sample(); fit = np.arange(45)
        one = fit_response(z, c, m, fit, 'angle_context')
        zz, cc, mm = z.copy(), c.copy(), m.copy()
        zz[45:] = 999; cc[45:] = -999; mm.loc[45:, 'class_id'] = 7
        two = fit_response(zz, cc, mm, fit, 'angle_context')
        np.testing.assert_array_equal(one['coefficient'], two['coefficient'])
        # Prediction API has no B image, label, or identity parameter.
        before = predict_delta(c[[45]], [32.], [42.], one)
        after = predict_delta(c[[45]], [32.], [42.], two)
        np.testing.assert_array_equal(before, after)

    def test_missing_metadata_is_exact_noop_and_same_angle_is_zero(self):
        z, c, m = sample(); model = fit_response(z, c, m, np.arange(45), 'angle_context')
        mm = m.iloc[45:].copy().reset_index(drop=True)
        mm.loc[0, 'incidence_deg'] = np.nan; mm.loc[1, 'pass_name'] = None
        delta, valid = canonical_delta(c[45:], mm, model)
        np.testing.assert_array_equal(delta[:2], 0)
        np.testing.assert_array_equal(predict_delta(c[:3], [35]*3, [35]*3, model), 0)

    def test_sham_keeps_each_vessel_distribution_and_outside_untouched(self):
        z, c, m = sample(); fit = np.arange(45)
        angle = shuffled_angles(m, fit)
        for ship, group in m.iloc[fit].groupby('mmsi'):
            np.testing.assert_array_equal(np.sort(angle[group.index]), np.sort(group.incidence_deg))
        np.testing.assert_array_equal(angle[45:], m.incidence_deg.to_numpy()[45:])


if __name__ == '__main__':
    unittest.main()
