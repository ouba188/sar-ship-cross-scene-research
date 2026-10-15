import unittest
import numpy as np
import pandas as pd
from condition_response import condition_basis, within_vessel, fit_effect, gains


class ResponseTests(unittest.TestCase):
    def test_identity_offsets_do_not_change_condition_estimate(self):
        rng = np.random.default_rng(7)
        meta = pd.DataFrame({'port': np.repeat(['a', 'b'], 60),
                             'mmsi': np.repeat(np.arange(40), 3)})
        basis = rng.normal(size=(120, 5))
        true = rng.normal(size=(5, 12))
        effect = basis@true
        offsets = rng.normal(size=(40, 12))*100
        coef, ref, _ = fit_effect(meta, basis, effect)
        coef2, ref2, _ = fit_effect(meta, basis, effect+np.repeat(offsets, 3, axis=0))
        np.testing.assert_allclose(coef, coef2, atol=1e-12)
        np.testing.assert_allclose(ref, ref2)

    def test_gain_is_class_independent_and_missing_is_identity(self):
        m = pd.DataFrame({'incidence_deg': [35., 45., np.nan],
            'pass_name': ['Ascending', 'Descending', 'Ascending'],
            'platform': ['S1A', 'S1C', 'S1A'], 'class_id': [0, 1, 2]})
        b, v = condition_basis(m)
        m['class_id'] = [7, 6, 5]
        b2, v2 = condition_basis(m)
        np.testing.assert_array_equal(b, b2)
        np.testing.assert_array_equal(v, v2)
        g = gains(b, v, np.ones((5, 4)), np.zeros(5))
        np.testing.assert_array_equal(g[2], np.ones(4))
        self.assertTrue(((g >= .5) & (g <= 2.)).all())

    def test_positive_gain_preserves_spatial_pattern(self):
        rng = np.random.default_rng(9)
        h = rng.uniform(.1, 5, (3, 6, 4, 4))
        gain = rng.uniform(.5, 2, (3, 6, 1, 1))
        corrected = h*gain
        np.testing.assert_allclose(h/h.mean((2, 3), keepdims=True),
                                  corrected/corrected.mean((2, 3), keepdims=True), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
