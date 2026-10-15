import unittest
import numpy as np
import pandas as pd
from probe_knowledge_attributes import clean_attributes, fit_ridge, predict_ridge


class AttributeProbeTests(unittest.TestCase):
    def test_mask_rejects_conflict_and_flagged_pairs(self):
        table = pd.DataFrame(dict(raw_exact_length=[100, 5, 1022, 100, 100, np.nan],
            raw_exact_beam=[20, 24, 6, 20, 20, np.nan],
            raw_identity_length=[100, 5, 1022, 101, 100, np.nan],
            raw_identity_beam=[20, 24, 6, 20, 20, np.nan],
            raw_identity_single_pair=[True]*4+[False, False], raw_exact_single_pair=[True]*5+[False]))
        values, valid, reasons = clean_attributes(table)
        np.testing.assert_array_equal(valid, [True, False, False, False, False, False])
        self.assertTrue(np.isnan(values[~valid]).all())
        self.assertEqual(reasons[3], 'product_identity_disagreement')

    def test_source_fitted_mapping_generalizes_without_test_statistics(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240, 4))
        coef = rng.normal(size=(4, 2))
        y = x@coef+np.array([3., 2.])
        model = fit_ridge(x[:160], y[:160], np.ones(160))
        score = predict_ridge(model, x[160:])
        self.assertLess(np.mean((score-y[160:])**2), .03*np.var(y[160:]))
        before = predict_ridge(model, x[160:161])
        together = predict_ridge(model, np.r_[x[160:161], np.full((1, 4), 1e6)])
        np.testing.assert_array_equal(before, together[:1])

    def test_zero_within_class_residual_cannot_create_individual_information(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(50, 5))
        model = fit_ridge(x, np.zeros((50, 2)), np.ones(50))
        np.testing.assert_array_equal(predict_ridge(model, rng.normal(size=(10, 5))), np.zeros((10, 2)))


if __name__ == '__main__':
    unittest.main()
