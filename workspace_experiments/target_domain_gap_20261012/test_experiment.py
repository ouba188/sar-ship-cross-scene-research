"""Checks for leakage, altered regularization, and residual implementation errors."""

import unittest
import numpy as np
import pandas as pd
from experiment import OUT, TASKS, SEEDS, audit_split, fit_head, residual


class ExperimentTests(unittest.TestCase):
    def test_saved_splits(self):
        frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
        for task in TASKS:
            for seed in SEEDS:
                audit_split(frame, dict(np.load(OUT/task/str(seed)/'split.npz')))

    def test_overlap_rejected(self):
        frame = pd.read_csv(OUT/'manifest.csv', dtype={'mmsi': str})
        split = dict(np.load(OUT/TASKS[0]/str(SEEDS[0])/'split.npz'))
        split['support'][0] = split['query'][0]
        with self.assertRaises(ValueError):
            audit_split(frame, split)

    def test_replication_does_not_change_regularization(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(80, 12)).astype(np.float32)
        y = np.arange(80) % 8
        a = fit_head(x, y, 80)
        b = fit_head(np.repeat(x, 2, axis=0), np.repeat(y, 2), 80)
        np.testing.assert_allclose(a['coef'], b['coef'], atol=2e-6)
        np.testing.assert_allclose(a['bias'], b['bias'], atol=2e-6)

    def test_dual_residual_matches_primal_weighted_ridge(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(20, 4))
        prior = rng.normal(size=(20, 8))
        si = np.arange(9)
        y = np.array([0, 0, 0, 1, 1, 2, 3, 4, 5])
        actual = residual(x, prior, si, y)
        phi = np.c_[x, np.ones(len(x))]
        counts = np.bincount(y, minlength=8)
        w = 1/counts[y]; w *= len(y)/w.sum()
        target = (2*np.eye(8)[y]-1-prior[si])*(counts > 0)
        beta = np.linalg.solve(phi[si].T@(w[:, None]*phi[si])+8*np.eye(5),
                               phi[si].T@(w[:, None]*target))
        np.testing.assert_allclose(actual, prior+phi@beta, atol=1e-12)
        np.testing.assert_array_equal(actual[:, 6:], prior[:, 6:])


if __name__ == '__main__':
    unittest.main()
