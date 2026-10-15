"""Checks against known factors, exact decompositions and the actual readout."""

import unittest
import numpy as np
import pandas as pd
import torch
from paired_evidence_factors import (
    acquisition_table, separate, complement, row_weights, support_operator,
    adapted_scores, fit_episode_correction,
)
from target_response_adapter import residual_head


def synthetic():
    rng = np.random.default_rng(412)
    rows, x = [], []
    for label in range(2):
        for ship in range(30):
            identity = rng.normal()
            for port in range(3):
                rows.append(dict(class_id=label, mmsi=100*label+ship,
                                 product=f'{ship}_{port}', port=str(port)))
                x.append([2*label-1+.03*rng.normal(), identity,
                          2*(2*label-1)+2*rng.normal(), rng.normal()])
    return np.array(x), pd.DataFrame(rows)


class FactorTests(unittest.TestCase):
    def test_known_stable_class_axis_not_stable_identity(self):
        x, m = synthetic()
        f = separate(x, m, np.random.default_rng(1).normal(size=(8, 4)), rank=1)
        direction = f['whitener']@f['bases']['paired'][:, 0]
        direction /= np.linalg.norm(direction)
        self.assertGreater(abs(direction[0]), .95)

    def test_rotation_equivariance_and_exact_complement(self):
        x, m = synthetic(); rng = np.random.default_rng(31)
        rotation = np.linalg.qr(rng.normal(size=(4, 4)))[0]
        head = rng.normal(size=(8, 4))
        a = separate(x, m, head, rank=1)
        b = separate(x@rotation, m, head@rotation, rank=1)
        u = (x-a['center'])@a['whitener']; q = a['bases']['paired']; r = complement(q)
        np.testing.assert_allclose(u@q@q.T+u@r@r.T, u, atol=1e-10)
        pa = q@q.T
        qb = rotation@b['bases']['paired']
        np.testing.assert_allclose(pa, qb@qb.T, atol=1e-8)

    def test_acquisition_duplicates_do_not_change_fit(self):
        x, m = synthetic(); head = np.random.default_rng(3).normal(size=(8, 4))
        a = separate(x, m, head, rank=1)
        ids = np.r_[np.arange(len(x)), 0, 0, 0]
        b = separate(x[ids], m.iloc[ids].reset_index(drop=True), head, rank=1)
        np.testing.assert_allclose(a['signal'], b['signal'], atol=1e-10)
        np.testing.assert_allclose(a['drift'], b['drift'], atol=1e-10)
        self.assertAlmostEqual(row_weights(m).sum(), 1.)

    def test_actual_support_readout_including_absent_class(self):
        rng = np.random.default_rng(4)
        ks, kq = rng.normal(size=(16, 5)), rng.normal(size=(9, 5))
        sy = np.arange(16)%3
        ps, pq = rng.normal(size=(16, 8)), rng.normal(size=(9, 8))
        op, present = support_operator(ks, kq, sy)
        expected = adapted_scores(ps, pq, op, present, sy)
        beta, _, _ = residual_head(torch.tensor(np.c_[ks, np.ones(16)]), torch.tensor(ps),
                                   torch.tensor(sy), balanced=True)
        actual = pq+np.c_[kq, np.ones(9)]@beta.numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(expected[:, 3:], pq[:, 3:], atol=1e-12)

    def test_episodic_solution_satisfies_deployment_gradient(self):
        rng = np.random.default_rng(5)
        f, k = rng.normal(size=(28, 3)), rng.normal(size=(28, 5))
        p = rng.normal(size=(28, 8)); y = np.arange(28)%8
        s, q = np.arange(12), np.arange(12, 28)
        coef = fit_episode_correction(f, p, y, [(s, q)], k)
        a = torch.tensor(coef, requires_grad=True)
        design = torch.tensor(np.c_[f, np.ones(28)])
        prior = torch.tensor(p)+design@a
        phi_s, phi_q = [torch.tensor(np.c_[k[ids], np.ones(len(ids))]) for ids in (s, q)]
        beta, _, _ = residual_head(phi_s, prior[s], torch.tensor(y[s]), balanced=True)
        pred = prior[q]+phi_q@beta
        truth = 2*torch.eye(8, dtype=torch.float64)[y[q]]-1
        counts = np.bincount(y[q], minlength=8)
        w = torch.tensor(len(q)/((counts > 0).sum()*counts[y[q]]))
        loss = (w[:, None]*(pred-truth).square()).sum()+8*a.square().sum()
        grad, = torch.autograd.grad(loss, a)
        self.assertLess(float(grad.abs().max()), 1e-8)


if __name__ == '__main__':
    unittest.main()
