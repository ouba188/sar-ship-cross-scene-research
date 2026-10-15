"""Consistency cannot equate wrong agreement with success or alter frozen geometry."""

import unittest
import numpy as np
import pandas as pd
import torch
from decision_condition_consistency import js_divergence, same_ship_pairs, pair_weights, pair_metrics, readout_inputs
from decision_consistency_softening import soften, fit_ratio


class DecisionConsistencyTests(unittest.TestCase):
    def test_js_is_symmetric_zero_at_identity_and_differentiable(self):
        a = torch.tensor([[2., -1., .5], [-2., 1., 0.]], requires_grad=True)
        b = torch.tensor([[0., 1., 0.], [1., 0., 2.]])
        torch.testing.assert_close(js_divergence(a, a), torch.zeros(2), atol=1e-7, rtol=0)
        torch.testing.assert_close(js_divergence(a, b), js_divergence(b, a))
        loss = js_divergence(a, b).mean()
        self.assertGreater(float(loss), 0.)
        self.assertLess(float(loss), np.log(2))
        loss.backward()
        self.assertGreater(float(a.grad.abs().sum()), 0.)

    def test_pair_weights_equalize_vessels_not_pair_counts(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1, 1, 2, 2], product=['a', 'b', 'c', 'a', 'b']))
        pairs = same_ship_pairs(meta)
        weights = pair_weights(meta, pairs)
        ships = meta.iloc[pairs[:, 0]].mmsi.to_numpy()
        self.assertAlmostEqual(float(weights[ships == 1].sum()), .5)
        self.assertAlmostEqual(float(weights[ships == 2].sum()), .5)

    def test_agreeing_wrong_predictions_are_not_both_correct(self):
        meta = pd.DataFrame(dict(mmsi=[1, 1], product=['a', 'b'], class_id=[0, 0]))
        row = pair_metrics(torch.tensor([[0., 2.], [0., 3.]]), meta)
        self.assertEqual(row['both_correct'], 0.)
        self.assertEqual(row['same_wrong'], 100.)
        self.assertEqual(row['disagree'], 0.)

    def test_prior_only_preserves_kernel_and_changes_scores(self):
        raw = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        changed = raw*torch.tensor([2., .5])
        phi, prior = readout_inputs(raw, changed, torch.eye(2), torch.zeros(2), 'prior')
        torch.testing.assert_close(phi[:, :-1], raw)
        torch.testing.assert_close(prior, changed)
        other_phi, other_prior = readout_inputs(raw, changed, torch.eye(2), torch.zeros(2), 'both')
        torch.testing.assert_close(other_prior, prior)
        self.assertFalse(torch.equal(phi@phi.T, other_phi@other_phi.T))

    def test_softening_calibration_ignores_common_score_offsets(self):
        reference = np.array([[1., 0., -1.], [2., -1., 0.], [0., 1., 3.]])
        changed = .8*reference+np.array([[10.], [-20.], [5.]])
        ratio = fit_ratio(reference, changed, np.array([[0, 1], [1, 2]]), np.array([.5, .5]))
        self.assertAlmostEqual(ratio, .8)
        original = torch.from_numpy(reference)
        result = soften(original, ratio)
        torch.testing.assert_close(result.mean(1), original.mean(1))
        torch.testing.assert_close(result.argmax(1), original.argmax(1))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
