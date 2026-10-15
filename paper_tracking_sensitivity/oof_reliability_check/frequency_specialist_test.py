"""Focused objective and support-only selection tests."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import unittest
import numpy as np
import torch
from frequency_specialist_core import rescue_responsibility, class_margin, branch_objective, per_band_scores
from frequency_specialist_policy import loo_scores, support_static, support_utility, local_choice
from frequency_reliability_core import coefficients


class SpecialistTest(unittest.TestCase):
    def test_frozen_rescue_allocation_has_floor_and_preserves_total(self):
        scores = torch.tensor([[[1., 0], [0., 1], [.5, 0], [.1, 0]], [[1., 0]]*4], requires_grad=True)
        ref, y = torch.tensor([[0., 1], [1., 0]]), torch.tensor([0, 0])
        r = rescue_responsibility(scores, ref, y)
        self.assertFalse(r.requires_grad)
        torch.testing.assert_close(r.mean(1), torch.ones(2))
        self.assertTrue(bool((r >= .5).all()))
        self.assertGreater(float(r[0, 0]), float(r[0, 1]))
        torch.testing.assert_close(r[1], torch.ones(4))

    def test_correct_boundary_protection_penalizes_a_flip(self):
        ref, y = torch.tensor([[1., 0], [0., 1]]), torch.tensor([0, 0])
        good = torch.tensor([[[.2, 0]]*4, [[0., 1]]*4])
        bad = good.clone()
        bad[0, 0] = torch.tensor([0., .2])
        _, _, pg = branch_objective(good, ref, y)
        _, _, pb = branch_objective(bad, ref, y)
        self.assertEqual(float(pg), 0.)
        self.assertGreater(float(pb), 0.)

    def test_all_bands_receive_classification_gradients(self):
        scores = torch.zeros(3, 4, 3, requires_grad=True)
        ref = torch.tensor([[1., 0, 0], [1., 0, 0], [1., 0, 0]])
        y = torch.tensor([0, 1, 2])
        r = rescue_responsibility(scores.detach(), ref, y)
        loss, _, _ = branch_objective(scores, ref, y, r)
        loss.backward()
        self.assertTrue(bool((scores.grad.abs().sum((0, 2)) > 0).all()))

    def test_single_band_solve_matches_numpy(self):
        from frequency_band_core import old_kernel, score_embedding
        rng = np.random.default_rng(7)
        data = dict(phi=rng.normal(size=(8, 3)), prior=rng.normal(size=(8, 3)),
                    mean=rng.normal(size=(8, 2)), conditions=rng.normal(size=(8, 2)))
        z = rng.normal(size=(8, 4, 64))
        z /= np.linalg.norm(z, axis=2, keepdims=True)
        y, support = np.array([0, 0, 1, 2]), np.arange(4)
        args = [torch.tensor(data[k]) for k in ('phi', 'prior', 'mean', 'conditions')]
        actual = per_band_scores(args, torch.tensor((z*.5).reshape(8, 256)), torch.tensor(y), .2, support_n=4)
        k, c = old_kernel(data, support, .2)
        for b in range(4):
            expected, _ = score_embedding(data, z[:, b], support, y, k, c)
            np.testing.assert_allclose(actual[:, b].numpy(), expected, atol=1e-12)

    def test_loo_recomputes_class_weights_with_remaining_labels(self):
        x = np.array([[1., 0], [2., 0], [0., 1], [0., 2]])
        y, prior = np.array([0, 0, 0, 1]), np.zeros((4, 2))
        scores = loo_scores(x@x.T, prior, y)
        held, keep = 0, np.array([1, 2, 3])
        weights = np.array([.75, .75, 1.5])
        target = 2*np.eye(2)[y[keep]]-1
        beta = np.linalg.solve(x[keep].T@(weights[:, None]*x[keep])+8*np.eye(2), x[keep].T@(weights[:, None]*target))
        np.testing.assert_allclose(scores[held], x[held]@beta)

    def test_support_selector_ties_and_no_utility_keep_main(self):
        main = np.array([[2., 0], [0, 2.]])
        cand = np.tile(main[:, None], (1, 4, 1))
        out, index = local_choice(main, cand, np.ones((2, 4)))
        np.testing.assert_array_equal(out, main)
        np.testing.assert_array_equal(index, 0)
        self.assertEqual(support_static(np.concatenate([main[:, None], cand], 1), np.array([0, 1])), 0)


if __name__ == '__main__':
    unittest.main()
