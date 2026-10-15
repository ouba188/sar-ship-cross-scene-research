"""Small synthetic CPU/float64 checks; no data, training, or GPU required."""

import unittest

import numpy as np
import torch
from torch.nn import functional as F

from support_task_loss import fusion_ce, torch_scores, torch_support


def numpy_support(raw, delta, cols, dose):
    result = raw.copy()
    if dose:
        original = raw[:, cols]
        center = original.mean(axis=0)
        changed = original + dose * delta
        changed -= changed.mean(axis=0)
        scale = np.linalg.norm(original - center) / np.linalg.norm(changed)
        result[:, cols] = center + scale * changed
    return result


def primal_scores(support, query, prior_s, prior_q, labels, strength):
    # Solve in feature space, independently of the implementation's dual kernel.
    weights = np.empty(len(labels), dtype=np.float64)
    present = np.unique(labels)
    for label in present:
        mask = labels == label
        weights[mask] = len(labels) / (len(present) * mask.sum())
    targets = np.full(prior_s.shape, -1.0)
    targets[np.arange(len(labels)), labels] = 1.0
    gram = support.T @ (weights[:, None] * support)
    rhs = support.T @ (weights[:, None] * (targets - prior_s))
    beta = np.linalg.solve(gram + strength * np.eye(support.shape[1]), rhs)
    return prior_q + query @ beta


def numpy_fusion_ce(base, expert, labels):
    log_base = base - np.logaddexp.reduce(base, axis=1, keepdims=True)
    log_expert = expert - np.logaddexp.reduce(expert, axis=1, keepdims=True)
    rows = np.arange(len(labels))
    losses = np.log(2.0) - np.logaddexp(
        log_base[rows, labels], log_expert[rows, labels])
    # Match the class-balanced query reduction via independent classwise means.
    return float(np.mean([losses[labels == label].mean()
                          for label in np.unique(labels)]))


class SupportTaskLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        self.rng = np.random.default_rng(314159)
        self.support = self.rng.normal(size=(9, 6)) + np.arange(6) * 0.3
        self.query = self.rng.normal(size=(5, 6)) - 0.2
        self.prior_s = self.rng.normal(scale=0.4, size=(9, 4))
        self.prior_q = self.rng.normal(scale=0.4, size=(5, 4))
        self.base = self.rng.normal(size=(5, 4))
        # Unequal counts and absent final class catch weighting/class-space errors.
        self.labels_s = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2], dtype=np.int64)
        self.labels_q = np.array([3, 0, 2, 1, 3], dtype=np.int64)
        self.cols = np.array([4, 1, 3], dtype=np.int64)
        self.delta = self.rng.normal(size=(9, 3)) + 0.4
        self.dose = 0.65
        self.strength = 1.7

    @staticmethod
    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, device='cpu')

    def scores(self, delta, cols=None, dose=None):
        return torch_scores(
            self.tensor(self.support), self.tensor(self.query),
            self.tensor(self.prior_s), self.tensor(self.prior_q),
            torch.tensor(self.labels_s, dtype=torch.long), delta,
            self.cols if cols is None else cols,
            self.dose if dose is None else dose, self.strength)

    def reference_scores(self, delta, dose=None):
        support = numpy_support(self.support, delta, self.cols,
                                self.dose if dose is None else dose)
        return primal_scores(support, self.query, self.prior_s, self.prior_q,
                             self.labels_s, self.strength)

    def test_weighted_scores_match_independent_primal_with_missing_class(self):
        for cols in (self.cols, torch.tensor(self.cols, dtype=torch.long)):
            for dose in (0.0, self.dose):
                with self.subTest(index_type=type(cols).__name__, dose=dose):
                    actual = self.scores(self.tensor(self.delta), cols, dose)
                    expected = self.reference_scores(self.delta, dose)
                    self.assertEqual(actual.shape, (5, 4))
                    self.assertEqual(actual.dtype, torch.float64)
                    self.assertEqual(actual.device.type, 'cpu')
                    np.testing.assert_allclose(actual.detach().numpy(), expected,
                                               atol=2e-11, rtol=2e-11)

    def test_support_preserves_identity_centroid_energy_and_other_columns(self):
        raw, delta = self.tensor(self.support), self.tensor(self.delta)
        original_raw, original_delta = raw.clone(), delta.clone()
        unchanged = np.setdiff1d(np.arange(raw.shape[1]), self.cols)
        for cols in (self.cols, torch.tensor(self.cols, dtype=torch.long)):
            with self.subTest(index_type=type(cols).__name__):
                identity = torch_support(raw, delta, cols, 0.0)
                torch.testing.assert_close(identity, original_raw, rtol=0, atol=0)
                changed = torch_support(raw, delta, cols, self.dose)
                self.assertEqual(changed.dtype, torch.float64)
                self.assertEqual(changed.device.type, 'cpu')
                torch.testing.assert_close(changed[:, unchanged], raw[:, unchanged],
                                           rtol=0, atol=0)
                before, after = raw[:, cols], changed[:, cols]
                torch.testing.assert_close(after.mean(0), before.mean(0),
                                           rtol=0, atol=2e-14)
                before_energy = (before - before.mean(0)).square().sum(1).mean()
                after_energy = (after - after.mean(0)).square().sum(1).mean()
                torch.testing.assert_close(after_energy, before_energy,
                                           rtol=2e-13, atol=2e-13)
                self.assertGreater(float((after - before).norm()), 0.1)
                np.testing.assert_allclose(
                    changed.numpy(),
                    numpy_support(self.support, self.delta, self.cols, self.dose),
                    atol=2e-13, rtol=2e-13)
                torch.testing.assert_close(raw, original_raw, rtol=0, atol=0)
                torch.testing.assert_close(delta, original_delta, rtol=0, atol=0)

    def test_delta_gradient_through_normalization_solve_and_fusion(self):
        delta = self.tensor(self.delta).requires_grad_()
        base = self.tensor(self.base)
        labels = torch.tensor(self.labels_q, dtype=torch.long)
        loss = fusion_ce(base, self.scores(delta), labels)
        self.assertTrue(loss.requires_grad, 'Fusion/solver detached delta entirely')
        gradient, = torch.autograd.grad(loss, delta)
        self.assertTrue(bool(torch.isfinite(gradient).all()))

        random_direction = self.rng.normal(size=self.delta.shape)
        one_row = np.zeros_like(self.delta)
        one_row[2] = [0.7, -0.4, 1.1]
        shifted = self.support[:, self.cols] + self.dose * self.delta
        # Translation and radial directions isolate centering and energy gradients.
        directions = dict(random=random_direction, one_row=one_row,
                          translation=np.broadcast_to([0.8, -0.3, 0.5], self.delta.shape),
                          radial=shifted - shifted.mean(axis=0))
        epsilon = 1e-5
        derivatives = []
        for name, direction in directions.items():
            direction = direction / np.linalg.norm(direction)
            numerical = []
            for sign in (-1, 1):
                offset = self.delta + sign * epsilon * direction
                expected = numpy_fusion_ce(
                    self.base, self.reference_scores(offset), self.labels_q)
                with torch.no_grad():
                    actual = float(fusion_ce(base, self.scores(self.tensor(offset)), labels))
                self.assertAlmostEqual(actual, expected, delta=2e-11)
                numerical.append(expected)
            finite_difference = (numerical[1] - numerical[0]) / (2 * epsilon)
            automatic = float((gradient * self.tensor(direction)).sum())
            with self.subTest(direction=name):
                self.assertAlmostEqual(
                    automatic, finite_difference,
                    delta=2e-8 + 2e-4 * abs(finite_difference),
                    msg='Check centering/energy, KSS solve and KQS gradient paths')
            derivatives.append(abs(finite_difference))
        self.assertGreater(max(derivatives[:2]), 1e-5,
                           'Fixture must exercise a nonzero task-loss derivative')

    def test_fusion_ce_uses_log_of_mean_probabilities(self):
        base = self.tensor([[6, -2, 1, 0], [-1, 4, 0, 2], [0, -3, 5, 1]])
        expert = self.tensor([[-2, 1, 5, 0], [4, -1, 0, 1], [3, 2, -1, 0]])
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        expected = numpy_fusion_ce(base.numpy(), expert.numpy(), labels.numpy())
        actual = fusion_ce(base, expert, labels)
        self.assertEqual(actual.ndim, 0)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertAlmostEqual(float(actual), expected, delta=2e-13)
        probabilities = 0.5 * base.softmax(1) + 0.5 * expert.softmax(1)
        wrong_loss = float(F.cross_entropy(probabilities, labels))
        self.assertGreater(abs(wrong_loss - expected), 0.05)


if __name__ == '__main__':
    unittest.main()
