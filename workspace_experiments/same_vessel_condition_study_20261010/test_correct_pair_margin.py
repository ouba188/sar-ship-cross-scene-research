import unittest
import numpy as np
import torch
from correct_pair_margin import normalized_margin, auxiliary_loss, rewire
from correct_pair_capacity import oracle_margin


class CorrectMarginTests(unittest.TestCase):
    def test_capacity_bound_distinguishes_correctable_errors(self):
        effects = np.array([[1., -1.]])
        near, _ = oracle_margin(np.array([0., .1]), effects, 0)
        far, _ = oracle_margin(np.array([0., 1.]), effects, 0)
        self.assertAlmostEqual(near, .1)
        self.assertAlmostEqual(far, -.8)

    def test_capacity_requires_beating_all_rivals_at_once(self):
        margin, gain = oracle_margin(np.array([0., .1, .1]), np.array([[0., 1., -1.]]), 0)
        self.assertAlmostEqual(margin, -.1)
        self.assertAlmostEqual(gain[0], 0.)

    def test_scale_and_offset_cannot_improve_margin(self):
        scores = torch.tensor([[1., 2., -.3], [2., -1., .4]])
        y = torch.tensor([0, 0])
        expected = normalized_margin(scores, y)
        torch.testing.assert_close(normalized_margin(.03*scores+4., y), expected, atol=2e-5, rtol=2e-5)
        self.assertLess(float(expected[0]), 0.)
        self.assertGreater(float(expected[1]), 0.)

    def test_wrong_agreement_penalized_and_weak_endpoint_trained(self):
        scores = torch.tensor([[[3., 0., 0.], [0., 3., 0.]]], requires_grad=True)
        y = torch.tensor([[0, 0]])
        loss = auxiliary_loss(scores, y, 'paired')
        self.assertGreater(float(loss), float(auxiliary_loss(scores, y, 'point')))
        loss.backward()
        self.assertEqual(float(scores.grad[0, 0].abs().sum()), 0.)
        self.assertLess(float(scores.grad[0, 1, 0]), 0.)
        wrong = torch.tensor([[[0., 3., 0.], [0., 3., 0.]]])
        self.assertGreater(float(auxiliary_loss(wrong, y, 'paired')), 0.)

    def test_rewire_preserves_endpoints_and_classes(self):
        pairs = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        ships = np.repeat([10, 20, 30, 40], 2)
        changed = rewire(pairs, labels, ships)
        np.testing.assert_array_equal(changed[:, 0], pairs[:, 0])
        np.testing.assert_array_equal(np.sort(changed[:, 1]), np.sort(pairs[:, 1]))
        np.testing.assert_array_equal(labels[changed[:, 0]], labels[changed[:, 1]])
        self.assertTrue(np.all(ships[changed[:, 0]] != ships[changed[:, 1]]))

    def test_unbreakable_identity_does_not_invent_samples(self):
        pairs = np.array([[0, 1], [0, 2]])
        np.testing.assert_array_equal(rewire(pairs, np.zeros(3), np.ones(3)), pairs)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
