"""Synthetic CPU contracts; run directly or collect with pytest, without data files."""

import math
import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from torch.nn import functional as F

from frequency_joint_core import comparison_loss, visual_scores
from learned_frequency_model import FrequencyResidual


def local_example():
    # Channel x position: unequal column lengths expose a wrong reshape/global cosine.
    tile = torch.zeros(16, 4)
    tile[torch.arange(4), torch.arange(4)] = torch.tensor([1., 2., 3., 4.])
    repeated = torch.zeros_like(tile)
    repeated[0] = torch.tensor([1., 2., 3., 4.])
    query = tile.flatten().repeat(4) / math.sqrt(30)
    support = torch.stack([
        tile, tile[:, [3, 1, 2, 0]], tile[:, [1, 2, 3, 0]], repeated,
    ]).flatten() / math.sqrt(30)
    # Last band: directed nearest means are 1/4 and 1, so symmetric mean is 5/8.
    evidence = torch.tensor([
        [1., 1., 1.], [13 / 30, .5, 1.], [0., 0., 1.], [1 / 30, .25, .625],
    ])
    return query, support, evidence


class FrequencyJointTest(unittest.TestCase):
    def assert_close(self, actual, expected):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)

    def assert_nonzero_gradient(self, tensor):
        self.assertIsNotNone(tensor.grad)
        self.assertTrue(torch.isfinite(tensor.grad).all().item())
        self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_hand_computed_local_matching_and_four_band_sum(self):
        query, support, evidence = local_example()
        for active in ([0], [1], [2], [3], [0, 1, 2, 3]):
            with self.subTest(active=active):
                mask = torch.zeros(4, 1)
                mask[active] = 1
                embedding = torch.stack([support, query]).reshape(2, 4, 64)
                scores, counts = visual_scores(
                    (embedding * mask).flatten(1), torch.zeros(2, 2),
                    torch.tensor([3]), support_n=1,
                )
                expected = torch.zeros(1, 8)
                expected[0, 3] = evidence[active].mean(1).sum()
                self.assert_close(scores, expected)
                self.assert_close(counts, torch.tensor([0, 0, 0, 1, 0, 0, 0, 0]))

    def test_band_gate_amplitude_survives_normalization(self):
        query, support, evidence = local_example()
        embedding = torch.stack([support, query]).reshape(2, 4, 64)
        gates = torch.tensor([[.64, .16, .16, .04], [.04, .16, .16, .64]])
        scores, _ = visual_scores(
            (embedding * gates.sqrt().unsqueeze(-1)).flatten(1),
            torch.zeros(2, 2), torch.tensor([0]), support_n=1,
        )
        expected = (evidence.mean(1) * (gates[0] * gates[1]).sqrt()).sum()
        self.assert_close(scores[0, 0], expected)
        uniform, _ = visual_scores(
            (embedding * .5).flatten(1), torch.zeros(2, 2),
            torch.tensor([0]), support_n=1,
        )
        self.assertGreater(abs((scores - uniform)[0, 0].item()), .05)
        scaled = (embedding * gates.sqrt().unsqueeze(-1)).flatten(1)
        scaled[0] *= 2
        scaled[1] *= 3
        amplified, _ = visual_scores(
            scaled, torch.zeros(2, 2), torch.tensor([0]), support_n=1,
        )
        self.assert_close(amplified, scores * 6)

    def test_global_condition_attention_shared_by_three_evidences(self):
        query, support, evidence = local_example()
        amplitudes = torch.tensor([[.7, .2, .8, .4], [.3, .9, .4, .6],
                                   [.5, .6, .3, .8]])
        embedding = torch.stack([query, support, query]).reshape(3, 4, 64)
        embedding = (embedding * amplitudes.unsqueeze(-1)).flatten(1)
        for support_conditions, distances in (
            ([[.5, 0.], [-.5, 1.]], [.25, 1.25]),
            ([[2., 0.], [0., 0.]], [4., 0.]),
        ):
            with self.subTest(distances=distances):
                conditions = torch.tensor(support_conditions + [[0., 0.]])
                scores, _ = visual_scores(
                    embedding, conditions, torch.tensor([5, 5]), support_n=2,
                )
                # Scalar logistic odds use analytic cosines, never the core's intermediates.
                expected = 0.
                for band, (global_cos, aligned, nearest) in enumerate(evidence.tolist()):
                    gap = (1 - global_cos) / .2 - (distances[0] - distances[1]) / .5
                    other_weight = 1 / (1 + math.exp(gap))
                    first = amplitudes[2, band] * amplitudes[0, band]
                    second = (amplitudes[2, band] * amplitudes[1, band]
                              * (global_cos + aligned + nearest) / 3)
                    expected += (1 - other_weight) * first + other_weight * second
                self.assert_close(scores[0, 5], expected)
                self.assertEqual(torch.count_nonzero(scores[:, [0, 1, 2, 3, 4, 6, 7]]), 0)

    def test_joint_support_permutation_and_class_relabeling(self):
        generator = torch.Generator(device="cpu").manual_seed(23)
        embedding = F.normalize(torch.randn(12, 256, generator=generator), dim=1)
        conditions = torch.rand(12, 2, generator=generator)
        labels = torch.tensor([0, 2, 0, 5, 2, 5, 0, 1, 2])
        scores, counts = visual_scores(embedding, conditions, labels, support_n=9)
        order = torch.tensor([8, 2, 5, 0, 7, 1, 6, 4, 3, 9, 10, 11])
        reordered, reordered_counts = visual_scores(
            embedding[order], conditions[order], labels[order[:9]], support_n=9,
        )
        self.assert_close(reordered, scores)
        self.assert_close(reordered_counts, counts)
        rename = torch.tensor([7, 5, 3, 1, 6, 4, 2, 0])
        renamed, renamed_counts = visual_scores(
            embedding, conditions, rename[labels], support_n=9,
        )
        self.assert_close(renamed[:, rename], scores)
        self.assert_close(renamed_counts[rename], counts)

    def test_default_support_n_and_unsupported_class_zero(self):
        generator = torch.Generator(device="cpu").manual_seed(31)
        embedding = F.normalize(torch.randn(67, 256, generator=generator), dim=1)
        conditions = torch.rand(67, 2, generator=generator)
        labels = torch.arange(64) % 7
        scores, counts = visual_scores(embedding, conditions, labels)
        explicit, explicit_counts = visual_scores(embedding, conditions, labels, support_n=64)
        self.assertEqual(scores.shape, (3, 8))
        self.assert_close(scores, explicit)
        self.assert_close(counts, explicit_counts)
        self.assertEqual(counts.sum().item(), 64)
        self.assertEqual(counts[7].item(), 0)
        self.assert_close(scores[:, 7], torch.zeros(3))

    def test_hard_negative_masks_true_and_under_supported_classes(self):
        counts = torch.tensor([2, 2, 1, 0, 3, 0, 0, 0])
        labels = torch.tensor([0, 1, 2, 3, 4])
        logits = torch.tensor([[5., 4., 100., 90., 3., 80., 70., 60.]]).repeat(5, 1)
        logits[torch.arange(5), labels] = 200
        logits.requires_grad_()
        visual = torch.zeros(5, 8, requires_grad=True)
        loss, valid, negative = comparison_loss(visual, counts, logits, labels)
        self.assert_close(valid, torch.tensor([True, True, False, False, True]))
        self.assert_close(negative, torch.tensor([1, 0, 0, 0, 0]))
        self.assertTrue((negative[valid] != labels[valid]).all().item())
        self.assertTrue((counts[negative[valid]] >= 2).all().item())
        self.assert_close(loss, torch.tensor(math.log(2)))
        loss.backward()
        expected_gradient = torch.zeros(5, 8)
        for row in (0, 1, 4):
            expected_gradient[row, labels[row]] = -5 / 3
            expected_gradient[row, negative[row]] = 5 / 3
        self.assert_close(visual.grad, expected_gradient)
        self.assertIsNone(logits.grad)

    def test_class_balanced_softplus_uses_visual_margin_and_temperature(self):
        labels = torch.tensor([0, 0, 0, 1])
        counts = torch.tensor([2, 2, 0, 0, 0, 0, 0, 0])
        visual = torch.zeros(4, 8)
        visual[torch.arange(4), labels] = torch.tensor([0., .1, .2, -.1])
        visual.requires_grad_()
        logits = torch.tensor([[9., 8., 100., 0., 0., 0., 0., 0.]]).repeat(4, 1)
        loss, valid, negative = comparison_loss(visual, counts, logits, labels)
        losses = [math.log1p(math.exp(value)) for value in (0., -1., -2., 1.)]
        expected = (sum(losses[:3]) / 3 + losses[3]) / 2
        self.assertTrue(valid.all().item())
        self.assert_close(negative, torch.tensor([1, 1, 1, 0]))
        self.assert_close(loss, torch.tensor(expected))
        self.assertGreater(abs(loss.item() - sum(losses) / 4), .1)
        loss.backward()
        self.assert_nonzero_gradient(visual)

    def test_no_valid_loss_is_finite_differentiable_zero(self):
        scenarios = (
            ([0] * 8, [0, 1]),
            ([1] * 8, [0, 1]),
            ([2, 0, 0, 0, 0, 0, 0, 0], [0, 0]),
            ([0, 2, 2, 0, 0, 0, 0, 0], [0, 3]),
        )
        for counts, labels in scenarios:
            with self.subTest(counts=counts, labels=labels):
                visual = torch.ones(2, 8, requires_grad=True)
                logits = torch.zeros(2, 8, requires_grad=True)
                loss, valid, _ = comparison_loss(
                    visual, torch.tensor(counts), logits, torch.tensor(labels),
                )
                self.assertFalse(valid.any().item())
                self.assertTrue(torch.isfinite(loss).item())
                self.assertEqual(loss.item(), 0.)
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assert_close(visual.grad, torch.zeros_like(visual))
                self.assertIsNone(logits.grad)

    def test_empty_support_scores_and_loss_backward(self):
        embedding = torch.ones(2, 256, requires_grad=True)
        scores, counts = visual_scores(
            embedding, torch.zeros(2, 2), torch.empty(0, dtype=torch.long), support_n=0,
        )
        self.assert_close(scores, torch.zeros(2, 8))
        self.assert_close(counts, torch.zeros(8, dtype=torch.long))
        loss, valid, _ = comparison_loss(
            scores, counts, torch.zeros(2, 8), torch.tensor([0, 1]),
        )
        self.assertFalse(valid.any().item())
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assert_close(embedding.grad, torch.zeros_like(embedding))

    def test_comparison_gradients_reach_query_and_both_candidate_supports(self):
        generator = torch.Generator(device="cpu").manual_seed(47)
        embedding = F.normalize(torch.randn(8, 256, generator=generator), dim=1)
        embedding.requires_grad_()
        scores, counts = visual_scores(
            embedding, torch.rand(8, 2, generator=generator),
            torch.tensor([0, 0, 1, 1, 2, 2]), support_n=6,
        )
        logits = torch.tensor([[9., 8., -5., 0., 0., 0., 0., 0.]]).repeat(2, 1)
        logits.requires_grad_()
        loss, valid, negative = comparison_loss(scores, counts, logits, torch.tensor([0, 1]))
        self.assertTrue(valid.all().item())
        self.assert_close(negative, torch.tensor([1, 0]))
        loss.backward()
        self.assert_nonzero_gradient(embedding)
        for row in (0, 1, 2, 3, 6, 7):
            self.assertGreater(embedding.grad[row].abs().sum().item(), 0)
        self.assert_close(embedding.grad[4:6], torch.zeros(2, 256))
        self.assertIsNone(logits.grad)

    def test_frequency_residual_blocks_and_gate_receive_comparison_gradients(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(53)
            model = FrequencyResidual("selective").cpu()
            features = torch.randn(6, 512, 8, 8, device="cpu", requires_grad=True)
            conditions = torch.rand(6, 2, device="cpu")
            embedding, weights = model(features, conditions)
            embedding.retain_grad()
            weights.retain_grad()
            scores, counts = visual_scores(
                embedding, conditions, torch.tensor([0, 0, 1, 1]), support_n=4,
            )
            logits = torch.tensor([[9., 8., 0., 0., 0., 0., 0., 0.],
                                   [8., 9., 0., 0., 0., 0., 0., 0.]], requires_grad=True)
            loss, valid, negative = comparison_loss(scores, counts, logits, torch.tensor([0, 1]))
            self.assertTrue(valid.all().item())
            self.assert_close(negative, torch.tensor([1, 0]))
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            for tensor in (features, embedding, weights, model.gate[-1].weight,
                           model.gate[-1].bias):
                self.assert_nonzero_gradient(tensor)
            for row in range(6):
                self.assertGreater(embedding.grad[row].abs().sum().item(), 0)
            for block in model.blocks:
                self.assert_nonzero_gradient(block[0].weight)
                self.assert_nonzero_gradient(block[3].weight)
            self.assertIsNone(logits.grad)
            self.assertEqual(embedding.device.type, "cpu")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
