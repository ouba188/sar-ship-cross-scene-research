"""Counterexamples for rejection, correspondence, axes and layer bookkeeping."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from sar_port.matching import (RegionalMatcher, absolute_reliability,
                               late_comparison, sinkhorn)


def configuration(**updates):
    values = dict(classes=2, third_layer=False, match_dim=4, layer_matching="concat",
                  pair_matching="paired", axis_matching=True, dustbin=True,
                  rejection_gate=True, class_competition=True, port_aggregation="flat",
                  geometry_weight=.25, scale_weight=0., layer_weight=0.,
                  residual_scale=.1, match_temperature=.3, sinkhorn_steps=35)
    values.update(updates)
    return SimpleNamespace(**values)


def tokens(batch, channels, gradients=False):
    return dict(desc=torch.randn(batch, 2, 28, channels, requires_grad=gradients),
                mass=torch.rand(batch, 2, 28) + .2,
                token_prior=torch.rand(batch, 2, 2, 28) + .3,
                available=torch.ones(batch, 2, dtype=torch.bool),
                write=torch.rand(batch, 2, 28, 3, 3) / 28,
                port_index=torch.arange(batch))


class MatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(53)

    def test_rejection_strength_scales_write_not_conditional_value(self):
        query_value = torch.tensor([[[1., 0.], [0., 1.]]])
        reference_value = torch.tensor([[[.2, 1.], [.8, -.5]]])
        plan = torch.tensor([[[[[.3, .1], [.1, .3]], [[.1, .3], [.3, .1]]]]])
        orientation = torch.tensor([[[.4, .6]]])
        metric = torch.ones(1, 1, 2)
        normal_distance = late_comparison(query_value, reference_value, plan, orientation, metric)
        tiny_distance = late_comparison(query_value, reference_value, plan * .02, orientation, metric)
        torch.testing.assert_close(normal_distance, tiny_distance)
        real = (plan * orientation[..., None, None]).sum(2)
        input_mass = torch.full((1, 1, 2), .5)
        torch.testing.assert_close(absolute_reliability(real * .02, input_mass),
                                   .02 * absolute_reliability(real, input_mass))

        matcher = RegionalMatcher(configuration())
        query, reference = tokens(1, 768), tokens(2, 768)
        labels, legal = torch.tensor([0, 1]), torch.ones(1, 2, dtype=torch.bool)

        def run(scale):
            def fixed_plan(cost, qa, ra, allowed, *args):
                return torch.diag_embed(qa * (.6 * scale)) * allowed
            with patch("sar_port.matching.sinkhorn", side_effect=fixed_plan):
                return matcher(query, reference, labels, legal)

        normal, tiny = run(1.), run(.02)
        self.assertGreater(normal["delta"].abs().max().item(), 1e-8)
        torch.testing.assert_close(tiny["delta"], normal["delta"] * .02, rtol=3e-5, atol=1e-8)
        torch.testing.assert_close(tiny["reliability"], normal["reliability"] * .02)
        # The same counterexample recreates the old failure when the absolute gate is disabled.
        matcher.config.rejection_gate = False
        normal, tiny = run(1.), run(.02)
        torch.testing.assert_close(tiny["delta"], normal["delta"], rtol=3e-5, atol=1e-7)

    def test_no_legal_references_produces_zero_residual(self):
        matcher = RegionalMatcher(configuration())
        result = matcher(tokens(2, 768), tokens(3, 768), torch.tensor([0, 0, 1]),
                         torch.zeros(2, 3, dtype=torch.bool))
        for key in ("delta", "evidence", "reliability"):
            self.assertTrue(torch.isfinite(result[key]).all())
            self.assertEqual(result[key].abs().max().item(), 0)

    def test_correspondence_survives_equal_marginals(self):
        query = torch.tensor([[[1., 0.], [0., 1.]]])
        reference = query.clone()
        aligned = torch.tensor([[[[[.5, 0.], [0., .5]]]]])
        crossed = torch.tensor([[[[[0., .5], [.5, 0.]]]]])
        aggregate = torch.full_like(aligned, .25)
        torch.testing.assert_close(aligned.sum(-1), crossed.sum(-1))
        torch.testing.assert_close(aligned.sum(-2), crossed.sum(-2))
        orientation, metric = torch.ones(1, 1, 1), torch.ones(1, 1, 2)
        distances = [late_comparison(query, reference, p, orientation, metric)
                     for p in (aligned, crossed, aggregate)]
        torch.testing.assert_close(distances[0], torch.zeros(1, 1, 2))
        torch.testing.assert_close(distances[1], torch.full((1, 1, 2), 2.))
        torch.testing.assert_close(distances[2], torch.full((1, 1, 2), .5))

    def test_reference_axis_flip_and_reference_order_are_invariant(self):
        matcher = RegionalMatcher(configuration())
        query, reference = tokens(1, 768), tokens(3, 768)
        labels = torch.tensor([0, 0, 1])
        legal = torch.ones(1, 3, dtype=torch.bool)
        original = matcher(query, reference, labels, legal)
        distance = (matcher.position[:, None] + matcher.position[None]).abs().sum(-1)
        distance += (matcher.extent[:, None] - matcher.extent[None]).abs().sum(-1)
        reverse = distance.argmin(-1)
        flipped = dict(reference, desc=reference["desc"][:, :, reverse],
                       mass=reference["mass"][:, :, reverse],
                       token_prior=reference["token_prior"][:, :, :, reverse])
        result = matcher(query, flipped, labels, legal)
        for key in ("delta", "evidence", "reliability"):
            torch.testing.assert_close(original[key], result[key], rtol=3e-5, atol=2e-6)
        order = torch.tensor([2, 0, 1])
        reordered = {key: value[order] for key, value in reference.items()}
        result = matcher(query, reordered, labels[order], legal[:, order])
        for key in ("delta", "evidence", "reliability"):
            torch.testing.assert_close(original[key], result[key], rtol=3e-5, atol=2e-6)

    def test_every_feature_layer_and_view_receives_finite_gradients(self):
        for third_layer in (False, True):
            channels = 1792 if third_layer else 768
            for mode in ("concat", "same", "cross"):
                with self.subTest(third_layer=third_layer, mode=mode):
                    matcher = RegionalMatcher(configuration(third_layer=third_layer,
                                              layer_matching=mode, sinkhorn_steps=8))
                    query, reference = tokens(1, channels, True), tokens(2, channels, True)
                    output = matcher(query, reference, torch.tensor([0, 1]),
                                     torch.ones(1, 2, dtype=torch.bool))
                    (output["delta"].square().sum() + output["evidence"].square().sum()).backward()
                    for collection in (query, reference):
                        pieces = collection["desc"].grad.split(matcher.channels, -1)
                        for piece in pieces:
                            self.assertTrue(torch.isfinite(piece).all())
                            self.assertTrue((piece.abs().sum((0, 2, 3)) > 0).all())

    def test_forbidden_edges_cannot_consume_mass_without_dustbin(self):
        # Two disjoint same-layer blocks are feasible and must each retain their mass.
        allowed = torch.tensor([[[[1, 1, 0, 0], [1, 1, 0, 0],
                                   [0, 0, 1, 1], [0, 0, 1, 1]]]], dtype=torch.bool)
        mass = torch.full((1, 1, 4), .25)
        plan = sinkhorn(torch.zeros(1, 1, 4, 4), mass, mass, allowed, dustbin=False)
        self.assertEqual(plan[~allowed].abs().max().item(), 0)
        torch.testing.assert_close(plan.sum(-1), mass)
        torch.testing.assert_close(plan.sum(-2), mass)

    def test_same_layer_aggregate_does_not_reintroduce_cross_layer_edges(self):
        matcher = RegionalMatcher(configuration(layer_matching="same", pair_matching="aggregate"))
        result = matcher(tokens(1, 768), tokens(2, 768), torch.tensor([0, 1]),
                         torch.ones(1, 2, dtype=torch.bool), details=True)
        for polarization in result["trace"]:
            for region in polarization:
                transport = region["transport"]
                half = transport.shape[-1] // 2
                self.assertEqual(transport[..., :half, half:].abs().max().item(), 0)
                self.assertEqual(transport[..., half:, :half].abs().max().item(), 0)

    def test_duplicate_layer_evidence_does_not_gain_extra_votes(self):
        outputs = []
        for third_layer, mode in ((False, "concat"), (False, "same"), (False, "cross"),
                                  (True, "concat"), (True, "same"), (True, "cross")):
            cfg = configuration(third_layer=third_layer, layer_matching=mode, dustbin=False,
                                axis_matching=False, class_competition=False, geometry_weight=0.)
            matcher = RegionalMatcher(cfg)
            with torch.no_grad():
                matcher.metric[-1].weight.zero_()
                matcher.metric[-1].bias.zero_()
                for output in matcher.output:
                    output.weight[:, 0].fill_(1.)
                    output.weight[:, 1].fill_(-1.)
            channels = sum(matcher.channels)
            query, reference = tokens(1, channels), tokens(2, channels)
            for collection in (query, reference):
                collection["mass"].zero_()
                collection["mass"][..., :2] = 1
                collection["token_prior"].fill_(1.)
                collection["write"].zero_()
                collection["write"][..., 0, 0, 0] = 1
                collection["write"][..., 1, 1, 1] = 1
            repeat = 1 if mode == "concat" else len(matcher.channels)

            def identical_layers(desc, pol):
                key = desc.new_zeros(len(desc), 28, 4)
                value = torch.zeros_like(key)
                key[..., 0] = 1
                if len(desc) == 2:
                    value[0, :, 0] = 1
                    value[1, :, 1] = 2
                return key.repeat(1, repeat, 1), value.repeat(1, repeat, 1)

            with patch.object(matcher, "encode", side_effect=identical_layers):
                outputs.append(matcher(query, reference, torch.tensor([0, 1]),
                                       torch.ones(1, 2, dtype=torch.bool)))
        self.assertGreater(outputs[0]["delta"].abs().max().item(), 0)
        for output in outputs[1:]:
            for key in ("delta", "evidence", "reliability"):
                torch.testing.assert_close(output[key], outputs[0][key], rtol=3e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
