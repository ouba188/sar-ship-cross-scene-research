"""Behavioral checks for area transport and the attention isolation boundary."""
import io
import unittest
from dataclasses import asdict
import numpy as np
import torch

from sar_port.two_stage.contracts import EncodedShip, LevelGeometry, ShipGeometry
from sar_port.two_stage.h import HConfig, HModel, partial_transport


def example(regions=2, seed=4, missing_vh=False, gradients=False):
    generator = torch.Generator().manual_seed(seed)
    levels, content, spatial = [], [], []
    for l, size in enumerate((16, 8, 4)):
        valid = torch.ones(2, 1, size, size)
        slots = torch.zeros(2, 3, size, size)
        slots[:, 0] = 1
        slots[:, 1, :, :size // 2] = 1
        slots[:, 2, :, size // 2:] = 1
        slot_valid = torch.ones(2, 3, dtype=torch.bool)
        if regions == 1:
            slot_valid[:, 1:] = False
        weights = torch.zeros(regions, 2, size, size)
        for i in range(regions):
            weights[i, :, :, i * size // regions:(i + 1) * size // regions] = 1
        match_valid = torch.ones(regions, 2, dtype=torch.bool)
        if regions > 1 and l > 0:
            match_valid[:] = False
        if missing_vh:
            valid[1] = 0
            slots[1] = 0
            slot_valid[1] = False
            weights[:, 1] = 0
            match_valid[:, 1] = False
        levels.append(LevelGeometry(valid, slots, slot_valid, weights, match_valid))
        content.append(torch.randn(2, 256, size, size, generator=generator) * valid)
        spatial.append((torch.randn(2, 64, size, size, generator=generator) * valid)
                       .requires_grad_(gradients))
    intervals = torch.tensor([[(i + .5) / regions - .5, 1 / regions]
                              for i in range(regions)])
    coverage = torch.ones(regions, 2, 3)
    if missing_vh:
        coverage[:, 1] = 0
    geometry = ShipGeometry(tuple(levels), torch.full((regions,), 4096 / regions), intervals, coverage,
                            True, regions, 64)
    return EncodedShip(tuple(content), tuple(spatial), geometry)


class TwoStageHTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(9)

    def model(self, **values):
        return HModel(HConfig(classes=2, descriptor_dim=12, sinkhorn_iterations=25,
                              **values)).set_stage("match")

    def test_transport_has_exact_capacities_and_zero_forbidden_edges(self):
        scores = torch.tensor([[.9, .8, -.5], [.4, .7, .2]], requires_grad=True)
        valid = torch.tensor([[True, False, False], [True, True, False]])
        a = torch.tensor([.8, .2])
        b = torch.tensor([.1, .3, .6])
        dustbin = torch.tensor(.1, requires_grad=True)
        plan = partial_transport(scores, valid, a, b, dustbin, iterations=80)
        torch.testing.assert_close(plan.sum(1), torch.cat((a, a.new_ones(1))))
        torch.testing.assert_close(plan.sum(0), torch.cat((b, b.new_ones(1))))
        self.assertEqual(float(plan[:-1, :-1][~valid].sum().detach()), 0.)
        self.assertTrue(bool((plan[:-1, :-1].sum(1) <= a + 1e-7).all()))
        loss = (plan[:-1, :-1] * scores).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(float(scores.grad[~valid].abs().sum()), 0.)

    def test_empty_edges_send_all_real_mass_to_dustbin(self):
        a, b = torch.tensor([.3, .7]), torch.tensor([1.])
        plan = partial_transport(torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bool),
                                 a, b, torch.tensor(0.))
        torch.testing.assert_close(plan[:-1, -1], a)
        torch.testing.assert_close(plan[-1, :-1], b)
        self.assertEqual(float(plan[:-1, :-1].sum()), 0.)

    def test_attention_cannot_change_transport_geometry_or_prototype_mixture(self):
        model = self.model().eval()
        query, refs = [example(2)], [example(2, 5), example(2, 6), example(2, 7)]
        before = model(query, refs, torch.tensor([0, 0, 1]), ["a", "a", "b"])
        with torch.no_grad():
            model.attention.score[-1].weight.normal_(0, 30)
        after = model(query, refs, torch.tensor([0, 0, 1]), ["a", "a", "b"])
        self.assertFalse(torch.allclose(before["weights"][0], after["weights"][0]))
        for ri in range(3):
            for name in ("transport", "geometry_cost", "beta", "evidence"):
                torch.testing.assert_close(before["details"][0]["pairs"][ri][name],
                                           after["details"][0]["pairs"][ri][name])
        for key in before["details"][0]["mixtures"]:
            torch.testing.assert_close(before["details"][0]["mixtures"][key]["weights"],
                                       after["details"][0]["mixtures"][key]["weights"])
        self.assertTrue(bool((after["weights"][0] >= .75 * query[0].geometry.mass).all()))
        self.assertTrue(bool((after["weights"][0] <= .75 * query[0].geometry.mass + .25).all()))

    def test_match_gradients_reach_both_sides_and_attention_but_not_h0(self):
        model = self.model()
        query = [example(2, gradients=True)]
        refs = [example(2, 8, gradients=True), example(2, 9, gradients=True)]
        output = model(query, refs, torch.tensor([0, 1]), ["a", "b"])
        losses = model.loss(output, torch.tensor([0]))
        losses["total"].backward()
        self.assertTrue(bool(query[0].spatial[0].grad.abs().sum() > 0))
        self.assertTrue(bool(refs[0].spatial[0].grad.abs().sum() > 0))
        self.assertTrue(bool(refs[1].spatial[0].grad.abs().sum() > 0))
        self.assertTrue(bool(model.attention.score[-1].weight.grad.abs().sum() > 0))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.h0.parameters()))
        model.train()
        self.assertFalse(model.h0.training)

    def test_no_support_falls_back_and_excludes_match_loss(self):
        model = self.model()
        output = model([example(1)], [example(2, 5)], torch.tensor([0]), ["a"],
                       torch.zeros(1, 1, dtype=torch.bool))
        torch.testing.assert_close(output["logits"], output["z0"])
        self.assertEqual(int(output["support"].sum()), 0)
        losses = model.loss(output, torch.tensor([0]))
        self.assertEqual(float(losses["match"].detach()), 0.)
        self.assertEqual(int(losses["match_samples"]), 0)

    def test_single_root_and_two_half_direction_equivalence(self):
        model = self.model()
        for nq, nr in ((1, 1), (1, 4), (2, 2)):
            query, ref = example(nq), example(nr, 6)
            pair = model.match_pair(query, ref)
            self.assertEqual(pair["transport"].shape, (nq + 1, nr + 1))
            self.assertEqual(pair["direction_solves"], 1)
            self.assertFalse(pair["direction_identifiable"])
        torch.testing.assert_close(model.region_weights(example(1)), torch.ones(1))

    def test_root_only_has_legal_same_or_adjacent_scale_combinations(self):
        model = self.model()
        pair = model.match_pair(example(1), example(2, 6, missing_vh=True))
        # Order: p, query layer, reference layer, skipping nonadjacent pairs.
        self.assertEqual(pair["beta"].shape[-1], 14)
        self.assertEqual(int((pair["beta"][0, 0] > 0).sum()), 2)
        self.assertEqual(float(pair["beta"][..., 7:].sum().detach()), 0.)

    def test_multiport_normalization_does_not_count_repeated_equal_evidence_as_votes(self):
        model = self.model().eval()
        query, a, b = [example(2)], example(2, 8), example(2, 9)
        two = model(query, [a, b], torch.tensor([0, 1]), ["p", "p"])
        duplicated = model(query, [a, a, b], torch.tensor([0, 0, 1]), ["p", "q", "p"])
        torch.testing.assert_close(two["evidence"], duplicated["evidence"])
        self.assertTrue(bool(((duplicated["evidence"] >= 0) & (duplicated["evidence"] <= 1)).all()))

    def test_base_rms_stage_selection_and_checkpoint_replay(self):
        model = self.model()
        ships = [example(1), example(2, 7)]
        rms = model.estimate_rms(ships)
        self.assertEqual(rms.shape, (2, 3))
        self.assertTrue(bool((rms > 0).all()))
        model.set_stage("base")
        self.assertTrue(all(p.requires_grad for p in model.h0.parameters()))
        self.assertFalse(model.dustbin.requires_grad)
        torch.nn.functional.cross_entropy(model.forward_base(ships), torch.tensor([0, 1])).backward()
        self.assertTrue(any(p.grad is not None for p in model.h0.parameters()))
        model.set_stage("match").eval()
        self.assertTrue(all(p.grad is None for p in model.h0.parameters()))
        expected = model([ships[0]], [ships[1]], torch.tensor([0]), ["p"])["logits"]
        buffer = io.BytesIO()
        torch.save(dict(config=asdict(model.config), state=model.state_dict()), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        loaded = HModel(HConfig(**saved["config"]))
        loaded.load_state_dict(saved["state"])
        loaded.set_stage("match").eval()
        actual = loaded([ships[0]], [ships[1]], torch.tensor([0]), ["p"])["logits"]
        torch.testing.assert_close(expected, actual)

    def test_alpha_zero_has_no_attention_parameters(self):
        model = self.model(alpha=0)
        self.assertIsNone(model.attention)
        self.assertFalse(any(name.startswith("attention.") for name, _ in model.named_parameters()))

    def test_zero_projection_cannot_remove_legal_support_or_match_supervision(self):
        model = self.model()
        query, reference = example(2), example(2, 8)
        before = model.match_pair(query, reference)
        with torch.no_grad():
            for parameter in model.projections.parameters():
                parameter.zero_()
        output = model([query], [reference], torch.tensor([0]), ["a"])
        after = output["details"][0]["pairs"][0]
        torch.testing.assert_close(before["edge_valid"], after["edge_valid"])
        self.assertTrue(bool(output["support"][0, 0]))
        self.assertEqual(int(model.loss(output, torch.tensor([0]))["match_samples"]), 1)
        self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_nonfinite_descriptor_is_reported_instead_of_changing_eligibility(self):
        model = self.model()
        query = example(2)
        query.spatial[0][0, 0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            model.match_pair(query, example(2, 9))

    def test_raw_fc_ablation_uses_content_and_has_correct_projection_width(self):
        model = self.model(descriptor_mode="raw_fc")
        query, ref = example(2), example(2, 9)
        for tensor in query.spatial + ref.spatial:
            tensor.zero_()
        for tensor in query.content + ref.content:
            tensor.requires_grad_(True)
        output = model([query], [ref], torch.tensor([0]), ["a"])
        model.loss(output, torch.tensor([0]))["total"].backward()
        self.assertTrue(bool(query.content[0].grad.abs().sum() > 0))
        self.assertTrue(bool(ref.content[0].grad.abs().sum() > 0))
        self.assertEqual(model.projections[0][0].in_features, 256)
        self.assertEqual(self.model().projections[0][0].in_features, 64)

    def test_disabled_scorers_have_no_parameters_and_uniform_legal_weights(self):
        model = self.model(learn_edge_weights=False, prototype_weighting="uniform")
        self.assertIsNone(model.edge_score)
        self.assertIsNone(model.prototype_score)
        self.assertFalse(any(name.startswith(("edge_score.", "prototype_score."))
                             for name, _ in model.named_parameters()))
        query, refs = [example(2)], [example(2, 5), example(2, 6)]
        output = model(query, refs, torch.tensor([0, 0]), ["a", "a"])
        beta = output["details"][0]["pairs"][0]["beta"]
        torch.testing.assert_close(beta[beta > 0], torch.full_like(beta[beta > 0], .5))
        pi = output["details"][0]["mixtures"][0, "a"]["weights"]
        torch.testing.assert_close(pi, torch.full_like(pi, .5))

    def test_attention_prototype_mode_omits_floor_but_preserves_shared_scorer(self):
        floor = self.model(prototype_floor=.4).eval()
        attention = self.model(prototype_weighting="attention", prototype_floor=.4).eval()
        with torch.no_grad():
            floor.prototype_score[-1].weight.normal_(0, 20)
        attention.load_state_dict(floor.state_dict())
        query, refs = [example(2)], [example(2, 5), example(2, 7)]
        result_floor = floor(query, refs, torch.tensor([0, 0]), ["a", "a"])
        result_attention = attention(query, refs, torch.tensor([0, 0]), ["a", "a"])
        pf = result_floor["details"][0]["mixtures"][0, "a"]["weights"]
        pa = result_attention["details"][0]["mixtures"][0, "a"]["weights"]
        torch.testing.assert_close(pf, .6 * pa + .2)
        self.assertIsNotNone(attention.prototype_score)

    def test_exact_union_fields_survive_device_transfer_and_control_pooling(self):
        model = self.model()
        query = example(2)
        ordinary = model.forward_base([query])
        slots = query.geometry.levels[0].slots.amax(0).clone()
        slots[:, :, 0:4] *= .1
        regions = query.geometry.levels[0].regions.amax(1).clone()
        regions[:, :, 0:4] *= .1
        query.geometry.union_slots = slots
        query.geometry.union_regions = regions
        moved = query.to("cpu")
        torch.testing.assert_close(moved.geometry.union_slots, slots)
        torch.testing.assert_close(moved.geometry.union_regions, regions)
        changed = model.forward_base([moved])
        self.assertFalse(torch.allclose(ordinary, changed))

    def test_complementary_polarizations_use_union_coverage_in_prototype_statistics(self):
        model = self.model()
        query = example(2)
        for level in query.geometry.levels:
            level.regions *= .5
        query.geometry.coverage.fill_(.5)
        # Each polarization covers half of each region, while their physical
        # union still covers it entirely (area remains the full region area).
        pair = model.match_pair(query, example(2, 8))
        torch.testing.assert_close(pair["stats"][:, 4], torch.ones(2))


if __name__ == "__main__":
    unittest.main()
