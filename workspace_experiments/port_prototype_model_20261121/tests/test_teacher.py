"""Native-operation preservation and teacher-measurement contract tests."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sar_port.teacher import CONDITIONS, make_interventions, measure_teacher, measure_utility_teacher


def sample_data():
    rng = np.random.default_rng(31)
    images = np.zeros((2, 2, 1, 16, 16), np.float32)
    images[:, :, :, 2:14, 2:14] = rng.uniform(.05, .95, (2, 2, 1, 12, 12))
    valid = np.zeros_like(images)
    valid[:, :, :, 2:14, 2:14] = 1
    valid[1, 1] = 0
    images[1, 1] = 0
    regions = np.zeros((2, 2, 3, 16, 16), np.float32)
    regions[:, :, 0, 5:11, 5:11] = 1
    regions[:, :, 1, 2:5, 5:11] = 1
    regions[:, :, 1, 11:14, 5:11] = 1
    regions[:, :, 2, 5:11, 2:5] = 1
    regions[:, :, 2, 5:11, 11:14] = 1
    return {"images": torch.tensor(images), "valid": torch.tensor(valid),
            "regions_native": torch.tensor(regions),
            "available": torch.tensor([[1., 1.], [1., 0.]])}


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.calls = []

    def forward(self, data, reference, reference_labels, legal=None):
        image = data["images"] * data["valid"]
        total = image.sum((1, 2, 3, 4))
        energy = image.square().sum((1, 2, 3, 4))
        logits = self.scale * torch.stack((total * .05, energy * .2, -total * .03), -1)
        pooled = F.adaptive_avg_pool2d(image.flatten(0, 1), (4, 4)).reshape(len(image), 2, 1, 4, 4)
        response = pooled * torch.tensor([1., -1., 2.])[None, None, :, None, None]
        self.calls.append(dict(logits=logits.detach().clone(), training=self.training,
                               grad=torch.is_grad_enabled(), legal=legal))
        return {"logits": logits, "responses": response}


class UtilityTeacher(ToyTeacher):
    def forward(self, data, reference, reference_labels, legal=None, *, selection_override=None):
        choice = data["base_choice"] if selection_override is None else selection_override
        direction = torch.tensor([[-2., -3., 2.], [1., -2., 0.]])
        logits = torch.tensor([[4., 3., 1.]]).expand(len(choice), -1).clone()
        logits += choice[:, 0, 0, None] * direction[0]
        logits += choice[:, 1, 2, None] * direction[1]
        self.calls.append(dict(choice=choice.clone(), data=data, reference=reference, legal=legal,
                               training=self.training, grad=torch.is_grad_enabled(), logits=logits))
        return dict(logits=logits, responses=torch.zeros(len(choice), 2, 3, 4, 4),
                    scale_trace=dict(choice=choice, eligible=data["eligible"]))


class TeacherTests(unittest.TestCase):
    def test_utility_changes_only_one_parent_and_keeps_reference_and_input(self):
        data = sample_data()
        data["base_choice"] = torch.full((2, 2, 8), .25)
        data["eligible"] = torch.zeros(2, 2, 8, dtype=torch.bool)
        data["eligible"][:, 0, 0] = True
        data["eligible"][0, 1, 2] = True
        teacher = UtilityTeacher().train()
        reference = {"fixed": torch.ones(1)}
        legal = torch.ones(2, 1, dtype=torch.bool)
        result = measure_utility_teacher(teacher, data, reference, torch.tensor([0]), legal,
                                         labels=torch.tensor([0, 2]), samples=16, seed=7)
        self.assertEqual(result["target_type"], "coarse_fine_margin_v1")
        self.assertEqual(result["teacher_forwards"], 5)
        self.assertEqual(tuple(result["utility"].shape), (2, 2, 3, 8))
        self.assertFalse(result["utility"].requires_grad)
        self.assertEqual(result["effect_mask"][1, 0, 0].item(), 1)
        self.assertEqual(result["effect_mask"][1, 1].sum().item(), 0)
        for i, (p, g) in enumerate(result["conditions"]):
            coarse, fine = teacher.calls[1 + 2 * i:3 + 2 * i]
            for call, value in ((coarse, 0), (fine, 1)):
                expected = data["base_choice"].clone()
                expected[:, p, g] = value
                torch.testing.assert_close(call["choice"], expected)
                self.assertIs(call["data"], data)
                self.assertIs(call["reference"], reference)
                self.assertIs(call["legal"], legal)
                self.assertFalse(call["training"] or call["grad"])
            z0, z1 = coarse["logits"][0].tolist(), fine["logits"][0].tolist()
            expected = [(z1[c] - max(z1[j] for j in range(3) if j != c)) -
                        (z0[c] - max(z0[j] for j in range(3) if j != c)) for c in range(3)]
            torch.testing.assert_close(result["utility"][0, p, :, g], torch.tensor(expected))
        self.assertGreater(result["utility"][..., 2, 0].max().item(), 0)

    def test_utility_sampling_is_seeded_and_empty_legal_support_is_skipped(self):
        data = sample_data()
        data["base_choice"] = torch.full((2, 2, 8), .4)
        data["eligible"] = data["available"].bool()[:, :, None].expand(-1, -1, 8)
        teacher = UtilityTeacher()
        common = dict(labels=torch.tensor([0, 0]), samples=2, seed=19)
        first = measure_utility_teacher(teacher, data, {}, torch.tensor([0]), **common)
        second = measure_utility_teacher(teacher, data, {}, torch.tensor([0]), **common)
        self.assertEqual(first["conditions"], second["conditions"])
        self.assertEqual(first["teacher_forwards"], 5)
        empty = measure_utility_teacher(teacher, data, {}, torch.tensor([0]),
                                        torch.zeros(2, 1, dtype=torch.bool), **common)
        self.assertEqual(empty["teacher_forwards"], 1)
        self.assertEqual(empty["effect_mask"].sum().item(), 0)

    def test_utility_requires_legal_reference_with_the_same_polarization(self):
        data = sample_data()
        data["base_choice"] = torch.full((2, 2, 8), .4)
        data["eligible"] = data["available"].bool()[:, :, None].expand(-1, -1, 8)
        reference = {"available": torch.tensor([[0., 1.], [1., 0.]])}
        legal = torch.tensor([[True, False], [True, False]])
        common = dict(labels=torch.tensor([0, 0]), samples=16, seed=19)
        result = measure_utility_teacher(UtilityTeacher(), data, reference, torch.tensor([0, 1]), legal, **common)
        self.assertEqual(result["effect_mask"][:, 0].sum().item(), 0)
        self.assertEqual(result["effect_mask"][0, 1].sum().item(), 8)
        self.assertEqual(result["effect_mask"][1].sum().item(), 0)

    def test_attention_uses_origin_zero_boundary_occupancy(self):
        class UniformResponseTeacher(ToyTeacher):
            def forward(self, *args, **kwargs):
                result = super().forward(*args, **kwargs)
                result["responses"] = torch.zeros_like(result["responses"])
                return result

        data = sample_data()
        data["valid"].zero_()
        data["valid"][:, 0, 0, 2, 2] = 1
        data["available"][:, 1] = 0
        data["images"] *= data["valid"]
        data["regions_native"].zero_()
        result = measure_teacher(UniformResponseTeacher(), data, None, None,
                                 labels=torch.zeros(2, dtype=torch.long))
        # Stride-4 origin-zero cells meet at coordinate 2. The native pixel
        # [1.5,2.5]^2 crosses four cells, whose widths are 2.5 and 4.
        # Adaptive average pooling would incorrectly put it in one cell.
        expected = torch.zeros(4, 4)
        expected[:2, :2] = torch.tensor([[.04, .025], [.025, .015625]])
        expected /= expected.sum()
        torch.testing.assert_close(result["attention"][0, 0, 0], expected)
        self.assertEqual(torch.count_nonzero(result["attention"][0, 0, 0]), 4)
        self.assertEqual(result["attention"][:, 1].abs().sum().item(), 0)

    def test_native_scope_and_brightness_control_signed_sum(self):
        data = sample_data()
        result = make_interventions(data)
        original = data["images"].numpy()
        before = original.copy()
        for action, (p, r, _) in enumerate(CONDITIONS):
            mask = (data["regions_native"][:, p, r].numpy() > 0) & (data["valid"][:, p, 0].numpy() > 0)
            allowed = np.zeros_like(original, dtype=bool)
            allowed[:, p, 0] = mask
            changed = result["interventions"][action] - original
            control = result["controls"][action] - original
            np.testing.assert_array_equal(changed[~allowed], 0)
            np.testing.assert_array_equal(control[~allowed], 0)
            np.testing.assert_allclose(changed.sum((1, 2, 3, 4), dtype=np.float64),
                                       control.sum((1, 2, 3, 4), dtype=np.float64), atol=1e-5)
            self.assertGreaterEqual(result["interventions"][action].min(), 0)
            self.assertLessEqual(result["controls"][action].max(), 1)
            if p == 1:
                self.assertFalse(result["eligible"][action, 1])
        np.testing.assert_array_equal(original, before)
        self.assertTrue(result["eligible"][:, 0].all())

    def test_measurement_masks_all_classes_and_exact_forward_count(self):
        data = sample_data()
        teacher = ToyTeacher().train()
        initial = teacher(data, None, None)["logits"]
        labels = initial.argmax(-1)
        labels[1] = (labels[1] + 1) % initial.shape[-1]
        teacher.calls.clear()
        legal = torch.tensor([[True], [True]])
        result = measure_teacher(teacher, data, None, torch.tensor([0]), legal, labels=labels)
        self.assertEqual(len(teacher.calls), 25)
        self.assertEqual(result["teacher_forwards"], 25)
        self.assertTrue(all(not call["training"] and not call["grad"] for call in teacher.calls))
        self.assertTrue(all(call["legal"] is legal for call in teacher.calls))
        self.assertFalse(teacher.scale.requires_grad)
        self.assertEqual(result["attention_mask"].sum().item(), 2)
        self.assertEqual(result["attention_mask"][1].sum().item(), 0)
        self.assertGreater(result["effect_mask"][1, 0].sum().item(), 0)
        self.assertEqual(result["effect_mask"][1, 1].sum().item(), 0)
        self.assertEqual(tuple(result["effects"].shape), (2, 2, 3, 3, 2))
        self.assertFalse(result["effects"].requires_grad)
        # Independent scalar calculation for action 0; all candidates participate.
        before = teacher.calls[0]["logits"][0].tolist()
        after = teacher.calls[1]["logits"][0].tolist()
        expected = []
        for c in range(3):
            original_margin = before[c] - max(before[j] for j in range(3) if j != c)
            altered_margin = after[c] - max(after[j] for j in range(3) if j != c)
            expected.append(original_margin - altered_margin)
        torch.testing.assert_close(result["effects"][0, 0, :, 0, 0], torch.tensor(expected))
        self.assertGreater(result["effects"][1, 0].abs().max().item(), 0)
        self.assertEqual(result["attention"][1, 1].abs().max().item(), 0)


if __name__ == "__main__":
    unittest.main()
