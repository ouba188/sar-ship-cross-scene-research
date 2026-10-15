import unittest

import torch

from directional_candidate_loss import directional_rows


class DirectionalRowsTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.tensor(
            [[0., 0.], [-2., 0.], [0., -3.]], dtype=torch.float64
        )
        self.labels = torch.tensor([0])

    def test_orthogonal_drift_is_unpenalized(self):
        anchors = torch.tensor([[0., 4.]], dtype=torch.float64, requires_grad=True)
        donors = torch.zeros_like(anchors, requires_grad=True)
        rows = directional_rows(
            anchors, donors, self.weight, self.labels,
            torch.tensor([[False, True, False]]),
        )
        torch.testing.assert_close(rows, torch.tensor([0.], dtype=torch.float64))
        rows.sum().backward()
        torch.testing.assert_close(anchors.grad, torch.zeros_like(anchors))
        torch.testing.assert_close(donors.grad, torch.zeros_like(donors))

    def test_parallel_loss_and_embedding_gradients_with_detached_head(self):
        anchors = torch.tensor([[4., 8.]], dtype=torch.float64, requires_grad=True)
        donors = torch.tensor([[1., 2.]], dtype=torch.float64, requires_grad=True)
        weight = self.weight.clone().requires_grad_()
        rows = directional_rows(
            anchors, donors, weight, self.labels,
            torch.tensor([[False, True, False]]),
        )
        torch.testing.assert_close(rows, torch.tensor([9.], dtype=torch.float64))
        rows.sum().backward()
        torch.testing.assert_close(
            anchors.grad, torch.tensor([[6., 0.]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            donors.grad, torch.tensor([[-6., 0.]], dtype=torch.float64)
        )
        self.assertIsNone(weight.grad)

    def test_active_selection_and_per_row_means(self):
        anchors = torch.tensor([[3., 4.], [3., 4.]], dtype=torch.float64)
        donors = torch.zeros_like(anchors)
        # For label 1, the direction to class 2 is (-2, 3) / sqrt(13).
        labels = torch.tensor([0, 1])
        selected = directional_rows(
            anchors, donors, self.weight, labels,
            torch.tensor([[False, True, False], [False, False, True]]),
        )
        all_rivals = directional_rows(
            anchors, donors, self.weight, labels, torch.ones(2, 3, dtype=torch.bool)
        )
        torch.testing.assert_close(
            selected, torch.tensor([9., 36. / 13.], dtype=torch.float64)
        )
        torch.testing.assert_close(
            all_rivals,
            torch.tensor([12.5, (9. + 36. / 13.) / 2.], dtype=torch.float64),
        )

    def test_self_is_excluded_from_numerator_and_denominator(self):
        anchors = torch.tensor([[3., 4.]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        original = mask.clone()
        rows = directional_rows(
            anchors, torch.zeros_like(anchors), self.weight, self.labels, mask
        )
        torch.testing.assert_close(rows, torch.tensor([9.], dtype=torch.float64))
        torch.testing.assert_close(mask, original)

    def test_empty_and_self_only_rows_have_zero_loss_and_gradients(self):
        anchors = torch.tensor(
            [[3., 4.], [5., 6.]], dtype=torch.float64, requires_grad=True
        )
        donors = torch.zeros_like(anchors, requires_grad=True)
        rows = directional_rows(
            anchors, donors, self.weight, torch.tensor([0, 1]),
            torch.tensor([[False, False, False], [False, True, False]]),
        )
        torch.testing.assert_close(rows, torch.zeros(2, dtype=torch.float64))
        rows.sum().backward()
        torch.testing.assert_close(anchors.grad, torch.zeros_like(anchors))
        torch.testing.assert_close(donors.grad, torch.zeros_like(donors))

    def test_identical_head_rows_are_stable_zero_directions(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                weight = torch.tensor(
                    [[0., 0.], [0., 0.], [-2., 0.]],
                    dtype=dtype, requires_grad=True,
                )
                anchors = torch.tensor(
                    [[3., 4.], [3., 4.]], dtype=dtype, requires_grad=True
                )
                donors = torch.zeros_like(anchors, requires_grad=True)
                rows = directional_rows(
                    anchors, donors, weight, torch.tensor([0, 0]),
                    torch.tensor([[False, True, False], [False, True, True]]),
                )
                torch.testing.assert_close(rows, torch.tensor([0., 4.5], dtype=dtype))
                rows.sum().backward()
                expected = torch.tensor([[0., 0.], [3., 0.]], dtype=dtype)
                torch.testing.assert_close(anchors.grad, expected)
                torch.testing.assert_close(donors.grad, -expected)
                self.assertIsNone(weight.grad)

    def test_head_shift_scale_and_joint_rotation_invariance(self):
        anchors = torch.tensor([[3., 4.], [2., -1.]], dtype=torch.float64)
        donors = torch.tensor([[1., -1.], [-2., 2.]], dtype=torch.float64)
        labels = torch.tensor([0, 2])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        baseline = directional_rows(anchors, donors, self.weight, labels, mask)
        # Expected from axis projections and the (2, -3) / sqrt(13) direction.
        torch.testing.assert_close(
            baseline,
            torch.tensor([14.5, (9. + 289. / 13.) / 2.], dtype=torch.float64),
        )
        shifted = self.weight + torch.tensor([7., -11.], dtype=torch.float64)
        for weight in (shifted, self.weight * 0.01, self.weight * 100.):
            torch.testing.assert_close(
                directional_rows(anchors, donors, weight, labels, mask), baseline
            )
        rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
        torch.testing.assert_close(
            directional_rows(
                anchors @ rotation, donors @ rotation, self.weight @ rotation,
                labels, mask,
            ),
            baseline,
        )
