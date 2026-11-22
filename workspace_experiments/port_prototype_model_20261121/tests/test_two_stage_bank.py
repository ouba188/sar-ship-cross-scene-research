import unittest
import numpy as np
import torch

from sar_port.two_stage.bank import (branch_distance, legal_reference_mask,
                                    select_reference_indices, whole_ship_descriptors)
from test_two_stage_h import example


class TwoStageBankTests(unittest.TestCase):
    def test_no_common_polarization_is_not_a_zero_distance(self):
        ship = example(1, missing_vh=True)
        descriptor, mask = whole_ship_descriptors(ship)
        other = (descriptor.flip(0), mask.flip(0))
        distance, count = branch_distance((descriptor, mask), other)
        self.assertTrue(bool(torch.isinf(distance)))
        self.assertEqual(int(count), 0)

    def test_source_only_budget_and_cross_port_vessel_deduplication(self):
        ships = [example(1, i) for i in range(7)]
        # The last held-out object must never be inspected.
        ships[-1] = None
        selected = select_reference_indices(
            ships, [0, 0, 0, 0, 1, 1, 0], ["a", "b", "a", "b", "a", "b", "target"],
            ["shared", "shared", "v2", "v3", "v4", "v5", "heldout"],
            ["g0", "g1", "g2", "g3", "g4", "g5", "g6"], 2,
            source_mask=[True] * 6 + [False])
        self.assertEqual(len(selected), 4)
        self.assertNotIn(6, selected)
        self.assertLessEqual(len(set(selected) & {0, 1}), 1)
        self.assertTrue(4 in selected and 5 in selected)
        selected_labels = [[0, 0, 0, 0, 1, 1, 0][i] for i in selected]
        self.assertEqual(selected_labels.count(0), 2)
        self.assertEqual(selected_labels.count(1), 2)

    def test_sparse_class_does_not_duplicate_and_conflicting_vessel_labels_fail(self):
        ships = [example(1), example(1, 5)]
        indices = select_reference_indices(ships, [0, 1], ["a", "a"], ["v0", "v1"],
                                           ["g0", "g1"], 5)
        self.assertEqual(set(indices), {0, 1})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            select_reference_indices(ships, [0, 1], ["a", "b"], ["same", "same"],
                                     ["g0", "g1"], 2)

    def test_same_vessel_or_same_acquisition_is_excluded(self):
        actual = legal_reference_mask(["a", "b", "c"], ["g1", "g2", ""],
                                      ["a", "d", "e"], ["g3", "g2", ""])
        expected = torch.tensor([[False, True, True], [True, False, True], [True, True, True]])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
