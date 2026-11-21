import unittest
import warnings

import numpy as np

from sar_port.protocol import legal_reference_mask, split_lopo, validate_bank_source
from sar_port.prototypes import select_prototypes


def row(identifier, vessel=None, port="A", group=None, label=0):
    return dict(id=str(identifier), label=label, vessel_id=vessel or f"v{identifier}",
                port=port, group_id=group or f"g{identifier}")


class PrototypeTests(unittest.TestCase):
    def test_one_vessel_one_vote_even_across_ports(self):
        rows = [row("a", "same", "A"), row("b", "same", "B"),
                row("c", port="B"), row("d", port="C")]
        features = np.eye(4)
        with warnings.catch_warnings(record=True) as observed:
            chosen = select_prototypes(rows, features, 4)
        self.assertEqual(len(chosen), 3)
        self.assertEqual(len({rows[i]["vessel_id"] for i in chosen}), 3)
        self.assertNotIn(1, chosen)
        self.assertIn("only 3 distinct", str(observed[0].message))

    def test_representative_is_nearest_real_observation_to_vessel_center(self):
        rows = [row(name, "same", port="A") for name in ("a", "b", "c", "d")]
        features = np.asarray([[-1., 0.], [1., 0.], [1., .1], [1., -.1]])
        for method in ("coverage", "random"):
            chosen = select_prototypes(rows, features, 1, method)
            self.assertEqual(chosen.tolist(), [1])
            legacy = select_prototypes(rows, features, 1, method, observation="first")
            self.assertEqual(legacy.tolist(), [0])

    def test_budget_per_class_and_port_coverage(self):
        rows = [row(i, port=p, label=c) for i, (p, c) in enumerate(
            [("A", 0)] * 5 + [("B", 0)] * 2 + [("C", 0)] + [("A", 1), ("B", 1), ("C", 1)])]
        features = np.random.default_rng(9).normal(size=(len(rows), 5))
        for method in ("random", "coverage"):
            chosen = select_prototypes(rows, features, 3, method, seed=4)
            for label in (0, 1):
                selected = [rows[i] for i in chosen if rows[i]["label"] == label]
                self.assertEqual(len(selected), 3)
                self.assertEqual({r["port"] for r in selected}, {"A", "B", "C"})
            np.testing.assert_array_equal(chosen, select_prototypes(rows, features, 3, method, seed=4))

    def test_coverage_preserves_separated_modes_as_real_anchors(self):
        rows = [row(i) for i in range(6)]
        x = np.asarray([[1, 0], [1, .02], [1, -.02], [-1, 0], [-1, .02], [-1, -.02]])
        chosen = select_prototypes(rows, x, 2, port_balance=False)
        self.assertEqual(set(np.sign(x[chosen, 0])), {-1, 1})
        self.assertEqual(set(chosen), {0, 3})

    def test_tied_descriptors_do_not_duplicate_medoids(self):
        rows = [row(i) for i in range(5)]
        chosen = select_prototypes(rows, np.ones((5, 4)), 4)
        self.assertEqual(len(set(chosen)), 4)

    def test_conflicting_vessel_labels_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            select_prototypes([row(0, "shared", label=0), row(1, "shared", label=1)], np.eye(2), 1)

    def test_lopo_excludes_cross_split_vessels_and_groups(self):
        rows = [row("t", "target_ship", "T", "target_group", 0),
                row("vt", "target_ship", "V"), row("vg", port="V", group="target_group"),
                row("v", "valid_ship", "V", "valid_group", 0),
                row("st", "target_ship", "S"), row("sg", port="S", group="target_group"),
                row("sv", "valid_ship", "S"), row("svg", port="S", group="valid_group"),
                row("s", port="S", label=0)]
        split = split_lopo(rows, ["T"], ["V"])
        self.assertEqual(split["test"].tolist(), [0])
        self.assertEqual(split["validation"].tolist(), [3])
        self.assertEqual(split["train"].tolist(), [8])
        self.assertEqual(split["excluded"].tolist(), [1, 2, 4, 5, 6, 7])
        self.assertEqual(split["missing_source_classes"], [])

    def test_target_only_class_is_reported_not_removed(self):
        rows = [row(0, port="S", label=0), row(1, port="V", label=1), row(2, port="T", label=2)]
        split = split_lopo(rows, ["T"], ["V"])
        self.assertEqual(split["missing_source_classes"], [1, 2])
        self.assertEqual(split["test"].tolist(), [2])

    def test_bank_is_source_only_and_unique_vessel(self):
        source = [row(0), row(1), row(2, "v0", port="B")]
        validate_bank_source(source[:2], source)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_bank_source([row(4, port="T")], source)
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_bank_source([dict(source[0], port="T")], source)
        with self.assertRaisesRegex(ValueError, "Repeated"):
            validate_bank_source([source[0], source[2]], source)

    def test_reference_mask_never_uses_query_label(self):
        queries = [{"vessel_id": "v0", "group_id": "g2"}]
        references = [row(0), row(1), row(2)]
        np.testing.assert_array_equal(legal_reference_mask(queries, references), [[False, True, False]])
        self.assertEqual(legal_reference_mask([], references).shape, (0, 3))
        self.assertEqual(legal_reference_mask(queries, []).shape, (1, 0))


if __name__ == "__main__":
    unittest.main()
