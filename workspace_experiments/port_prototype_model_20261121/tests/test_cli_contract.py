import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sar_port import cli


def record(identifier, port, label=0):
    return dict(id=str(identifier), vessel_id=f"v{identifier}", group_id=f"g{identifier}", port=port, label=label)


class CLIContractTests(unittest.TestCase):
    def test_help_does_not_import_model_or_start_training(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as stopped:
                cli.main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("select-bank", output.getvalue())
        self.assertIn("export", output.getvalue())

    def test_train_requires_explicit_steps_and_conflicting_initializers_fail(self):
        common = ["train", "--manifest", "m.json", "--split", "s.json", "--bank", "b.json", "--output", "out"]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(common)
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(common + ["--steps", "1", "--initialize", "x.pt", "--imagenet", "y.pt"])

    def test_split_and_random_bank_need_no_images_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [record(0, "S"), record(1, "S"), record(2, "V"), record(3, "T")]
            manifest, split, bank = (root / name for name in ("manifest.json", "split.json", "bank.json"))
            manifest.write_text(json.dumps(rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["split", "--manifest", str(manifest), "--target-ports", "T", "--validation-ports", "V", "--output", str(split)])
                cli.main(["select-bank", "--manifest", str(manifest), "--split", str(split), "--method", "random", "--per-class", "1", "--output", str(bank)])
            split_data = cli._load_split(split, rows)
            selected = cli._load_bank(bank, rows, split_data)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["port"], "S")

    def test_modified_split_and_bank_fail_source_contract(self):
        rows = [record(0, "S"), record(1, "V"), record(2, "T")]
        with tempfile.TemporaryDirectory() as directory:
            split_path, bank_path = Path(directory) / "s.json", Path(directory) / "b.json"
            source = dict(train=[0], validation=[1], test=[2], excluded=[],
                          target_ports=["T"], validation_ports=["V"], record_ids=[r["id"] for r in rows])
            cli._write_json(split_path, dict(source, train=[0, 2]))
            with self.assertRaisesRegex(ValueError, "violates"):
                cli._load_split(split_path, rows)
            cli._write_json(bank_path, dict(indices=[2], record_ids=["2"]))
            with self.assertRaisesRegex(ValueError, "outside"):
                cli._load_bank(bank_path, rows, source)

    def test_checkpoint_provenance_rejects_all_port_exposure(self):
        rows = [record(0, "S"), record(1, "T")]
        with self.assertRaisesRegex(ValueError, "no source-training"):
            cli._check_source_checkpoint({"model": {}}, rows[:1])
        with self.assertRaisesRegex(ValueError, "outside"):
            cli._check_source_checkpoint({"source_records": rows, "bank_records": rows[:1]}, rows[:1])
        cli._check_source_checkpoint({"source_records": rows[:1], "bank_records": rows[:1]}, rows[:1])

    def test_previous_validation_port_cannot_become_new_test_port(self):
        source = [record(0, "S")]
        for protocol in ({"validation_ports": ["T"]},
                         {"validation_ports": ["V"], "selection_ports": ["V", "T"]}):
            checkpoint = dict(source_records=source, bank_records=source, protocol=protocol)
            with self.assertRaisesRegex(ValueError, "selection used"):
                cli._check_source_checkpoint(checkpoint, source, {"target_ports": ["T"]})

    def test_freeze_prefix_changes_grad_flags_and_requires_real_match(self):
        model = torch.nn.ModuleDict({"backbone": torch.nn.Linear(2, 3), "student": torch.nn.Linear(3, 2)})
        counts = cli._freeze_parameters(model, ["backbone"])
        self.assertEqual(counts, {"backbone": 9})
        self.assertTrue(all(not p.requires_grad for p in model["backbone"].parameters()))
        self.assertTrue(all(p.requires_grad for p in model["student"].parameters()))
        with self.assertRaisesRegex(ValueError, "matches no"):
            cli._freeze_parameters(model, ["absent_module"])

    def test_label_free_legal_mask_handles_unknown_and_known_identity(self):
        queries = [{"id": "new"}, {"id": "known", "vessel_id": "v0"}]
        bank = [record(0, "S"), record(1, "S")]
        mask = cli._legal(queries, bank, "cpu", allow_unidentified=True)
        self.assertEqual(mask.tolist(), [[True, True], [False, True]])

    def test_export_tree_is_detached_and_metadata_has_no_image_paths(self):
        tensor = torch.tensor([1.], requires_grad=True)
        value = cli._tree_to({"a": [tensor], "b": (tensor * 2,)}, "cpu", detach=True)
        self.assertFalse(value["a"][0].requires_grad)
        self.assertFalse(value["b"][0].requires_grad)
        self.assertEqual(cli._metadata([dict(record(0, "S"), vv={"image": "private.png"})]), [record(0, "S")])

    def test_metrics_define_imbalance_and_absent_class_denominators(self):
        result = cli.classification_metrics([0, 0, 0, 1], [0, 0, 2, 0], 3)
        self.assertEqual(result["oa"], 50.)
        self.assertAlmostEqual(result["ba"], 100 / 3)
        self.assertAlmostEqual(result["macro_f1"], 100 * (4 / 6) / 3)
        self.assertEqual(result["present_classes"], [0, 1])
        self.assertEqual(result["f1_classes"], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
