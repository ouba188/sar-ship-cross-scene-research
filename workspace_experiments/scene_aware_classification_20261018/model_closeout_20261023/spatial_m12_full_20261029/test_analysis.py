"""Small CPU synthetic checks; never opens real predictions or trains a model."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("full_analysis_under_test", Path(__file__).with_name("analysis.py"))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def rows_for(labels):
    return [dict(id=str(i), label=y, port="port", pass_group="A" if i < 2 else "B",
                 mmsi=str(100 + i), analysis_available="both") for i, y in enumerate(labels)]


class AnalysisTests(unittest.TestCase):
    def test_metrics_known_probabilities_and_missing_class(self):
        y = np.array([0, 0, 1])
        p = np.full((3, 8), .05)
        p[0, 0] = .65
        p[1, 1], p[1, 0] = .60, .10
        p[2, 1] = .65
        result = a.metrics(np.log(p), y)
        self.assertAlmostEqual(result["OA"], 200 / 3)
        self.assertAlmostEqual(result["BA"], 75)
        self.assertAlmostEqual(result["NLL"], -(2 * np.log(.65) + np.log(.1)) / 3)
        self.assertAlmostEqual(result["Brier"], (2 * (.35**2 + 7 * .05**2) + .9**2 + .6**2 + 6 * .05**2) / 3)
        self.assertEqual(a.metrics(np.zeros((0, 8)), np.array([], int))["BA"], None)
        uniform = a.metrics(np.full((3, 8), 10000.), y)
        self.assertAlmostEqual(uniform["NLL"], np.log(8))
        self.assertAlmostEqual(uniform["Brier"], .875)
        self.assertAlmostEqual(a.moments([1, 2, 3])["sd"], 1.)

    def test_id_join_and_label_mismatch(self):
        rows = rows_for([0, 1, 2])
        score = np.arange(24).reshape(3, 8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            order = [2, 0, 1]
            np.savez(path, score=score[order], ids=np.array(order).astype(str), y=np.array(order))
            np.testing.assert_array_equal(a.base.load_score(path, rows), score)
            np.savez(path, score=score, ids=np.array(["0", "1", "2"]), y=np.array([1, 0, 2]))
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                a.base.load_score(path, rows)

    def test_flips_groups_and_empty_class(self):
        rows = rows_for([0, 0, 1, 1])
        old = np.full((4, 8), -5.)
        new = old.copy()
        old[np.arange(4), [0, 1, 1, 0]] = 5
        new[np.arange(4), [0, 0, 0, 0]] = 5
        scores = {(c, s): (new if c == "spatial_after" else old) for c in a.CONDITIONS for s in a.SEEDS}
        events = a.pair_events(rows, scores, "legacy_after", "spatial_after")
        self.assertEqual([e["state"] for e in events[:4]], ["CC", "rescued", "harmed", "WW"])
        detailed, summary = a.summarize_comparison(rows, scores, "legacy_after", "spatial_after", events)
        self.assertEqual(summary[0]["rescued_seed_events"], 3)
        self.assertEqual(summary[0]["rescued_unique_ids"], 1)
        self.assertEqual(summary[0]["delta_OA_mean"], 0)
        self.assertEqual(summary[0]["delta_BA_mean"], 0)
        absent = next(r for r in detailed if r["dimension"] == "label" and r["group"] == "7")
        self.assertIsNone(absent["delta_BA"])
        self.assertIsNone(absent["reference_OA"])

    def test_pass_bootstrap_matches_explicit_resampling(self):
        rows = rows_for([0, 1, 0, 0, 1])
        y = np.array([r["label"] for r in rows])
        rng = np.random.default_rng(92)
        scores = {(c, s): rng.normal(size=(5, 8)) for c in a.CONDITIONS for s in a.SEEDS}
        result = a.pass_bootstrap(rows, scores, "legacy_after", "spatial_after", repeats=60, seed=81)
        draw_rng = np.random.default_rng(81)
        pass_indices = [np.array([0, 1]), np.array([2, 3, 4])]
        brute = []
        for _ in range(60):
            indices = np.concatenate([pass_indices[g] for g in draw_rng.integers(0, 2, 2)])
            delta = []
            for s in a.SEEDS:
                old = a.metrics(scores["legacy_after", s][indices], y[indices])
                new = a.metrics(scores["spatial_after", s][indices], y[indices])
                delta.append([new[m] - old[m] for m in a.METRICS])
            brute.append(np.mean(delta, axis=0))
        expected = np.quantile(brute, [.025, .975], axis=0)
        for i, m in enumerate(a.METRICS):
            self.assertAlmostEqual(result["metrics"][m]["low"], expected[0, i])
            self.assertAlmostEqual(result["metrics"][m]["high"], expected[1, i])
        unchanged = a.pass_bootstrap(rows, scores, "legacy_after", "legacy_after", repeats=20)
        self.assertTrue(all(v["low"] == v["high"] == 0 for v in unchanged["metrics"].values()))

    def test_selection_final_flag_and_intersection(self):
        rows = rows_for([0, 1, 2])
        inputs = {}
        for arm in a.ARMS:
            for seed in a.SEEDS:
                inputs[arm, seed] = [dict(id=r["id"], label=r["label"],
                    useful=(i < 2 if arm == "legacy" else i > 0), original=[0.] * 8, altered=[0.] * 8,
                    choice=dict(useful=True, rival=7, structural_margin_gain=.02, chosen_index=1, control_index=2),
                    selected_intervention=dict(kind="structure", pol="vv", region="body", band="fine")) for i, r in enumerate(rows)]
        with tempfile.TemporaryDirectory() as directory:
            summary = a.selection_analysis({"adapt": rows}, inputs, Path(directory))
        for intersection in summary["intersections"]:
            self.assertEqual(intersection["intersection_ids"], ["1"])
            self.assertAlmostEqual(intersection["jaccard"], 1 / 3)
            self.assertEqual(intersection["exact_intervention_intersection"], 1)
        for entry in summary["count_summary"]:
            self.assertEqual(entry["useful_mean"], 2)
            self.assertEqual(entry["choice_useful_mean"], 3)
            self.assertEqual(entry["choice_rejected_by_final_mean"], 1)

    def test_case_selection_net_outcomes_and_missing_polarization(self):
        events = []
        for index in range(12):
            state = a.STATES[index // 3]
            for seed in a.SEEDS:
                events.append(dict(id=str(index), seed=seed, state=state, delta_fixed_margin=12 - index,
                                   available="vv_only" if index == 2 else "both"))
        cases = a.select_cases(events)
        self.assertEqual(len(cases), 8)
        self.assertEqual(len({e["id"] for e in cases}), 8)
        self.assertEqual({s: sum(e["selection_category"] == s for e in cases) for s in a.STATES}, {s: 2 for s in a.STATES})
        self.assertIn("2", [e["id"] for e in cases])
        mixed = copy.deepcopy(events)
        for e in mixed:
            if e["id"] == "0" and e["seed"] == 43:
                e["state"] = "harmed"
        isolated = a.select_cases([e for e in mixed if e["id"] == "0"])[0]
        self.assertEqual(isolated["selection_category"], "rescued")
        self.assertEqual(isolated["net_rescue_seed_events"], 1)

    def test_native_panel_availability_and_no_resize(self):
        native = a.module_at("synthetic_native_input", a.base.PACKAGE / "sar_m123/input.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chip = root / "chip.png"
            a.base.Image.new("L", (37, 29), 128).save(chip)
            view = dict(image=str(chip), target_index=0, detections=[dict(
                corners=[[5, 5], [15, 5], [15, 20], [5, 20]], confidence=.9)])
            row = dict(id="synthetic", label=0, vv=view, vh=None)
            cohorts = {"expanded": [row]}
            a.annotate_available(cohorts, native)
            self.assertEqual(row["analysis_available"], "vv_only")
            tile, receipt = a.base.native_tile(row, "vv", native, "QUERY")
            self.assertEqual(receipt["raw_size"], [37, 29])
            self.assertEqual(receipt["pixel_scale"], 1)
            self.assertEqual(tile.getpixel((8, 63)), (128, 128, 128))
            self.assertEqual(tile.getpixel((44, 91)), (128, 128, 128))
            tile.close()
            self.assertFalse(a.base.native_tile(row, "vh", native, "QUERY")[1]["available"])

    def test_syntax(self):
        for name in ("analysis.py", "test_analysis.py"):
            ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
