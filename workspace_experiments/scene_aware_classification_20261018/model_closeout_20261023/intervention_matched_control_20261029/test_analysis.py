"""Synthetic CPU checks. Writes temporary fixtures only below analysis/."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("matched_analysis_test", ROOT / "analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def rows_for(labels):
    return [dict(id=str(i), label=y, port="port", pass_group="A" if i < 2 else "B",
                 mmsi=str(100 + i), analysis_available="both") for i, y in enumerate(labels)]


def fixture_scores():
    old, new = np.full((4, 8), -5.), np.full((4, 8), -5.)
    old[np.arange(4), [0, 1, 1, 0]] = 5
    new[np.arange(4), [0, 0, 0, 0]] = 5
    return {(c, s): (new.copy() if c == "guided" else old.copy()) for c in a.CONDITIONS for s in a.SEEDS}


class AnalysisTests(unittest.TestCase):
    def temporary(self):
        folder = ROOT / "analysis"
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="cpu_test_", dir=folder)

    def test_input_routes_and_incomplete_inputs_do_not_read_predictions(self):
        with self.temporary() as directory:
            root, source = Path(directory) / "run", Path(directory) / "full"
            paths, receipts = a.input_paths(root, source)
            self.assertEqual(len(paths), 18)
            self.assertEqual(len(receipts), 6)
            self.assertEqual(paths["expanded", "control", 17], root / "predictions/expanded_control_s17.npz")
            self.assertEqual(paths["development", "guided", 29], source / "predictions/development_after_spatial_s29.npz")
            self.assertEqual(paths["expanded", "before", 43], source / "predictions/expanded_before_spatial_s43.npz")
            with patch.object(a.base, "load_score") as read:
                with self.assertRaisesRegex(SystemExit, "Incomplete inputs"):
                    a.load_inputs(root, source)
                read.assert_not_called()
            self.assertFalse((root / "analysis").exists())

    def test_id_join_reorders_and_rejects_label_mismatch(self):
        rows = rows_for([0, 1, 2])
        score = np.arange(24).reshape(3, 8)
        with self.temporary() as directory:
            path = Path(directory) / "synthetic.npz"
            order = [2, 0, 1]
            np.savez(path, score=score[order], ids=np.array(order).astype(str), y=np.array(order))
            np.testing.assert_array_equal(a.base.load_score(path, rows), score)
            np.savez(path, score=score, ids=np.array(["0", "1", "2"]), y=np.array([1, 0, 2]))
            with self.assertRaisesRegex(ValueError, "labels disagree"):
                a.base.load_score(path, rows)

    def test_three_condition_events_and_paired_direction(self):
        rows, scores = rows_for([0, 0, 1, 1]), fixture_scores()
        events = a.pair_events(rows, scores, "control", "guided")
        self.assertEqual([e["state"] for e in events[:4]], ["CC", "rescued", "harmed", "WW"])
        self.assertEqual(events[1]["delta_fixed_margin"], 20)
        self.assertEqual(events[1]["before_prediction"], 1)
        self.assertEqual(events[1]["control_prediction"], 1)
        self.assertEqual(events[1]["guided_prediction"], 0)
        detail, summary = a.full.summarize_comparison(rows, scores, "control", "guided", events)
        self.assertEqual(summary[0]["rescued_seed_events"], 3)
        self.assertEqual(summary[0]["rescued_unique_ids"], 1)
        self.assertEqual(summary[0]["delta_OA_mean"], 0)
        absent = next(r for r in detail if r["dimension"] == "label" and r["group"] == "7")
        self.assertIsNone(absent["delta_BA"])
        reversed_events = a.pair_events(rows, scores, "guided", "control")
        self.assertEqual(reversed_events[1]["state"], "harmed")
        for left, right in zip(events, reversed_events):
            self.assertEqual(left["fixed_rival"], right["fixed_rival"])
            self.assertAlmostEqual(left["delta_fixed_margin"], -right["delta_fixed_margin"])

    def test_metrics_and_sample_sd(self):
        y = np.array([0, 0, 1])
        p = np.full((3, 8), .05)
        p[0, 0], p[1, 0], p[1, 1], p[2, 1] = .65, .10, .60, .65
        result = a.base.metrics(np.log(p), y)
        self.assertAlmostEqual(result["OA"], 200 / 3)
        self.assertAlmostEqual(result["BA"], 75)
        self.assertAlmostEqual(result["NLL"], -(2 * np.log(.65) + np.log(.1)) / 3)
        self.assertAlmostEqual(result["Brier"], (2 * (.35**2 + 7 * .05**2) + .9**2 + .6**2 + 6 * .05**2) / 3)
        self.assertEqual(a.base.moments([1, 2, 3])["sd"], 1)

    def test_pass_bootstrap_against_explicit_whole_pass_draws(self):
        rows = rows_for([0, 1, 0, 0, 1])
        y = np.array([r["label"] for r in rows])
        rng = np.random.default_rng(92)
        scores = {(c, s): rng.normal(size=(5, 8)) for c in a.CONDITIONS for s in a.SEEDS}
        result = a.full.pass_bootstrap(rows, scores, "control", "guided", repeats=60, seed=81)
        draws, brute = np.random.default_rng(81), []
        groups = [np.array([0, 1]), np.array([2, 3, 4])]
        for _ in range(60):
            indices = np.concatenate([groups[g] for g in draws.integers(0, 2, 2)])
            deltas = []
            for seed in a.SEEDS:
                old, new = [a.base.metrics(scores[c, seed][indices], y[indices]) for c in ("control", "guided")]
                deltas.append([new[m] - old[m] for m in a.METRICS])
            brute.append(np.mean(deltas, axis=0))
        expected = np.quantile(brute, [.025, .975], axis=0)
        for index, metric in enumerate(a.METRICS):
            self.assertAlmostEqual(result["metrics"][metric]["low"], expected[0, index])
            self.assertAlmostEqual(result["metrics"][metric]["high"], expected[1, index])

    def test_cases_cover_states_and_all_conditions_actual_rivals(self):
        rows, scores = rows_for([0, 0, 1, 1]), fixture_scores()
        events = a.pair_events(rows, scores, "control", "guided")
        selected = a.full.select_cases(events)
        self.assertEqual(len(selected), 4)
        self.assertEqual({e["selection_category"] for e in selected}, set(a.STATES))
        # A before-only wrong class in a nonrepresentative seed must have refs.
        scores["before", 43][0, 2] = 20
        refs = [dict(id="r0", label=0, mmsi="200"), dict(id="r1", label=1, mmsi="201"),
                dict(id="r2", label=2, mmsi="202"), dict(id="excluded", label=2, mmsi=rows[0]["mmsi"])]
        case = next(e for e in selected if e["id"] == "0")
        detail, _, _ = a.case_details(case, 0, rows[0], refs, scores)
        self.assertEqual(detail["visual_status"], "not_reviewed")
        self.assertEqual(detail["true_reference_ids"], ["r0"])
        self.assertIn("r2", detail["rival_reference_ids"])
        self.assertNotIn("excluded", detail["eligible_reference_ids"])
        self.assertEqual(detail["per_seed"][2]["conditions"]["before"]["prediction"], 2)
        self.assertEqual(len(detail["per_seed"]), 3)

    def test_native_gallery_pixels_availability_and_three_condition_margins(self):
        native = a.full.module_at("synthetic_matched_native", a.base.PACKAGE / "sar_m123/input.py")
        with self.temporary() as directory:
            out = Path(directory)
            chip = out / "chip.png"
            a.base.Image.new("L", (37, 29), 128).save(chip)
            view = dict(image=str(chip), target_index=0, detections=[dict(
                corners=[[5, 5], [15, 5], [15, 20], [5, 20]], confidence=.9)])
            rows = rows_for([0, 0, 1, 1])
            for row in rows:
                row.update(vv=view, vh=None)
            refs = [dict(id="r" + str(i), label=i, mmsi=str(200 + i), vv=view, vh=None) for i in (0, 1)]
            cohorts = dict(expanded=rows, references=refs)
            a.full.annotate_available(cohorts, native)
            self.assertEqual(rows[0]["analysis_available"], "vv_only")
            scores = fixture_scores()
            cases = a.gallery(out, cohorts, scores, a.pair_events(rows, scores, "control", "guided"), native)
            self.assertEqual(len(cases), 4)
            self.assertTrue(all(c["visual_status"] == "not_reviewed" for c in cases))
            with a.base.Image.open(cases[0]["panels"]["vv"]) as panel:
                self.assertEqual(panel.getpixel((8, 63)), (128, 128, 128))
                self.assertEqual(panel.getpixel((44, 91)), (128, 128, 128))
                self.assertEqual(panel.getpixel((45, 91)), (255, 255, 255))
            sources = a.base.read_json(out / "cases/native_sources.json")
            self.assertTrue(all(r["pixel_scale"] == 1 for r in sources))
            self.assertTrue(all(not r["available"] for r in sources if r["pol"] == "vh"))
            page = (out / "cases/index.html").read_text(encoding="utf-8")
            for condition in a.CONDITIONS:
                self.assertIn(condition + ": pred / true-top-rival margin / fixed margin", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)
