"""Small synthetic CPU tests; all temporary writes stay under analysis/."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("private_analysis_test", ROOT / "analysis.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def rows_for(labels):
    return [dict(id=str(i), label=int(y), port="port", pass_group="A" if i < 2 else "B",
                 mmsi=str(100 + i), analysis_available="both") for i, y in enumerate(labels)]


def fixture_scores(n=12):
    old, new = np.full((n, 8), -5.), np.full((n, 8), -5.)
    old[np.arange(n), np.resize([0, 1, 1, 0], n)] = 5
    new[np.arange(n), 0] = 5
    return {(c, s): (new.copy() if c in ("full_after", "full_control_after") else old.copy())
            for c in a.CONDITIONS for s in a.SEEDS}


def event_sets(rows, scores):
    return {name: a.pair_events(rows, scores, *a.COMPARISONS[name]) for name in (*a.PRIMARY, a.OBJECTIVE_CONTROL)}


class AnalysisTests(unittest.TestCase):
    def temporary(self):
        folder = ROOT / "analysis"
        folder.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix="cpu_test_", dir=folder)

    def test_explicit_ready_required_before_any_load(self):
        with patch.object(a, "load_inputs") as load:
            with self.assertRaisesRegex(SystemExit, "explicit main-agent ready"):
                a.analyze()
            load.assert_not_called()
        self.assertNotIn("torch", sys.modules)

    def test_routes_and_missing_inputs_do_not_read_predictions(self):
        with self.temporary() as directory:
            root = Path(directory)
            source, control, refit = [root / name for name in ("full", "control", "refit")]
            paths, receipts = a.input_paths(root, source, control, refit)
            self.assertEqual(len(paths), 72)
            self.assertEqual(len(set(paths.values())), 66)
            self.assertEqual(len(set(receipts.values())), 33)
            self.assertEqual(paths["expanded", "no_m2_before", 17], root / "predictions/expanded_no_m2_before_s17.npz")
            self.assertEqual(paths["development", "full_after", 29], source / "predictions/development_after_spatial_s29.npz")
            self.assertEqual(paths["expanded", "full_control_after", 43], control / "predictions/expanded_control_s43.npz")
            self.assertEqual(paths["expanded", "full_before", 17], paths["expanded", "full_control_before", 17])
            self.assertEqual(receipts["no_m1", "before", 17], root / "source_fit/training/no_m1_s17/training.json")
            self.assertEqual(receipts["no_m1", "after", 17], root / "training/no_m1_s17/training.json")
            with patch.object(a.base, "load_score") as load:
                with self.assertRaisesRegex(SystemExit, "evaluation_check.json"):
                    a.load_inputs(root, source, control, refit)
                load.assert_not_called()

    def test_synthetic_receipts_gate_and_shared_before_loaded_once(self):
        with self.temporary() as directory:
            root = Path(directory)
            source, control, refit = [root / name for name in ("full", "control", "refit")]
            paths, receipts = a.input_paths(root, source, control, refit)
            rows = rows_for([0, 1])
            source.mkdir()
            a.base.dump(source / "cohorts.json", {k: rows for k in a.matched.EXPECTED})
            for path in set(paths.values()):
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, score=np.zeros((2, 8)), ids=np.array(["0", "1"]), y=np.array([0, 1]))
            for (_, stage, seed), path in receipts.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                a.base.dump(path, dict(seed=seed, steps=408 if stage == "before" else 54,
                                      completed_epochs=8 if stage == "before" else 6))
            for name in ("checks", "evaluation_check"):
                a.base.dump(root / f"{name}.json", dict(synthetic=True))
            with patch.dict(a.matched.EXPECTED, {k: 2 for k in a.matched.EXPECTED}):
                with patch.object(a.base, "load_score", wraps=a.base.load_score) as load:
                    _, scores, training, gates = a.load_inputs(root, source, control, refit)
                    self.assertEqual(load.call_count, 66)
                self.assertIs(scores["expanded", "full_before", 17], scores["expanded", "full_control_before", 17])
                self.assertEqual(len(training), 36)
                self.assertTrue(gates["checks"]["receipt"]["synthetic"])
                path = receipts["no_m1", "after", 17]
                a.base.dump(path, dict(seed=17, steps=9, completed_epochs=1))
                with patch.object(a.base, "load_score") as load:
                    with self.assertRaisesRegex(ValueError, "expected completed 54-step"):
                        a.load_inputs(root, source, control, refit)
                    load.assert_not_called()

    def test_id_alignment_labels_and_nonfinite_scores(self):
        rows = rows_for([0, 1, 2])
        score = np.arange(24).reshape(3, 8)
        with self.temporary() as directory:
            path = Path(directory) / "fixture.npz"
            order = [2, 0, 1]
            np.savez(path, score=score[order], ids=np.array(order).astype(str), y=np.array(order))
            np.testing.assert_array_equal(a.base.load_score(path, rows), score)
            for ids, y, values, error in (
                (["0", "1", "2"], [1, 0, 2], score, "labels disagree"),
                (["0", "0", "2"], [0, 1, 2], score, "IDs"),
                (["0", "1", "2"], [0, 1, 2], np.full((3, 8), np.nan), "nonfinite"),
            ):
                np.savez(path, score=values, ids=np.array(ids), y=np.array(y))
                with self.assertRaisesRegex(ValueError, error):
                    a.base.load_score(path, rows)

    def test_known_metrics_sample_sd_and_absent_class(self):
        y = np.array([0, 0, 1])
        p = np.full((3, 8), .05)
        p[0, 0], p[1, 0], p[1, 1], p[2, 1] = .65, .10, .60, .65
        value = a.base.metrics(np.log(p), y)
        self.assertAlmostEqual(value["OA"], 200 / 3)
        self.assertAlmostEqual(value["BA"], 75)
        self.assertAlmostEqual(value["NLL"], -(2 * np.log(.65) + np.log(.1)) / 3)
        self.assertAlmostEqual(value["Brier"], (2 * (.35**2 + 7 * .05**2) + .9**2 + .6**2 + 6 * .05**2) / 3)
        self.assertEqual(a.base.moments([1, 2, 3])["sd"], 1)
        self.assertIsNone(a.base.metrics(np.empty((0, 8)), np.array([], int))["BA"])

    def test_contrast_direction_states_fixed_rival_and_missing_class(self):
        rows, scores = rows_for([0, 0, 1, 1]), fixture_scores(4)
        events = a.pair_events(rows, scores, "plain_after", "full_after")
        self.assertEqual([e["state"] for e in events[:4]], ["CC", "rescued", "harmed", "WW"])
        self.assertEqual(events[1]["delta_fixed_margin"], 20)
        detail, summary = a.full.summarize_comparison(rows, scores, "plain_after", "full_after", events)
        self.assertEqual(summary[0]["rescued_seed_events"], 3)
        self.assertEqual(summary[0]["rescued_unique_ids"], 1)
        absent = next(r for r in detail if r["dimension"] == "label" and r["group"] == "7")
        self.assertIsNone(absent["delta_BA"])
        reverse = a.pair_events(rows, scores, "full_after", "plain_after")
        for old, new in zip(events, reverse):
            self.assertEqual(old["fixed_rival"], new["fixed_rival"])
            self.assertAlmostEqual(old["delta_fixed_margin"], -new["delta_fixed_margin"])
        self.assertEqual(a.COMPARISONS[a.FIT_CONTROL], ("plain_before", "full_before"))
        self.assertEqual(a.COMPARISONS[a.OBJECTIVE_CONTROL], ("plain_after", "full_control_after"))

    def test_pass_bootstrap_against_explicit_resampling(self):
        rows = rows_for([0, 0, 1, 1, 1])
        y = np.array([r["label"] for r in rows])
        rng = np.random.default_rng(19)
        scores = {(c, s): rng.normal(size=(5, 8)) for c in a.CONDITIONS for s in a.SEEDS}
        result = a.full.pass_bootstrap(rows, scores, "no_m1_after", "full_after", repeats=60, seed=81)
        rng, brute, missing = np.random.default_rng(81), [], 0
        groups = [np.array([0, 1]), np.array([2, 3, 4])]
        for _ in range(60):
            indices = np.concatenate([groups[g] for g in rng.integers(0, 2, 2)])
            missing += int(len(np.unique(y[indices])) < 2)
            differences = []
            for seed in a.SEEDS:
                old, new = [a.base.metrics(scores[c, seed][indices], y[indices]) for c in ("no_m1_after", "full_after")]
                differences.append([new[m] - old[m] for m in a.METRICS])
            brute.append(np.mean(differences, axis=0))
        expected = np.quantile(brute, [.025, .975], axis=0)
        self.assertEqual(result["missing_class_replicates"], missing)
        for i, metric in enumerate(a.METRICS):
            self.assertAlmostEqual(result["metrics"][metric]["low"], expected[0, i])
            self.assertAlmostEqual(result["metrics"][metric]["high"], expected[1, i])

    def test_case_selection_modules_states_unique_vessels_and_no_flips(self):
        rows, scores = rows_for(np.resize([0, 0, 1, 1], 16)), fixture_scores(16)
        rows[-1]["mmsi"] = rows[3]["mmsi"]
        rows[0]["analysis_available"] = "vv_only"
        cases = a.select_cases(event_sets(rows, scores))
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({c["mmsi"] for c in cases}), 12)
        self.assertTrue(set(a.STATES) <= {c["selection_category"] for c in cases})
        self.assertTrue(set(a.PRIMARY) <= {c["comparison"] for c in cases})
        self.assertTrue(any(c["available"] == "vv_only" for c in cases))
        no_flips = {(c, s): scores["plain_before", s] for c in a.CONDITIONS for s in a.SEEDS}
        cases = a.select_cases(event_sets(rows, no_flips))
        self.assertEqual(len(cases), 12)
        self.assertTrue({"CC", "WW"} <= {c["selection_category"] for c in cases})
        self.assertTrue(all(c["state"] in ("CC", "WW") for c in cases))

    def test_all_conditions_rivals_reference_exclusion_and_plain_context(self):
        rows, scores = rows_for([0, 0, 1, 1]), fixture_scores(4)
        scores["no_m2_before", 43][0, 2] = 20
        events = event_sets(rows, scores)
        case = next(e for e in a.select_cases(events) if e["id"] == "0")
        refs = [dict(id="r0", label=0, mmsi="200"), dict(id="r1", label=1, mmsi="201"),
                dict(id="r2", label=2, mmsi="202"), dict(id="excluded", label=2, mmsi=rows[0]["mmsi"])]
        detail, _, _ = a.case_details(case, 0, rows[0], refs, scores, events)
        self.assertEqual(detail["visual_status"], "not_reviewed")
        self.assertIn("r2", detail["rival_reference_ids"])
        self.assertNotIn("excluded", detail["eligible_reference_ids"])
        self.assertEqual(detail["per_seed"][2]["conditions"]["no_m2_before"]["prediction"], 2)
        self.assertEqual(len(detail["per_seed"][0]["conditions"]), 12)
        self.assertIn("no reference contribution", detail["reference_usage"]["plain"])
        self.assertIn("no M3 reference transport", detail["reference_usage"]["no_m3"])

    def test_synthetic_summary_and_native_gallery(self):
        native = a.full.module_at("private_synthetic_native", a.base.PACKAGE / "sar_m123/input.py")
        with self.temporary() as directory:
            out = Path(directory)
            chip = out / "chip.png"
            a.base.Image.new("L", (37, 29), 128).save(chip)
            view = dict(image=str(chip), target_index=0, detections=[dict(
                corners=[[5, 5], [15, 5], [15, 20], [5, 20]], confidence=.9)])
            rows = rows_for([0, 0, 1, 1])
            for row in rows:
                row.update(vv=view, vh=None)
            refs = [dict(id=f"r{i}", label=i, mmsi=str(200+i), vv=view, vh=None) for i in (0, 1)]
            cohorts = dict(expanded=rows, development=rows, references=refs)
            a.full.annotate_available(cohorts, native)
            self.assertEqual(rows[0]["analysis_available"], "vv_only")
            scores = fixture_scores(4)
            combined = {(scope, c, s): score for scope in a.SCOPES for (c, s), score in scores.items()}
            summary, tables, events = a.summarize_scores(cohorts, combined, repeats=8)
            self.assertEqual(len(summary["expanded"]["comparisons"]), 12)
            self.assertEqual(len(tables["metrics"]), 72)
            cases, coverage = a.gallery(out, cohorts, scores, events, native)
            self.assertEqual(len(cases), 4)
            self.assertEqual(coverage["visual_status"], "not_reviewed")
            with a.base.Image.open(cases[0]["panels"]["vv"]) as panel:
                self.assertEqual(panel.getpixel((8, 63)), (128, 128, 128))
                self.assertEqual(panel.getpixel((44, 91)), (128, 128, 128))
                self.assertEqual(panel.getpixel((45, 91)), (255, 255, 255))
            sources = a.base.read_json(out / "cases/native_sources.json")
            self.assertTrue(all(r["pixel_scale"] == 1 for r in sources))
            self.assertTrue(all(not r["available"] for r in sources if r["pol"] == "vh"))
            self.assertTrue(all(r["plain_reference_contribution"] is False for r in sources))
            page = (out / "cases/index.html").read_text(encoding="utf-8")
            self.assertIn("Plain: NO reference contribution", page)
            a.numeric_report(out, dict(scopes=summary), len(cases))
            report = (out / "NUMERIC_REPORT.md").read_text(encoding="utf-8")
            main_table = report.split("### Main final endpoints")[1].split("### Matched subsequent objective")[0]
            self.assertNotIn("full_control", main_table)
            self.assertNotIn("_before", main_table)
            self.assertIn(a.FIT_CONTROL, report)
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
