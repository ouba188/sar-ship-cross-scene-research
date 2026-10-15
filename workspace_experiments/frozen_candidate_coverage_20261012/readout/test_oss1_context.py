"""CPU-only archive replay; no training, CV selection, or result-file writes.

Run with the msfa sklearn-1.7.2 Python and -B to leave old artifacts untouched.
"""

import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import softmax
from threadpoolctl import threadpool_limits


PATH = Path(__file__).with_name("oss1_context.py")
_spec = importlib.util.spec_from_file_location("_test_frozen_oss1_adapter", PATH)
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)
from unified_head_core import coefficients


class OSS1ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.linear_model import RidgeClassifier
        # A source refit in the adapter must fail rather than silently reproduce it.
        with patch.object(RidgeClassifier, "fit", side_effect=AssertionError("No source refit allowed")):
            cls.ctx = adapter.OSS1Context()

    def test_contract_and_ship_groups(self):
        c = self.ctx
        self.assertEqual(c.info["operator"], "compact_c4")
        self.assertEqual((c.info["dataset"], c.info["task"]), ("oss1", "oss1"))
        self.assertEqual(c.hs.shape, (37343, 193))
        self.assertEqual(c.ht.shape, (1349, 193))
        for h in (c.hs, c.ht):
            self.assertEqual(h.dtype, np.dtype("float64"))
            np.testing.assert_array_equal(h[:, -1], 1.)
            self.assertTrue(np.isfinite(h).all())
        self.assertEqual(c.data["support"].shape, (5, 64))
        self.assertEqual(c.data["support_labels"].shape, (5, 64))
        self.assertEqual(c.data["eval_masks"].shape, (5, 1349))
        np.testing.assert_array_equal(c.data["rows"], np.arange(1349))
        groups, known = adapter._ship_groups(pd.DataFrame({"mmsi_or_ship_id":
            ["123456789", "000000000", "unknown", "123456789.0"]}), np.array(["a", "b", "c", "d"]))
        np.testing.assert_array_equal(known, [True, False, False, False])
        np.testing.assert_array_equal(groups, ["123456789", "sample:b", "sample:c", "sample:d"])
        self.assertEqual(c.groups.dtype.kind, "U")
        self.assertEqual(c.known.dtype, np.dtype(bool))

    def test_source_and_target_head_coordinates(self):
        c = self.ctx
        rows, raw, _, extra = adapter._common.source_arrays("task_trained", "oss1")
        s = slice(0, adapter._common.BATCH)
        np.testing.assert_array_equal(c.hs[s, :-1], adapter._expert_inputs(raw[rows[s]], extra[s], c.model))
        np.testing.assert_array_equal(c.ht[s, :-1], adapter._expert_inputs(c._raw[s], c._extra[s], c.model))
        expected = c.heads["dual"].decision_function(c.ht[:, :-1]).astype(np.float32).astype(float)
        np.testing.assert_array_equal(c._expert_prior, expected)

    def test_all_five_draws_reproduce_archived_outputs(self):
        c = self.ctx
        archived_coefficients = adapter._common.archive(c.reference_dir / "coefficients.npz")
        max_error = dict(full=0., expert=0., base_recovered=0., coefficients=0.)
        for draw, support in enumerate(c.data["support"]):
            pe, ke, pb, kb = c.components(support)
            for v, shape in ((pe, (1349, 8)), (ke, (1349, 64)),
                             (pb, (1349, 8)), (kb, (1349, 64))):
                self.assertEqual(v.shape, shape)
                self.assertEqual(v.dtype, np.dtype("float64"))
            labels = c.data["support_labels"][draw]
            ae = coefficients(ke[support], pe[support], labels, 8.)
            ab = coefficients(kb[support], pb[support], labels, 8.)
            old_alpha = archived_coefficients[f"linear_{draw}_DD"]
            # The archived model was pickled under sklearn 1.8 / NumPy 2;
            # the formal runner uses sklearn 1.7.2 / NumPy 1.26. Preserve the
            # operator and exact classes, while allowing FP32 roundoff.
            np.testing.assert_allclose(ae, old_alpha, atol=1e-7, rtol=1e-7)
            # Archived alpha identifies the old support targets independently.
            _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
            weights = len(labels) / (len(counts) * counts[inverse])
            old_targets = pe[support] + ke[support] @ old_alpha + 8. * old_alpha / weights[:, None]
            np.testing.assert_allclose(old_targets, 2. * np.eye(8)[labels] - 1., atol=2e-6, rtol=0.)
            bp = softmax(pb + kb @ ab, axis=1)
            ep = softmax(pe + ke @ ae, axis=1)
            full = .5 * (bp + ep)
            ref = c.reference(draw)
            for name, probability in (("full", full), ("expert", ep), ("base", bp)):
                np.testing.assert_array_equal(probability.argmax(1), ref[name + "_predictions"])
                np.testing.assert_array_equal(probability.argmax(1), probability.astype(np.float32).argmax(1))
                np.testing.assert_allclose(probability, ref[name + "_probabilities"], atol=4e-7, rtol=1e-6)
                key = "base_recovered" if name == "base" else name
                max_error[key] = max(max_error[key], float(np.max(np.abs(probability-ref[name+"_probabilities"]))))
            max_error["coefficients"] = max(max_error["coefficients"], float(np.max(np.abs(ae-old_alpha))))
        import sklearn
        print(json.dumps(dict(operator=c.info["operator"], dataset="oss1", source_n=len(c.hs),
                              target_n=len(c.ht), draws=5, support_budget=64,
                              sklearn=sklearn.__version__, device="cpu", source_refits=0,
                              exact_predictions_all_three_stages=True,
                              max_abs_error=max_error), indent=2), flush=True)

    def test_no_generic_run_import(self):
        sentinel = types.ModuleType("run")
        with patch.dict(sys.modules, {"run": sentinel}):
            spec = importlib.util.spec_from_file_location("_oss1_collision_probe", PATH)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertIs(sys.modules["run"], sentinel)
            self.assertTrue(hasattr(module, "OSS1Context"))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        unittest.main(verbosity=2)
