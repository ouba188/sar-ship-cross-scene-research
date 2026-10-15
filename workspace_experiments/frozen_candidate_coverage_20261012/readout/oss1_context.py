"""Read-only OSS1 adapter for the archived task-trained DD + compact C4 model.

This is not a learned-frequency context. No source fit, feature extraction,
support sampling, or CV splitting is performed here.
"""

import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


PUBLIC = Path(__file__).resolve().parents[2] / "public_oss_comparison_20261012"
_spec = importlib.util.spec_from_file_location(
    "_oss1_frozen_public_common", PUBLIC / "common.py")
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)

from compact_fusion_source import _expert_inputs
from projection_residual_expert import discarded_coordinates


def _ship_groups(frame, ids):
    ship = frame["mmsi_or_ship_id"].astype(str)
    known = (ship.str.fullmatch(r"[0-9]{9}") & ship.ne("000000000")).to_numpy()
    return np.where(known, ship.to_numpy(str), np.char.add("sample:", ids)), known


class OSS1Context:
    """Frozen source-head inputs and support kernels in original row order."""

    def __init__(self):
        # Bound CPU BLAS without changing the caller's CV or sklearn version.
        with threadpool_limits(limits=2):
            self._load()

    def _load(self):
        original = _common.original("oss1")
        joint = _common.joint("oss1")
        geometry_path = _common.fit_dir("task_trained", "oss1") / "geometry.pkl"
        self.reference_dir = PUBLIC / "targets/task_trained/oss1"
        with geometry_path.open("rb") as stream:
            bundle = pickle.load(stream)
        self.model, self.stats, self.heads = (
            bundle[key] for key in ("model", "stats", "heads"))
        source_rows, raw, _, source_extra = _common.source_arrays("task_trained", "oss1")
        np.testing.assert_array_equal(source_rows, bundle["source_rows"])
        np.testing.assert_array_equal(source_rows, np.load(geometry_path.parent / "source_rows.npy"))
        old_source = _common.archive(joint / "source_rows.npz")
        np.testing.assert_array_equal(source_rows, old_source["source_rows"])
        source_frame = pd.read_csv(_common.SOURCE / "manifest.csv",
                                  usecols=["instance_id", "class_id"])
        self.source_ids = source_frame.instance_id.to_numpy(str)[source_rows]
        self.source_labels = source_frame.class_id.to_numpy(int)[source_rows]
        np.testing.assert_array_equal(self.source_labels, old_source["labels"][source_rows])
        old_oof = _common.archive(original / "source_oof.npz")
        np.testing.assert_array_equal(old_oof["rows"], source_rows)
        np.testing.assert_array_equal(old_oof["labels"], self.source_labels)
        np.testing.assert_array_equal(source_frame.instance_id.to_numpy(str)[old_oof["rows"]],
                                      self.source_ids)
        if len(np.unique(self.source_ids)) != len(source_rows):
            raise ValueError("Duplicate source instance IDs")

        self.hs = np.ones((len(source_rows), 193), dtype=np.float64)
        for lo in range(0, len(source_rows), _common.BATCH):
            s = slice(lo, min(lo + _common.BATCH, len(source_rows)))
            self.hs[s, :-1] = _expert_inputs(raw[source_rows[s]], source_extra[s], self.model)

        features = _common.archive(original / "features.npz")
        ids = features["sample_ids"]
        n = len(ids)
        self._raw = features["paired_r50"]
        self._extra = np.load(joint / "joint/target_extra.npy", mmap_mode="r")
        np.testing.assert_array_equal(np.load(joint / "target_sample_ids.npy"), ids)
        c4dir = _common.ROOT / "cross_dataset_c4_20261009/oss1"
        self._c4 = np.load(c4dir / "pooled_c4.npy", mmap_mode="r")
        np.testing.assert_array_equal(np.load(c4dir / "sample_ids.npy"), ids)
        policies = _common.archive(original / "policies.npz")
        np.testing.assert_array_equal(policies["sample_ids"], ids)
        frame = pd.read_csv(original / "target_manifest.csv", keep_default_na=False,
                            dtype={"mmsi_or_ship_id": str},
                            usecols=["sample_id", "class_id", "mmsi_or_ship_id"])
        np.testing.assert_array_equal(frame.sample_id.to_numpy(str), ids)
        support = policies["support_rows"]
        self.data = dict(rows=np.arange(n), ids=ids, support=support,
                         support_labels=frame.class_id.to_numpy(int)[support],
                         eval_masks=policies["eval_masks"], source_rows=source_rows)
        self.groups, self.known = _ship_groups(frame, ids)
        self._archive = _common.archive(self.reference_dir / "predictions.npz")
        for key in ("ids", "support", "eval_masks"):
            np.testing.assert_array_equal(self.data[key], self._archive[key])
        self._dd = list(self._archive["arms"]).index("DD")
        self._kind = list(self._archive["kinds"]).index("linear")
        self._probabilities = np.load(self.reference_dir / "probabilities.npy", mmap_mode="r")
        if self._dd != 6 or self._probabilities.shape != (2, 5, 7, 2, n, 8):
            raise ValueError("Unexpected archived DD/C4 probability layout")
        for labels, mask, rows in zip(self.data["support_labels"], self.data["eval_masks"], support):
            if len(np.unique(rows)) != 64 or mask[rows].any():
                raise ValueError("Changed support budget or row-disjoint query protocol")
            if (labels < 0).any() or (labels > 7).any():
                raise ValueError("Support labels outside the frozen eight classes")
        for head in self.heads.values():
            np.testing.assert_array_equal(head.classes_, np.arange(8))
        self.info = dict(dataset="oss1", task="oss1", operator="compact_c4",
                         regime="task_trained", support_budget=64, source_n=len(source_rows),
                         target_n=n, geometry_path=str(geometry_path),
                         reference_path=str(self.reference_dir), source_refit=False,
                         learned_frequency=False, support_query_protocol="historical_row_disjoint",
                         source_identity_check="frozen source rows plus manifest IDs and archived labels")
        self.ht = np.ones((n, 193), dtype=np.float64)
        self._expert_prior = np.empty((n, 8), dtype=np.float64)
        self._base_prior = np.empty((n, 8), dtype=np.float64)
        for lo in range(0, n, _common.BATCH):
            s = slice(lo, min(lo + _common.BATCH, n))
            f = self._features(s)
            self.ht[s, :-1] = f["x"]
            self._expert_prior[s], self._base_prior[s] = f["pd"], f["pb"]

    def _features(self, index):
        model, stats = self.model, self.stats
        raw, extra = self._raw[index], np.asarray(self._extra[index])
        x = _expert_inputs(raw, extra, model)
        # evaluate.py::features, retaining each cast/divisor and concatenation.
        xs = ((model["pca"].transform(raw) - model["center"]) / model["scale"]).astype(np.float32)
        rr = (xs - model["expert_center"][:128]) / model["expert_scale"][:128]
        rd = (discarded_coordinates(raw, model["pca"]) - stats["discarded_center"][:2048])
        rd = rd / stats["discarded_scale"][:2048]
        r = np.c_[np.asarray(rr / np.sqrt(192), float), rd / np.sqrt(2560)] * np.sqrt(.5)
        projected = model["pca_ssl"].transform(extra).astype(np.float32)
        retained = (projected - model["expert_center"][128:192]) / model["expert_scale"][128:192]
        complement = ((discarded_coordinates(extra, model["pca_ssl"])
                       - stats["discarded_center"][2048:]) / stats["discarded_scale"][2048:])
        t = np.c_[retained / np.sqrt(192), complement / np.sqrt(2560)] * np.sqrt(.5)
        x4 = ((model["pca"].transform(self._c4[index]) - model["center"]) / model["scale"]).astype(np.float32)
        b = (x4 - model["residual_center"]) / model["residual_scale"] / np.sqrt(128)
        return dict(x=x, r=r, t=t, b=b,
                    pd=self.heads["dual"].decision_function(x).astype(np.float32).astype(float),
                    pb=self.heads["base"].decision_function(x4).astype(np.float32).astype(float))

    def components(self, support):
        """Return expert prior/kernel and base prior/kernel, all float64.

        Accept subsets too: the caller's internal CV may use fewer than 64 rows.
        DD's T kernel is independent of the reconstruction-mapper arm; directly
        using T omits only roundoff from the archived P + (T - P) identity.
        """
        support = np.asarray(support, dtype=np.int64)
        n = len(self.data["rows"])
        ke, kb = np.empty((n, len(support))), np.empty((n, len(support)))
        pe, pb = self._expert_prior.copy(), self._base_prior.copy()
        with threadpool_limits(limits=2):
            # Match the archived support-union projection, not a new PCA batch.
            reference = np.union1d(np.unique(self.data["support"]), support)
            locations = np.searchsorted(reference, support)
            ref = self._features(reference)
            for lo in range(0, n, _common.BATCH):
                s = slice(lo, min(lo + _common.BATCH, n))
                f = self._features(s)
                ke[s] = (f["r"] @ ref["r"].T + 1. + f["t"] @ ref["t"].T)[:, locations]
                kb[s] = (f["b"] @ ref["b"].T + 1.)[:, locations]
            # The old solver used support-union projection, not query-batch
            # projection. These differ by FP32 rounding even for the same row.
            kss = ref["r"] @ ref["r"].T + 1. + ref["t"] @ ref["t"].T
            bss = ref["b"] @ ref["b"].T + 1.
            ke[support] = kss[np.ix_(locations, locations)]
            kb[support] = bss[np.ix_(locations, locations)]
            pe[support], pb[support] = ref["pd"][locations], ref["pb"][locations]
        return pe, ke, pb, kb

    def reference(self, draw):
        """Read archived DD predictions; recover the unarchived C4-only view."""
        full = self._probabilities[self._kind, draw, self._dd, 1]
        expert = self._probabilities[self._kind, draw, self._dd, 0]
        # The archive stores .5 * (base + expert), not the standalone base.
        base = 2. * full.astype(float) - expert.astype(float)
        return dict(full_predictions=self._archive["predictions"][self._kind, draw, self._dd, 1],
                    expert_predictions=self._archive["predictions"][self._kind, draw, self._dd, 0],
                    base_predictions=base.argmax(1), full_probabilities=full,
                    expert_probabilities=expert, base_probabilities=base)
