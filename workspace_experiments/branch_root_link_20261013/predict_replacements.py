"""Reuse the archived mapper/readout for metadata-fixed geometry controls."""

import sys
from pathlib import Path
import pickle

OUT = Path(__file__).resolve().parent
ROOT_CAUSE = OUT.parent / "ports_root_cause_20261013"
sys.path.insert(0, str(ROOT_CAUSE))
import settings as cfg
import slc_geometry
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    receipts = []
    for fold in cfg.FOLDS:
        fits = cfg.pilot.PREVIOUS / f"fold_{fold}/fits"
        with (fits / "geometry.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        np.testing.assert_array_equal(bundle["source_rows"], cfg.crossfit.permitted_rows(fold))
        for condition in ("original", "range_geometry", "axis_sham"):
            # Materialize before configuring evaluate: data loaders reset its globals.
            data = (cfg.target("oss2", "original") if condition == "original"
                    else slc_geometry.data_for("oss2", condition))
            ev = cfg.crossfit.configure_evaluation(fold)
            ev.OUT = OUT / "replacement_predictions" / condition / f"fold_{fold}"
            ev.fit_dir = lambda regime, name: fits
            ev.mapper_path = lambda regime, name: fits / "mapper.npz"
            ev.target_arrays = lambda regime, name, grdh_only=False: data
            ev.predict("shared_frozen", "oss2", bundle)
            dst = ev.OUT / "targets/shared_frozen/oss2"
            saved = cfg.archive(dst / "predictions.npz")
            for key in ("ids", "support", "eval_masks"):
                np.testing.assert_array_equal(saved[key], data[key])
            error = None
            if condition == "original":
                old = cfg.INPUT / "predictions/original" / f"fold_{fold}/targets/shared_frozen/oss2"
                previous = cfg.archive(old / "predictions.npz")
                np.testing.assert_array_equal(saved["predictions"], previous["predictions"])
                p = np.load(dst / "probabilities.npy", mmap_mode="r")
                q = np.load(old / "probabilities.npy", mmap_mode="r")
                error = float(np.max(np.abs(p-q)))
                np.testing.assert_allclose(p, q, atol=2e-7, rtol=0.)
            receipts.append(dict(fold=fold, condition=condition, original_probability_max_error=error))
    cfg.write(OUT / "REPLACEMENT_PREDICTIONS_FROZEN.json", dict(
        contexts=6, source_version="archive", new_neural_or_mapper_training=0,
        new_source_fits=0, support_coefficients_resolved=True,
        query_metrics_evaluated=False, replays=receipts))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
